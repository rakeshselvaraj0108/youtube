"""Pipeline orchestration.

Phase 2 wires ingest plus the four offline perception agents. Retrieval, the
adversarial triad, fusion, scoring and remediation land in later phases behind
the same `PipelineResult`.

The scheduling rule that matters: only `ingest` and `speech` are load-bearing.
Every other agent runs inside a guard, and a failure becomes a DEGRADED or
FAILED AgentResult with a reason rather than a traceback. A tool that dies on a
missing optional dependency loses on functionality.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from preflight import cas
from preflight.agents.nim import NimClient
from preflight.agents.triad import CrossModalContext, run_triad, to_agent_result
from preflight.budget import CallBudget
from preflight.chunking import Window, build_windows
from preflight.config import Settings
from preflight.ingest.pipeline import Ingested, ingest
from preflight.models import AgentResult, Finding
from preflight.orchestrator import Orchestrator
from preflight.perception import (
    accessibility,
    audio,
    audio_intel,
    disclosure,
    metadata,
    ocr,
    quality,
    vision,
)
from preflight.perception import asr as asr_mod
from preflight.perception import speech_intel
from preflight.perception.asr import Transcript
from preflight.providers.registry import Registry
from preflight.policy.corpus import Corpus, load_corpus
from preflight.policy.index import build_scoped_indexes
from preflight.scoring.fusion import apply_fusion
from preflight.scoring.readiness import Readiness, compute_readiness, sub_scores

# id -> (tier, parents). Mirrors the DAG the UI's agent flow renders.
TOPOLOGY: dict[str, tuple[int, list[str]]] = {
    "orchestrator": (0, []),
    "ingest": (1, ["orchestrator"]),
    "speech": (2, ["ingest"]),
    "vision": (2, ["ingest"]),
    "audio": (2, ["ingest"]),
    "meta": (2, ["ingest"]),
    "ocr": (3, ["vision"]),
    "access": (3, ["speech"]),
    "policy": (4, ["speech", "vision", "ocr", "audio"]),
    "score": (5, ["policy", "access", "meta", "audio"]),
    "remedy": (6, ["score"]),
    "report": (7, ["remedy"]),
}

# Share of the analysis surface each agent is responsible for. Coverage is a
# weighted mean over these, so a degraded vision agent costs far more than a
# degraded report writer.
#
# Deliberately does NOT include "remedy" or "report", even though both are
# real TOPOLOGY stages a judge sees in the agent-flow diagram. `run_perception`
# — everything `compute_coverage` is measuring — never populates either: A12
# runs from `preflight fix`, and the report is written by `_emit` after this
# score is already computed. Carrying them here with non-zero weight meant
# `check` could never structurally report 100% coverage, even for a video
# where every applicable agent succeeded — 3% of the denominator was
# permanently unreachable. Their weight folds into "score", the actual
# synthesis stage of what `check` produces.
SURFACE_WEIGHT: dict[str, float] = {
    "orchestrator": 0.0,
    "ingest": 0.05,
    "speech": 0.20,
    "vision": 0.22,
    "ocr": 0.13,
    "audio": 0.15,
    "access": 0.06,
    "meta": 0.04,
    "policy": 0.10,
    "score": 0.05,
}


@dataclass
class PipelineResult:
    source: Path
    ingested: Ingested
    transcript: Transcript | None
    agents: list[AgentResult] = field(default_factory=list)
    started_at: float = 0.0
    windows: list[Window] = field(default_factory=list)
    corpus: Corpus | None = None
    retrieval_backend: str = "none"
    fusion_log: list[str] = field(default_factory=list)
    visual_tracks: list = field(default_factory=list)
    ocr_items: list = field(default_factory=list)
    budget: CallBudget = field(default_factory=lambda: CallBudget())

    @property
    def sub_scores(self) -> dict[str, float]:
        return sub_scores(self.findings)

    @property
    def readiness(self) -> Readiness:
        return compute_readiness(self.sub_scores)

    @property
    def findings(self) -> list[Finding]:
        out: list[Finding] = []
        for agent in self.agents:
            out.extend(agent.findings)
        return out

    @property
    def coverage(self) -> float:
        return compute_coverage(self.agents)

    @property
    def temporal_coverage(self):
        """Where on the timeline each modality actually looked.

        The scalar above says how much of its own sample set an agent got
        through; it cannot say *which minutes* those samples came from. On a
        long upload those are different questions, and only the second one
        supports an absence claim about a particular stretch of video.
        """
        from preflight import coverage as coverage_mod

        frames = self.ingested.keyframes if self.ingested else []
        evidence: dict[str, Any] = {}
        # Only modalities that actually ran contribute a row. An agent that
        # never ran must be absent rather than shown as a line of zeroes —
        # "did not run" and "ran and saw nothing" are different facts.
        if frames:
            evidence["frames"] = frames
        if self.ocr_items:
            evidence["ocr"] = self.ocr_items
        # Frames vision inspected, not tracks it produced. A frame examined
        # and found empty is still coverage of that moment; keying this on
        # output would report a working vision pass over a clean stretch of
        # video as though it had never run.
        vision_agent = self.agent("vision")
        examined = (vision_agent.artifacts or {}).get("examined_ms") if vision_agent else None
        if examined:
            evidence["vision"] = [{"ts_ms": ts} for ts in examined]
        segments = getattr(self.transcript, "segments", None) if self.transcript else None
        if segments:
            evidence["speech"] = segments
        return coverage_mod.build(
            self.ingested.meta.durationMs if self.ingested else 0, evidence
        )

    @property
    def total_calls(self) -> int:
        return sum(agent.calls for agent in self.agents)

    def agent(self, agent_id: str) -> AgentResult | None:
        return next((a for a in self.agents if a.agent_id == agent_id), None)


def compute_coverage(agents: list[AgentResult]) -> float:
    """Weighted mean of per-agent coverage over the analysis surface."""
    seen = {agent.agent_id: agent for agent in agents}
    weight_sum = 0.0
    accumulated = 0.0
    for agent_id, weight in SURFACE_WEIGHT.items():
        if weight == 0.0:
            continue
        weight_sum += weight
        result = seen.get(agent_id)
        # An agent that never ran contributes zero coverage, not zero weight.
        # Silently dropping it would let a report that skipped vision entirely
        # still claim 100% coverage.
        accumulated += weight * (result.coverage if result else 0.0)
    return accumulated / weight_sum if weight_sum else 1.0


def _guard(
    orch: Orchestrator, agent_id: str, name: str, fn: Callable[[], AgentResult]
) -> AgentResult:
    """Run one optional stage through the real orchestrator.

    This used to catch-and-degrade on its own, duplicating exactly what
    `Orchestrator.run_stage` already does — retry on a transient failure,
    record a `StageOutcome`, degrade rather than raise — except without the
    retry or the record. `preflight/orchestrator.py` was fully built and
    tested but never driven by a real run; routing through it here means the
    retry budget and the execution timeline it was designed to produce are no
    longer decorative. `required` is left for the orchestrator to derive from
    `REQUIRED_STAGES` itself (true only for "speech" among `_guard`'s
    callers) — this function never overrides that.
    """
    return orch.run_stage(agent_id, fn, name=name)


def run_perception(
    source: Path,
    store: cas.Store,
    *,
    asr_model: str = asr_mod.DEFAULT_MODEL,
    skip_speech: bool = False,
    settings: Settings | None = None,
    budget: CallBudget | None = None,
    on_event: Callable[[dict], None] | None = None,
) -> PipelineResult:
    source = Path(source)
    started_at = time.perf_counter()
    settings = settings or Settings.load()
    budget = budget or CallBudget()

    orch = Orchestrator(max_attempts=2, on_event=on_event)

    if on_event is not None:
        # The plan first, so a viewer sees the shape of the work before any
        # of it happens rather than inferring it from what arrives.
        on_event(
            {
                "type": "run.start",
                "source": source.name,
                "topology": [
                    {"id": key, "tier": tier, "parents": parents}
                    for key, (tier, parents) in TOPOLOGY.items()
                ],
            }
        )
        # The orchestrator is scheduling from here until the last stage
        # settles, so it shows as working rather than as a node that never
        # lit up. `remedy` and `report` are deliberately absent: A12 runs
        # from `preflight fix` and the report is written after this returns,
        # and inventing progress for stages that are not running is exactly
        # the dishonesty the rest of this pipeline refuses.
        on_event(
            {
                "type": "stage.start",
                "stage": "orchestrator",
                "agentId": "A01",
                "name": "Orchestrator",
                "startedMs": 0,
            }
        )

    orchestrator = AgentResult(
        agent_id="orchestrator",
        name="Orchestrator",
        log=[f"scheduling {len(TOPOLOGY)} agents · 30 rpm token bucket"],
    )

    ingested = ingest(source, store)
    ingest_agent = AgentResult(
        agent_id="ingest",
        name="Video Processing",
        status="OK",
        elapsed_ms=ingested.elapsed_ms,
        log=list(ingested.log),
        artifacts={"keyframes": len(ingested.keyframes), "cached": ingested.cached},
    )
    if ingested.profile is not None:
        ingest_agent.artifacts["profile"] = ingested.profile.to_json()

    # Picture quality, motion and thumbnail candidates — one decode, bounded
    # by a fixed sample budget rather than by duration, so a thirty-minute
    # upload costs what a two-minute clip costs. Guarded like every optional
    # stage: a file that will not sample degrades the technical profile, it
    # does not fail the run.
    intel = _guard(
        orch,
        "ingest_quality",
        # Distinct from "Video Processing" on purpose. Both fold into the
        # ingest node, but the live event stream lists them separately, and
        # two stages sharing a display name reads as the same agent running
        # twice — which is exactly the duplicate-decode bug this project
        # already had, so it should not be simulated by a label.
        "Picture Quality",
        lambda: _quality(source, ingested.meta.durationMs),
    )
    if intel.artifacts.get("intelligence"):
        ingest_agent.artifacts.update(intel.artifacts["intelligence"])
        ingest_agent.log.extend(intel.log)
    # Handed to vision so the frame budget follows the motion. Kept out of
    # the artifacts dict deliberately: it is a per-sample numpy array, and
    # the artifacts go into report.json.
    _motion_signal = intel.artifacts.get("motion_signal")
    # Ingest itself stays unwrapped above — its specific exceptions
    # (FileNotFoundError, UnsupportedInput, FfmpegFailed) are caught by name at
    # the CLI boundary, and a retry loop swallowing them into a generic FAILED
    # result would break that contract. It only reaches here once it has
    # already succeeded, so this call registers the outcome in the timeline
    # without repeating or risking any work.
    orch.run_stage("ingest", lambda: ingest_agent, required=True, name="Video Processing")

    if skip_speech:
        speech_agent = AgentResult.skipped("speech", "Speech Agent", "disabled by --no-speech")
        orch.run_stage("speech", lambda: speech_agent, required=True, name="Speech Agent")
        transcript = None
        quotation_spans: list = []
        framing_cues: list = []
    else:
        speech_agent, transcript, quotation_spans, framing_cues = _speech(
            orch, ingested, store, asr_model
        )

    audio_agent = _guard(
        orch,
        "audio",
        "Audio Agent",
        lambda: _audio(source, ingested),
    )

    access_agent = _guard(
        orch,
        "access",
        "Accessibility Agent",
        lambda: accessibility.analyse(source, ingested.meta.durationMs, transcript),
    )

    meta_agent = _guard(
        orch,
        "meta",
        "Metadata Agent",
        lambda: metadata.analyse(source, ingested.meta.durationMs, transcript),
    )

    # Vision and OCR carry 35% of the analysis surface between them. Both
    # resolve their own capability, and both report SKIPPED with a reason
    # rather than failing when it is unavailable — which is the whole point of
    # coverage being reported instead of assumed.
    registry = _registry(settings)
    flagged = _flagged_spans(speech_agent)

    vision_agent, tracks = _vision(orch, ingested, registry, flagged, _motion_signal)
    ocr_agent, ocr_report = _ocr(orch, ingested, registry, transcript)

    # Policy grounding + adversarial adjudication.
    windows = build_windows(
        transcript,
        ingested.meta.durationMs,
        ingested.keyframes,
        chunk_ms=settings.chunk_ms if settings else 30_000,
        overlap_ms=settings.overlap_ms if settings else 5_000,
        ocr_items=ocr_report.items,
    )
    cross_modal = CrossModalContext(
        quotation_spans=quotation_spans,
        framing_cues=framing_cues,
        visual_tracks=tracks,
        vision_coverage=vision_agent.coverage,
        declared_category=str(meta_agent.artifacts.get("category", "")),
        declared_audience=str(meta_agent.artifacts.get("declared_audience", "")),
        music_deliberate_bed=(
            audio_agent.artifacts.get("quality", {}).get("ducking", {}).get("deliberate_bed")
        ),
    )
    policy_agent, corpus, backend = _policy(
        orch, windows, store, settings, transcript, cross_modal, registry, budget
    )

    agents = [
        orchestrator,
        ingest_agent,
        speech_agent,
        vision_agent,
        audio_agent,
        ocr_agent,
        access_agent,
        meta_agent,
        policy_agent,
    ]

    # Fusion runs across every agent's findings at once — corroboration is only
    # meaningful when the modalities can see each other.
    per_agent_coverage = {a.agent_id: a.coverage for a in agents}
    all_findings = [f for agent in agents for f in agent.findings]
    fusion_log = apply_fusion(all_findings, per_agent_coverage)

    # Fusion and scoring are real work with a real duration, and they were the
    # one stage doing something a viewer would want to watch that never said
    # so. Routed through the orchestrator like everything else rather than
    # emitted by hand, so it lands in the timeline too.
    scoring_agent = orch.run_stage(
        "score",
        lambda: AgentResult(
            agent_id="score",
            name="Scoring Agent",
            status="OK",
            log=fusion_log
            + [
                f"{len(all_findings)} finding(s) fused · "
                f"readiness {compute_readiness(sub_scores(all_findings)).overall}"
            ],
        ),
        required=False,
        name="Scoring Agent",
    )
    agents.append(scoring_agent)

    orchestrator.elapsed_ms = int((time.perf_counter() - started_at) * 1000)

    if on_event is not None:
        on_event(
            {
                "type": "stage.end",
                "stage": "orchestrator",
                "agentId": "A01",
                "name": "Orchestrator",
                "status": "OK",
                "coverage": 1.0,
                "elapsedMs": orchestrator.elapsed_ms,
                "attempts": 1,
                "findings": 0,
                "calls": 0,
                "detail": f"{len(orch.report.timeline)} stage(s) scheduled",
            }
        )

    orchestrator.artifacts["pipeline_id"] = orch.report.pipeline_id
    orchestrator.artifacts["timeline"] = orch.timeline_json()
    orchestrator.log.append(
        f"{len(orch.report.timeline)} stage(s) recorded, "
        f"{sum(o.attempts for o in orch.report.timeline) - len(orch.report.timeline)} retried"
    )

    return PipelineResult(
        source=source,
        ingested=ingested,
        transcript=transcript,
        agents=agents,
        started_at=started_at,
        windows=windows,
        corpus=corpus,
        retrieval_backend=backend,
        fusion_log=fusion_log,
        visual_tracks=tracks,
        ocr_items=list(ocr_report.items),
        budget=budget,
    )


def _policy(
    orch: Orchestrator,
    windows: list[Window],
    store: cas.Store,
    settings: Settings | None,
    transcript: Transcript | None = None,
    cross_modal: CrossModalContext | None = None,
    registry: Registry | None = None,
    budget: CallBudget | None = None,
) -> tuple[AgentResult, Corpus | None, str]:
    """Retrieval plus the triad, guarded like every other optional stage."""
    settings = settings or Settings.load()
    corpus: Corpus | None = None
    backend = "none"

    def run() -> AgentResult:
        nonlocal corpus, backend
        corpus = load_corpus(settings.policy_dir)
        client = NimClient(settings, store)

        # One index per scope. The triad searches only the advertiser-friendly
        # clauses: retrieving the paid-promotion clause for a transcript window
        # can never be correct, and it occupies a slot the adjudicator needs.
        indexes = build_scoped_indexes(
            corpus, settings, store, client, registry=registry
        )
        backend = indexes.backend

        policy_index = indexes.get("policy")
        if policy_index is None:
            agent = AgentResult.skipped(
                "policy", "Policy Agent", "no policy-scoped clauses in the corpus"
            )
            agent.log = indexes.log + agent.log
            return agent

        result = run_triad(
            windows,
            corpus.scoped("policy"),
            policy_index,
            client,
            settings,
            transcript,
            cross_modal=cross_modal,
            budget=budget,
        )
        agent = to_agent_result(result)
        agent.log = indexes.log + agent.log
        agent.calls += indexes.calls
        return agent

    return _guard(orch, "policy", "Policy Agent", run), corpus, backend


def _quality(source: Path, duration_ms: int) -> AgentResult:
    """Picture quality, motion and thumbnail candidates from one decode.

    Folded into the ingest node rather than given its own agent: it answers
    technical questions about the file, which is what A01 already reports,
    and the roster does not declare a thirteenth agent.
    """
    # The event stream reports `result.name`, not the label `_guard` was
    # given, so the rename has to happen here to take effect.
    result = AgentResult(agent_id="ingest_quality", name="Picture Quality")
    intel = quality.analyse(source, duration_ms)
    if intel is None:
        result.status = "SKIPPED"
        result.error = "could not sample frames for quality analysis"
        result.log.append(result.error)
        return result

    result.artifacts["intelligence"] = intel.to_json()
    result.artifacts["motion_signal"] = intel.signal
    result.log.append(
        f"picture: {intel.quality.blur_label} · "
        f"brightness {intel.quality.brightness:.0f} · "
        f"contrast {intel.quality.contrast:.0f} · "
        f"{intel.quality.frames_sampled} frames sampled"
    )
    result.log.append(
        f"motion: {intel.motion.scene_count} scene(s) · "
        f"{len(intel.motion.hard_cuts)} hard cut(s), "
        f"{len(intel.motion.gradual_transitions)} gradual"
    )
    return result


def _registry(settings: Settings | None) -> Registry | None:
    """One registry for the run, or none if it cannot be constructed.

    Built here rather than inside each agent because resolution is the
    expensive part — probing for tesseract, checking a key's shape, deciding
    whether Qdrant answers — and doing it twice would double the startup cost
    to reach the same answer. Agents receive it and ask for capabilities; they
    still never see a credential.
    """
    try:
        return Registry(offline=bool(settings and settings.offline))
    except Exception:  # noqa: BLE001 - a registry that cannot resolve is a skip
        return None


def _flagged_spans(speech_agent: AgentResult) -> list[tuple[int, int]]:
    """Spans the text layer already found something in.

    Vision pays per frame, so the frames worth paying for are the ones another
    modality has pointed at, plus a uniform baseline. Passing these through is
    what turns A03's frame gating from a claim into a saving.
    """
    return [
        (f.startMs, f.endMs)
        for f in speech_agent.findings
        if f.endMs > f.startMs
    ]


def _audio(source: Path, ingested: Ingested) -> AgentResult:
    """A04, both tiers.

    `audio.analyse` produces the findings — loudness, clipping, dead air, a
    dead channel, a music bed. `audio_intel.analyse` produces the acoustic
    evidence underneath them: segmentation, transients, hum, tempo, and the
    ducking measurement that says whether a bed was placed on a timeline or
    picked up in the room. One agent node in the topology, so the second tier
    folds its artifacts and log into the first rather than inventing a
    thirteenth agent the roster does not declare.
    """
    result = audio.analyse(ingested.fingerprint_wav, source, ingested.meta.durationMs)
    if result.status in {"SKIPPED", "FAILED"}:
        return result

    intel, events, segments = audio_intel.analyse(ingested.fingerprint_wav, source)
    if intel.status != "OK":
        result.log.append(f"acoustic analysis unavailable: {intel.error or intel.status}")
        return result

    result.artifacts.update(
        {
            "acoustic_events": intel.artifacts.get("events", []),
            "segments": intel.artifacts.get("segments", []),
            "quality": intel.artifacts.get("quality", {}),
            "unresolvable_without_classifier": intel.artifacts.get(
                "unresolvable_without_classifier", []
            ),
        }
    )
    result.log.extend(intel.log)
    result.elapsed_ms += intel.elapsed_ms
    return result


def _vision(
    orch: Orchestrator,
    ingested: Ingested,
    registry: Registry | None,
    flagged: list[tuple[int, int]],
    motion: Any = None,
) -> tuple[AgentResult, list[vision.Track]]:
    def run() -> AgentResult:
        agent, tracks = vision.analyse(
            ingested.keyframes,
            registry,
            flagged_spans=flagged,
            # Free: the quality pass already produced this from a decode the
            # run paid for, so concentrating the vision budget where the
            # picture moves costs arithmetic rather than another decode.
            motion=motion,
            duration_ms=ingested.meta.durationMs,
        )
        run.tracks = tracks  # type: ignore[attr-defined]
        return agent

    run.tracks = []  # type: ignore[attr-defined]
    agent = _guard(orch, "vision", "Vision Agent", run)
    return agent, list(getattr(run, "tracks", []))


def _ocr(
    orch: Orchestrator,
    ingested: Ingested,
    registry: Registry | None,
    transcript: Transcript | None,
) -> tuple[AgentResult, ocr.OcrReport]:
    def run() -> AgentResult:
        agent, report = ocr.analyse(
            ingested.keyframes,
            registry,
            duration_ms=ingested.meta.durationMs,
            transcript=_transcript_segments(transcript),
        )
        run.report = report  # type: ignore[attr-defined]
        return agent

    run.report = ocr.OcrReport()  # type: ignore[attr-defined]
    agent = _guard(orch, "ocr", "OCR Agent", run)
    report = getattr(run, "report", ocr.OcrReport())

    # A05 reads the text; nothing looked at what the text *was*. A creator's
    # API key legible in a terminal behind a demo is invisible to every other
    # agent — speech never hears it, vision has no vocabulary term for it —
    # and it is the one finding here whose consequence lands within hours of
    # upload rather than at review time.
    if agent.status not in {"FAILED"} and report.items:
        found = disclosure.analyse(report.items)
        if found:
            agent.findings.extend(disclosure.to_findings(found))
            agent.artifacts["disclosures"] = [d.to_json() for d in found]
            agent.log.append(
                f"{len(found)} sensitive string(s) on screen: "
                + ", ".join(sorted({d.kind for d in found}))
            )

    return agent, report


def _transcript_segments(transcript: Transcript | None) -> list[dict] | None:
    """Transcript in the shape A05's caption correlation expects."""
    if transcript is None:
        return None
    segments = getattr(transcript, "segments", None) or []
    return [
        {
            "start_ms": int(getattr(s, "start_ms", getattr(s, "startMs", 0))),
            "end_ms": int(getattr(s, "end_ms", getattr(s, "endMs", 0))),
            "text": str(getattr(s, "text", "")),
        }
        for s in segments
    ]


def _speech(
    orch: Orchestrator, ingested: Ingested, store: cas.Store, asr_model: str
) -> tuple[AgentResult, Transcript | None, list, list]:
    """A02, both tiers.

    `asr_mod.transcribe` produces the transcript — words, timing, language.
    `speech_intel.analyse` produces the intelligence layer on top of it:
    lexicon-matched events (profanity, PII, sensitive-event mentions, twelve
    categories in all), quotation spans, and framing cues (EDSA + harm
    reduction). The second tier existed as a 722-line module with its own
    test suite and was never called from here — `speech_intel` was imported
    for a helper function elsewhere in this file and nothing ever reached
    `analyse()`. Folded into one agent node, the same way A04's acoustic tier
    folds into `audio`, rather than inventing a thirteenth agent the roster
    does not declare.
    """

    def run() -> AgentResult:
        agent, transcript = asr_mod.transcribe(
            ingested.asr_wav,
            store,
            model_id=asr_model,
            duration_ms=ingested.meta.durationMs,
        )
        run.transcript = transcript  # type: ignore[attr-defined]

        if agent.status not in {"SKIPPED", "FAILED"} and transcript is not None:
            intel, _events, spans, framing = speech_intel.analyse(transcript)
            run.quotation_spans = spans  # type: ignore[attr-defined]
            run.framing_cues = framing  # type: ignore[attr-defined]
            if intel.status == "OK":
                agent.artifacts.update(intel.artifacts)
                agent.log.extend(intel.log)
                agent.elapsed_ms += intel.elapsed_ms

        return agent

    run.transcript = None  # type: ignore[attr-defined]
    run.quotation_spans = []  # type: ignore[attr-defined]
    run.framing_cues = []  # type: ignore[attr-defined]
    agent = _guard(orch, "speech", "Speech Agent", run)
    return (
        agent,
        getattr(run, "transcript", None),
        getattr(run, "quotation_spans", []),
        getattr(run, "framing_cues", []),
    )

