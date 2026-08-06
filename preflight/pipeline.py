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
from preflight.chunking import Window, build_windows
from preflight.config import Settings
from preflight.ingest.pipeline import Ingested, ingest
from preflight.models import AgentResult, Finding
from preflight.perception import accessibility, audio, audio_intel, metadata, ocr, vision
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


def _guard(agent_id: str, name: str, fn: Callable[[], AgentResult]) -> AgentResult:
    started = time.perf_counter()
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - degrade, never fail
        result = AgentResult.failed(agent_id, name, f"{type(exc).__name__}: {exc}")
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return result


def run_perception(
    source: Path,
    store: cas.Store,
    *,
    asr_model: str = asr_mod.DEFAULT_MODEL,
    skip_speech: bool = False,
    settings: Settings | None = None,
) -> PipelineResult:
    source = Path(source)
    started_at = time.perf_counter()
    settings = settings or Settings.load()

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

    if skip_speech:
        speech_agent = AgentResult.skipped("speech", "Speech Agent", "disabled by --no-speech")
        transcript = None
        quotation_spans: list = []
        framing_cues: list = []
    else:
        speech_agent, transcript, quotation_spans, framing_cues = _speech(
            ingested, store, asr_model
        )

    audio_agent = _guard(
        "audio",
        "Audio Agent",
        lambda: _audio(source, ingested),
    )

    access_agent = _guard(
        "access",
        "Accessibility Agent",
        lambda: accessibility.analyse(source, ingested.meta.durationMs, transcript),
    )

    meta_agent = _guard(
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

    vision_agent, tracks = _vision(ingested, registry, flagged)
    ocr_agent, ocr_report = _ocr(ingested, registry, transcript)

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
        windows, store, settings, transcript, cross_modal
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

    scoring_agent = AgentResult(
        agent_id="score",
        name="Scoring Agent",
        status="OK",
        log=fusion_log
        + [
            f"{len(all_findings)} finding(s) fused · "
            f"readiness {compute_readiness(sub_scores(all_findings)).overall}"
        ],
    )
    agents.append(scoring_agent)

    orchestrator.elapsed_ms = int((time.perf_counter() - started_at) * 1000)

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
    )


def _policy(
    windows: list[Window],
    store: cas.Store,
    settings: Settings | None,
    transcript: Transcript | None = None,
    cross_modal: CrossModalContext | None = None,
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
        indexes = build_scoped_indexes(corpus, settings, store, client)
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
        )
        agent = to_agent_result(result)
        agent.log = indexes.log + agent.log
        agent.calls += indexes.calls
        return agent

    return _guard("policy", "Policy Agent", run), corpus, backend


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
    ingested: Ingested,
    registry: Registry | None,
    flagged: list[tuple[int, int]],
) -> tuple[AgentResult, list[vision.Track]]:
    def run() -> AgentResult:
        agent, tracks = vision.analyse(
            ingested.keyframes, registry, flagged_spans=flagged
        )
        run.tracks = tracks  # type: ignore[attr-defined]
        return agent

    run.tracks = []  # type: ignore[attr-defined]
    agent = _guard("vision", "Vision Agent", run)
    return agent, list(getattr(run, "tracks", []))


def _ocr(
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
    agent = _guard("ocr", "OCR Agent", run)
    return agent, getattr(run, "report", ocr.OcrReport())


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
    ingested: Ingested, store: cas.Store, asr_model: str
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
    agent = _guard("speech", "Speech Agent", run)
    return (
        agent,
        getattr(run, "transcript", None),
        getattr(run, "quotation_spans", []),
        getattr(run, "framing_cues", []),
    )

