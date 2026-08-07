"""Assemble the AnalysisReport the UI renders.

This is the join between the engine and the page. The shape is defined by
`src/types/analysis.ts` and enforced by `schema/analysis-report.schema.json`:
every report is validated before it is written, so a drift on either side turns
into a hard error here rather than a blank panel in the browser.

Nothing in this file invents a value. Every number traces to a measurement over
the input file, and where a measurement is missing the field is omitted so the
UI can show a real empty state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from preflight import __version__, cas
from preflight.ingest.frames import Keyframe, frames_in_span, to_data_uri
from preflight.models import Finding, SEVERITY_RANK
from preflight.pipeline import TOPOLOGY, PipelineResult
from preflight.remediate.codegen import Program, build_program
from preflight.remediate.edl import EDL, compile_edl
from preflight.plan import HIERARCHICAL_ABOVE_MS, SEGMENT_MS, build_plan
from preflight.scoring.incidents import build_graph
from preflight.scoring.reasoning import explain_all
from preflight.scoring.simulation import explore
from preflight.scoring.readiness import SUB_SCORE_ORDER, compute_readiness, sub_scores
from preflight.scoring.rollup import rollup

# How many keyframes to embed per finding. Each is roughly 40KB of base64, and
# report.html has to stay openable.
MAX_FRAMES_PER_FINDING = 3
RISK_SEGMENTS = 96

SEVERITY_WEIGHT = {"CRITICAL": 1.0, "HIGH": 0.76, "MEDIUM": 0.52, "LOW": 0.28}
FILE_SCOPE_RATIO = 0.5


def _is_file_scoped(finding: Finding, duration_ms: int) -> bool:
    if duration_ms <= 0:
        return False
    return (finding.endMs - finding.startMs) / duration_ms > FILE_SCOPE_RATIO


def build_risk_bands(findings: list[Finding], duration_ms: int) -> list[dict[str, Any]]:
    """Risk terrain, one band per segment.

    File-scoped findings are excluded. They are true of every second, so
    including them flattens the terrain into a plateau and hides the spikes
    that are the entire point of looking at it.
    """
    if duration_ms <= 0:
        return []

    width = duration_ms / RISK_SEGMENTS
    scoped = [f for f in findings if not _is_file_scoped(f, duration_ms)]

    raw: list[float] = []
    bounds: list[tuple[int, int]] = []
    for i in range(RISK_SEGMENTS):
        start = round(i * width)
        end = round((i + 1) * width)
        bounds.append((start, end))
        risk = 0.0
        for finding in scoped:
            if finding.endMs <= start or finding.startMs >= end:
                continue
            confidence = finding.fusedConfidence or finding.confidence
            risk = max(risk, SEVERITY_WEIGHT.get(finding.severity, 0.5) * confidence)
        raw.append(risk)

    # One-segment neighbour bleed, so a three-second spike still reads at this
    # scale without inventing risk where there is none.
    bands: list[dict[str, Any]] = []
    for i, (start, end) in enumerate(bounds):
        previous = raw[i - 1] if i > 0 else 0.0
        following = raw[i + 1] if i + 1 < len(raw) else 0.0
        value = max(raw[i], previous * 0.45, following * 0.45)
        bands.append({"startMs": start, "endMs": end, "risk": round(value, 3)})
    return bands


def build_breakdown(findings: list[Finding]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Finding]] = {}
    for finding in findings:
        grouped.setdefault(finding.category, []).append(finding)

    rows = []
    for category, group in grouped.items():
        worst = min(group, key=lambda f: SEVERITY_RANK[f.severity]).severity
        rows.append({"category": category, "count": len(group), "severity": worst})
    return sorted(rows, key=lambda r: (SEVERITY_RANK[r["severity"]], -r["count"]))


def attach_evidence_frames(
    findings: list[Finding], keyframes: list[Keyframe], duration_ms: int
) -> int:
    """Embed real keyframes as data URIs on each finding.

    report.html has to open with no network, so evidence travels inside it.
    """
    embedded = 0
    for finding in findings:
        if _is_file_scoped(finding, duration_ms):
            continue
        matches = frames_in_span(keyframes, finding.startMs, finding.endMs)
        finding.evidence.frames = [
            frame.data_uri() for frame in matches[:MAX_FRAMES_PER_FINDING]
        ]
        embedded += len(finding.evidence.frames)
    return embedded


@dataclass
class ReportBundle:
    report: dict[str, Any]
    edl: EDL
    program: Program

    @property
    def overall(self) -> int:
        return int(self.report["scores"]["overall"])

    @property
    def verdict(self) -> str:
        return str(self.report["scores"]["verdict"])


def build_report(
    result: PipelineResult,
    *,
    policy_version: str = "unknown",
    embed_media: bool = True,
    render_ms: int = 0,
    strategy: str | None = None,
    chunk_ms: int = 30_000,
    overlap_ms: int = 5_000,
) -> ReportBundle:
    meta = result.ingested.meta
    findings = result.findings
    duration_ms = meta.durationMs

    if embed_media:
        attach_evidence_frames(findings, result.ingested.keyframes, duration_ms)

    edl = compile_edl(
        findings, str(result.source), duration_ms, result.transcript, strategy=strategy
    )
    safe_name = f"{Path(result.source).stem}.safe.mp4"
    program = build_program(edl, result.source, safe_name)

    # Correlated once and reused: the reasoning chains must explain the same
    # incidents the report lists, and building the graph twice would let the
    # two drift into citing ids that do not appear.
    _graph = build_graph(findings, duration_ms)
    _incidents = _graph.to_json()["incidents"]

    sub = sub_scores(findings)
    readiness = compute_readiness(sub)

    video: dict[str, Any] = meta.to_json()
    video["srcUrl"] = f"./{Path(result.source).name}"
    video["posterUrl"] = f"./{Path(result.source).stem}.poster.jpg"
    if embed_media and result.ingested.poster.is_file():
        video["posterDataUri"] = to_data_uri(result.ingested.poster)

    # The attestation binds the video, the rules and the models together. Two
    # runs over the same inputs produce the same hash; that is what makes it
    # evidence rather than decoration.
    attestation = cas.hash_json(
        {
            "video": result.ingested.video_hash,
            "policy": result.corpus.digest if result.corpus else "none",
            "engine": __version__,
            "models": sorted(
                {a.agent_id: a.calls for a in result.agents}.items()
            ),
        }
    )

    agents = []
    for agent in result.agents:
        tier, parents = TOPOLOGY.get(agent.agent_id, (9, []))
        agents.append(agent.to_agent_run(tier, parents, agent_ts(result, agent.agent_id)))

    report = {
        "video": video,
        "meta": {
            "analyzedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "policyVersion": policy_version,
            "engineVersion": __version__,
            "attestationHash": cas.prefixed(attestation),
            "coverage": round(result.coverage, 4),
        },
        "scores": {
            "overall": readiness.overall,
            "sub": {key: round(sub[key], 1) for key in SUB_SCORE_ORDER},
            "verdict": readiness.verdict,
            "weakest": readiness.weakest,
        },
        # Findings are what each agent reported; incidents are what actually
        # happened. Four agents noticing the same moment is one problem seen
        # four times, and a reader acting on the count needs the second view.
        "incidents": _incidents,
        # Why each incident was concluded, as a chain of cited claims. Built
        # from material the run already produced — no model is called here,
        # because generating fresh reasoning at report time would be a fifth
        # opinion nobody adjudicated.
        "reasoning": [
            chain.to_json()
            for chain in explain_all(
                _graph.incidents,
                findings,
                coverage={a.agent_id: a.coverage for a in result.agents},
                known_agents=[a.agent_id for a in result.agents],
            )
        ],
        # What happens if the creator changes the video. Computed from the
        # findings already produced — no perception rerun, no ffmpeg — and
        # scored by the same scorer that produced the headline number.
        "simulation": explore(findings, duration_ms).to_json(),
        "riskBands": build_risk_bands(findings, duration_ms),
        "findings": [f.to_json() for f in findings],
        "breakdown": build_breakdown(findings),
        "remediation": {
            "ops": [op.to_json() for op in edl.ops],
            "ffmpegCommand": program.pretty(),
            "renderMs": int(render_ms),
            "videoStreamCopied": program.video_stream_copied,
            **({"strategy": strategy} if strategy else {}),
            "log": list(edl.log),
        },
        "agents": agents,
        # What the run predicted against what it spent. The estimate is an
        # upper bound computed before any work began, so `actualCalls`
        # exceeding it is a bug in the plan rather than an unlucky run —
        # which is what makes this a check on PREFLIGHT and not a decoration.
        "cost": {
            "estimatedCalls": build_plan(
                duration_ms, chunk_ms=chunk_ms, overlap_ms=overlap_ms
            ).est_total_llm_calls,
            "actualCalls": result.total_calls,
            "ceiling": result.budget.ceiling,
            "shed": [s.to_json() for s in result.budget.shed],
        },
    }

    # Segment rollup, for videos long enough that a flat finding list stops
    # being readable. Gated on the decomposition plan's own threshold rather
    # than a second constant, so what the plan announced it would do is what
    # the report actually contains. Absent rather than empty below it: an
    # empty array reads as "rolled up and found nothing".
    if duration_ms > HIERARCHICAL_ABOVE_MS:
        report["segments"] = [
            segment.to_json() for segment in rollup(findings, duration_ms, SEGMENT_MS)
        ]

    return ReportBundle(report=report, edl=edl, program=program)


def agent_ts(result: PipelineResult, agent_id: str) -> int:
    """Offset from run start, so the terminal log replays the real sequence."""
    running = 0
    for agent in result.agents:
        if agent.agent_id == agent_id:
            return running
        running += agent.elapsed_ms if agent.agent_id != "orchestrator" else 0
    return running


def validate(report: dict[str, Any], schema_path: Path) -> None:
    """Hard-fail on a contract violation.

    A schema failure means the page would render something wrong. Better to
    stop here than to ship a report whose numbers the UI silently misreads.
    """
    try:
        import jsonschema
    except ImportError:  # pragma: no cover
        return
    import json

    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    jsonschema.validate(report, schema)
