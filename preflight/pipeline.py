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
from preflight.ingest.pipeline import Ingested, ingest
from preflight.models import AgentResult, Finding
from preflight.perception import accessibility, audio, metadata
from preflight.perception import asr as asr_mod
from preflight.perception.asr import Transcript

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
    "score": 0.02,
    "remedy": 0.02,
    "report": 0.01,
}


@dataclass
class PipelineResult:
    source: Path
    ingested: Ingested
    transcript: Transcript | None
    agents: list[AgentResult] = field(default_factory=list)
    started_at: float = 0.0

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
) -> PipelineResult:
    source = Path(source)
    started_at = time.perf_counter()

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
    else:
        speech_agent, transcript = _speech(ingested, store, asr_model)

    audio_agent = _guard(
        "audio",
        "Audio Agent",
        lambda: audio.analyse(
            ingested.fingerprint_wav, source, ingested.meta.durationMs
        ),
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

    orchestrator.elapsed_ms = int((time.perf_counter() - started_at) * 1000)

    return PipelineResult(
        source=source,
        ingested=ingested,
        transcript=transcript,
        agents=[
            orchestrator,
            ingest_agent,
            speech_agent,
            audio_agent,
            access_agent,
            meta_agent,
        ],
        started_at=started_at,
    )


def _speech(
    ingested: Ingested, store: cas.Store, asr_model: str
) -> tuple[AgentResult, Transcript | None]:
    def run() -> AgentResult:
        agent, transcript = asr_mod.transcribe(
            ingested.asr_wav,
            store,
            model_id=asr_model,
            duration_ms=ingested.meta.durationMs,
        )
        run.transcript = transcript  # type: ignore[attr-defined]
        return agent

    run.transcript = None  # type: ignore[attr-defined]
    agent = _guard("speech", "Speech Agent", run)
    return agent, getattr(run, "transcript", None)
