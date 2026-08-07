"""A01 — ORCHESTRATOR.

The workflow controller. Specified in `prompts/A01_orchestrator.md`, and
verified against that specification by `tests/test_roster.py`.

Deterministic by construction. It classifies nothing, infers nothing, and never
touches another agent's output — the last of those is enforced structurally
rather than by instruction: stage results are held by reference and copied, not
mutated.

The `agents_completed` list this emits is a record of what ran, derived from
stage results rather than asserted. That distinction is the entire reason this
is code: a model asked to orchestrate produces the same list whether the agents
ran or not.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from preflight.agents.roster import Roster, load_roster
from preflight.models import AgentResult

# Stages the pipeline cannot produce a meaningful report without. Everything
# else is optional by design: a report honest about what it could not inspect
# is worth more than one that refuses to exist.
REQUIRED_STAGES = {"ingest", "speech"}

# Transient conditions worth another attempt. A 401 is not here — it will never
# succeed, and retrying burns budget the rest of the run needs.
TRANSIENT = ("timeout", "429", "5xx", "temporarily", "unreachable", "connection")


@dataclass(frozen=True)
class StageOutcome:
    """One stage's execution record. Frozen: the orchestrator cannot rewrite it."""

    stage: str
    agent_id: str
    status: str
    started_ms: int
    elapsed_ms: int
    attempts: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in {"OK", "DEGRADED"}

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "stage": self.stage,
            "agentId": self.agent_id,
            "status": self.status,
            "startedMs": self.started_ms,
            "elapsedMs": self.elapsed_ms,
            "attempts": self.attempts,
        }
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass
class ExecutionReport:
    """A01's only output: execution metadata.

    No findings, no scores, no verdicts. Those belong to the agents that
    produced them, and an orchestrator that summarises them is an orchestrator
    that can misreport them.
    """

    pipeline_id: str
    status: str = "RUNNING"
    started_at: str = ""
    timeline: list[StageOutcome] = field(default_factory=list)
    report_path: str | None = None
    failure_reason: str | None = None
    roster_digest: str = ""

    @property
    def agents_completed(self) -> list[str]:
        return [o.agent_id for o in self.timeline if o.ok]

    @property
    def execution_time_seconds(self) -> float:
        return round(sum(o.elapsed_ms for o in self.timeline) / 1000, 1)

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "pipeline_id": self.pipeline_id,
            "agents_completed": self.agents_completed,
            "execution_time_seconds": self.execution_time_seconds,
            "report_path": self.report_path,
            "started_at": self.started_at,
            "roster_digest": self.roster_digest[:16],
            "timeline": [o.to_json() for o in self.timeline],
            **({"failure_reason": self.failure_reason} if self.failure_reason else {}),
        }


class InputInvalid(ValueError):
    """Responsibility 1. Fails before any work is scheduled."""


def new_pipeline_id(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"PF-{now.year}-{uuid.uuid4().hex[:6].upper()}"


def validate_input(video: Path) -> Path:
    video = Path(video)
    if not video.exists():
        raise InputInvalid(f"no such file: {video}")
    if not video.is_file():
        raise InputInvalid(f"not a file: {video}")
    if video.stat().st_size == 0:
        raise InputInvalid(f"empty file: {video}")
    return video


def _is_transient(error: str | None) -> bool:
    if not error:
        return False
    lowered = error.lower()
    return any(token in lowered for token in TRANSIENT)


class Orchestrator:
    """Dispatch, collect, validate, retry, record.

    Stages are registered rather than hardcoded so the executor has no
    knowledge of what any of them do — which is what keeps "never classify
    content" true by construction rather than by discipline.
    """

    def __init__(
        self,
        *,
        roster: Roster | None = None,
        max_attempts: int = 2,
        clock: Callable[[], float] = time.perf_counter,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.roster = roster or load_roster()
        self.max_attempts = max_attempts
        self._clock = clock
        # Fired as each stage starts and settles. The orchestrator already
        # sequences every stage and records the timeline; a live view is the
        # same information delivered while it is still happening rather than
        # after. Never raises into the pipeline — a broken listener must not
        # take down a run.
        self._on_event = on_event
        self._stage_to_agent = {
            "orchestrator": "A01", "ingest": "A01", "speech": "A02",
            "vision": "A03", "audio": "A04", "ocr": "A05", "meta": "A06",
            "policy": "A07", "access": "A02", "score": "A08", "remedy": "A12",
            "report": "A12",
        }
        self.report = ExecutionReport(
            pipeline_id=new_pipeline_id(),
            started_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            roster_digest=self.roster.digest,
        )
        self._origin = self._clock()

    # ---------------------------------------------------------------- #

    def run_stage(
        self,
        stage: str,
        fn: Callable[[], AgentResult],
        *,
        required: bool | None = None,
        name: str | None = None,
    ) -> AgentResult:
        """Execute one stage exactly once, with retries on transient failure.

        Returns the agent's own result untouched. The orchestrator records
        *around* it and never reaches inside.
        """
        if required is None:
            required = stage in REQUIRED_STAGES

        started_ms = int((self._clock() - self._origin) * 1000)
        attempts = 0
        result: AgentResult | None = None

        self._emit(
            {
                "type": "stage.start",
                "stage": stage,
                "agentId": self._stage_to_agent.get(stage, "A??"),
                "name": name or stage.title(),
                "startedMs": started_ms,
            }
        )

        while attempts < self.max_attempts:
            attempts += 1
            began = self._clock()
            try:
                result = fn()
            except Exception as exc:  # noqa: BLE001 - a stage must not take the run
                result = AgentResult.failed(
                    stage, name or stage.title(), f"{type(exc).__name__}: {exc}"
                )
            elapsed = int((self._clock() - began) * 1000)
            result.elapsed_ms = result.elapsed_ms or elapsed

            if result.status != "FAILED" or not _is_transient(result.error):
                break
            if attempts >= self.max_attempts:
                break

        assert result is not None
        self.report.timeline.append(
            StageOutcome(
                stage=stage,
                agent_id=self._stage_to_agent.get(stage, "A??"),
                status=result.status,
                started_ms=started_ms,
                elapsed_ms=result.elapsed_ms,
                attempts=attempts,
                error=result.error,
            )
        )

        self._emit(
            {
                "type": "stage.end",
                "stage": stage,
                "agentId": self._stage_to_agent.get(stage, "A??"),
                "name": result.name,
                "status": result.status,
                "coverage": round(result.coverage, 4),
                "elapsedMs": result.elapsed_ms,
                "attempts": attempts,
                "findings": len(result.findings),
                "calls": result.calls,
                "detail": result.detail,
            }
        )

        # A required stage that failed stops the pipeline cleanly. An optional
        # one is recorded, coverage falls, and the run continues — which is the
        # behaviour that lets a report ship at 52% coverage instead of not at
        # all.
        if required and result.status in {"FAILED", "SKIPPED"}:
            self.report.status = "FAILED"
            self.report.failure_reason = (
                f"required stage {stage} did not complete: {result.error}"
            )

        return result

    @property
    def halted(self) -> bool:
        return self.report.status == "FAILED"

    def complete(self, report_path: Path | str | None = None) -> ExecutionReport:
        """Responsibility 12. Verifies the quality checklist before succeeding."""
        if self.report.status != "FAILED":
            problems = self.quality_checklist(report_path)
            self.report.status = "SUCCESS" if not problems else "PARTIAL"
            if problems:
                self.report.failure_reason = "; ".join(problems)
        self.report.report_path = str(report_path) if report_path else None
        return self.report

    def quality_checklist(self, report_path: Path | str | None) -> list[str]:
        """Verified programmatically, not asserted in prose."""
        problems: list[str] = []

        ran = {o.stage for o in self.report.timeline if o.ok}
        for stage in REQUIRED_STAGES:
            if stage not in ran:
                problems.append(f"required stage {stage} did not complete")

        if not self.report.timeline:
            problems.append("no stages executed")

        if report_path is not None and not Path(report_path).exists():
            problems.append(f"report not written to {report_path}")

        return problems

    def timeline_json(self) -> list[dict[str, Any]]:
        return [outcome.to_json() for outcome in self.report.timeline]

    def _emit(self, event: dict[str, Any]) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(event)
        except Exception:  # noqa: BLE001 - a listener must never fail a run
            pass
