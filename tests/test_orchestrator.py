"""A01 — the orchestrator's behavioural contract.

Every test here corresponds to a line in `prompts/A01_orchestrator.md`. The
spec says "ensure every stage executes exactly once" and "never invent missing
data"; these are what make those statements true rather than aspirational.
"""

from __future__ import annotations

import pytest

from preflight.models import AgentResult
from preflight.orchestrator import (
    InputInvalid,
    Orchestrator,
    StageOutcome,
    new_pipeline_id,
    validate_input,
)


def ok(stage: str = "speech") -> AgentResult:
    return AgentResult(agent_id=stage, name=stage.title(), status="OK", elapsed_ms=5)


def failed(stage: str, reason: str) -> AgentResult:
    return AgentResult.failed(stage, stage.title(), reason)


class TestInputValidation:
    """Responsibility 1: validate before scheduling any work."""

    def test_accepts_a_real_file(self, tmp_path):
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x" * 64)
        assert validate_input(video) == video

    def test_rejects_a_missing_file(self, tmp_path):
        with pytest.raises(InputInvalid, match="no such file"):
            validate_input(tmp_path / "absent.mp4")

    def test_rejects_a_directory(self, tmp_path):
        with pytest.raises(InputInvalid, match="not a file"):
            validate_input(tmp_path)

    def test_rejects_an_empty_file(self, tmp_path):
        empty = tmp_path / "empty.mp4"
        empty.touch()
        with pytest.raises(InputInvalid, match="empty file"):
            validate_input(empty)


class TestPipelineIdentity:
    def test_ids_are_unique(self):
        assert len({new_pipeline_id() for _ in range(200)}) == 200

    def test_id_carries_the_year(self):
        from datetime import datetime, timezone

        assert new_pipeline_id().startswith(f"PF-{datetime.now(timezone.utc).year}-")


class TestExactlyOnce:
    """The spec's central invariant."""

    def test_a_successful_stage_runs_once(self):
        calls = []
        orch = Orchestrator()
        orch.run_stage("speech", lambda: calls.append(1) or ok())
        assert len(calls) == 1
        assert orch.report.timeline[0].attempts == 1

    def test_each_stage_appears_once_in_the_timeline(self):
        orch = Orchestrator()
        for stage in ("ingest", "speech", "audio"):
            orch.run_stage(stage, lambda s=stage: ok(s))
        stages = [o.stage for o in orch.report.timeline]
        assert stages == ["ingest", "speech", "audio"]
        assert len(set(stages)) == 3


class TestRetryPolicy:
    def test_retries_a_transient_failure(self):
        attempts = []

        def flaky() -> AgentResult:
            attempts.append(1)
            if len(attempts) == 1:
                return failed("speech", "connection timeout")
            return ok()

        orch = Orchestrator(max_attempts=2)
        result = orch.run_stage("speech", flaky)
        assert len(attempts) == 2
        assert result.status == "OK"
        assert orch.report.timeline[0].attempts == 2

    def test_does_not_retry_a_permanent_failure(self):
        """A 401 will never succeed, and retrying burns budget."""
        attempts = []

        def rejected() -> AgentResult:
            attempts.append(1)
            return failed("policy", "auth rejected (401) — check the key")

        Orchestrator(max_attempts=3).run_stage("policy", rejected)
        assert len(attempts) == 1

    def test_gives_up_after_the_attempt_budget(self):
        attempts = []

        def always_flaky() -> AgentResult:
            attempts.append(1)
            return failed("policy", "429 rate limited")

        orch = Orchestrator(max_attempts=3)
        orch.run_stage("policy", always_flaky)
        assert len(attempts) == 3
        assert orch.report.timeline[0].status == "FAILED"


class TestFailureHandling:
    def test_an_optional_stage_failing_does_not_halt_the_run(self):
        """A report honest about what it could not inspect beats no report."""
        orch = Orchestrator()
        orch.run_stage("ingest", lambda: ok("ingest"))
        orch.run_stage("speech", lambda: ok("speech"))
        orch.run_stage("vision", lambda: failed("vision", "no API key"))
        assert not orch.halted
        assert orch.complete().status == "SUCCESS"

    def test_a_required_stage_failing_halts_cleanly(self):
        orch = Orchestrator()
        orch.run_stage("speech", lambda: failed("speech", "no backend installed"))
        assert orch.halted
        report = orch.complete()
        assert report.status == "FAILED"
        assert "speech" in (report.failure_reason or "")

    def test_an_exception_becomes_a_recorded_failure_not_a_crash(self):
        def explodes() -> AgentResult:
            raise RuntimeError("boom")

        orch = Orchestrator()
        result = orch.run_stage("vision", explodes)
        assert result.status == "FAILED"
        assert "boom" in (result.error or "")
        assert orch.report.timeline[0].status == "FAILED"

    def test_never_fabricates_a_missing_output(self):
        """An agent that did not run is reported as not having run."""
        orch = Orchestrator()
        orch.run_stage("ingest", lambda: ok("ingest"))
        orch.run_stage("speech", lambda: ok("speech"))
        orch.run_stage("vision", lambda: failed("vision", "no key"))
        completed = orch.complete().agents_completed
        assert "A03" not in completed  # vision
        assert "A02" in completed  # speech


class TestOutputContract:
    def test_emits_execution_metadata_only(self):
        orch = Orchestrator()
        orch.run_stage("ingest", lambda: ok("ingest"))
        orch.run_stage("speech", lambda: ok("speech"))
        payload = orch.complete().to_json()

        assert set(payload) >= {
            "status", "pipeline_id", "agents_completed",
            "execution_time_seconds", "report_path",
        }
        # No findings, no scores, no verdicts. Those belong to the agents that
        # produced them, and an orchestrator that summarises them can misreport
        # them.
        for forbidden in ("findings", "scores", "verdict", "severity", "readiness"):
            assert forbidden not in payload

    def test_agents_completed_is_derived_not_asserted(self):
        """The distinction that makes this code rather than a prompt."""
        orch = Orchestrator()
        orch.run_stage("ingest", lambda: ok("ingest"))
        orch.run_stage("speech", lambda: failed("speech", "no backend"))
        assert orch.complete().agents_completed == ["A01"]

    def test_records_the_roster_digest(self):
        payload = Orchestrator().complete().to_json()
        assert len(payload["roster_digest"]) == 16

    def test_timeline_is_ordered_and_monotonic(self):
        orch = Orchestrator()
        for stage in ("ingest", "speech", "audio", "meta"):
            orch.run_stage(stage, lambda s=stage: ok(s))
        starts = [o.started_ms for o in orch.report.timeline]
        assert starts == sorted(starts)


class TestImmutability:
    def test_a_stage_outcome_cannot_be_rewritten(self):
        """'Never change another agent's output' — enforced structurally."""
        orch = Orchestrator()
        orch.run_stage("speech", lambda: ok())
        with pytest.raises(Exception):
            orch.report.timeline[0].status = "OK"  # type: ignore[misc]

    def test_the_orchestrator_returns_the_agents_own_result(self):
        produced = ok("audio")
        returned = Orchestrator().run_stage("audio", lambda: produced)
        assert returned is produced


class TestQualityChecklist:
    def test_missing_required_stage_is_reported(self):
        orch = Orchestrator()
        orch.run_stage("ingest", lambda: ok("ingest"))
        problems = orch.quality_checklist(None)
        assert any("speech" in p for p in problems)

    def test_a_promised_report_that_was_not_written_is_caught(self, tmp_path):
        orch = Orchestrator()
        orch.run_stage("ingest", lambda: ok("ingest"))
        orch.run_stage("speech", lambda: ok("speech"))
        report = orch.complete(tmp_path / "report.html")
        assert report.status == "PARTIAL"
        assert "not written" in (report.failure_reason or "")

    def test_a_written_report_passes(self, tmp_path):
        path = tmp_path / "report.html"
        path.write_text("<html></html>", encoding="utf-8")
        orch = Orchestrator()
        orch.run_stage("ingest", lambda: ok("ingest"))
        orch.run_stage("speech", lambda: ok("speech"))
        assert orch.complete(path).status == "SUCCESS"

    def test_a_run_with_no_stages_is_not_a_success(self):
        assert Orchestrator().complete().status == "PARTIAL"
