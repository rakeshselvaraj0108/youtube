"""Section 18's restart experiment, driven through the real server module.

The point is not that a database persists rows — `test_lineage` covers that.
It is that the *workflow* picks up correctly: after a process dies mid-render,
a second process finds the operation, knows where it stopped, reuses only what
is still valid, and refuses to reuse anything that is not.

A restart is exactly "the objects are gone, the files remain", so these tests
drop every handle and construct a fresh store — the same state a new process
would find, without a subprocess to make flaky.
"""

from __future__ import annotations

import json

import pytest

from preflight import lifecycle, server
from preflight.lineage import Lineage


def report() -> dict:
    return {
        "video": {"filename": "clip.mp4", "durationMs": 20_000},
        "meta": {"coverage": 0.8, "policyVersion": "2026.08.01"},
        "scores": {"overall": 42, "verdict": "PUBLISH_WITH_FIXES"},
        "findings": [{"id": "p_00", "clauseId": "AF-09"}],
        "incidents": [{"id": "INC-001", "findingIds": ["p_00"]}],
    }


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"original bytes standing in for a video")
    return path


@pytest.fixture
def rendered(tmp_path):
    path = tmp_path / "clip.safe.mp4"
    path.write_bytes(b"rendered bytes, definitely not the original")
    return path


def interrupted_at(db, source, rendered, state: str) -> str:
    """Walk a remediation to `state` and abandon it there."""
    store = Lineage(db)
    store.record_run("run-1", report(), video_path=str(source))
    record = store.open_remediation(
        "run-1", source_path=str(source), finding_ids=["p_00"]
    )
    walk = [
        "RENDERING",
        "RENDERED",
        "STRUCTURAL_VERIFYING",
        "STRUCTURALLY_VALID",
        "REANALYSIS_QUEUED",
        "REANALYSING",
        "REANALYSIS_COMPLETE",
        "COMPARING",
    ]
    for step in walk:
        fields: dict = {}
        if step == "RENDERING":
            fields["edl_json"] = json.dumps(
                [{"op": "MUTE", "startMs": 1_000, "endMs": 2_000}]
            )
        if step == "RENDERED":
            artifact = store.record_artifact(rendered)
            fields["artifact_id"] = artifact.artifact_id
            fields["output_path"] = str(rendered)
        store.transition(record.remediation_id, step, **fields)
        if step == state:
            break
    return record.remediation_id


class TestFindingInterruptedWork:
    @pytest.mark.parametrize("state", ["RENDERING", "REANALYSING", "COMPARING"])
    def test_a_new_process_finds_the_operation(self, tmp_path, source, rendered, state):
        db = tmp_path / "lineage.db"
        remediation_id = interrupted_at(db, source, rendered, state)

        # Everything above is gone. This is what a restart sees.
        found = server.find_resumable(Lineage(db), source)
        assert found is not None
        assert found.remediation_id == remediation_id
        assert found.state == state

    def test_it_can_say_what_was_interrupted(self, tmp_path, source, rendered):
        db = tmp_path / "lineage.db"
        interrupted_at(db, source, rendered, "REANALYSING")
        found = server.find_resumable(Lineage(db), source)
        assert found.describe() == "REM-0001 was interrupted during REANALYSING"

    def test_a_finished_remediation_is_not_resumed(self, tmp_path, source, rendered):
        db = tmp_path / "lineage.db"
        store = Lineage(db)
        remediation_id = interrupted_at(db, source, rendered, "COMPARING")
        store.transition(
            remediation_id, "PARTIALLY_REMEDIATED", verdict="PARTIALLY_REMEDIATED"
        )
        assert server.find_resumable(Lineage(db), source) is None

    def test_a_remediation_for_another_video_is_not_offered(
        self, tmp_path, source, rendered
    ):
        db = tmp_path / "lineage.db"
        interrupted_at(db, source, rendered, "REANALYSING")
        other = tmp_path / "different.mp4"
        other.write_bytes(b"a different video entirely")
        assert server.find_resumable(Lineage(db), other) is None


class TestArtifactReuseIsEarned:
    """The one way persistence could make this system *less* correct."""

    def test_an_intact_artifact_is_reusable(self, tmp_path, source, rendered):
        db = tmp_path / "lineage.db"
        interrupted_at(db, source, rendered, "REANALYSING")
        store = Lineage(db)
        record = server.find_resumable(store, source)
        assert server.reusable_artifact(store, record) == rendered

    def test_an_artifact_changed_underneath_the_record_is_refused(
        self, tmp_path, source, rendered
    ):
        db = tmp_path / "lineage.db"
        interrupted_at(db, source, rendered, "REANALYSING")

        # Something replaced the file between the two processes. The row still
        # names it; the bytes are no longer what was verified.
        rendered.write_bytes(b"someone else wrote this file after the record")

        store = Lineage(db)
        record = server.find_resumable(store, source)
        assert server.reusable_artifact(store, record) is None

    def test_a_deleted_artifact_is_refused(self, tmp_path, source, rendered):
        db = tmp_path / "lineage.db"
        interrupted_at(db, source, rendered, "REANALYSING")
        rendered.unlink()
        store = Lineage(db)
        record = server.find_resumable(store, source)
        assert server.reusable_artifact(store, record) is None

    def test_an_interrupted_render_has_nothing_to_reuse(
        self, tmp_path, source, rendered
    ):
        """A crash during RENDERING never recorded an artifact, so there is
        nothing to trust and the render is repeated."""
        db = tmp_path / "lineage.db"
        interrupted_at(db, source, rendered, "RENDERING")
        store = Lineage(db)
        record = server.find_resumable(store, source)
        assert record.artifact_id is None
        assert server.reusable_artifact(store, record) is None


class TestResumeEntersTheRightState:
    def test_an_interrupted_render_starts_over(self, tmp_path, source, rendered):
        db = tmp_path / "lineage.db"
        interrupted_at(db, source, rendered, "RENDERING")
        assert Lineage(db).resume("REM-0001").state == "REMEDIATION_REQUESTED"

    def test_an_interrupted_reanalysis_keeps_the_render(
        self, tmp_path, source, rendered
    ):
        db = tmp_path / "lineage.db"
        interrupted_at(db, source, rendered, "REANALYSING")
        resumed = Lineage(db).resume("REM-0001")
        assert resumed.state == "STRUCTURALLY_VALID"
        assert resumed.output_path == str(rendered)

    def test_an_interrupted_comparison_keeps_both_runs(
        self, tmp_path, source, rendered
    ):
        db = tmp_path / "lineage.db"
        interrupted_at(db, source, rendered, "COMPARING")
        assert Lineage(db).resume("REM-0001").state == "REANALYSIS_COMPLETE"

    def test_the_resume_is_itself_recorded(self, tmp_path, source, rendered):
        """The audit trail must show the interruption, not paper over it."""
        db = tmp_path / "lineage.db"
        interrupted_at(db, source, rendered, "REANALYSING")
        store = Lineage(db)
        store.resume("REM-0001")
        history = store.remediation("REM-0001").transitions
        resume_step = history[-1]
        assert resume_step.from_state == "REANALYSING"
        assert resume_step.to_state == "STRUCTURALLY_VALID"
        assert "REANALYSING" in resume_step.detail

    def test_resuming_does_not_lose_the_relationships(
        self, tmp_path, source, rendered
    ):
        """Section 18's list: the run, the remediation id, the artifact and
        the targeted findings all survive."""
        db = tmp_path / "lineage.db"
        interrupted_at(db, source, rendered, "REANALYSING")
        resumed = Lineage(db).resume("REM-0001")
        assert resumed.remediation_id == "REM-0001"
        assert resumed.source_run_id == "run-1"
        assert resumed.artifact_id is not None
        assert resumed.finding_ids == ("p_00",)
        assert resumed.ops[0]["op"] == "MUTE"


class TestTheStateMachineStillHoldsAfterARestart:
    def test_a_resumed_remediation_cannot_skip_to_a_verdict(
        self, tmp_path, source, rendered
    ):
        db = tmp_path / "lineage.db"
        interrupted_at(db, source, rendered, "REANALYSING")
        store = Lineage(db)
        store.resume("REM-0001")
        with pytest.raises(lifecycle.InvalidTransition):
            store.transition("REM-0001", "VERIFIED")

    def test_a_resumed_remediation_can_complete_normally(
        self, tmp_path, source, rendered
    ):
        db = tmp_path / "lineage.db"
        interrupted_at(db, source, rendered, "REANALYSING")
        store = Lineage(db)
        store.resume("REM-0001")
        for step in (
            "REANALYSIS_QUEUED",
            "REANALYSING",
            "REANALYSIS_COMPLETE",
            "COMPARING",
        ):
            store.transition("REM-0001", step)
        store.transition("REM-0001", "VERIFIED", verdict="VERIFIED_SAFE")

        final = Lineage(db).remediation("REM-0001")
        assert final.state == "VERIFIED"
        assert final.terminal
        # The interruption is still in the history rather than erased by the
        # successful completion.
        assert any(t.to_state == "STRUCTURALLY_VALID" for t in final.transitions[1:])
