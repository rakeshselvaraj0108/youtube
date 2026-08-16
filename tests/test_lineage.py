"""The durable run graph.

The property under test is survival: everything the loop produces must still
be there, and still be linked, after the process that produced it is gone. So
every test that matters here closes the store and opens a new one, because an
object still in memory proves nothing about what is on disk.
"""

from __future__ import annotations

import pytest

from preflight import lifecycle
from preflight.lineage import Lineage


def report(overall: int = 42, findings: int = 3, incidents: int = 2) -> dict:
    return {
        "video": {"filename": "clip.mp4", "durationMs": 20_000},
        "meta": {"coverage": 0.81, "policyVersion": "2026.08.01"},
        "scores": {"overall": overall, "verdict": "PUBLISH_WITH_FIXES"},
        "findings": [{"id": f"F{i}", "clauseId": "AF-01"} for i in range(findings)],
        "incidents": [{"id": f"INC-{i:03d}"} for i in range(incidents)],
    }


@pytest.fixture
def store(tmp_path):
    return Lineage(tmp_path / "lineage.db")


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not really a video, but real bytes with a real hash")
    return path


class TestIdentifiersAreDurable:
    def test_remediation_ids_are_sequential_and_padded(self, store, video):
        store.record_run("run-1", report(), video_path=str(video))
        first = store.open_remediation("run-1", source_path=str(video))
        second = store.open_remediation("run-1", source_path=str(video))
        assert first.remediation_id == "REM-0001"
        assert second.remediation_id == "REM-0002"

    def test_an_id_survives_reopening_the_store(self, tmp_path, video):
        path = tmp_path / "lineage.db"
        first = Lineage(path)
        first.record_run("run-1", report(), video_path=str(video))
        first.open_remediation("run-1", source_path=str(video))
        del first

        # A different process would see exactly this: the file, nothing else.
        reopened = Lineage(path)
        assert reopened.remediation("REM-0001") is not None
        # And the counter continues rather than restarting, which is the bug a
        # counter held in memory produces on every restart.
        assert (
            reopened.open_remediation("run-1", source_path=str(video)).remediation_id
            == "REM-0002"
        )

    def test_artifact_ids_are_content_addressed(self, store, tmp_path):
        same_a = tmp_path / "a.mp4"
        same_b = tmp_path / "b.mp4"
        same_a.write_bytes(b"identical bytes")
        same_b.write_bytes(b"identical bytes")
        assert (
            store.record_artifact(same_a).artifact_id
            == store.record_artifact(same_b).artifact_id
        )

    def test_different_bytes_are_different_artifacts(self, store, tmp_path):
        one, two = tmp_path / "1.mp4", tmp_path / "2.mp4"
        one.write_bytes(b"first")
        two.write_bytes(b"second")
        assert store.record_artifact(one).artifact_id != store.record_artifact(two).artifact_id


class TestArtifactsAreCheckedNotTrusted:
    def test_an_unchanged_artifact_still_matches(self, store, video):
        assert store.record_artifact(video).still_matches()

    def test_an_artifact_edited_underneath_the_record_does_not_match(
        self, store, video
    ):
        """The one way persistence could be a correctness regression: reusing
        a file because a row mentions it, after something else replaced it."""
        artifact = store.record_artifact(video)
        video.write_bytes(b"completely different content written by someone else")
        assert not artifact.still_matches()

    def test_a_deleted_artifact_does_not_match(self, store, video):
        artifact = store.record_artifact(video)
        video.unlink()
        assert not artifact.still_matches()


class TestTheGraphIsTraceable:
    def test_a_verification_run_names_its_parent(self, store, video):
        store.record_run("original", report(), video_path=str(video))
        store.record_run(
            "verification",
            report(overall=55),
            role="VERIFICATION",
            parent_run_id="original",
            video_path=str(video),
        )
        graph = store.graph("original")
        assert [n.run_id for n in graph.verification_runs] == ["verification"]

    def test_the_graph_resolves_the_whole_chain(self, store, video):
        store.record_run("original", report(), video_path=str(video))
        simulation_id = store.record_simulation(
            "original",
            {"best": "mute all", "baseline": {"overall": 42},
             "scenarios": [{"name": "mute all", "overall": 61}]},
        )
        artifact = store.record_artifact(video)
        record = store.open_remediation(
            "original", source_path=str(video), simulation_id=simulation_id
        )
        _advance_to_verdict(store, record.remediation_id, artifact.artifact_id)
        verification_id = store.record_verification(
            record.remediation_id,
            original_run_id="original",
            verification_run_id=None,
            comparison={"verdict": "VERIFIED_SAFE"},
        )
        store.attach_verification(record.remediation_id, verification_id)
        store.record_certificate(verification_id, {"a": 1}, "b3:deadbeef")

        graph = store.graph("original")
        assert graph.root is not None
        assert len(graph.simulations) == 1
        assert len(graph.remediations) == 1
        assert len(graph.certificates) == 1
        assert graph.remediations[0].simulation_id == simulation_id

    def test_reports_are_referenced_not_copied(self, store, video, tmp_path):
        """A lineage row must not become a second copy of a report that can
        disagree with the first."""
        path = tmp_path / "report.json"
        store.record_run(
            "original", report(), video_path=str(video), report_path=str(path)
        )
        node = store.run("original")
        assert node is not None
        assert node.report_path == str(path)
        # Counts, not contents.
        assert node.findings == 3 and node.incidents == 2


def _advance_to_verdict(store: Lineage, remediation_id: str, artifact_id: str) -> None:
    """Walk the legal path. Every step is a real transition."""
    store.transition(remediation_id, "RENDERING")
    store.transition(remediation_id, "RENDERED", artifact_id=artifact_id)
    store.transition(remediation_id, "STRUCTURAL_VERIFYING")
    store.transition(remediation_id, "STRUCTURALLY_VALID")
    store.transition(remediation_id, "REANALYSIS_QUEUED")
    store.transition(remediation_id, "REANALYSING")
    store.transition(remediation_id, "REANALYSIS_COMPLETE")
    store.transition(remediation_id, "COMPARING")
    store.transition(remediation_id, "VERIFIED", verdict="VERIFIED_SAFE")


class TestTheStoreRefusesImpossibleHistories:
    def test_a_render_cannot_be_recorded_as_verified(self, store, video):
        store.record_run("run-1", report(), video_path=str(video))
        record = store.open_remediation("run-1", source_path=str(video))
        store.transition(record.remediation_id, "RENDERING")
        with pytest.raises(lifecycle.InvalidTransition):
            store.transition(record.remediation_id, "VERIFIED")

    def test_a_rejected_transition_leaves_the_state_alone(self, store, video):
        """A refused write must not half-apply — the row is the audit trail."""
        store.record_run("run-1", report(), video_path=str(video))
        record = store.open_remediation("run-1", source_path=str(video))
        store.transition(record.remediation_id, "RENDERING")
        with pytest.raises(lifecycle.InvalidTransition):
            store.transition(record.remediation_id, "VERIFIED")
        assert store.remediation(record.remediation_id).state == "RENDERING"

    def test_a_finished_remediation_cannot_be_reopened(self, store, video):
        store.record_run("run-1", report(), video_path=str(video))
        artifact = store.record_artifact(video)
        record = store.open_remediation("run-1", source_path=str(video))
        _advance_to_verdict(store, record.remediation_id, artifact.artifact_id)
        with pytest.raises(lifecycle.InvalidTransition):
            store.transition(record.remediation_id, "RENDERING")

    def test_an_unknown_column_is_refused(self, store, video):
        store.record_run("run-1", report(), video_path=str(video))
        record = store.open_remediation("run-1", source_path=str(video))
        with pytest.raises(ValueError):
            store.transition(record.remediation_id, "RENDERING", verdict_typo="x")

    def test_stored_operations_round_trip_as_operations(self, store, video):
        """`ops` is a list of operation dicts, not the EDL envelope.

        Storing `{source, durationMs, ops, warnings}` here still parses as
        JSON and still iterates — it yields the *key strings*, so every
        consumer downstream silently receives characters where it expected
        operations. The failure surfaces far from the cause, which is why the
        shape is asserted at the boundary.
        """
        import json as json_mod

        store.record_run("run-1", report(), video_path=str(video))
        record = store.open_remediation("run-1", source_path=str(video))
        store.transition(
            record.remediation_id,
            "RENDERING",
            edl_json=json_mod.dumps(
                [{"op": "CUT", "startMs": 1_000, "endMs": 2_000}]
            ),
        )
        ops = store.remediation(record.remediation_id).ops
        assert [dict(op) for op in ops] == [
            {"op": "CUT", "startMs": 1_000, "endMs": 2_000}
        ]

    @pytest.mark.parametrize(
        "stored",
        [
            # The whole EDL envelope, as an earlier version wrote it.
            '{"source":"a.mp4","durationMs":20000,'
            '"ops":[{"op":"CUT","startMs":1000,"endMs":2000}],"warnings":[]}',
            "not json at all",
            '"a bare string"',
            "[1, 2, 3]",
        ],
    )
    def test_a_malformed_operations_column_does_not_break_the_listing(
        self, store, video, stored
    ):
        """One bad row must not hide every healthy row beside it.

        A row written before the column settled holds the EDL envelope, which
        parses as JSON and iterates as key strings — so a strict reader raises
        inside a comprehension and takes the whole remediation listing with
        it. The same rule the run listing already follows applies here.
        """
        import sqlite3
        from contextlib import closing

        store.record_run("run-1", report(), video_path=str(video))
        record = store.open_remediation("run-1", source_path=str(video))
        with closing(sqlite3.connect(store.path)) as conn:
            conn.execute(
                "UPDATE remediations SET edl_json = ? WHERE remediation_id = ?",
                (stored, record.remediation_id),
            )
            conn.commit()

        listed = store.remediations()
        assert len(listed) == 1
        assert all(isinstance(op, dict) for op in listed[0].ops)
        listed[0].to_json()  # must not raise

    def test_a_legacy_envelope_still_yields_its_operations(self, store, video):
        """Degrading to empty is the floor, not the goal — where the ops are
        recoverable they are recovered."""
        import sqlite3
        from contextlib import closing

        store.record_run("run-1", report(), video_path=str(video))
        record = store.open_remediation("run-1", source_path=str(video))
        with closing(sqlite3.connect(store.path)) as conn:
            conn.execute(
                "UPDATE remediations SET edl_json = ? WHERE remediation_id = ?",
                (
                    '{"source":"a.mp4","ops":[{"op":"CUT","startMs":1000,'
                    '"endMs":2000}],"warnings":[]}',
                    record.remediation_id,
                ),
            )
            conn.commit()
        assert store.remediation(record.remediation_id).ops == (
            {"op": "CUT", "startMs": 1000, "endMs": 2000},
        )

    def test_every_transition_is_recorded(self, store, video):
        store.record_run("run-1", report(), video_path=str(video))
        artifact = store.record_artifact(video)
        record = store.open_remediation("run-1", source_path=str(video))
        _advance_to_verdict(store, record.remediation_id, artifact.artifact_id)
        history = store.remediation(record.remediation_id).transitions
        # Nine steps plus the opening transition.
        assert len(history) == 10
        assert history[0].to_state == "REMEDIATION_REQUESTED"
        assert history[-1].to_state == "VERIFIED"
        assert all(t.at for t in history)


class TestRestartAndResume:
    """Section 18's experiment, without the subprocess.

    A restart is exactly "the object is gone, the file remains", so dropping
    the handle and opening a second store is the same test with less to go
    wrong in the harness.
    """

    def _interrupt_during(self, path, video, state: str) -> str:
        store = Lineage(path)
        store.record_run("run-1", report(), video_path=str(video))
        artifact = store.record_artifact(video)
        record = store.open_remediation("run-1", source_path=str(video))
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
            fields = {"artifact_id": artifact.artifact_id} if step == "RENDERED" else {}
            store.transition(record.remediation_id, step, **fields)
            if step == state:
                break
        return record.remediation_id

    @pytest.mark.parametrize("state", ["RENDERING", "REANALYSING", "COMPARING"])
    def test_an_interrupted_remediation_is_found_after_restart(
        self, tmp_path, video, state
    ):
        path = tmp_path / "lineage.db"
        remediation_id = self._interrupt_during(path, video, state)

        reopened = Lineage(path)
        open_records = reopened.interrupted()
        assert [r.remediation_id for r in open_records] == [remediation_id]
        assert open_records[0].state == state

    def test_the_system_can_say_what_was_interrupted(self, tmp_path, video):
        """The brief's requirement, verbatim: the operation is not lost."""
        path = tmp_path / "lineage.db"
        self._interrupt_during(path, video, "REANALYSING")
        record = Lineage(path).interrupted()[0]
        assert record.describe() == "REM-0001 was interrupted during REANALYSING"

    def test_resuming_a_reanalysis_keeps_the_render(self, tmp_path, video):
        path = tmp_path / "lineage.db"
        self._interrupt_during(path, video, "REANALYSING")
        reopened = Lineage(path)
        resumed = reopened.resume("REM-0001")
        assert resumed.state == "STRUCTURALLY_VALID"
        # And the artifact is still named, so the render need not be repeated.
        assert resumed.artifact_id is not None
        assert reopened.artifact(resumed.artifact_id).still_matches()

    def test_resuming_a_render_repeats_it(self, tmp_path, video):
        """A half-written file proves nothing; the render is redone."""
        path = tmp_path / "lineage.db"
        self._interrupt_during(path, video, "RENDERING")
        assert Lineage(path).resume("REM-0001").state == "REMEDIATION_REQUESTED"

    def test_the_resume_itself_is_recorded(self, tmp_path, video):
        path = tmp_path / "lineage.db"
        self._interrupt_during(path, video, "COMPARING")
        store = Lineage(path)
        store.resume("REM-0001")
        last = store.remediation("REM-0001").transitions[-1]
        assert last.from_state == "COMPARING"
        assert last.to_state == "REANALYSIS_COMPLETE"
        assert "resumed" in last.detail

    def test_a_finished_remediation_cannot_be_resumed(self, tmp_path, video):
        path = tmp_path / "lineage.db"
        store = Lineage(path)
        store.record_run("run-1", report(), video_path=str(video))
        artifact = store.record_artifact(video)
        record = store.open_remediation("run-1", source_path=str(video))
        _advance_to_verdict(store, record.remediation_id, artifact.artifact_id)
        assert Lineage(path).interrupted() == []
        with pytest.raises(lifecycle.InvalidTransition):
            Lineage(path).resume(record.remediation_id)

    def test_a_failed_remediation_is_not_offered_for_resume(self, tmp_path, video):
        path = tmp_path / "lineage.db"
        store = Lineage(path)
        store.record_run("run-1", report(), video_path=str(video))
        record = store.open_remediation("run-1", source_path=str(video))
        store.transition(record.remediation_id, "RENDERING")
        store.fail(record.remediation_id, "ffmpeg exited 1")
        assert Lineage(path).interrupted() == []
        assert Lineage(path).remediation("REM-0001").error == "ffmpeg exited 1"


class TestReferenceIntegrity:
    def test_a_verification_resolves_its_comparison(self, store, video):
        store.record_run("run-1", report(), video_path=str(video))
        record = store.open_remediation("run-1", source_path=str(video))
        verification_id = store.record_verification(
            record.remediation_id,
            original_run_id="run-1",
            verification_run_id=None,
            comparison={"verdict": "PARTIALLY_REMEDIATED", "resolved": 2},
            telemetry={"totalMs": 1234},
        )
        stored = store.verification(verification_id)
        assert stored["comparison"]["resolved"] == 2
        assert stored["telemetry"]["totalMs"] == 1234

    def test_the_newest_certificate_wins(self, store, video):
        store.record_run("run-1", report(), video_path=str(video))
        record = store.open_remediation("run-1", source_path=str(video))
        verification_id = store.record_verification(
            record.remediation_id,
            original_run_id="run-1",
            verification_run_id=None,
            comparison={"verdict": "NO_CHANGE"},
        )
        store.record_certificate(verification_id, {"v": 1}, "b3:one")
        store.record_certificate(verification_id, {"v": 2}, "b3:two")
        assert store.certificate_for(verification_id)["payload"]["v"] == 2
