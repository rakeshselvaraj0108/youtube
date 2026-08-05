"""The findings archive and the Policy Drift Watcher."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from preflight.archive import Archive
from preflight.drift import (
    SEMANTIC_FLOOR,
    ClauseChange,
    detect,
    diff_corpus,
    read_snapshot,
    snapshot,
    write_snapshot,
)
from preflight.policy.corpus import load_corpus

POLICY = Path("data/policy")


def make_report(filename="demo.mp4", overall=45, clauses=("AF-10",)) -> dict:
    return {
        "video": {"filename": filename, "durationMs": 51_000},
        "meta": {
            "analyzedAt": "2026-08-05T11:00:00Z",
            "policyVersion": "2026-08",
            "engineVersion": "0.1.0",
            "coverage": 0.83,
        },
        "scores": {"overall": overall, "verdict": "DO_NOT_PUBLISH"},
        "findings": [
            {"clauseId": c, "severity": "MEDIUM", "confidence": 0.8} for c in clauses
        ],
    }


class TestArchive:
    def test_records_and_reads_back(self, tmp_path):
        archive = Archive(tmp_path / "a.db")
        archive.record(make_report(), video_hash="h1", policy_digest="d1")

        videos = archive.latest_reports()
        assert len(videos) == 1
        assert videos[0].filename == "demo.mp4"
        assert videos[0].clauses == ("AF-10",)

    def test_keeps_only_the_latest_report_per_video(self, tmp_path):
        archive = Archive(tmp_path / "a.db")
        archive.record(make_report(overall=45), video_hash="h1", policy_digest="d1")
        archive.record(make_report(overall=88), video_hash="h1", policy_digest="d2")

        videos = archive.latest_reports()
        assert len(videos) == 1
        assert videos[0].overall == 88

    def test_history_keeps_every_run(self, tmp_path):
        archive = Archive(tmp_path / "a.db")
        archive.record(make_report(overall=45), video_hash="h1", policy_digest="d1")
        archive.record(make_report(overall=88), video_hash="h1", policy_digest="d2")
        assert [row[1] for row in archive.score_history("h1")] == [45, 88]

    def test_records_clauses_considered_but_not_fired(self, tmp_path):
        """A clause that nearly fired is what a tightening will newly catch."""
        archive = Archive(tmp_path / "a.db")
        archive.record(
            make_report(clauses=("AF-10",)),
            video_hash="h1",
            policy_digest="d1",
            considered={"AF-08", "AF-10"},
        )
        video = archive.latest_reports()[0]
        assert video.clauses == ("AF-10",)
        assert video.near_miss_clauses == ("AF-08",)

    def test_affected_by_matches_fired_clauses(self, tmp_path):
        archive = Archive(tmp_path / "a.db")
        archive.record(make_report(clauses=("AF-10",)), video_hash="h1", policy_digest="d")
        assert len(archive.affected_by({"AF-10"})) == 1
        assert archive.affected_by({"AF-01"}) == []

    def test_affected_by_also_matches_near_misses(self, tmp_path):
        archive = Archive(tmp_path / "a.db")
        archive.record(
            make_report(clauses=("AF-10",)),
            video_hash="h1",
            policy_digest="d",
            considered={"AF-08"},
        )
        assert len(archive.affected_by({"AF-08"})) == 1

    def test_selective_invalidation_leaves_untouched_videos_alone(self, tmp_path):
        archive = Archive(tmp_path / "a.db")
        for i in range(10):
            archive.record(
                make_report(filename=f"v{i}.mp4", clauses=("AF-01",)),
                video_hash=f"h{i}",
                policy_digest="d",
            )
        archive.record(
            make_report(filename="risky.mp4", clauses=("AF-08",)),
            video_hash="hx",
            policy_digest="d",
        )
        affected = archive.affected_by({"AF-08"})
        assert len(affected) == 1
        assert affected[0].filename == "risky.mp4"

    def test_stats(self, tmp_path):
        archive = Archive(tmp_path / "a.db")
        archive.record(make_report(), video_hash="h1", policy_digest="d")
        assert archive.stats()["videos"] == 1
        assert archive.stats()["reports"] == 1


@pytest.mark.skipif(not POLICY.is_dir(), reason="run scripts/build_corpus.py")
class TestSnapshot:
    def test_captures_every_clause_with_sections(self):
        corpus = load_corpus(POLICY)
        captured = snapshot(corpus)
        assert len(captured) == len(corpus.clauses)
        first = next(iter(captured.values()))
        assert first["sections"]
        assert first["sha256"]

    def test_round_trips_through_disk(self, tmp_path):
        corpus = load_corpus(POLICY)
        path = write_snapshot(corpus, tmp_path / "snap.json")
        restored = read_snapshot(path)
        assert restored["digest"] == corpus.digest
        assert len(restored["clauses"]) == len(corpus.clauses)

    def test_identical_corpus_produces_no_changes(self, tmp_path):
        corpus = load_corpus(POLICY)
        previous = read_snapshot(write_snapshot(corpus, tmp_path / "s.json"))
        assert diff_corpus(previous, corpus) == []


class TestDiff:
    def _snap(self, clauses: dict[str, dict]) -> dict:
        return {"version": "2026-08", "clauses": clauses}

    def _clause(self, sections: dict[str, str], title="Firearms") -> dict:
        text = "\n".join(f"## {k}\n{v}" for k, v in sections.items())
        return {
            "sha256": str(hash(text)),
            "title": title,
            "text": text,
            "sections": sections,
            "version": "2026-08",
        }

    def test_detects_an_added_clause(self, tmp_path):
        corpus = load_corpus(POLICY)
        previous = self._snap({})
        changes = diff_corpus(previous, corpus)
        assert all(c.kind == "ADDED" for c in changes)
        assert len(changes) == len(corpus.clauses)

    def test_detects_a_removed_clause(self, tmp_path):
        corpus = load_corpus(POLICY)
        captured = snapshot(corpus)
        captured["AF-99"] = self._clause({"Scope": "gone"}, title="Retired")
        changes = diff_corpus(self._snap(captured), corpus)
        assert [c.kind for c in changes] == ["REMOVED"]
        assert changes[0].clause_id == "AF-99"

    def test_section_level_delta_sees_a_change_whole_clause_comparison_misses(self):
        """A Yellow->Red move rewrites a fifth of the text. Compared whole, it
        scores like a typo; compared by section, it does not."""
        old = self._clause(
            {
                "Scope": "Firearms content." + " padding." * 40,
                "Yellow": "Range and demonstration content",
                "Red": "Manufacturing instructions",
            }
        )
        new = self._clause(
            {
                "Scope": "Firearms content." + " padding." * 40,
                "Yellow": "Safety instruction with no discharge",
                "Red": "Manufacturing instructions. Range and demonstration of any kind.",
            }
        )
        changes = diff_corpus(self._snap({"AF-08": old}), _FakeCorpus({"AF-08": new}))
        assert len(changes) == 1
        assert changes[0].semantic_delta >= SEMANTIC_FLOOR
        assert set(changes[0].sections_changed) == {"Yellow", "Red"}

    def test_a_cosmetic_edit_is_not_material(self):
        old = self._clause({"Scope": "Firearms and firearm-adjacent content."})
        new = self._clause({"Scope": "Firearms and firearm adjacent content."})
        changes = diff_corpus(self._snap({"AF-08": old}), _FakeCorpus({"AF-08": new}))
        assert changes[0].semantic_delta < SEMANTIC_FLOOR
        assert changes[0].material is False

    def test_added_and_removed_are_always_material(self):
        assert ClauseChange("AF-15", "New", "ADDED").material
        assert ClauseChange("AF-99", "Old", "REMOVED").material


class _FakeCorpus:
    """Minimal stand-in so diff tests do not need files on disk."""

    def __init__(self, clauses: dict[str, dict]) -> None:
        self._clauses = clauses
        self.version = "2026-09"
        self.digest = "fake"

    @property
    def clauses(self):
        from types import SimpleNamespace

        return [
            SimpleNamespace(
                clause_id=cid,
                title=data["title"],
                sections=data["sections"],
                sha256=data["sha256"],
                version=data["version"],
            )
            for cid, data in self._clauses.items()
        ]


@pytest.mark.skipif(not POLICY.is_dir(), reason="run scripts/build_corpus.py")
class TestDetect:
    def test_no_drift_against_its_own_snapshot(self, tmp_path):
        corpus = load_corpus(POLICY)
        path = write_snapshot(corpus, tmp_path / "s.json")
        archive = Archive(tmp_path / "a.db")

        report = detect(path, POLICY, archive)
        assert report.changes == []
        assert report.affected == []

    def test_reports_selectivity_against_the_archive(self, tmp_path):
        corpus = load_corpus(POLICY)
        path = write_snapshot(corpus, tmp_path / "s.json")
        archive = Archive(tmp_path / "a.db")
        for i in range(5):
            archive.record(
                make_report(filename=f"v{i}.mp4", clauses=("AF-01",)),
                video_hash=f"h{i}",
                policy_digest="d",
            )

        report = detect(path, POLICY, archive)
        assert report.archive_size == 5
        assert report.to_json()["selectivity"] == 0.0

    def test_serialises_to_json(self, tmp_path):
        corpus = load_corpus(POLICY)
        path = write_snapshot(corpus, tmp_path / "s.json")
        payload = detect(path, POLICY, Archive(tmp_path / "a.db")).to_json()
        assert json.dumps(payload)
        assert "detectedAt" in payload
        assert "selectivity" in payload
