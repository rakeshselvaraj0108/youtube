"""Before / after evidence, against real rendered files.

These tests build actual videos with ffmpeg and actually cut them, because the
one claim this module makes — "the after frame came out of the remediated
file" — cannot be tested against a mock. A mocked extractor would happily
return the same bytes for both sides, which is precisely the bug.
"""

from __future__ import annotations

import subprocess

import pytest

from preflight import evidence, ffmpeg
from preflight.verify import FindingChange, TimeMap, compare

pytestmark = pytest.mark.skipif(
    not ffmpeg.available(), reason="ffmpeg is required for real evidence extraction"
)


class Op:
    def __init__(self, op: str, start_ms: int, end_ms: int) -> None:
        self.op, self.start_ms, self.end_ms = op, start_ms, end_ms

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """Six seconds in three flat colours, two seconds each.

    Flat colours on purpose: it makes "which second is this frame from"
    checkable by reading one pixel, so a test can prove an after-frame came
    from the shifted timeline rather than trusting that it did.
    """
    path = tmp_path_factory.mktemp("evidence") / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=red:s=320x240:d=2",
            "-f", "lavfi", "-i", "color=c=green:s=320x240:d=2",
            "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=2",
            "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[out]",
            "-map", "[out]", "-r", "25", "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture(scope="module")
def cut_clip(clip, tmp_path_factory):
    """The same video with the green two seconds removed: red then blue."""
    path = tmp_path_factory.mktemp("evidence-cut") / "clip.safe.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(clip),
            "-vf", "select='not(between(t,2,4))',setpts=N/25/TB",
            "-r", "25", "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def dominant_channel(path) -> str:
    """Which of R/G/B this still is mostly made of."""
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True,
        capture_output=True,
    ).stdout
    totals = [sum(raw[i::3]) for i in range(3)]
    return "RGB"[totals.index(max(totals))]


def finding(fid, *, clause="AF-01", start=0, end=1_000, category="Language"):
    return {
        "id": fid,
        "clauseId": clause,
        "category": category,
        "severity": "HIGH",
        "startMs": start,
        "endMs": end,
        "confidence": 0.8,
        "modalities": {"vision": 0.9},
        "evidence": {"transcript": "some words", "highlightSpan": [0, 4]},
    }


class TestFrameExtraction:
    def test_a_frame_comes_back_from_a_real_timestamp(self, clip, tmp_path):
        out = evidence.extract_frame(clip, 1_000, tmp_path / "f.jpg")
        assert out is not None and out.stat().st_size > 0

    def test_the_frame_is_from_the_instant_asked_for(self, clip, tmp_path):
        """Seeking must be accurate, not merely fast. An input-seek alone
        lands on the previous keyframe, which for this clip is a different
        colour and therefore a different claim about the video."""
        assert dominant_channel(evidence.extract_frame(clip, 1_000, tmp_path / "a.jpg")) == "R"
        assert dominant_channel(evidence.extract_frame(clip, 3_000, tmp_path / "b.jpg")) == "G"
        assert dominant_channel(evidence.extract_frame(clip, 5_000, tmp_path / "c.jpg")) == "B"

    def test_a_timestamp_past_the_end_returns_none_not_a_substitute(
        self, clip, tmp_path
    ):
        """None is the honest answer. The nearest decodable frame is a
        different moment, close enough to look right and wrong enough to
        mislead."""
        assert evidence.extract_frame(clip, 600_000, tmp_path / "d.jpg") is None

    def test_a_missing_file_returns_none(self, tmp_path):
        assert evidence.extract_frame(
            tmp_path / "nope.mp4", 1_000, tmp_path / "e.jpg"
        ) is None


class TestAfterFramesComeFromTheRenderedFile:
    """The single rule the module exists to enforce."""

    def test_the_after_frame_is_the_remediated_timeline_not_the_original(
        self, clip, cut_clip, tmp_path
    ):
        # A finding at 05.0s in the original. Two seconds were cut before it,
        # so it lives at 03.0s in the output — and both are blue, while 03.0s
        # in the *original* is green. Reading green here would prove the after
        # frame was taken from the wrong file.
        ops = [Op("CUT", 2_000, 4_000)]
        changes = [
            FindingChange(
                status="PERSISTING",
                clause_id="AF-01",
                category="Language",
                severity="HIGH",
                original_id="f1",
                remediated_id="x1",
            )
        ]
        pairs = evidence.build_pairs(
            changes,
            [finding("f1", start=4_800, end=5_200)],
            [finding("x1", start=2_800, end=3_200)],
            original_path=clip,
            remediated_path=cut_clip,
            ops=ops,
            remediation_id="REM-0001",
            original_run_id="run-a",
            verification_run_id="run-b",
            out_dir=tmp_path / "ev",
        )
        pair = pairs[0]
        assert pair.before is not None and pair.after is not None
        assert pair.before_ts_ms == 5_000
        assert pair.after_ts_ms == 3_000
        assert dominant_channel(pair.before.path) == "B"
        assert dominant_channel(pair.after.path) == "B"
        assert pair.after.source == "remediated"

    def test_a_cut_span_has_no_after_frame(self, clip, cut_clip, tmp_path):
        """The evidence was removed. That is a good outcome and there is
        nothing to photograph — so the pair says so rather than showing the
        original frame under an "after" label."""
        changes = [
            FindingChange(
                status="RESOLVED",
                clause_id="AF-01",
                category="Language",
                severity="HIGH",
                original_id="f1",
            )
        ]
        pairs = evidence.build_pairs(
            changes,
            [finding("f1", start=2_800, end=3_200)],
            [],
            original_path=clip,
            remediated_path=cut_clip,
            ops=[Op("CUT", 2_000, 4_000)],
            remediation_id="REM-0001",
            original_run_id="run-a",
            verification_run_id="run-b",
            out_dir=tmp_path / "ev",
        )
        pair = pairs[0]
        assert pair.removed_by_remediation is True
        assert pair.after is None
        assert pair.after_unavailable == "EVIDENCE REMOVED BY REMEDIATION"
        # The before frame is still real — the moment existed, it was cut.
        assert pair.before is not None
        assert dominant_channel(pair.before.path) == "G"

    def test_a_new_finding_has_no_before_frame(self, clip, cut_clip, tmp_path):
        """It did not exist in the input. Inventing a before frame would
        assert it did."""
        changes = [
            FindingChange(
                status="NEW",
                clause_id="AUD-01",
                category="Audio Delivery",
                severity="HIGH",
                remediated_id="x9",
            )
        ]
        pairs = evidence.build_pairs(
            changes,
            [],
            [finding("x9", clause="AUD-01", start=1_000, end=1_400)],
            original_path=clip,
            remediated_path=cut_clip,
            ops=[],
            remediation_id="REM-0001",
            original_run_id="run-a",
            verification_run_id="run-b",
            out_dir=tmp_path / "ev",
        )
        assert pairs[0].before is None
        assert pairs[0].after is not None

    def test_no_rendered_artifact_means_no_after_frame(self, clip, tmp_path):
        changes = [
            FindingChange(
                status="RESOLVED", clause_id="AF-01", category="Language",
                severity="HIGH", original_id="f1",
            )
        ]
        pairs = evidence.build_pairs(
            changes,
            [finding("f1")],
            [],
            original_path=clip,
            remediated_path=None,
            ops=[],
            remediation_id="REM-0001",
            original_run_id="run-a",
            verification_run_id=None,
            out_dir=tmp_path / "ev",
        )
        assert pairs[0].after is None
        assert "NOT MEASURED" in pairs[0].after_unavailable


class TestLineage:
    def test_each_side_names_the_run_it_came_from(self, clip, cut_clip, tmp_path):
        changes = [
            FindingChange(
                status="PERSISTING", clause_id="AF-01", category="Language",
                severity="HIGH", original_id="f1", remediated_id="x1",
            )
        ]
        pairs = evidence.build_pairs(
            changes,
            [finding("f1", start=800, end=1_200)],
            [finding("x1", start=800, end=1_200)],
            original_path=clip,
            remediated_path=cut_clip,
            ops=[Op("MUTE", 800, 1_200)],
            remediation_id="REM-0001",
            original_run_id="run-a",
            verification_run_id="run-b",
            out_dir=tmp_path / "ev",
            incidents=[{"id": "INC-001", "findingIds": ["f1"]}],
        )
        pair = pairs[0]
        assert pair.before_run_id == "run-a"
        assert pair.after_run_id == "run-b"
        assert pair.incident_id == "INC-001"
        assert pair.remediation is not None
        assert pair.remediation.op == "MUTE"
        assert pair.remediation.remediation_id == "REM-0001"

    def test_coverage_that_was_not_measured_is_none_not_zero(
        self, clip, cut_clip, tmp_path
    ):
        changes = [
            FindingChange(
                status="PERSISTING", clause_id="AF-01", category="Language",
                severity="HIGH", original_id="f1", remediated_id="x1",
            )
        ]
        pairs = evidence.build_pairs(
            changes,
            [finding("f1")],
            [finding("x1")],
            original_path=clip,
            remediated_path=cut_clip,
            ops=[],
            remediation_id="REM-0001",
            original_run_id="run-a",
            verification_run_id="run-b",
            out_dir=tmp_path / "ev",
            coverage={},
        )
        assert pairs[0].coverage is None
        assert pairs[0].to_json(embed=False)["before"]["coverage"] is None


class TestOrderingAndBudget:
    def test_the_interesting_findings_come_first(self, clip, cut_clip, tmp_path):
        """A reader asks about what appeared before what was fixed."""
        statuses = ["RESOLVED", "NEW", "PERSISTING", "INCONCLUSIVE"]
        changes = [
            FindingChange(
                status=s, clause_id=f"C-{i}", category="Language", severity="HIGH",
                original_id=None if s == "NEW" else f"f{i}",
                remediated_id=f"x{i}" if s in {"NEW", "PERSISTING"} else None,
            )
            for i, s in enumerate(statuses)
        ]
        pairs = evidence.build_pairs(
            changes,
            [finding(f"f{i}") for i in range(4)],
            [finding(f"x{i}") for i in range(4)],
            original_path=clip,
            remediated_path=cut_clip,
            ops=[],
            remediation_id="REM-0001",
            original_run_id="run-a",
            verification_run_id="run-b",
            out_dir=tmp_path / "ev",
        )
        assert [p.status for p in pairs][:2] == ["NEW", "PERSISTING"]

    def test_the_pair_count_is_bounded(self, clip, cut_clip, tmp_path):
        """Each pair costs two ffmpeg invocations; a verdict is carried by a
        handful of findings, not by all of them."""
        changes = [
            FindingChange(
                status="PERSISTING", clause_id=f"C-{i}", category="Language",
                severity="HIGH", original_id=f"f{i}", remediated_id=f"x{i}",
            )
            for i in range(30)
        ]
        pairs = evidence.build_pairs(
            changes,
            [finding(f"f{i}") for i in range(30)],
            [finding(f"x{i}") for i in range(30)],
            original_path=clip,
            remediated_path=cut_clip,
            ops=[],
            remediation_id="REM-0001",
            original_run_id="run-a",
            verification_run_id="run-b",
            out_dir=tmp_path / "ev",
            limit=4,
        )
        assert len(pairs) == 4


class TestSummary:
    def test_the_summary_counts_what_was_actually_extracted(
        self, clip, cut_clip, tmp_path
    ):
        result = compare(
            [finding("f1", start=2_800, end=3_200)],
            [],
            [Op("CUT", 2_000, 4_000)],
        )
        pairs = evidence.build_pairs(
            result.changes,
            [finding("f1", start=2_800, end=3_200)],
            [],
            original_path=clip,
            remediated_path=cut_clip,
            ops=[Op("CUT", 2_000, 4_000)],
            remediation_id="REM-0001",
            original_run_id="run-a",
            verification_run_id="run-b",
            out_dir=tmp_path / "ev",
        )
        summary = evidence.summarise(pairs)
        assert summary["pairs"] == 1
        assert summary["beforeFramesExtracted"] == 1
        assert summary["afterFramesExtracted"] == 0
        assert summary["removedByRemediation"] == 1


class TestTimeMapAgreement:
    def test_evidence_and_comparison_use_the_same_mapping(self):
        """Two mappings would eventually disagree about where a finding went,
        and the evidence would then illustrate a different claim than the one
        the verdict rests on."""
        ops = [Op("CUT", 2_000, 4_000)]
        assert TimeMap.from_ops(ops).to_remediated(5_000) == 3_000
