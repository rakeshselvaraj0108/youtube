"""Segment rollup — where the risk actually lives in a long video."""

from __future__ import annotations

import pytest

from preflight.models import Adversarial, Evidence, Finding, PolicyRef
from preflight.scoring.rollup import concentration, rollup


def finding(
    start: int,
    end: int,
    *,
    severity: str = "HIGH",
    clause: str = "AF-01",
    fid: str = "f",
) -> Finding:
    return Finding(
        id=fid,
        clauseId=clause,
        category="Language",
        title="t",
        description="d",
        startMs=start,
        endMs=end,
        severity=severity,  # type: ignore[arg-type]
        confidence=0.9,
        modalities={"speech": 0.9},
        evidence=Evidence(transcript=""),
        policy=PolicyRef("AF-01", "t", "s", "x"),
        adversarial=Adversarial(charge="c", rationale="r", confidence=0.9),
    )


HOUR = 3_600_000
TEN_MIN = 600_000


class TestSegmentation:
    def test_segments_tile_the_whole_timeline_without_gaps(self):
        segments = sorted(rollup([], HOUR), key=lambda s: s.index)
        assert segments[0].start_ms == 0
        assert segments[-1].end_ms == HOUR
        for earlier, later in zip(segments, segments[1:]):
            assert earlier.end_ms == later.start_ms

    def test_a_short_final_segment_is_not_padded_past_the_video(self):
        segments = sorted(rollup([], TEN_MIN + 1000), key=lambda s: s.index)
        assert segments[-1].end_ms == TEN_MIN + 1000

    def test_no_findings_means_no_risk_anywhere(self):
        assert all(s.risk_share == 0.0 for s in rollup([], HOUR))

    def test_a_zero_length_video_rolls_up_to_nothing(self):
        assert rollup([], 0) == []


class TestRiskAttribution:
    def test_shares_sum_to_one_when_anything_was_found(self):
        segments = rollup(
            [finding(60_000, 70_000, fid="a"), finding(2_000_000, 2_010_000, fid="b")],
            HOUR,
        )
        assert sum(s.risk_share for s in segments) == pytest.approx(1.0, abs=0.001)

    def test_the_segment_holding_the_findings_ranks_first(self):
        segments = rollup(
            [finding(2_000_000, 2_010_000, fid="a", severity="CRITICAL")], HOUR
        )
        assert segments[0].start_ms <= 2_000_000 < segments[0].end_ms
        assert segments[0].risk_share == pytest.approx(1.0, abs=0.001)

    def test_a_finding_spanning_a_boundary_counts_in_both_segments(self):
        """It genuinely affects both, and attributing it to whichever
        segment happens to hold its start would under-report the other."""
        segments = {
            s.index: s for s in rollup([finding(TEN_MIN - 5_000, TEN_MIN + 5_000)], HOUR)
        }
        assert segments[0].finding_count == 1
        assert segments[1].finding_count == 1

    def test_severity_drives_the_ranking_not_finding_count(self):
        """Three advisories must not outrank one critical — the same
        anti-averaging property the headline score has."""
        many_low = [
            finding(60_000 + i * 1000, 61_000 + i * 1000, severity="LOW", fid=f"l{i}")
            for i in range(3)
        ]
        one_critical = [finding(2_000_000, 2_010_000, severity="CRITICAL", fid="c")]
        segments = rollup(many_low + one_critical, HOUR)
        assert segments[0].start_ms <= 2_000_000 < segments[0].end_ms

    def test_dominant_clause_is_the_most_frequent_in_that_segment(self):
        findings = [
            finding(60_000, 61_000, clause="AF-02", fid="a"),
            finding(62_000, 63_000, clause="AF-02", fid="b"),
            finding(64_000, 65_000, clause="AF-09", fid="c"),
        ]
        first = next(s for s in rollup(findings, HOUR) if s.index == 0)
        assert first.dominant_clause == "AF-02"

    def test_worst_severity_is_reported_per_segment(self):
        findings = [
            finding(60_000, 61_000, severity="LOW", fid="a"),
            finding(62_000, 63_000, severity="CRITICAL", fid="b"),
        ]
        first = next(s for s in rollup(findings, HOUR) if s.index == 0)
        assert first.worst_severity == "CRITICAL"


class TestConcentration:
    def test_names_the_few_segments_carrying_most_of_the_risk(self):
        findings = [
            finding(60_000, 70_000, severity="CRITICAL", fid="a"),
            finding(2_000_000, 2_010_000, severity="CRITICAL", fid="b"),
            finding(3_000_000, 3_001_000, severity="LOW", fid="c"),
        ]
        hot = concentration(rollup(findings, HOUR))
        assert 0 < len(hot) < 6
        assert sum(s.risk_share for s in hot) >= 0.8

    def test_returns_them_in_timeline_order(self):
        findings = [
            finding(3_000_000, 3_010_000, severity="CRITICAL", fid="late"),
            finding(60_000, 70_000, severity="CRITICAL", fid="early"),
        ]
        hot = concentration(rollup(findings, HOUR))
        assert [s.index for s in hot] == sorted(s.index for s in hot)

    def test_a_clean_video_concentrates_nowhere(self):
        assert concentration(rollup([], HOUR)) == []
