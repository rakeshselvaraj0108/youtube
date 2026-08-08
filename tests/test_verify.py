"""Closed-loop verification.

The claim under test is the product's whole point: that PREFLIGHT can say
"the problem is gone" on evidence rather than on ffmpeg's exit code. So the
tests that matter are the ones where a naive implementation would claim
success — a cut that shifted every later timestamp, a finding that vanished
because nothing looked for it, a fix that introduced a new problem.
"""

from __future__ import annotations

import pytest

from preflight.verify import (
    MATCH_IOU,
    TimeMap,
    compare,
    prediction_outcome,
    verdict,
)


class Op:
    """The shape `compare` reads off an EDL operation."""

    def __init__(self, op: str, start_ms: int, end_ms: int) -> None:
        self.op, self.start_ms, self.end_ms = op, start_ms, end_ms


def finding(
    fid: str,
    *,
    clause: str = "AF-01",
    category: str = "Language",
    severity: str = "HIGH",
    start: int = 10_000,
    end: int = 12_000,
) -> dict:
    return {
        "id": fid,
        "clauseId": clause,
        "category": category,
        "severity": severity,
        "startMs": start,
        "endMs": end,
    }


class TestTimeMapping:
    """A CUT moves everything after it. Comparing raw timestamps across one
    compares unrelated moments."""

    def test_no_cuts_is_the_identity(self):
        mapping = TimeMap.from_ops([Op("MUTE", 0, 5_000), Op("BLEEP", 9_000, 9_500)])
        assert mapping.identity
        assert mapping.to_remediated(30_000) == 30_000

    def test_content_after_a_cut_shifts_earlier(self):
        """The brief's own example: 00:10–00:20 removed, so a finding at
        00:30 lands at 00:20."""
        mapping = TimeMap.from_ops([Op("CUT", 10_000, 20_000)])
        assert mapping.to_remediated(30_000) == 20_000

    def test_content_before_a_cut_does_not_move(self):
        mapping = TimeMap.from_ops([Op("CUT", 10_000, 20_000)])
        assert mapping.to_remediated(5_000) == 5_000

    def test_a_moment_inside_a_cut_has_no_counterpart(self):
        """None is a real answer. Searching for a counterpart anyway is how
        a removed span gets misreported as an unresolved finding."""
        mapping = TimeMap.from_ops([Op("CUT", 10_000, 20_000)])
        assert mapping.to_remediated(15_000) is None

    def test_several_cuts_accumulate(self):
        mapping = TimeMap.from_ops(
            [Op("CUT", 10_000, 20_000), Op("CUT", 40_000, 45_000)]
        )
        assert mapping.to_remediated(50_000) == 35_000

    def test_a_span_crossing_a_cut_is_unmappable(self):
        mapping = TimeMap.from_ops([Op("CUT", 10_000, 20_000)])
        assert mapping.map_span(15_000, 16_000) is None

    def test_only_cuts_change_the_timeline(self):
        mapping = TimeMap.from_ops(
            [Op("MUTE", 0, 30_000), Op("BLUR_REGION", 0, 30_000)]
        )
        assert mapping.to_remediated(40_000) == 40_000


class TestResolution:
    def test_a_finding_that_is_gone_is_resolved(self):
        result = compare([finding("f1")], [], [])
        assert [c.status for c in result.changes] == ["RESOLVED"]

    def test_a_finding_that_remains_is_persisting(self):
        """Ids differ between runs, so matching on them would report this
        as one resolved plus one new — simultaneously wrong twice."""
        result = compare([finding("f1")], [finding("SECOND-RUN-ID")], [])
        assert [c.status for c in result.changes] == ["PERSISTING"]

    def test_a_cut_span_resolves_by_removal(self):
        result = compare(
            [finding("f1", start=12_000, end=14_000)],
            [],
            [Op("CUT", 10_000, 20_000)],
        )
        assert result.resolved
        assert "cut" in result.resolved[0].detail

    def test_a_shifted_finding_is_matched_not_double_counted(self):
        """The failure a naive diff produces: the cut moved the finding, so
        raw timestamps miss it and it is reported resolved AND new."""
        original = [finding("f1", start=30_000, end=32_000)]
        remediated = [finding("x1", start=20_000, end=22_000)]
        result = compare(original, remediated, [Op("CUT", 10_000, 20_000)])
        assert [c.status for c in result.changes] == ["PERSISTING"]

    def test_a_different_clause_at_the_same_time_is_not_the_same_problem(self):
        result = compare(
            [finding("f1", clause="AF-01")],
            [finding("x1", clause="VID-02", category="Accessibility")],
            [],
        )
        assert {c.status for c in result.changes} == {"RESOLVED", "NEW"}

    def test_a_softened_finding_is_changed_not_resolved(self):
        result = compare(
            [finding("f1", severity="CRITICAL")],
            [finding("x1", severity="LOW")],
            [],
        )
        assert result.changes[0].status == "CHANGED"
        assert "CRITICAL" in result.changes[0].detail


class TestNewRisk:
    """Remediation must answer both questions: did the problem go, and did
    the fix create another."""

    def test_a_finding_only_in_the_output_is_new(self):
        result = compare([], [finding("x1")], [])
        assert [c.status for c in result.changes] == ["NEW"]

    def test_a_new_serious_finding_outranks_every_resolution(self):
        result = compare(
            [finding("f1", severity="CRITICAL")],
            [finding("x1", clause="AUD-01", category="Audio Delivery",
                     severity="CRITICAL", start=50_000, end=52_000)],
            [],
        )
        assert verdict(result) == "NEW_RISK_DETECTED"

    def test_a_minor_new_finding_does_not_claim_verified_safe(self):
        result = compare(
            [finding("f1")],
            [finding("x1", clause="AUD-01", category="Audio Delivery",
                     severity="LOW", start=50_000, end=52_000)],
            [],
        )
        assert verdict(result) == "PARTIALLY_REMEDIATED"


class TestVerdict:
    def test_everything_resolved_and_nothing_new_is_verified_safe(self):
        result = compare([finding("f1"), finding("f2", start=40_000, end=42_000)], [], [])
        assert verdict(result) == "VERIFIED_SAFE"

    def test_some_resolved_some_persisting_is_partial(self):
        result = compare(
            [finding("f1"), finding("f2", clause="AF-09", start=40_000, end=42_000)],
            [finding("x1", clause="AF-09", start=40_000, end=42_000)],
            [],
        )
        assert verdict(result) == "PARTIALLY_REMEDIATED"

    def test_nothing_resolved_is_a_failure(self):
        result = compare([finding("f1")], [finding("x1")], [])
        assert verdict(result) == "REMEDIATION_FAILED"

    def test_no_findings_either_side_is_no_change(self):
        assert verdict(compare([], [], [])) == "NO_CHANGE"

    def test_a_structural_failure_is_never_verified(self):
        """ffmpeg exiting zero having written a broken file must not reach
        a success verdict."""
        result = compare([finding("f1")], [], [], structural_ok=False)
        assert verdict(result) == "REMEDIATION_FAILED"

    def test_a_failed_reanalysis_is_inconclusive_not_successful(self):
        """A finding that vanished from a run that never happened has not
        been fixed."""
        result = compare([finding("f1")], [], [], reanalysis_ok=False)
        assert verdict(result) == "INCONCLUSIVE"
        assert all(c.status == "INCONCLUSIVE" for c in result.changes)


class TestPredictionOutcome:
    """No percentage. Accuracy over one run has no defined denominator, and
    inventing one is the fabrication this loop exists to avoid."""

    def test_a_close_prediction_matched(self):
        assert prediction_outcome(82, 80) == "MATCHED"

    def test_a_distant_prediction_is_classified_by_direction(self):
        assert prediction_outcome(82, 50) == "OVERESTIMATED"
        assert prediction_outcome(50, 82) == "UNDERESTIMATED"

    def test_resolving_fewer_than_promised_overestimates(self):
        assert prediction_outcome(82, 82, predicted_resolved=3, actual_resolved=1) == (
            "OVERESTIMATED"
        )

    def test_resolving_more_than_promised_underestimates(self):
        assert prediction_outcome(82, 82, predicted_resolved=1, actual_resolved=3) == (
            "UNDERESTIMATED"
        )

    def test_no_prediction_is_inconclusive_not_zero(self):
        assert prediction_outcome(None, 80) == "INCONCLUSIVE"


class TestReportShape:
    def test_the_comparison_serialises_with_its_verdict(self):
        import json

        result = compare(
            [finding("f1")], [], [], original_score=45, remediated_score=78
        )
        payload = result.to_json()
        json.dumps(payload)
        assert payload["scoreDelta"] == 33
        assert payload["verdict"] == "VERIFIED_SAFE"
        assert payload["resolved"] == 1

    def test_scores_are_reported_separately_never_merged(self):
        """Original, predicted and actual are three different numbers and
        the original is never overwritten."""
        result = compare([], [], [], original_score=45, remediated_score=78)
        payload = result.to_json()
        assert payload["originalScore"] == 45
        assert payload["remediatedScore"] == 78


class TestScale:
    @pytest.mark.parametrize("count", [1, 10, 50, 200])
    def test_comparison_stays_linear(self, count):
        import time

        original = [
            finding(f"f{i}", start=i * 3_000, end=i * 3_000 + 500) for i in range(count)
        ]
        remediated = [
            finding(f"x{i}", start=i * 3_000, end=i * 3_000 + 500) for i in range(count)
        ]
        started = time.perf_counter()
        result = compare(original, remediated, [])
        assert time.perf_counter() - started < 1.0
        assert len(result.changes) == count

    def test_every_finding_is_classified_exactly_once(self):
        original = [finding(f"f{i}", start=i * 5_000, end=i * 5_000 + 400) for i in range(20)]
        remediated = original[:10]
        result = compare(original, remediated, [])
        assert len(result.changes) == 20
        assert len(result.resolved) + len(result.persisting) == 20


class TestCoverageGatesAbsence:
    """Making re-analysis cheaper must not make success more likely.

    Bounding the verification budget is what makes a closed loop terminate
    on a long video. Without this gate it would also make the loop
    dishonest: fewer frames examined means fewer findings detected means
    more findings apparently "resolved". Absence is only evidence when
    something actually looked.
    """

    def _vision_finding(self):
        f = finding("f1", clause="AF-04", category="Violence")
        f["modalities"] = {"vision": 0.9}
        return f

    def test_absence_under_thin_coverage_is_inconclusive(self):
        result = compare(
            [self._vision_finding()], [], [], coverage={"vision": 0.09}
        )
        assert result.changes[0].status == "INCONCLUSIVE"
        assert "too little" in result.changes[0].detail

    def test_absence_under_real_coverage_is_resolved(self):
        result = compare(
            [self._vision_finding()], [], [], coverage={"vision": 0.95}
        )
        assert result.changes[0].status == "RESOLVED"

    def test_a_thin_run_cannot_reach_verified_safe(self):
        """The property that matters: a cheap re-analysis must not be able
        to certify a video."""
        result = compare(
            [self._vision_finding()], [], [], coverage={"vision": 0.09}
        )
        assert verdict(result) != "VERIFIED_SAFE"

    def test_missing_coverage_information_does_not_block_resolution(self):
        """Callers that cannot supply coverage — the CLI, older reports —
        keep the previous behaviour rather than having every finding
        downgraded to inconclusive."""
        result = compare([self._vision_finding()], [], [])
        assert result.changes[0].status == "RESOLVED"

    def test_a_persisting_finding_is_unaffected_by_coverage(self):
        """The gate only governs absence. A finding still detected is still
        detected however little else was examined."""
        result = compare(
            [self._vision_finding()],
            [self._vision_finding()],
            [],
            coverage={"vision": 0.05},
        )
        assert result.changes[0].status == "PERSISTING"
