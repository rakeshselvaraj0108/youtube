"""Incident-level comparison, and the adversarial cases around it.

An incident is a group, so it can be genuinely half-fixed in a way a single
finding cannot — and that is where a comparison is most tempted to round in
the flattering direction. The tests here are mostly attempts to make the
system say VERIFIED_SAFE when it should not: renumbered ids, shifted
timestamps, a fix that introduces a new event, a re-analysis that barely
looked.

The one invariant behind all of them: **insufficient evidence must never
produce a clean verdict.**
"""

from __future__ import annotations

import pytest

from preflight.verify import compare, compare_incidents, TimeMap, verdict


class Op:
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
    modalities: dict | None = None,
) -> dict:
    return {
        "id": fid,
        "clauseId": clause,
        "category": category,
        "severity": severity,
        "startMs": start,
        "endMs": end,
        "modalities": modalities or {"speech": 0.9},
    }


def incident(
    iid: str,
    finding_ids: list[str],
    *,
    category: str = "Language",
    severity: str = "HIGH",
    start: int = 10_000,
    end: int = 12_000,
    clauses: list[str] | None = None,
) -> dict:
    return {
        "id": iid,
        "findingIds": finding_ids,
        "category": category,
        "severity": severity,
        "startMs": start,
        "endMs": end,
        "clauses": clauses or ["AF-01"],
    }


def compare_with(original_f, remediated_f, original_i, remediated_i, ops=(), **kw):
    return compare(
        original_f,
        remediated_f,
        ops,
        original_incidents=original_i,
        remediated_incidents=remediated_i,
        **kw,
    )


class TestIncidentIdentityIsNotById:
    """The second run renumbers INC-001..n by timestamp. After a cut moves
    everything, the numbers are actively misleading."""

    def test_a_persisting_incident_is_matched_across_renumbering(self):
        result = compare_with(
            [finding("f1")],
            [finding("SECOND-RUN-ID")],
            [incident("INC-001", ["f1"])],
            [incident("INC-007", ["SECOND-RUN-ID"])],
        )
        assert [i.status for i in result.incidents] == ["PERSISTING"]
        assert result.incidents[0].original_id == "INC-001"
        assert result.incidents[0].remediated_id == "INC-007"

    def test_the_same_number_on_a_different_event_is_not_a_match(self):
        """INC-001 in both runs, describing unrelated problems. Matching on
        the id would report this as persisting when it is one resolved and
        one new."""
        result = compare_with(
            [finding("f1", clause="AF-01", category="Language")],
            [finding("x1", clause="AUD-01", category="Audio Delivery",
                     start=50_000, end=52_000)],
            [incident("INC-001", ["f1"])],
            [incident("INC-001", ["x1"], category="Audio Delivery",
                      start=50_000, end=52_000, clauses=["AUD-01"])],
        )
        assert {i.status for i in result.incidents} == {"RESOLVED", "NEW"}

    def test_a_shifted_incident_is_matched_through_the_cut(self):
        result = compare_with(
            [finding("f1", start=30_000, end=32_000)],
            [finding("x1", start=20_000, end=22_000)],
            [incident("INC-001", ["f1"], start=30_000, end=32_000)],
            [incident("INC-001", ["x1"], start=20_000, end=22_000)],
            ops=[Op("CUT", 10_000, 20_000)],
        )
        assert [i.status for i in result.incidents] == ["PERSISTING"]
        assert result.incidents[0].mapped_span == (20_000, 22_000)


class TestIncidentRollup:
    def test_every_finding_resolved_resolves_the_incident(self):
        result = compare_with(
            [finding("f1"), finding("f2", start=10_500, end=11_500)],
            [],
            [incident("INC-001", ["f1", "f2"])],
            [],
        )
        assert result.incidents[0].status == "RESOLVED"
        assert set(result.incidents[0].resolved_findings) == {"f1", "f2"}

    def test_some_resolved_some_not_is_partially_remediated(self):
        """The status a finding cannot have, and the reason incidents get
        their own comparison."""
        result = compare_with(
            [finding("f1"), finding("f2", clause="AF-09", start=10_500, end=11_500)],
            [finding("x2", clause="AF-09", start=10_500, end=11_500)],
            [incident("INC-001", ["f1", "f2"], clauses=["AF-01", "AF-09"])],
            [incident("INC-001", ["x2"], clauses=["AF-09"])],
        )
        change = result.incidents[0]
        assert change.status == "PARTIALLY_REMEDIATED"
        assert change.resolved_findings == ("f1",)
        assert change.persisting_findings == ("f2",)

    def test_nothing_resolved_is_persisting(self):
        result = compare_with(
            [finding("f1")],
            [finding("x1")],
            [incident("INC-001", ["f1"])],
            [incident("INC-001", ["x1"])],
        )
        assert result.incidents[0].status == "PERSISTING"

    def test_a_softened_incident_is_changed(self):
        result = compare_with(
            [finding("f1", severity="CRITICAL")],
            [finding("x1", severity="LOW")],
            [incident("INC-001", ["f1"], severity="CRITICAL")],
            [incident("INC-001", ["x1"], severity="LOW")],
        )
        assert result.incidents[0].status == "CHANGED"

    def test_an_unchecked_finding_makes_the_whole_incident_inconclusive(self):
        """Two resolved and one nobody looked for is not two-thirds fixed. The
        incident's fate is unknown, and that has its own word."""
        vision = {"vision": 0.9}
        result = compare_with(
            [
                finding("f1", start=10_000, end=11_000),
                finding("f2", clause="AF-04", category="Violence",
                        start=11_000, end=12_000, modalities=vision),
            ],
            [],
            [incident("INC-001", ["f1", "f2"], clauses=["AF-01", "AF-04"])],
            [],
            coverage={"speech": 0.95, "vision": 0.05},
        )
        change = result.incidents[0]
        assert change.status == "INCONCLUSIVE"
        assert change.resolved_findings == ("f1",)
        assert change.inconclusive_findings == ("f2",)


class TestNewIncidents:
    def test_an_incident_only_in_the_output_is_new(self):
        result = compare_with(
            [finding("f1")],
            [finding("f1-again"),
             finding("x9", clause="AUD-01", category="Audio Delivery",
                     start=50_000, end=52_000)],
            [incident("INC-001", ["f1"])],
            [incident("INC-001", ["f1-again"]),
             incident("INC-002", ["x9"], category="Audio Delivery",
                      start=50_000, end=52_000, clauses=["AUD-01"])],
        )
        new = result.incidents_of("NEW")
        assert [i.remediated_id for i in new] == ["INC-002"]
        assert new[0].new_findings == ("x9",)

    def test_a_serious_new_incident_prevents_verified_safe(self):
        """Section 7's requirement. The demo's strongest moment: everything
        asked for was fixed, and something else appeared."""
        result = compare_with(
            [finding("f1")],
            [finding("x9", clause="AUD-01", category="Audio Delivery",
                     severity="HIGH", start=50_000, end=52_000)],
            [incident("INC-001", ["f1"])],
            [incident("INC-004", ["x9"], category="Audio Delivery",
                      severity="HIGH", start=50_000, end=52_000,
                      clauses=["AUD-01"])],
        )
        assert result.incidents_of("RESOLVED")
        assert verdict(result) == "NEW_RISK_DETECTED"

    def test_a_minor_new_incident_still_blocks_verified_safe(self):
        result = compare_with(
            [finding("f1")],
            [finding("x9", clause="AUD-01", category="Audio Delivery",
                     severity="LOW", start=50_000, end=52_000)],
            [incident("INC-001", ["f1"])],
            [incident("INC-004", ["x9"], category="Audio Delivery",
                      severity="LOW", start=50_000, end=52_000,
                      clauses=["AUD-01"])],
        )
        assert verdict(result) == "PARTIALLY_REMEDIATED"

    def test_a_regrouping_is_not_a_new_incident(self):
        """Correlation may split one original incident into two after an edit
        changes the timing. Both halves are matched findings, so calling the
        unclaimed half NEW would invent a problem that did not appear."""
        result = compare_with(
            [finding("f1", start=10_000, end=11_000),
             finding("f2", start=11_000, end=12_000)],
            [finding("x1", start=10_000, end=11_000),
             finding("x2", start=11_000, end=12_000)],
            [incident("INC-001", ["f1", "f2"])],
            [incident("INC-001", ["x1"], start=10_000, end=11_000),
             incident("INC-002", ["x2"], start=11_000, end=12_000)],
        )
        assert result.incidents_of("NEW") == []


class TestRemovedEvidence:
    def test_an_incident_inside_a_cut_is_marked_removed(self):
        result = compare_with(
            [finding("f1", start=12_000, end=14_000)],
            [],
            [incident("INC-001", ["f1"], start=12_000, end=14_000)],
            [],
            ops=[Op("CUT", 10_000, 20_000)],
        )
        change = result.incidents[0]
        assert change.status == "RESOLVED"
        assert change.removed_by_cut is True
        assert change.mapped_span is None
        assert "cut" in change.detail


class TestFailureModesNeverCertify:
    """Section 20's list. Each of these has an obvious wrong answer."""

    def test_a_remediation_that_fixes_everything_verifies(self):
        result = compare_with(
            [finding("f1"), finding("f2", start=40_000, end=41_000)],
            [],
            [incident("INC-001", ["f1"]),
             incident("INC-002", ["f2"], start=40_000, end=41_000)],
            [],
            coverage={"speech": 0.95},
        )
        assert verdict(result) == "VERIFIED_SAFE"
        assert len(result.incidents_of("RESOLVED")) == 2

    def test_a_remediation_that_fixes_nothing_fails(self):
        result = compare_with(
            [finding("f1")],
            [finding("x1")],
            [incident("INC-001", ["f1"])],
            [incident("INC-001", ["x1"])],
        )
        assert verdict(result) == "REMEDIATION_FAILED"

    def test_a_failed_structural_check_is_never_verified(self):
        result = compare_with(
            [finding("f1")], [], [incident("INC-001", ["f1"])], [],
            structural_ok=False,
        )
        assert verdict(result) == "REMEDIATION_FAILED"

    def test_an_incomplete_reanalysis_is_inconclusive(self):
        result = compare_with(
            [finding("f1")], [], [incident("INC-001", ["f1"])], [],
            reanalysis_ok=False,
        )
        assert verdict(result) == "INCONCLUSIVE"
        assert [i.status for i in result.incidents] == ["INCONCLUSIVE"]

    def test_low_coverage_cannot_certify(self):
        """The single most important negative: making the verification pass
        cheaper must not make its verdict kinder."""
        result = compare_with(
            [finding("f1", clause="AF-04", category="Violence",
                     modalities={"vision": 0.9})],
            [],
            [incident("INC-001", ["f1"], category="Violence")],
            [],
            coverage={"vision": 0.05},
        )
        assert result.incidents[0].status == "INCONCLUSIVE"
        assert verdict(result) != "VERIFIED_SAFE"

    def test_thin_coverage_on_one_agent_does_not_condemn_another(self):
        """The gate must be per-modality. A thin vision pass says nothing
        about a speech finding that speech examined thoroughly."""
        result = compare_with(
            [finding("f1", modalities={"speech": 0.9})],
            [],
            [incident("INC-001", ["f1"])],
            [],
            coverage={"speech": 0.95, "vision": 0.02},
        )
        assert result.incidents[0].status == "RESOLVED"
        assert verdict(result) == "VERIFIED_SAFE"


class TestSerialisation:
    def test_the_incident_comparison_reaches_the_payload(self):
        import json

        result = compare_with(
            [finding("f1")],
            [finding("x9", clause="AUD-01", category="Audio Delivery",
                     severity="LOW", start=50_000, end=52_000)],
            [incident("INC-001", ["f1"])],
            [incident("INC-004", ["x9"], category="Audio Delivery",
                      severity="LOW", start=50_000, end=52_000,
                      clauses=["AUD-01"])],
        )
        payload = result.to_json()
        json.dumps(payload)
        assert payload["incidentsResolved"] == 1
        assert payload["incidentsNew"] == 1
        assert len(payload["incidentChanges"]) == 2

    def test_a_run_with_no_incidents_still_serialises(self):
        payload = compare([], [], []).to_json()
        assert payload["incidentChanges"] == []
        assert payload["incidentsNew"] == 0


class TestScale:
    @pytest.mark.parametrize("count", [1, 25, 120])
    def test_incident_comparison_stays_linear(self, count):
        import time

        findings = [
            finding(f"f{i}", start=i * 3_000, end=i * 3_000 + 500)
            for i in range(count)
        ]
        incidents = [
            incident(f"INC-{i:03d}", [f"f{i}"], start=i * 3_000, end=i * 3_000 + 500)
            for i in range(count)
        ]
        started = time.perf_counter()
        result = compare_with(findings, [], incidents, [])
        assert time.perf_counter() - started < 1.0
        assert len(result.incidents) == count


class TestDirectEntryPoint:
    def test_compare_incidents_is_usable_on_its_own(self):
        """The rollup is a pure function of the finding verdicts, so it can be
        exercised without running a whole comparison."""
        base = compare([finding("f1")], [], [])
        changes = compare_incidents(
            [incident("INC-001", ["f1"])], [], base.changes, TimeMap()
        )
        assert [c.status for c in changes] == ["RESOLVED"]
