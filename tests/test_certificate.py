"""The verification certificate and its integrity digest.

Two properties carry the whole thing. The document must be **deterministic** —
re-issuing it for the same verification reproduces the same bytes, or the hash
certifies nothing. And it must **never invent a value** — a missing artifact
reads NOT MEASURED, an absent simulation reads null, and neither is quietly
rendered as a zero that looks like a measurement.
"""

from __future__ import annotations

import json

import pytest

from preflight import certificate as cert
from preflight.lineage import Artifact, RemediationRecord


def artifact(digest: str = "b3:aaaa", size: int = 1024, duration: int = 20_000):
    return Artifact(
        artifact_id=f"ART-{digest[-4:]}",
        path=f"/tmp/{digest[-4:]}.mp4",
        content_hash=digest,
        size_bytes=size,
        duration_ms=duration,
        created_at="2026-08-15T00:00:00+00:00",
    )


def remediation(**overrides) -> RemediationRecord:
    base = dict(
        remediation_id="REM-0001",
        source_run_id="run-original",
        simulation_id="SIM-0001",
        verification_run_id="run-verification",
        verification_id="VER-0001",
        artifact_id="ART-bbbb",
        source_path="/tmp/clip.mp4",
        output_path="/tmp/clip.safe.mp4",
        finding_ids=("AF-09", "AF-14"),
        incident_ids=("INC-001",),
        ops=({"op": "CUT", "startMs": 1_000, "endMs": 2_000},),
        state="PARTIALLY_REMEDIATED",
        previous_state="COMPARING",
        verdict="PARTIALLY_REMEDIATED",
        error=None,
        created_at="2026-08-15T00:00:00+00:00",
        updated_at="2026-08-15T00:05:00+00:00",
    )
    base.update(overrides)
    return RemediationRecord(**base)


def comparison(**overrides) -> dict:
    base = {
        "verdict": "PARTIALLY_REMEDIATED",
        "originalScore": 42,
        "remediatedScore": 42,
        "scoreDelta": 0,
        "predictedScore": 43,
        "predictionOutcome": "MATCHED",
        "resolved": 2,
        "persisting": 3,
        "new": 1,
        "inconclusive": 0,
        "structuralOk": True,
        "reanalysisOk": True,
        "changes": [
            {"status": "RESOLVED", "clauseId": "AF-09"},
            {"status": "RESOLVED", "clauseId": "AF-14"},
            {"status": "PERSISTING", "clauseId": "AUD-01"},
            {"status": "NEW", "clauseId": "AF-01"},
        ],
        "incidentChanges": [
            {"status": "RESOLVED"},
            {"status": "PERSISTING"},
            {"status": "NEW"},
        ],
    }
    base.update(overrides)
    return base


def report(findings: int = 6, incidents: int = 3, coverage: float = 0.81) -> dict:
    return {
        "meta": {"coverage": coverage, "policyVersion": "2026.08.01"},
        "findings": [{"id": f"F{i}"} for i in range(findings)],
        "incidents": [{"id": f"INC-{i:03d}"} for i in range(incidents)],
        "scores": {"overall": 42},
    }


def build(**overrides):
    kwargs = dict(
        remediation=remediation(),
        verification_id="VER-0001",
        comparison=comparison(),
        original_report=report(),
        remediated_report=report(findings=5, incidents=3),
        original_artifact=artifact("b3:aaaa"),
        remediated_artifact=artifact("b3:bbbb", size=899_746, duration=18_000),
        coverage={"speech": 0.95, "vision": 0.62},
        telemetry={"totalMs": 258_000},
        issued_at="2026-08-15T00:05:00+00:00",
        evidence={"pairs": 6, "afterFramesExtracted": 5},
    )
    kwargs.update(overrides)
    return cert.build(**kwargs)


class TestDeterminism:
    def test_the_same_verification_produces_the_same_document(self):
        assert build() == build()

    def test_the_same_document_produces_the_same_hash(self):
        assert cert.digest(build()) == cert.digest(build())

    def test_the_hash_does_not_depend_on_key_order(self):
        payload = build()
        shuffled = dict(reversed(list(payload.items())))
        assert cert.digest(payload) == cert.digest(shuffled)

    def test_the_issue_time_comes_from_the_record_not_the_clock(self):
        """A certificate re-issued for the same verification must be the same
        document. Stamping it with `now` would change the hash on every read,
        and a hash that always changes proves nothing."""
        assert build()["issuedAt"] == "2026-08-15T00:05:00+00:00"


class TestIntegrity:
    def test_a_sealed_certificate_verifies(self):
        assert cert.verify_integrity(cert.seal(build(), "CERT-0001"))

    def test_an_edited_certificate_stops_verifying(self):
        sealed = cert.seal(build(), "CERT-0001")
        sealed["verification"]["verdict"] = "VERIFIED_SAFE"
        assert not cert.verify_integrity(sealed)

    def test_a_changed_score_changes_the_hash(self):
        assert cert.digest(build()) != cert.digest(
            build(comparison=comparison(remediatedScore=91))
        )

    def test_a_changed_artifact_changes_the_hash(self):
        assert cert.digest(build()) != cert.digest(
            build(remediated_artifact=artifact("b3:cccc"))
        )

    def test_the_storage_id_is_outside_the_hash(self):
        """The id is allocated after the payload exists, so a hash covering it
        could never be recomputed from the payload alone."""
        payload = build()
        assert cert.seal(payload, "CERT-0001")["certificateHash"] == cert.seal(
            payload, "CERT-0099"
        )["certificateHash"]

    def test_an_unsealed_certificate_does_not_verify(self):
        assert not cert.verify_integrity(build())


class TestNothingIsInvented:
    def test_a_missing_artifact_reads_not_measured(self):
        payload = build(remediated_artifact=None)
        assert payload["artifacts"]["remediated"]["contentHash"] == cert.NOT_MEASURED
        assert payload["artifacts"]["remediated"]["sizeBytes"] == cert.NOT_MEASURED

    def test_an_absent_simulation_is_null_not_invented(self):
        payload = build(remediation=remediation(simulation_id=None))
        assert payload["lineage"]["simulationId"] is None

    def test_an_absent_prediction_is_not_a_prediction_of_zero(self):
        payload = build(comparison=comparison(predictedScore=None))
        assert payload["scores"]["predicted"] is None

    def test_a_failed_reanalysis_reports_no_actual_score(self):
        """A score from a run that did not happen is the one number this
        certificate must never carry."""
        payload = build(
            comparison=comparison(reanalysisOk=False, remediatedScore=0),
            remediated_report=None,
        )
        assert payload["scores"]["actual"] == cert.NOT_MEASURED
        assert payload["scores"]["delta"] == cert.NOT_MEASURED
        assert payload["verification"]["postAnalysis"] == "INCOMPLETE"
        assert any("did not complete" in line for line in payload["limitations"])

    def test_a_forecast_for_a_different_edit_is_flagged(self):
        """The compiler picks its own balanced operation set, which routinely
        differs from the highest-scoring scenario. Presenting that scenario's
        number as a forecast of *this* edit without saying so invites a reader
        to grade a prediction that was never made about it."""
        payload = build(
            comparison=comparison(
                predictedScenario="BLEEP AF-14", predictionIsForThisEdit=False
            )
        )
        assert payload["scores"]["predictedScenario"] == "BLEEP AF-14"
        assert payload["scores"]["predictionIsForThisEdit"] is False
        assert any("not from the operation set" in x for x in payload["limitations"])

    def test_a_forecast_for_this_edit_carries_no_caveat(self):
        payload = build(
            comparison=comparison(
                predictedScenario="apply every recommendation",
                predictionIsForThisEdit=True,
            )
        )
        assert not any("not from the operation set" in x for x in payload["limitations"])

    def test_the_certificate_claims_no_cryptographic_identity(self):
        payload = build()
        assert any(
            "not a signature" in line for line in payload["limitations"]
        )


class TestContentIsCopiedNotRecomputed:
    def test_every_lineage_id_is_carried(self):
        lineage = build()["lineage"]
        assert lineage == {
            "originalRunId": "run-original",
            "simulationId": "SIM-0001",
            "remediationId": "REM-0001",
            "verificationRunId": "run-verification",
            "verificationId": "VER-0001",
        }

    def test_the_scores_are_three_separate_numbers(self):
        scores = build()["scores"]
        assert (scores["original"], scores["predicted"], scores["actual"]) == (
            42,
            43,
            42,
        )

    def test_incident_counts_come_from_the_comparison(self):
        incidents = build()["incidents"]
        assert incidents["resolved"] == 1
        assert incidents["persisting"] == 1
        assert incidents["new"] == 1
        assert incidents["original"] == 3

    def test_finding_clauses_are_named_not_just_counted(self):
        findings = build()["findings"]
        assert findings["resolvedClauses"] == ["AF-09", "AF-14"]
        assert findings["newClauses"] == ["AF-01"]

    def test_the_lifecycle_state_travels_with_the_verdict(self):
        assert build()["verification"]["lifecycleState"] == "PARTIALLY_REMEDIATED"


class TestCoverageDisclosure:
    def test_a_thin_modality_is_named(self):
        payload = build(coverage={"speech": 0.95, "vision": 0.09})
        assert payload["coverage"]["belowFloor"] == ["vision"]
        assert any("vision" in line for line in payload["limitations"])

    def test_a_healthy_run_names_nothing(self):
        assert build(coverage={"speech": 0.95, "vision": 0.88})["coverage"][
            "belowFloor"
        ] == []

    def test_the_floor_travels_with_the_certificate(self):
        """A reader must be able to see the rule that produced the judgement,
        not just the judgement."""
        from preflight.verify import MIN_COVERAGE_FOR_ABSENCE

        assert build()["coverage"]["absenceFloor"] == MIN_COVERAGE_FOR_ABSENCE


class TestSerialisationAndRendering:
    def test_the_certificate_is_json(self):
        json.dumps(cert.seal(build(), "CERT-0001"))

    def test_rendering_names_every_required_field(self):
        text = cert.render(cert.seal(build(), "CERT-0001"))
        for label in (
            "PREFLIGHT VERIFICATION CERTIFICATE",
            "Certificate ID",
            "Certificate Hash",
            "Original Run",
            "Remediation",
            "Verification Run",
            "Original Artifact",
            "Remediated Artifact",
            "Original Score",
            "Predicted Score",
            "Actual Score",
            "Score Delta",
            "Original Incidents",
            "Resolved Incidents",
            "New Incidents",
            "Original Findings",
            "Resolved Findings",
            "Structural Verification",
            "Post-Analysis",
            "Coverage",
            "Prediction Outcome",
            "FINAL VERDICT",
        ):
            assert label in text, label

    def test_rendering_an_incomplete_certificate_says_so(self):
        text = cert.render(
            cert.seal(build(comparison=comparison(reanalysisOk=False)), "CERT-0002")
        )
        assert "NOT MEASURED" in text
        assert "INCOMPLETE" in text
