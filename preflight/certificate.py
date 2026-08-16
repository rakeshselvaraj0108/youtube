"""The verification certificate.

A verdict that lives in one HTTP response is an opinion. A verdict with the
hashes of both artifacts, both run ids, the coverage the second run actually
reached and the prediction it was checked against is a document someone else
can audit — including a reader who does not trust the system that produced it,
which is the only reader worth designing for.

Three properties are deliberate.

**Every field is copied, never computed.** The certificate does not re-score,
re-count or re-derive anything. It reads numbers the verification already
produced. A certificate that recomputes its own figures can disagree with the
verification it certifies, and then there are two answers.

**Absent is absent.** A run with no simulation has `simulationId: null`, not
an invented one; an artifact that could not be hashed says `NOT MEASURED`.
The certificate's whole value is that a blank means nobody measured it.

**The digest is an integrity check, not a signature.** It proves the payload
has not changed since it was issued, to anyone holding both. It proves nothing
about *who* issued it — there is no key here, and claiming otherwise would be
the one dishonest sentence in an artifact built to be honest.
"""

from __future__ import annotations

from typing import Any

from preflight import __version__, cas
from preflight.verify import MIN_COVERAGE_FOR_ABSENCE

CERTIFICATE_VERSION = "1.0"

# Fields assigned when the certificate is stored, so they cannot be inside
# what the digest covers — the id is allocated from the row count *after* the
# payload exists, and a hash covering its own storage id is unverifiable.
UNHASHED = ("certificateId", "certificateHash")

NOT_MEASURED = "NOT MEASURED"


def _artifact_json(artifact: Any) -> dict[str, Any]:
    if artifact is None:
        return {
            "artifactId": None,
            "contentHash": NOT_MEASURED,
            "sizeBytes": NOT_MEASURED,
            "durationMs": NOT_MEASURED,
            "path": None,
        }
    return {
        "artifactId": artifact.artifact_id,
        "contentHash": artifact.content_hash,
        "sizeBytes": artifact.size_bytes,
        "durationMs": artifact.duration_ms or NOT_MEASURED,
        "path": artifact.path,
    }


def build(
    *,
    remediation: Any,
    verification_id: str,
    comparison: dict[str, Any],
    original_report: dict[str, Any],
    remediated_report: dict[str, Any] | None,
    original_artifact: Any,
    remediated_artifact: Any,
    coverage: dict[str, float] | None,
    telemetry: dict[str, Any] | None,
    issued_at: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the payload. Deterministic given its inputs.

    `issued_at` comes from the stored verification record, not from the clock.
    Re-issuing a certificate for the same verification must produce the same
    document — otherwise the hash changes for a payload whose content did not,
    and the integrity check stops meaning anything.
    """
    original_incidents = original_report.get("incidents") or []
    remediated_incidents = (
        (remediated_report or {}).get("incidents") or [] if remediated_report else []
    )

    def count(status: str) -> int:
        return sum(
            1 for c in comparison.get("incidentChanges", []) if c.get("status") == status
        )

    def findings_of(status: str) -> list[str]:
        return [
            str(c.get("clauseId"))
            for c in comparison.get("changes", [])
            if c.get("status") == status
        ]

    reanalysis_ok = bool(comparison.get("reanalysisOk", False))
    structural_ok = bool(comparison.get("structuralOk", False))
    by_agent = {k: round(float(v), 4) for k, v in (coverage or {}).items()}
    gated = sorted(k for k, v in by_agent.items() if v < MIN_COVERAGE_FOR_ABSENCE)

    payload: dict[str, Any] = {
        "certificateVersion": CERTIFICATE_VERSION,
        "issuedAt": issued_at,
        "engineVersion": __version__,
        "policyVersion": original_report.get("meta", {}).get("policyVersion"),

        "lineage": {
            "originalRunId": remediation.source_run_id,
            "simulationId": remediation.simulation_id,
            "remediationId": remediation.remediation_id,
            "verificationRunId": remediation.verification_run_id,
            "verificationId": verification_id,
        },

        "artifacts": {
            "original": _artifact_json(original_artifact),
            "remediated": _artifact_json(remediated_artifact),
            "operations": [dict(op) for op in remediation.ops],
        },

        "scores": {
            "original": comparison.get("originalScore"),
            # None, not 0. A run with no simulation made no prediction, and a
            # predicted score of zero is a prediction.
            "predicted": comparison.get("predictedScore"),
            "actual": comparison.get("remediatedScore") if reanalysis_ok else NOT_MEASURED,
            "delta": comparison.get("scoreDelta") if reanalysis_ok else NOT_MEASURED,
            "predictionOutcome": comparison.get("predictionOutcome", "INCONCLUSIVE"),
            # Which scenario the forecast came from, and whether that scenario
            # is the edit that was actually rendered. A predicted score for a
            # different operation set is still informative, but a reader
            # comparing it against the actual is entitled to know which it is.
            "predictedScenario": comparison.get("predictedScenario"),
            "predictionIsForThisEdit": bool(
                comparison.get("predictionIsForThisEdit", False)
            ),
        },

        "incidents": {
            "original": len(original_incidents),
            "remediated": len(remediated_incidents),
            "resolved": count("RESOLVED"),
            "persisting": count("PERSISTING"),
            "partiallyRemediated": count("PARTIALLY_REMEDIATED"),
            "changed": count("CHANGED"),
            "new": count("NEW"),
            "inconclusive": count("INCONCLUSIVE"),
        },

        "findings": {
            "original": len(original_report.get("findings", [])),
            "remediated": (
                len((remediated_report or {}).get("findings", []))
                if remediated_report
                else NOT_MEASURED
            ),
            "resolved": comparison.get("resolved", 0),
            "persisting": comparison.get("persisting", 0),
            "new": comparison.get("new", 0),
            "inconclusive": comparison.get("inconclusive", 0),
            "resolvedClauses": sorted(set(findings_of("RESOLVED"))),
            "persistingClauses": sorted(set(findings_of("PERSISTING"))),
            "newClauses": sorted(set(findings_of("NEW"))),
        },

        "verification": {
            "structural": "PASSED" if structural_ok else "FAILED",
            "postAnalysis": "COMPLETE" if reanalysis_ok else "INCOMPLETE",
            "verdict": comparison.get("verdict", "INCONCLUSIVE"),
            "lifecycleState": remediation.state,
        },

        "coverage": {
            "byAgent": by_agent,
            "overall": (
                round(
                    (remediated_report or {}).get("meta", {}).get("coverage", 0.0), 4
                )
                if remediated_report
                else NOT_MEASURED
            ),
            "absenceFloor": MIN_COVERAGE_FOR_ABSENCE,
            # Named, not just counted. A reader deciding whether to trust a
            # RESOLVED needs to know which modality was thin, not that one was.
            "belowFloor": gated,
        },

        "evidence": evidence or {},
        "telemetry": telemetry or {},

        "limitations": [
            "PREFLIGHT predicts risk against published policy. It is not "
            "YouTube classifier output.",
            "The certificate hash is an integrity digest over this payload. It "
            "is not a signature and attests to no identity.",
            (
                "Findings absent from a modality below "
                f"{MIN_COVERAGE_FOR_ABSENCE:.0%} coverage are reported "
                "INCONCLUSIVE, never RESOLVED."
            ),
        ],
    }

    if not reanalysis_ok:
        payload["limitations"].append(
            "Re-analysis did not complete. No claim is made about whether the "
            "original findings were resolved."
        )
    if gated:
        payload["limitations"].append(
            "Coverage fell below the absence floor for: " + ", ".join(gated)
        )
    if comparison.get("predictedScenario") and not comparison.get(
        "predictionIsForThisEdit"
    ):
        payload["limitations"].append(
            f"The predicted score is from scenario "
            f"'{comparison['predictedScenario']}', not from the operation set "
            "that was rendered. Only the scores are comparable."
        )

    return payload


def digest(payload: dict[str, Any]) -> str:
    """A content hash over the certificate, excluding its own storage fields.

    Canonical JSON with sorted keys, so a payload that differs only in key
    order hashes the same and a payload that differs in any *value* does not.
    """
    body = {k: v for k, v in payload.items() if k not in UNHASHED}
    return cas.prefixed(cas.hash_json(body))


def seal(payload: dict[str, Any], certificate_id: str) -> dict[str, Any]:
    """Attach the id and the digest. The payload is not otherwise touched."""
    sealed = dict(payload)
    sealed["certificateId"] = certificate_id
    sealed["certificateHash"] = digest(payload)
    return sealed


def verify_integrity(sealed: dict[str, Any]) -> bool:
    """Does the stored hash still match the stored payload?

    The whole point of storing the digest: a certificate edited after issue —
    by hand, by a bad merge, by a bug — stops verifying, and says so.
    """
    claimed = sealed.get("certificateHash")
    return bool(claimed) and claimed == digest(sealed)


def render(sealed: dict[str, Any]) -> str:
    """The certificate as text, for the terminal and for `preflight verify`."""
    lineage = sealed.get("lineage", {})
    scores = sealed.get("scores", {})
    incidents = sealed.get("incidents", {})
    findings = sealed.get("findings", {})
    verification = sealed.get("verification", {})
    artifacts = sealed.get("artifacts", {})

    def row(label: str, value: Any) -> str:
        return f"  {label:<26} {value}"

    lines = [
        "",
        "  PREFLIGHT VERIFICATION CERTIFICATE",
        "  " + "-" * 58,
        row("Certificate ID", sealed.get("certificateId", NOT_MEASURED)),
        row("Certificate Hash", sealed.get("certificateHash", NOT_MEASURED)),
        row("Issued", sealed.get("issuedAt", NOT_MEASURED)),
        "",
        row("Original Run", lineage.get("originalRunId")),
        row("Simulation", lineage.get("simulationId") or NOT_MEASURED),
        row("Remediation", lineage.get("remediationId")),
        row("Verification Run", lineage.get("verificationRunId") or NOT_MEASURED),
        row("Verification", lineage.get("verificationId")),
        "",
        row("Original Artifact", artifacts.get("original", {}).get("contentHash")),
        row("Remediated Artifact", artifacts.get("remediated", {}).get("contentHash")),
        row("Original Duration", artifacts.get("original", {}).get("durationMs")),
        row("Remediated Duration", artifacts.get("remediated", {}).get("durationMs")),
        "",
        row("Original Score", scores.get("original")),
        row("Predicted Score", scores.get("predicted") if scores.get("predicted") is not None else NOT_MEASURED),
        row("Actual Score", scores.get("actual")),
        row("Score Delta", scores.get("delta")),
        row("Prediction Outcome", scores.get("predictionOutcome")),
        "",
        row("Original Incidents", incidents.get("original")),
        row("Resolved Incidents", incidents.get("resolved")),
        row("Persisting Incidents", incidents.get("persisting")),
        row("Partial Incidents", incidents.get("partiallyRemediated")),
        row("New Incidents", incidents.get("new")),
        "",
        row("Original Findings", findings.get("original")),
        row("Resolved Findings", findings.get("resolved")),
        row("Persisting Findings", findings.get("persisting")),
        row("New Findings", findings.get("new")),
        row("Inconclusive Findings", findings.get("inconclusive")),
        "",
        row("Structural Verification", verification.get("structural")),
        row("Post-Analysis", verification.get("postAnalysis")),
        row("Coverage", sealed.get("coverage", {}).get("overall")),
        "",
        row("FINAL VERDICT", verification.get("verdict")),
        "",
    ]
    return "\n".join(lines)
