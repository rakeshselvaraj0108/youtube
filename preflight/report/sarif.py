"""SARIF 2.1.0 and the release certificate.

SARIF is what CodeQL and Semgrep speak, so these findings render natively in
GitHub's Security tab and a video repository can genuinely fail CI.

SARIF has no time axis. Rather than invent one, seconds stand in for
`startLine` so GitHub renders a position, and true millisecond spans travel in
`properties` where a consumer that understands video can read them. That is a
deliberate mapping, not a side effect of copying the schema — and it mirrors
`src/lib/sarif.ts` exactly so both emitters agree.
"""

from __future__ import annotations

from typing import Any

SARIF_SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"
INFORMATION_URI = "https://github.com/rakeshselvaraj0108/youtube"

SARIF_LEVEL = {
    "CRITICAL": "error",
    "HIGH": "error",
    "MEDIUM": "warning",
    "LOW": "note",
}

PASSING_VERDICTS = {"READY_TO_PUBLISH", "PUBLISH_WITH_FIXES"}


def _rule_name(title: str) -> str:
    cleaned = "".join(c if c.isalnum() or c == " " else "" for c in title)
    return "".join(word.capitalize() for word in cleaned.split())


def _timecode(ms: int) -> str:
    total = ms // 1000
    return f"{total // 60:02d}:{total % 60:02d}.{(ms % 1000) // 100}"


def build_sarif(report: dict[str, Any]) -> dict[str, Any]:
    findings = report["findings"]

    # One rule per distinct clause. SARIF rules are a set; two Language
    # findings must not produce two AF-01 entries.
    rules: dict[str, dict[str, Any]] = {}
    for finding in findings:
        clause = finding["clauseId"]
        if clause in rules:
            continue
        policy = finding["policy"]
        rules[clause] = {
            "id": clause,
            "name": _rule_name(policy["title"]),
            "shortDescription": {"text": policy["title"]},
            "fullDescription": {"text": policy["text"]},
            "defaultConfiguration": {
                "level": SARIF_LEVEL.get(finding["severity"], "warning")
            },
            "properties": {
                "tags": [
                    "monetization",
                    "advertiser-friendly",
                    finding["category"].lower(),
                ],
                "section": policy["section"],
            },
        }

    results = []
    for finding in findings:
        fix = finding["suggestedFix"]
        fix_text = (
            "No automated fix available."
            if fix == "NONE"
            else f"Fix: {fix} {finding['startMs']}-{finding['endMs']}ms."
        )
        results.append(
            {
                "ruleId": finding["clauseId"],
                "level": SARIF_LEVEL.get(finding["severity"], "warning"),
                "message": {
                    "text": (
                        f"{finding['title']} at {_timecode(finding['startMs'])} — "
                        f"{finding['severity']} (conf {finding['confidence']:.2f}). "
                        f"{fix_text}"
                    )
                },
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": report["video"]["filename"]},
                            "region": {
                                # Seconds stand in for lines. True timings below.
                                "startLine": max(1, finding["startMs"] // 1000),
                                "endLine": max(1, finding["endMs"] // 1000),
                                "startColumn": 1,
                                "snippet": {"text": finding["evidence"]["transcript"]},
                            },
                        }
                    }
                ],
                "properties": {
                    "startMs": finding["startMs"],
                    "endMs": finding["endMs"],
                    "confidence": finding["confidence"],
                    "fusedConfidence": finding["fusedConfidence"],
                    "modalities": finding["modalities"],
                    "severity": finding["severity"],
                    "category": finding["category"],
                    "suggestedFix": fix,
                    "policySection": finding["policy"]["section"],
                    "auditorCharge": finding["adversarial"]["auditor"]["charge"],
                    "advocateDefense": finding["adversarial"]["advocate"]["defense"],
                    "advocateStrength": finding["adversarial"]["advocate"]["strength"],
                    "adjudicatorVerdict": finding["adversarial"]["adjudicator"]["verdict"],
                    "adjudicatorRationale": finding["adversarial"]["adjudicator"][
                        "rationale"
                    ],
                },
            }
        )

    return {
        "$schema": SARIF_SCHEMA,
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "PREFLIGHT",
                        "version": report["meta"]["engineVersion"],
                        "informationUri": INFORMATION_URI,
                        "rules": list(rules.values()),
                    }
                },
                "results": results,
                "invocations": [
                    {
                        "executionSuccessful": True,
                        "endTimeUtc": report["meta"]["analyzedAt"],
                        "properties": {
                            # Coverage is reported, never hidden. A consumer can
                            # see this run did not inspect everything.
                            "coverage": report["meta"]["coverage"],
                            "policyVersion": report["meta"]["policyVersion"],
                            "degradedAgents": [
                                {
                                    "id": a["id"],
                                    "status": a["status"],
                                    "coverage": a["coverage"],
                                }
                                for a in report["agents"]
                                if a["status"] != "OK"
                            ],
                        },
                    }
                ],
            }
        ],
    }


def exit_code(report: dict[str, Any]) -> int:
    return 0 if report["scores"]["verdict"] in PASSING_VERDICTS else 1


def build_certificate(
    report: dict[str, Any],
    *,
    models: dict[str, str],
    policy_digest: str,
    video_hash: str,
    retrieval_backend: str,
) -> dict[str, Any]:
    """Proof of what was checked, against which rules, with which models.

    A certificate that says only "94/100" is marketing. One that says "94/100,
    against policy 2026-08, having inspected 83% of the analysis surface, with
    these model ids" is evidence — so the scoring rule and the coverage matrix
    travel with the number and a client can recompute it.
    """
    from preflight.scoring.readiness import CLAMP_HEADROOM, WEIGHTS, compute_readiness

    sub = {k: float(v) for k, v in report["scores"]["sub"].items()}
    readiness = compute_readiness(sub)
    severities = [f["severity"] for f in report["findings"]]

    return {
        "certificateVersion": "1.0",
        "generatedAt": report["meta"]["analyzedAt"],
        "subject": {
            "filename": report["video"]["filename"],
            "durationMs": report["video"]["durationMs"],
            "resolution": f"{report['video']['width']}x{report['video']['height']}",
            "fps": report["video"]["fps"],
            "sizeBytes": report["video"]["sizeBytes"],
            "videoHash": video_hash,
        },
        "attestation": {
            "attestationHash": report["meta"]["attestationHash"],
            "policyVersion": report["meta"]["policyVersion"],
            "policyCorpusDigest": policy_digest,
            "engineVersion": report["meta"]["engineVersion"],
            "models": models,
            "retrievalBackend": retrieval_backend,
        },
        "readiness": {
            "overall": report["scores"]["overall"],
            "verdict": report["scores"]["verdict"],
            "weakestDimension": report["scores"]["weakest"],
            "subScores": report["scores"]["sub"],
            "weights": WEIGHTS,
            "clamp": f"overall = min(weightedMean, worst + {CLAMP_HEADROOM:.0f})",
            "weightedMean": round(readiness.weighted, 2),
            "clampBound": readiness.capped,
        },
        "coverage": {
            "overall": report["meta"]["coverage"],
            "agents": [
                {
                    "id": a["id"],
                    "name": a["name"],
                    "status": a["status"],
                    "coverage": a["coverage"],
                    "elapsedMs": a["elapsedMs"],
                    "calls": a["calls"],
                }
                for a in report["agents"]
            ],
        },
        "findings": {
            "total": len(severities),
            "bySeverity": {
                level: severities.count(level)
                for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
            },
            "clauses": sorted({f["clauseId"] for f in report["findings"]}),
        },
        "remediation": {
            "operations": len(report["remediation"]["ops"]),
            "renderMs": report["remediation"]["renderMs"],
            "videoStreamCopied": report["remediation"]["videoStreamCopied"],
            "ffmpegCommand": report["remediation"]["ffmpegCommand"],
        },
        "ci": {"exitCode": exit_code(report)},
        "limitations": [
            "PREFLIGHT predicts risk against published policy. It is not YouTube "
            "classifier output and is not affiliated with YouTube.",
            "A public fingerprint match predicts a Content ID claim. The absence of "
            "a match does not prove safety: Content ID's reference database is "
            "private and larger than any public one.",
            f"Analysis covered {report['meta']['coverage'] * 100:.0f}% of the "
            "surface; see coverage.agents for what each agent actually inspected.",
        ],
    }
