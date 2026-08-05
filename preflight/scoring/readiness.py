"""Release Readiness scoring.

This is a deliberate port of `computeReadiness` in src/lib/scoring.ts. The two
implementations must agree to the decimal, because the page renders one and the
JSON carries the other — a disagreement would show up as a report whose headline
number contradicts its own data. `tests/test_scoring.py` and the matching Vitest
test pin them together against shared vectors.

Two rules govern the number:

1. Every dimension runs the same direction. 100 is always good.
2. Overall is capped at `worst + 15`. A confirmed Content ID match scores 19 on
   copyright; without the cap four healthy dimensions average that away into a
   passing grade. One fatal flaw is never averaged away.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from preflight.models import Finding


def js_round(value: float) -> int:
    """Round half UP, the way JavaScript's Math.round does.

    Python's built-in round() is banker's rounding: round(84.5) is 84 here and
    85 in the browser. That single difference is enough to flip a verdict at a
    boundary and produce a page whose headline number disagrees with its own
    JSON. The cross-language test pins this.
    """
    return int(math.floor(value + 0.5))

WEIGHTS: dict[str, float] = {
    "policy": 0.40,
    "copyright": 0.30,
    "metadata": 0.12,
    "accessibility": 0.10,
    "audio": 0.08,
}
SUB_SCORE_ORDER = ["policy", "copyright", "metadata", "accessibility", "audio"]
CLAMP_HEADROOM = 15.0

# Risk contributed by one finding at full confidence.
SEVERITY_RISK: dict[str, float] = {
    "CRITICAL": 1.00,
    "HIGH": 0.55,
    "MEDIUM": 0.28,
    "LOW": 0.10,
}

# A violation in the opening weighs more: it is what a classifier and a human
# reviewer both see first.
EARLY_MS = 30_000
EARLY_MULTIPLIER = 1.35

# Which dimension each clause family scores against.
CLAUSE_DIMENSION: dict[str, str] = {
    "AF": "policy",
    "COPY": "copyright",
    "CID": "copyright",  # retained: earlier reports used this prefix
    "META": "metadata",
    "ACC": "accessibility",
    "AUD": "audio",
}


def dimension_for(clause_id: str) -> str:
    return CLAUSE_DIMENSION.get(clause_id.split("-")[0].upper(), "policy")


def finding_risk(finding: Finding) -> float:
    """Risk contribution of one finding, in [0, 1)."""
    severity = SEVERITY_RISK.get(finding.severity, 0.28)
    confidence = max(0.0, min(1.0, finding.fusedConfidence or finding.confidence))

    # Duration weighting models "how much of the video is affected", which is
    # the right lens for cumulative nuisance — a long stretch of mild language
    # is worse than one word. It is the wrong lens for a categorical breach: a
    # three-second graphic injury demonetises the whole upload, and discounting
    # it to 48% because it was brief would score that video as merely mediocre.
    # CRITICAL findings are therefore not duration-discounted at all.
    if finding.severity == "CRITICAL":
        weighted_duration = 1.0
    else:
        span_ms = max(0, finding.endMs - finding.startMs)
        duration = min(1.0, span_ms / 10_000) if span_ms else 1.0
        weighted_duration = 0.35 + 0.65 * duration

    position = EARLY_MULTIPLIER if finding.startMs < EARLY_MS else 1.0

    return min(0.99, severity * confidence * weighted_duration * position)


def risk_score(findings: list[Finding]) -> float:
    """Saturating combiner, 0-100.

    Many small findings can never exceed one severe one, which is the property
    that stops a video full of advisories outranking a video with a Content ID
    match.
    """
    product = 1.0
    for finding in findings:
        product *= 1.0 - finding_risk(finding)
    return 100.0 * (1.0 - product)


def sub_scores(findings: list[Finding]) -> dict[str, float]:
    """Derive all five dimensions from findings. Higher is always better."""
    buckets: dict[str, list[Finding]] = {key: [] for key in SUB_SCORE_ORDER}
    for finding in findings:
        buckets[dimension_for(finding.clauseId)].append(finding)
    return {key: round(100.0 - risk_score(group), 1) for key, group in buckets.items()}


def verdict_for(overall: int, worst: float) -> str:
    if overall >= 85 and worst >= 70:
        return "READY_TO_PUBLISH"
    if overall >= 70 and worst >= 50:
        return "PUBLISH_WITH_FIXES"
    if overall >= 50:
        return "NOT_READY"
    return "DO_NOT_PUBLISH"


@dataclass
class Readiness:
    overall: int
    weighted: float
    worst: float
    weakest: str
    verdict: str
    capped: bool

    def to_json(self, sub: dict[str, float]) -> dict:
        return {
            "overall": self.overall,
            "sub": {k: round(sub[k], 1) for k in SUB_SCORE_ORDER},
            "verdict": self.verdict,
            "weakest": self.weakest,
        }


def compute_readiness(sub: dict[str, float]) -> Readiness:
    weighted = sum(WEIGHTS[key] * sub[key] for key in SUB_SCORE_ORDER)

    # Ties resolve to the earlier key in SUB_SCORE_ORDER, which is descending by
    # weight — so `weakest` is stable and matches the TypeScript side exactly.
    weakest = SUB_SCORE_ORDER[0]
    for key in SUB_SCORE_ORDER:
        if sub[key] < sub[weakest]:
            weakest = key
    worst = sub[weakest]

    clamped = min(weighted, worst + CLAMP_HEADROOM)
    overall = js_round(max(0.0, min(100.0, clamped)))

    return Readiness(
        overall=overall,
        weighted=weighted,
        worst=worst,
        weakest=weakest,
        verdict=verdict_for(overall, worst),
        capped=(worst + CLAMP_HEADROOM) < weighted,
    )
