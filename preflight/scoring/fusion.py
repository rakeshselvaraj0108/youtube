"""Cross-modal confidence fusion.

Independent agents agreeing is evidence; one agent shouting is not. Fusion
combines per-modality confidences into a single number, then applies
corroboration rules that adjust severity based on *how many* modalities agree.

The rule that earns its place is DEMOTE. Vision-language models hallucinate
objects — they will report a weapon in a frame containing a tripod. A lone
vision claim below the confidence floor drops to advisory rather than driving a
demonetisation verdict, and that is a measurable precision gain rather than a
philosophical position.
"""

from __future__ import annotations

from dataclasses import dataclass

from preflight.models import Finding, Severity

# How much each modality's confidence is worth. Speech is exact — the word was
# said or it was not. Vision is inferential. Audio DSP is a proxy for a proxy.
MODALITY_WEIGHT: dict[str, float] = {
    "speech": 1.00,
    "music": 0.95,
    "access": 0.95,
    "meta": 0.90,
    "vision": 0.85,
    "ocr": 0.80,
    "metadata": 0.70,
    "audio": 0.60,
}
DEFAULT_WEIGHT = 0.5

PROMOTE_FUSED = 0.90
PROMOTE_MODALITIES = 2
DEMOTE_VISION_ONLY = 0.70
CONTRADICTION_SCALE = 0.75
CONTRADICTION_GAP = 0.5

SEVERITY_ORDER: list[Severity] = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def fuse(modalities: dict[str, float], coverage: dict[str, float] | None = None) -> float:
    """Noisy-or over weighted per-modality confidences.

    Each modality's contribution is scaled by that agent's actual coverage.
    When the vision agent only reached 42% of keyframes, a vision confidence of
    0.9 is not worth 0.9 — pretending otherwise reports certainty the run never
    earned.
    """
    product = 1.0
    for modality, confidence in modalities.items():
        if confidence <= 0:
            continue
        weight = MODALITY_WEIGHT.get(modality, DEFAULT_WEIGHT)
        if coverage is not None:
            weight *= max(0.0, min(1.0, coverage.get(modality, 1.0)))
        product *= 1.0 - weight * max(0.0, min(1.0, confidence))
    return round(1.0 - product, 4)


def _bump(severity: Severity, steps: int) -> Severity:
    index = SEVERITY_ORDER.index(severity) if severity in SEVERITY_ORDER else 1
    return SEVERITY_ORDER[max(0, min(len(SEVERITY_ORDER) - 1, index + steps))]


@dataclass
class FusionOutcome:
    fused: float
    severity: Severity
    rule: str | None = None
    requires_review: bool = False
    note: str | None = None


def corroborate(finding: Finding, coverage: dict[str, float] | None = None) -> FusionOutcome:
    """Apply the corroboration rules to one finding."""
    active = {m: c for m, c in finding.modalities.items() if c > 0}
    fused = fuse(active, coverage)
    severity = finding.severity
    rule: str | None = None
    requires_review = False
    note: str | None = None

    # CONTRADICTION — modalities that should agree do not. Something is wrong
    # with the evidence, so confidence drops and a human is asked to look.
    if len(active) >= 2:
        values = sorted(active.values())
        if values[-1] - values[0] >= CONTRADICTION_GAP:
            fused = round(fused * CONTRADICTION_SCALE, 4)
            requires_review = True
            rule = "CONTRADICTION"
            low = min(active, key=lambda m: active[m])
            high = max(active, key=lambda m: active[m])
            note = (
                f"{high} and {low} disagree ({active[high]:.2f} vs {active[low]:.2f}); "
                "confidence reduced and flagged for review"
            )

    # DEMOTE — a lone visual claim below the floor. VLMs hallucinate objects.
    elif set(active) <= {"vision"} and fused < DEMOTE_VISION_ONLY:
        if severity != "LOW":
            severity = "LOW"
            rule = "DEMOTE"
            note = (
                "single visual modality below the corroboration floor; "
                "held at advisory rather than acted on"
            )

    # PROMOTE — several independent modalities agreeing at high confidence.
    if (
        rule is None
        and len(active) >= PROMOTE_MODALITIES
        and fused >= PROMOTE_FUSED
        and severity in {"MEDIUM", "HIGH"}
    ):
        severity = _bump(severity, 1)
        rule = "PROMOTE"
        note = (
            f"{len(active)} independent modalities agree at {fused:.2f}; "
            "severity raised"
        )

    return FusionOutcome(
        fused=fused,
        severity=severity,
        rule=rule,
        requires_review=requires_review,
        note=note,
    )


def apply_fusion(
    findings: list[Finding], coverage: dict[str, float] | None = None
) -> list[str]:
    """Fuse every finding in place. Returns a log of the rules that fired."""
    log: list[str] = []
    for finding in findings:
        outcome = corroborate(finding, coverage)
        finding.fusedConfidence = outcome.fused
        if outcome.rule:
            before = finding.severity
            finding.severity = outcome.severity
            if outcome.note:
                finding.adversarial.rationale = (
                    f"{finding.adversarial.rationale} [{outcome.rule}: {outcome.note}]"
                ).strip()
            log.append(
                f"{outcome.rule} {finding.clauseId} {before}->{finding.severity} "
                f"(fused {outcome.fused:.2f})"
            )
    return log
