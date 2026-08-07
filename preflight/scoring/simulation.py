"""What happens if the creator changes the video.

The report says a video is risky. The question a creator actually has is
what to do about it, and which of the things they could do is worth the
edit. This answers that without touching the video: every scenario is
computed from findings the run already produced, so a what-if costs
microseconds and no ffmpeg, no Whisper, no hosted call.

Two design decisions carry the whole module.

**Edits remove evidence, not findings.** Muting 10–12s does not delete a
finding; it deletes the *speech* observation inside that span. A finding
corroborated by vision survives the mute with lower confidence, and one that
existed only in the audio disappears. Simulating at the finding level —
"remove the finding, subtract its risk" — would get both of those wrong, and
would be wrong in the flattering direction, promising reductions the edit
cannot deliver.

**The predicted score is computed by the real scorer.** `sub_scores` and
`compute_readiness` are the same functions that produced the current score,
called on the modified findings. There is no second risk model to drift out
of step, which matters because the real one is deliberately non-linear: the
combiner saturates, so removing one of two CRITICAL findings barely moves
the number, and the anti-masking clamp means removing the *worst* finding
can move it a great deal. Any independent estimate would have to reproduce
both behaviours exactly, and would eventually fail to.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from preflight.models import Finding
from preflight.scoring.fusion import corroborate
from preflight.scoring.readiness import compute_readiness, sub_scores

# Which observations an edit destroys. This is the physical claim the whole
# simulation rests on, so each one is the conservative reading.
#
# BLEEP replaces a word with a tone: the speech evidence is gone, but the
# picture and any on-screen text are untouched. MUTE additionally takes the
# music bed and the acoustic events. BLUR_REGION covers part of the frame,
# so vision and OCR go and the audio stays. CUT removes the span entirely
# and therefore every modality in it.
EDIT_REMOVES: dict[str, frozenset[str]] = {
    "MUTE": frozenset({"speech", "audio", "music"}),
    "BLEEP": frozenset({"speech"}),
    "BLUR_REGION": frozenset({"vision", "ocr"}),
    "REPLACE_AUDIO": frozenset({"music", "audio"}),
    "CUT": frozenset({"speech", "audio", "music", "vision", "ocr", "access", "meta"}),
    # Disclosure and thumbnail changes address metadata rather than a moment
    # in the video, so they carry no span and clear the metadata observation.
    "INSERT_DISCLOSURE": frozenset({"meta"}),
    "REPLACE_THUMBNAIL": frozenset({"meta"}),
    "ADD_CAPTIONS": frozenset({"access"}),
}

# An edit cannot repair what it does not reach. A finding is affected only
# where the spans genuinely overlap, with a small tolerance for the fact
# that agents disagree about timing by a beat.
SPAN_TOLERANCE_MS = 250

# The impact a recommendation may reach. Taken from the remediation
# compiler's "balanced" ceiling rather than chosen here, because a scenario
# the compiler would refuse to lower is not a recommendation — it is advice
# the rest of the system cannot carry out.
DEFAULT_IMPACT_CEILING = 0.45


@dataclass(frozen=True)
class Edit:
    """One change a creator could make. Immutable — scenarios share these."""

    kind: str
    start_ms: int = 0
    end_ms: int = 0
    label: str = ""
    # File-scoped edits (captions, disclosure, thumbnail) apply everywhere
    # rather than to a span.
    whole_file: bool = False

    @property
    def removes(self) -> frozenset[str]:
        return EDIT_REMOVES.get(self.kind, frozenset())

    def covers(self, finding: Finding) -> bool:
        if self.whole_file:
            return True
        return (
            self.start_ms - SPAN_TOLERANCE_MS <= finding.endMs
            and self.end_ms + SPAN_TOLERANCE_MS >= finding.startMs
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "label": self.label or self.kind,
            "wholeFile": self.whole_file,
            "removes": sorted(self.removes),
        }


@dataclass(frozen=True)
class Scenario:
    """One hypothetical version of the video, scored by the real scorer."""

    name: str
    edits: tuple[Edit, ...]
    overall: int
    verdict: str
    sub: dict[str, float]
    surviving: int
    removed_finding_ids: tuple[str, ...]
    weakened_finding_ids: tuple[str, ...]
    delta: int = 0
    # Cumulative viewer impact, from the remediation compiler's own cost
    # table. A simulation ranking edits by a different impact model than the
    # compiler that renders them would give contradictory advice.
    impact: float = 0.0
    # Why an edit that clearly repaired something did not move the score.
    gated_by: str | None = None

    @property
    def value(self) -> float:
        """Score gained per unit of video sacrificed.

        Ranking on raw score alone recommends silencing every flagged span,
        which scores best and destroys the video. What a creator is choosing
        between is repairs worth making, so the ordering has to price what
        the repair costs them.
        """
        if self.delta <= 0:
            return 0.0
        return self.delta / max(self.impact, 0.01)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "edits": [e.to_json() for e in self.edits],
            "overall": self.overall,
            "verdict": self.verdict,
            "sub": self.sub,
            "delta": self.delta,
            "impact": round(self.impact, 3),
            "value": round(self.value, 2),
            "gatedBy": self.gated_by,
            "survivingFindings": self.surviving,
            "removedFindingIds": list(self.removed_finding_ids),
            "weakenedFindingIds": list(self.weakened_finding_ids),
        }


def apply_edits(
    findings: Iterable[Finding], edits: Iterable[Edit]
) -> tuple[list[Finding], list[str], list[str]]:
    """Rebuild the finding list as it would be after these edits.

    Returns the survivors, the ids that disappeared, and the ids that
    survived with less evidence than before. Pure: the input findings are
    never mutated, because a scenario that quietly edited the real run would
    make every later scenario wrong.
    """
    # Union the modalities removed at each finding, so two edits that both
    # strip speech over the same span strip it once. Accumulating
    # per-edit reductions instead would double-count the same repair — the
    # single most likely way for a simulation to promise a score it cannot
    # reach.
    survivors: list[Finding] = []
    removed: list[str] = []
    weakened: list[str] = []

    for finding in findings:
        stripped: set[str] = set()
        for edit in edits:
            if edit.covers(finding):
                stripped |= edit.removes

        if not stripped:
            survivors.append(finding)
            continue

        remaining = {
            name: value
            for name, value in (finding.modalities or {}).items()
            if name not in stripped and value > 0
        }

        if not remaining:
            # Every observation that supported this finding is gone. There is
            # nothing left to report — not a weaker finding, no finding.
            removed.append(finding.id)
            continue

        # Re-fuse through the real fusion path so a weakened finding is
        # scored exactly as a natively weak one would be.
        rebuilt = replace(finding, modalities=remaining)
        outcome = corroborate(rebuilt)
        survivors.append(
            replace(
                rebuilt,
                fusedConfidence=outcome.fused,
                severity=outcome.severity,
            )
        )
        weakened.append(finding.id)

    return survivors, removed, weakened


def viewer_impact(edits: Iterable[Edit]) -> float:
    """What these edits cost the video, on the compiler's own scale.

    Imported from `remediate.edl` rather than restated: the simulation and
    the compiler that actually renders the fix must agree about what a CUT
    costs, or the engine recommends an edit the compiler would then refuse
    to make under its own impact ceiling.
    """
    from preflight.remediate.edl import COST

    total = 0.0
    for edit in edits:
        cost = COST.get(edit.kind)
        total += cost[0] if cost else 0.05
    return round(total, 4)


def score(findings: list[Finding]) -> tuple[int, str, dict[str, float]]:
    """The real scorer, on hypothetical findings."""
    sub = sub_scores(findings)
    readiness = compute_readiness(sub)
    return readiness.overall, readiness.verdict, {k: round(v, 1) for k, v in sub.items()}


def simulate(
    findings: list[Finding],
    edits: Iterable[Edit],
    *,
    name: str = "scenario",
    baseline: int | None = None,
) -> Scenario:
    """Score one hypothetical version of the video."""
    edits = tuple(edits)
    survivors, removed, weakened = apply_edits(findings, edits)
    overall, verdict, sub = score(survivors)
    delta = overall - baseline if baseline is not None else 0

    # An edit that removed a real finding and moved nothing is not useless —
    # it is gated. The clamp holds the overall at the weakest dimension, so
    # repairing a healthier one cannot show until the weakest is addressed.
    # Reporting "+0" without saying that reads as "this edit is worthless",
    # which is both wrong and the opposite of the advice a creator needs.
    gated_by: str | None = None
    if (removed or weakened) and delta <= 0:
        weakest = min(sub, key=lambda key: sub[key])
        gated_by = weakest

    return Scenario(
        name=name,
        edits=edits,
        overall=overall,
        verdict=verdict,
        sub=sub,
        surviving=len(survivors),
        removed_finding_ids=tuple(removed),
        weakened_finding_ids=tuple(weakened),
        delta=delta,
        impact=viewer_impact(edits),
        gated_by=gated_by,
    )


def edit_for(finding: Finding) -> Edit | None:
    """The edit the finding itself recommends, if any."""
    kind = finding.suggestedFix
    if kind == "NONE" or kind not in EDIT_REMOVES:
        return None
    return Edit(
        kind=kind,
        start_ms=finding.startMs,
        end_ms=finding.endMs,
        label=f"{kind} {finding.clauseId}",
    )


@dataclass
class SimulationReport:
    baseline: Scenario
    scenarios: list[Scenario] = field(default_factory=list)

    def best_within(self, ceiling: float = DEFAULT_IMPACT_CEILING) -> Scenario:
        """The repair most worth making, among those actually renderable.

        Two wrong answers came before this one.

        Ranking on score alone recommended "silence every flagged span",
        which scored highest precisely because it destroyed the most audio.
        Ranking on score *per unit of impact* did not fix it either: muting
        three spans really is efficient by that measure, +14 for 0.75, and
        the ratio happily recommends gutting the video efficiently.

        The coherent constraint is the compiler's own. `remediate.edl` will
        not lower an edit set above its strategy ceiling, so a scenario at
        0.75 impact is one the renderer would refuse — recommending it is
        advice the rest of the system cannot carry out. Filtering to what is
        renderable first, then ranking by value, is the version where the
        engine and the compiler agree.
        """
        candidates = [
            s for s in self.scenarios if s.delta > 0 and s.impact <= ceiling
        ]
        if not candidates:
            return self.baseline
        return max(candidates, key=lambda s: (s.value, -len(s.edits)))

    @property
    def best(self) -> Scenario:
        return self.best_within()

    @property
    def highest_score(self) -> Scenario:
        """The best achievable number, whatever it costs. Kept separate
        from `best` so both questions can be answered without either
        pretending to be the other."""
        return max([self.baseline, *self.scenarios], key=lambda s: s.overall)

    def to_json(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.to_json(),
            "scenarios": [s.to_json() for s in self.scenarios],
            "best": self.best.name,
            "highestScore": self.highest_score.name,
        }


def explore(findings: list[Finding], duration_ms: int = 0) -> SimulationReport:
    """Build the scenario set a creator is actually choosing between.

    One scenario per recommended edit, so the marginal value of each is
    visible on its own, plus the combined case. Deliberately not the power
    set: with a dozen findings that is four thousand scenarios, almost all of
    them nonsense like "apply the third and seventh edits only", and the
    decision a creator faces is which edits to make rather than which subset
    is theoretically optimal.
    """
    baseline_overall, baseline_verdict, baseline_sub = score(findings)
    baseline = Scenario(
        name="current",
        edits=(),
        overall=baseline_overall,
        verdict=baseline_verdict,
        sub=baseline_sub,
        surviving=len(findings),
        removed_finding_ids=(),
        weakened_finding_ids=(),
    )

    recommended = [e for e in (edit_for(f) for f in findings) if e is not None]
    scenarios: list[Scenario] = []

    for edit in recommended:
        scenarios.append(
            simulate(
                findings,
                [edit],
                name=edit.label or edit.kind,
                baseline=baseline_overall,
            )
        )

    if len(recommended) > 1:
        scenarios.append(
            simulate(
                findings,
                recommended,
                name="apply every recommendation",
                baseline=baseline_overall,
            )
        )

    # Modality-isolated scenarios answer "what if only my audio changes" —
    # the question a creator with one editing skill actually has.
    for label, kind in (("silence every flagged span", "MUTE"),
                        ("blur every flagged region", "BLUR_REGION")):
        spans = [
            Edit(kind=kind, start_ms=f.startMs, end_ms=f.endMs, label=label)
            for f in findings
            if f.endMs > f.startMs
        ]
        if spans:
            scenarios.append(
                simulate(findings, spans, name=label, baseline=baseline_overall)
            )

    return SimulationReport(baseline=baseline, scenarios=scenarios)
