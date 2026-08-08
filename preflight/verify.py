"""Closed-loop verification — did the remediation actually work?

A successful render is not a successful remediation. ffmpeg exiting zero
proves a file was written; it proves nothing about whether the problem the
edit was meant to fix is still in the picture. The only evidence that a fix
worked is the same pipeline finding fewer problems in the output than it
found in the input.

So this module never deletes findings from the original report to simulate
success. The rendered file goes back through the real pipeline, and this
compares the two reports.

Three things make the comparison harder than diffing two lists.

**Ids do not survive.** The second run assigns its own finding ids, so
matching on them would report every finding as resolved and every one as
new, simultaneously. Identity here is the clause, the category and the time
span — what the finding *is*, not what it was called.

**Timestamps move.** A CUT removes ten seconds, so everything after it
shifts earlier by ten seconds. Comparing raw timestamps across a cut
compares unrelated moments. The EDL is the authority on that shift, and
`TimeMap` applies it.

**Absence is not proof.** A finding that vanished from a region the second
run never examined has not been fixed; it has been unobserved. Where the
mapping cannot place a finding, or coverage collapsed, the comparison says
INCONCLUSIVE rather than claiming a resolution it cannot support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

Status = Literal[
    "RESOLVED", "PERSISTING", "NEW", "CHANGED", "INCONCLUSIVE"
]

Verdict = Literal[
    "VERIFIED_SAFE",
    "PARTIALLY_REMEDIATED",
    "REMEDIATION_FAILED",
    "NEW_RISK_DETECTED",
    "INCONCLUSIVE",
    "NO_CHANGE",
]

PredictionOutcome = Literal[
    "MATCHED", "PARTIALLY_MATCHED", "OVERESTIMATED", "UNDERESTIMATED",
    "FAILED", "INCONCLUSIVE",
]

# Two findings describe the same problem when their mapped spans overlap
# this much. Loose on purpose: a re-encode moves a detected span by a frame
# or two, and demanding exact equality would report a persisting finding as
# one resolved plus one new.
MATCH_IOU = 0.3

# Below this, a mapped span is close enough to count as the same moment even
# when IoU is poor — a short finding inside a long one, most often.
MATCH_TOLERANCE_MS = 1_500

# Coverage an agent must have reached in the re-analysis before its silence
# counts as evidence of absence. Set at half because that is the same line
# the reasoning engine already draws for "this agent looked hard enough for
# not-finding-it to mean something", and two different thresholds for the
# same judgement would eventually disagree.
MIN_COVERAGE_FOR_ABSENCE = 0.5


@dataclass(frozen=True)
class TimeMap:
    """Original timestamps to remediated ones, through the cuts.

    Only CUT changes the timeline. MUTE, BLEEP, BLUR_REGION and
    REPLACE_AUDIO all preserve duration, so a report full of those maps
    one-to-one and this is the identity function.
    """

    cuts: tuple[tuple[int, int], ...] = ()

    @classmethod
    def from_ops(cls, ops: Iterable[Any]) -> "TimeMap":
        cuts = tuple(
            sorted(
                (int(op.start_ms), int(op.end_ms))
                for op in ops
                if getattr(op, "op", None) == "CUT" and op.end_ms > op.start_ms
            )
        )
        return cls(cuts=cuts)

    @property
    def identity(self) -> bool:
        return not self.cuts

    def removed_before(self, ms: int) -> int:
        return sum(
            min(end, ms) - start for start, end in self.cuts if start < ms
        )

    def to_remediated(self, ms: int) -> int | None:
        """Where this instant ended up, or None if it was cut out.

        None is a real answer, not a failure: a finding inside a removed
        span has no counterpart to look for, and searching for one anyway is
        how a cut gets misreported as an unresolved finding.
        """
        for start, end in self.cuts:
            if start <= ms < end:
                return None
        return max(0, ms - self.removed_before(ms))

    def map_span(self, start_ms: int, end_ms: int) -> tuple[int, int] | None:
        mapped_start = self.to_remediated(start_ms)
        mapped_end = self.to_remediated(max(start_ms, end_ms - 1))
        if mapped_start is None or mapped_end is None:
            return None
        return mapped_start, max(mapped_start + 1, mapped_end + 1)


def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    start, end = max(a[0], b[0]), min(a[1], b[1])
    overlap = max(0, end - start)
    union = (a[1] - a[0]) + (b[1] - b[0]) - overlap
    return overlap / union if union > 0 else 0.0


def _same_problem(
    original: dict[str, Any], candidate: dict[str, Any], mapped: tuple[int, int] | None
) -> bool:
    """Is this the same problem, seen again?

    Clause first: a different clause is a different problem however close in
    time. Then the span, mapped through the cuts.
    """
    if original.get("clauseId") != candidate.get("clauseId"):
        return False
    if mapped is None:
        return False
    other = (int(candidate.get("startMs", 0)), int(candidate.get("endMs", 0)))
    if _iou(mapped, other) >= MATCH_IOU:
        return True
    # A short finding inside a long one scores poorly on IoU while plainly
    # being the same moment.
    return (
        abs(mapped[0] - other[0]) <= MATCH_TOLERANCE_MS
        and abs(mapped[1] - other[1]) <= MATCH_TOLERANCE_MS
    )


@dataclass(frozen=True)
class FindingChange:
    status: Status
    clause_id: str
    category: str
    severity: str
    original_id: str | None = None
    remediated_id: str | None = None
    detail: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "clauseId": self.clause_id,
            "category": self.category,
            "severity": self.severity,
            "originalId": self.original_id,
            "remediatedId": self.remediated_id,
            "detail": self.detail,
        }


@dataclass
class Comparison:
    changes: list[FindingChange] = field(default_factory=list)
    original_score: int = 0
    remediated_score: int = 0
    structural_ok: bool = True
    reanalysis_ok: bool = True
    notes: list[str] = field(default_factory=list)

    def of(self, status: Status) -> list[FindingChange]:
        return [c for c in self.changes if c.status == status]

    @property
    def resolved(self) -> list[FindingChange]:
        return self.of("RESOLVED")

    @property
    def persisting(self) -> list[FindingChange]:
        return self.of("PERSISTING")

    @property
    def new(self) -> list[FindingChange]:
        return self.of("NEW")

    def to_json(self) -> dict[str, Any]:
        return {
            "changes": [c.to_json() for c in self.changes],
            "originalScore": self.original_score,
            "remediatedScore": self.remediated_score,
            "scoreDelta": self.remediated_score - self.original_score,
            "resolved": len(self.resolved),
            "persisting": len(self.persisting),
            "new": len(self.new),
            "inconclusive": len(self.of("INCONCLUSIVE")),
            "structuralOk": self.structural_ok,
            "reanalysisOk": self.reanalysis_ok,
            "verdict": verdict(self),
            "notes": list(self.notes),
        }


def compare(
    original: list[dict[str, Any]],
    remediated: list[dict[str, Any]],
    ops: Iterable[Any] = (),
    *,
    original_score: int = 0,
    remediated_score: int = 0,
    structural_ok: bool = True,
    reanalysis_ok: bool = True,
    coverage: dict[str, float] | None = None,
) -> Comparison:
    """Classify every finding across the two runs. Deterministic — no model.

    `coverage` is the second run's per-agent coverage, and it gates what
    absence is allowed to mean. A finding that vanished from a modality the
    re-analysis barely examined has not been shown to be fixed — it has been
    unobserved, which is a different claim. Without this gate, making the
    re-analysis cheaper would silently make it more likely to report
    success, which is the worst possible incentive to build into a
    verification system.
    """
    time_map = TimeMap.from_ops(ops)
    result = Comparison(
        original_score=original_score,
        remediated_score=remediated_score,
        structural_ok=structural_ok,
        reanalysis_ok=reanalysis_ok,
    )

    if not reanalysis_ok:
        result.notes.append(
            "Re-analysis did not complete; nothing can be concluded about "
            "whether the original findings were resolved."
        )
        for f in original:
            result.changes.append(
                FindingChange(
                    status="INCONCLUSIVE",
                    clause_id=str(f.get("clauseId", "")),
                    category=str(f.get("category", "")),
                    severity=str(f.get("severity", "MEDIUM")),
                    original_id=str(f.get("id", "")),
                    detail="re-analysis unavailable",
                )
            )
        return result

    unmatched = list(remediated)

    for f in original:
        span = (int(f.get("startMs", 0)), int(f.get("endMs", 0)))
        mapped = time_map.map_span(*span)

        if mapped is None:
            # The span was cut out. There is nothing left to detect, which
            # is a resolution by removal rather than by repair.
            result.changes.append(
                FindingChange(
                    status="RESOLVED",
                    clause_id=str(f.get("clauseId", "")),
                    category=str(f.get("category", "")),
                    severity=str(f.get("severity", "MEDIUM")),
                    original_id=str(f.get("id", "")),
                    detail="the span carrying this finding was cut",
                )
            )
            continue

        match = next(
            (c for c in unmatched if _same_problem(f, c, mapped)), None
        )
        if match is None:
            # Absence only counts as resolution when the re-analysis
            # actually looked. A vision finding gone from a run whose vision
            # agent covered 9% of frames is unobserved, not fixed.
            observers = [
                name for name, value in (f.get("modalities") or {}).items() if value > 0
            ]
            seen = [
                (coverage or {}).get(name, 1.0) for name in observers
            ] or [1.0]
            if max(seen) < MIN_COVERAGE_FOR_ABSENCE:
                thin = ", ".join(
                    f"{n} {((coverage or {}).get(n, 0.0)):.0%}" for n in observers
                )
                result.changes.append(
                    FindingChange(
                        status="INCONCLUSIVE",
                        clause_id=str(f.get("clauseId", "")),
                        category=str(f.get("category", "")),
                        severity=str(f.get("severity", "MEDIUM")),
                        original_id=str(f.get("id", "")),
                        detail=(
                            f"not detected, but the re-analysis examined too "
                            f"little to call it resolved ({thin})"
                        ),
                    )
                )
                result.notes.append(
                    f"{f.get('clauseId')} unobserved rather than resolved — "
                    "coverage too low for an absence claim"
                )
                continue

            result.changes.append(
                FindingChange(
                    status="RESOLVED",
                    clause_id=str(f.get("clauseId", "")),
                    category=str(f.get("category", "")),
                    severity=str(f.get("severity", "MEDIUM")),
                    original_id=str(f.get("id", "")),
                    detail="not detected in the remediated file",
                )
            )
            continue

        unmatched.remove(match)
        before = str(f.get("severity", "MEDIUM"))
        after = str(match.get("severity", "MEDIUM"))
        result.changes.append(
            FindingChange(
                status="CHANGED" if before != after else "PERSISTING",
                clause_id=str(f.get("clauseId", "")),
                category=str(f.get("category", "")),
                severity=after,
                original_id=str(f.get("id", "")),
                remediated_id=str(match.get("id", "")),
                detail=(
                    f"severity {before} → {after}"
                    if before != after
                    else "still detected after remediation"
                ),
            )
        )

    # Whatever the second run found that the first did not. This is the
    # question a render-only check never asks: did the fix break something?
    for c in unmatched:
        result.changes.append(
            FindingChange(
                status="NEW",
                clause_id=str(c.get("clauseId", "")),
                category=str(c.get("category", "")),
                severity=str(c.get("severity", "MEDIUM")),
                remediated_id=str(c.get("id", "")),
                detail="appeared only after remediation",
            )
        )

    return result


def _critical(changes: list[FindingChange]) -> int:
    return sum(1 for c in changes if c.severity in {"CRITICAL", "HIGH"})


def verdict(comparison: Comparison) -> Verdict:
    """The final call, from post-analysis evidence only.

    Deliberately conservative. VERIFIED_SAFE requires that structural
    verification passed, re-analysis completed, nothing new appeared and
    nothing serious persisted — because the word means the system is willing
    to stand behind the output, and every weaker outcome has its own name so
    it never has to be said loosely.
    """
    if not comparison.structural_ok:
        return "REMEDIATION_FAILED"
    if not comparison.reanalysis_ok:
        return "INCONCLUSIVE"

    resolved, persisting, new = (
        comparison.resolved,
        comparison.persisting + comparison.of("CHANGED"),
        comparison.new,
    )

    # A new serious problem outranks every resolution. "We fixed the thing
    # you asked about and introduced another" is not a success.
    if _critical(new) > 0:
        return "NEW_RISK_DETECTED"
    if not resolved and not persisting and not new:
        return "NO_CHANGE"
    if not resolved and persisting:
        return "REMEDIATION_FAILED"
    if persisting or new:
        return "PARTIALLY_REMEDIATED"
    return "VERIFIED_SAFE"


def prediction_outcome(
    predicted_score: int | None,
    actual_score: int,
    predicted_resolved: int | None = None,
    actual_resolved: int | None = None,
) -> PredictionOutcome:
    """How the simulation's prediction held up.

    No percentage. "Accuracy" over a single run is a number with no defined
    denominator, and inventing one would be the fabrication the whole
    verification loop exists to avoid. The classification says what
    happened; the raw delta is reported beside it.
    """
    if predicted_score is None:
        return "INCONCLUSIVE"

    delta = actual_score - predicted_score

    if predicted_resolved is not None and actual_resolved is not None:
        if actual_resolved < predicted_resolved:
            return "OVERESTIMATED"
        if actual_resolved > predicted_resolved:
            return "UNDERESTIMATED"

    if abs(delta) <= 2:
        return "MATCHED"
    if abs(delta) <= 10:
        return "PARTIALLY_MATCHED"
    return "OVERESTIMATED" if delta < 0 else "UNDERESTIMATED"
