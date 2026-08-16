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

# Incidents get one status findings do not: an incident is a group, so it can
# be genuinely half-fixed in a way a single finding cannot. Collapsing that
# into PERSISTING would hide real progress, and into RESOLVED would hide a
# real remaining problem.
IncidentStatus = Literal[
    "RESOLVED", "PERSISTING", "PARTIALLY_REMEDIATED", "CHANGED", "NEW",
    "INCONCLUSIVE",
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


@dataclass(frozen=True)
class IncidentChange:
    """One incident, across the two runs.

    Carries the finding-level verdicts it was rolled up from, so a reader
    following the verdict downwards never hits a claim without its evidence:
    the incident says PARTIALLY_REMEDIATED *because* these two findings
    resolved and that one did not.
    """

    status: IncidentStatus
    category: str
    severity: str
    original_id: str | None = None
    remediated_id: str | None = None
    original_span: tuple[int, int] | None = None
    mapped_span: tuple[int, int] | None = None
    remediated_span: tuple[int, int] | None = None
    clauses: tuple[str, ...] = ()
    resolved_findings: tuple[str, ...] = ()
    persisting_findings: tuple[str, ...] = ()
    new_findings: tuple[str, ...] = ()
    inconclusive_findings: tuple[str, ...] = ()
    removed_by_cut: bool = False
    detail: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "category": self.category,
            "severity": self.severity,
            "originalId": self.original_id,
            "remediatedId": self.remediated_id,
            "originalSpan": list(self.original_span) if self.original_span else None,
            "mappedSpan": list(self.mapped_span) if self.mapped_span else None,
            "remediatedSpan": (
                list(self.remediated_span) if self.remediated_span else None
            ),
            "clauses": list(self.clauses),
            "resolvedFindings": list(self.resolved_findings),
            "persistingFindings": list(self.persisting_findings),
            "newFindings": list(self.new_findings),
            "inconclusiveFindings": list(self.inconclusive_findings),
            "removedByCut": self.removed_by_cut,
            "detail": self.detail,
        }


@dataclass
class Comparison:
    changes: list[FindingChange] = field(default_factory=list)
    incidents: list[IncidentChange] = field(default_factory=list)
    original_score: int = 0
    remediated_score: int = 0
    structural_ok: bool = True
    reanalysis_ok: bool = True
    notes: list[str] = field(default_factory=list)

    def of(self, status: Status) -> list[FindingChange]:
        return [c for c in self.changes if c.status == status]

    def incidents_of(self, status: IncidentStatus) -> list[IncidentChange]:
        return [i for i in self.incidents if i.status == status]

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
            "incidentChanges": [i.to_json() for i in self.incidents],
            "originalScore": self.original_score,
            "remediatedScore": self.remediated_score,
            "scoreDelta": self.remediated_score - self.original_score,
            "resolved": len(self.resolved),
            "persisting": len(self.persisting),
            "new": len(self.new),
            "inconclusive": len(self.of("INCONCLUSIVE")),
            "incidentsResolved": len(self.incidents_of("RESOLVED")),
            "incidentsPersisting": len(self.incidents_of("PERSISTING")),
            "incidentsPartial": len(self.incidents_of("PARTIALLY_REMEDIATED")),
            "incidentsChanged": len(self.incidents_of("CHANGED")),
            "incidentsNew": len(self.incidents_of("NEW")),
            "incidentsInconclusive": len(self.incidents_of("INCONCLUSIVE")),
            "structuralOk": self.structural_ok,
            "reanalysisOk": self.reanalysis_ok,
            "verdict": verdict(self),
            "notes": list(self.notes),
        }


SEVERITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def _incident_status(
    resolved: list[str],
    persisting: list[str],
    changed: list[str],
    inconclusive: list[str],
) -> tuple[IncidentStatus, str]:
    """Roll finding verdicts up into one incident verdict.

    The ordering is deliberately pessimistic. An unresolved question outranks
    a resolution, because "we could not tell" and "it is gone" are different
    statements and only one of them is safe to act on. An incident with two
    resolved findings and one nobody looked for is INCONCLUSIVE, not
    PARTIALLY_REMEDIATED — the honest answer is that its fate is unknown.
    """
    still_there = persisting + changed

    if inconclusive and not still_there:
        return (
            "INCONCLUSIVE",
            f"{len(resolved)} of its findings resolved, but {len(inconclusive)} "
            "could not be checked with the coverage available",
        )
    if inconclusive and still_there and not resolved:
        return (
            "PERSISTING",
            f"{len(still_there)} finding(s) still detected; "
            f"{len(inconclusive)} could not be checked",
        )
    if not still_there and not inconclusive:
        if not resolved:
            return "INCONCLUSIVE", "no finding-level evidence either way"
        return "RESOLVED", f"all {len(resolved)} of its findings are gone"
    if not resolved:
        if changed and not persisting:
            return (
                "CHANGED",
                f"{len(changed)} finding(s) changed severity but none resolved",
            )
        return "PERSISTING", f"{len(still_there)} finding(s) still detected"
    return (
        "PARTIALLY_REMEDIATED",
        f"{len(resolved)} resolved, {len(still_there)} still detected"
        + (f", {len(inconclusive)} unchecked" if inconclusive else ""),
    )


def compare_incidents(
    original: list[dict[str, Any]],
    remediated: list[dict[str, Any]],
    changes: list[FindingChange],
    time_map: TimeMap,
    *,
    reanalysis_ok: bool = True,
) -> list[IncidentChange]:
    """Incident-level comparison, rolled up from the finding comparison.

    Incidents are *not* re-matched independently. The backend's correlation is
    authoritative and already decided which findings describe one event; a
    second matching pass here would be a parallel grouping implementation that
    could disagree with it, and two answers to "how many problems are there"
    is worse than either answer alone.

    Identity is therefore transitive: original incident → its findings →
    their matched counterparts → the remediated incident that contains them.
    That is why ids never enter the comparison. The second run renumbers
    INC-001 through INC-00n by timestamp, so after a cut moves everything the
    numbers are actively misleading — INC-002 in the output is routinely a
    different event from INC-002 in the input.
    """
    by_original = {c.original_id: c for c in changes if c.original_id}
    owner: dict[str, dict[str, Any]] = {}
    for incident in remediated:
        for fid in incident.get("findingIds", []):
            owner[str(fid)] = incident

    out: list[IncidentChange] = []
    claimed: set[str] = set()

    for incident in original:
        span = (int(incident.get("startMs", 0)), int(incident.get("endMs", 0)))
        mapped = time_map.map_span(*span)
        resolved: list[str] = []
        persisting: list[str] = []
        changed_ids: list[str] = []
        inconclusive: list[str] = []
        counterparts: list[dict[str, Any]] = []

        for fid in incident.get("findingIds", []):
            change = by_original.get(str(fid))
            if change is None:
                inconclusive.append(str(fid))
                continue
            if change.status == "RESOLVED":
                resolved.append(str(fid))
            elif change.status == "PERSISTING":
                persisting.append(str(fid))
            elif change.status == "CHANGED":
                changed_ids.append(str(fid))
            else:
                inconclusive.append(str(fid))
            if change.remediated_id and change.remediated_id in owner:
                counterparts.append(owner[change.remediated_id])

        if not reanalysis_ok:
            status, detail = "INCONCLUSIVE", "re-analysis did not complete"
        else:
            status, detail = _incident_status(
                resolved, persisting, changed_ids, inconclusive
            )

        # The counterpart is the remediated incident holding most of this
        # one's surviving findings. Ties go to the earliest, which is stable.
        counterpart: dict[str, Any] | None = None
        if counterparts:
            counts: dict[str, int] = {}
            for c in counterparts:
                counts[str(c["id"])] = counts.get(str(c["id"]), 0) + 1
            best = max(sorted(counts), key=lambda k: counts[k])
            counterpart = next(c for c in remediated if str(c["id"]) == best)
            claimed.add(best)

        severity = str(incident.get("severity", "MEDIUM"))
        if counterpart is not None:
            after_severity = str(counterpart.get("severity", severity))
            if (
                status == "PERSISTING"
                and SEVERITY_RANK.get(after_severity, 1)
                < SEVERITY_RANK.get(severity, 1)
            ):
                status = "CHANGED"
                detail = f"severity {severity} → {after_severity}"
            severity = after_severity

        # A span the edit removed entirely. Recorded rather than inferred:
        # the reader is told the evidence is gone because it was cut, not
        # shown an "after" frame that does not exist.
        removed = mapped is None and not time_map.identity
        if removed and status == "RESOLVED":
            detail = "the span carrying this incident was cut"

        out.append(
            IncidentChange(
                status=status,
                category=str(incident.get("category", "")),
                severity=severity,
                original_id=str(incident.get("id", "")),
                remediated_id=str(counterpart["id"]) if counterpart else None,
                original_span=span,
                mapped_span=mapped,
                remediated_span=(
                    (
                        int(counterpart.get("startMs", 0)),
                        int(counterpart.get("endMs", 0)),
                    )
                    if counterpart
                    else None
                ),
                clauses=tuple(str(c) for c in incident.get("clauses", [])),
                resolved_findings=tuple(resolved),
                persisting_findings=tuple(persisting + changed_ids),
                inconclusive_findings=tuple(inconclusive),
                removed_by_cut=removed,
                detail=detail,
            )
        )

    if not reanalysis_ok:
        return out

    # Anything in the output that no original incident accounts for. This is
    # the question a render-only check never asks, and the one that produced
    # this project's most useful result: an edit that fixed what it targeted
    # and introduced an event that was not there before.
    new_ids = {c.remediated_id for c in changes if c.status == "NEW"}
    for incident in remediated:
        incident_id = str(incident.get("id", ""))
        if incident_id in claimed:
            continue
        members = [str(f) for f in incident.get("findingIds", [])]
        if not any(m in new_ids for m in members):
            # Every member was matched to some original finding, but no
            # original incident claimed the group. That is a regrouping, not
            # a new event, and calling it NEW would invent a problem.
            continue
        out.append(
            IncidentChange(
                status="NEW",
                category=str(incident.get("category", "")),
                severity=str(incident.get("severity", "MEDIUM")),
                remediated_id=incident_id,
                remediated_span=(
                    int(incident.get("startMs", 0)),
                    int(incident.get("endMs", 0)),
                ),
                clauses=tuple(str(c) for c in incident.get("clauses", [])),
                new_findings=tuple(m for m in members if m in new_ids),
                detail="appeared only after remediation",
            )
        )

    return out


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
    original_incidents: list[dict[str, Any]] | None = None,
    remediated_incidents: list[dict[str, Any]] | None = None,
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
        result.incidents = compare_incidents(
            original_incidents or [],
            [],
            result.changes,
            time_map,
            reanalysis_ok=False,
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

    # Incidents last: the rollup reads the finding verdicts above, so it can
    # only run once every finding has one.
    result.incidents = compare_incidents(
        original_incidents or [],
        remediated_incidents or [],
        result.changes,
        time_map,
        reanalysis_ok=True,
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
    # you asked about and introduced another" is not a success. Checked at
    # both levels: a serious new *incident* is a serious new event even when
    # its individual findings each look survivable, and the incident layer is
    # where correlated evidence raises severity.
    new_incidents = comparison.incidents_of("NEW")
    if _critical(new) > 0 or any(
        i.severity in {"CRITICAL", "HIGH"} for i in new_incidents
    ):
        return "NEW_RISK_DETECTED"

    # An incident nobody could check is not a clean bill of health. Findings
    # already gate this individually; incidents catch the case where a whole
    # correlated event fell into an unexamined region.
    unchecked_incidents = comparison.incidents_of("INCONCLUSIVE")

    if not resolved and not persisting and not new:
        # Nothing moved. Whether that is "nothing to do" or "nobody looked"
        # depends entirely on whether anything was left unchecked, and the two
        # must not share a name.
        if comparison.of("INCONCLUSIVE") or unchecked_incidents:
            return "INCONCLUSIVE"
        return "NO_CHANGE"
    if not resolved and persisting:
        return "REMEDIATION_FAILED"
    if persisting or new or new_incidents or unchecked_incidents:
        return "PARTIALLY_REMEDIATED"
    if comparison.of("INCONCLUSIVE"):
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
