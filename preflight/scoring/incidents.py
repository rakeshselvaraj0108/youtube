"""Correlation — from independent findings to one explainable incident.

Four agents that each noticed something at 02:14 have not found four
problems. They have found one problem four times, and a report that lists it
four times is both wrong about the count and worse at explaining any of it.

This is the layer that decides which findings describe the same event. It
sits after `fusion`, which is a different question: fusion combines the
*modalities within one finding* into a confidence, and this combines
*separate findings* into an incident. Running fusion's noisy-or a second
time here would be double-counting the same agreement, which is exactly the
confidence inflation this module has to avoid rather than cause.

Three traps govern the design, and the real corpus contains all of them.

**File-scoped findings correlate with everything.** "No caption track" and
"integrated loudness below target" span the entire video, so any rule based
on temporal overlap merges them into every incident on the timeline and
produces one meaningless super-incident containing the whole run. They are
properties of the upload, not events in it, and they are never correlated.

**Proximity is not relatedness.** Profanity at 02:14 and a frozen frame at
02:14 are two unrelated problems that happen to share a timestamp. Merging
them would invent a category that describes neither. Findings must be
temporally close *and* semantically compatible.

**One agent cannot corroborate itself.** Two speech findings a second apart
are two events; corroboration means *independent* observers agreeing, so a
second finding from an agent already in the incident adds a member without
adding confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from preflight.models import Finding

# Findings this close are candidates for the same event. Agents disagree
# about timing by a beat: ASR times a word to the syllable, a keyframe lands
# on the nearest sampled frame, so the same moment routinely arrives with a
# second between observations.
PROXIMITY_MS = 2_500

# A finding covering this share of the video is describing the upload, not a
# moment in it. Deliberately below 1.0 — the corpus carries a "frozen video"
# finding spanning 100ms to 20000ms of a 20000ms file, which is file-scoped
# in everything but arithmetic.
FILE_SCOPED_SHARE = 0.9

# What can be the same event. Keyed by finding category, and deliberately
# narrow: an unlisted pair is two incidents, which is the safe direction to
# be wrong in. A false split reports two real problems; a false merge hides
# one behind a label that does not describe it.
COMPATIBLE: dict[str, set[str]] = {
    "Language": {"Language", "Violence", "Controversial", "Sensitive"},
    "Violence": {"Violence", "Language", "Sensitive"},
    "Controversial": {"Controversial", "Language", "Sensitive"},
    "Sensitive": {"Sensitive", "Language", "Violence", "Controversial"},
    "Substances": {"Substances", "Language", "Metadata"},
    "Metadata": {"Metadata", "Substances"},
    "Accessibility": {"Accessibility"},
    "Audio Delivery": {"Audio Delivery"},
    "Copyright": {"Copyright", "Audio Delivery"},
}

# Corroboration bonus per additional independent agent, and the ceiling it
# may reach. Bounded because certainty is not additive: three agents at 0.8
# is strong evidence, not proof, and a scheme that walks to 0.99 on volume
# alone has stopped measuring anything.
CORROBORATION_STEP = 0.05
CONFIDENCE_CEILING = 0.97

SEVERITY_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


@dataclass(frozen=True)
class Incident:
    """One event, as every agent that saw it described it."""

    id: str
    start_ms: int
    end_ms: int
    category: str
    severity: str
    confidence: float
    finding_ids: list[str]
    agents: list[str]
    clauses: list[str]
    suggested_fix: str
    reasoning: str
    corroborated: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "category": self.category,
            "severity": self.severity,
            "confidence": self.confidence,
            "findingIds": self.finding_ids,
            "agents": self.agents,
            "clauses": self.clauses,
            "suggestedFix": self.suggested_fix,
            "reasoning": self.reasoning,
            "corroborated": self.corroborated,
        }


def is_file_scoped(finding: Finding, duration_ms: int) -> bool:
    """Does this describe the upload rather than a moment in it?"""
    if duration_ms <= 0:
        return False
    span = max(0, finding.endMs - finding.startMs)
    return span >= duration_ms * FILE_SCOPED_SHARE


def agents_of(finding: Finding) -> set[str]:
    """Which agents observed this. `modalities` is the record of that."""
    return {name for name, value in (finding.modalities or {}).items() if value > 0}


def compatible(left: Finding, right: Finding) -> bool:
    """Could these two describe the same event?

    Same clause is always compatible — one violation seen twice. Otherwise
    the category families must be listed as co-occurring.
    """
    if left.clauseId == right.clauseId:
        return True
    allowed = COMPATIBLE.get(left.category, {left.category})
    return right.category in allowed


def _gap(left: Finding, right: Finding) -> int:
    """Milliseconds between two spans; zero when they overlap."""
    if left.endMs >= right.startMs and right.endMs >= left.startMs:
        return 0
    return right.startMs - left.endMs if right.startMs > left.endMs else left.startMs - right.endMs


def _severity(findings: list[Finding]) -> str:
    ranks = [
        SEVERITY_ORDER.index(f.severity) if f.severity in SEVERITY_ORDER else 1
        for f in findings
    ]
    return SEVERITY_ORDER[max(ranks)] if ranks else "MEDIUM"


def _confidence(findings: list[Finding], agents: set[str]) -> float:
    """Incident confidence, bounded and never inflated by repetition.

    The strongest single observation sets the floor, because an incident
    cannot be less certain than the best evidence in it. Each *additional
    independent agent* adds a small bounded step. Two findings from the same
    agent add nothing — an observer repeating itself is not corroboration,
    and treating it as such is how a single noisy detector talks itself into
    a near-certain incident.
    """
    best = max((f.fusedConfidence or f.confidence) for f in findings)
    bonus = CORROBORATION_STEP * max(0, len(agents) - 1)
    return round(min(CONFIDENCE_CEILING, best + bonus), 4)


def _reasoning(findings: list[Finding], agents: set[str]) -> str:
    if len(findings) == 1:
        return f"Single observation from {', '.join(sorted(agents)) or 'one agent'}."
    named = ", ".join(sorted(agents))
    clauses = sorted({f.clauseId for f in findings})
    return (
        f"{len(findings)} findings within {PROXIMITY_MS / 1000:.1f}s "
        f"observed by {len(agents)} independent agent(s) ({named}); "
        f"clauses {', '.join(clauses)}."
    )


def _build(index: int, findings: list[Finding]) -> Incident:
    ordered = sorted(findings, key=lambda f: f.startMs)
    agents: set[str] = set()
    for finding in ordered:
        agents |= agents_of(finding)

    # The category of the most severe, most confident member — the incident
    # is named for what it most is, not for whichever finding sorted first.
    lead = max(
        ordered,
        key=lambda f: (
            SEVERITY_ORDER.index(f.severity) if f.severity in SEVERITY_ORDER else 1,
            f.fusedConfidence or f.confidence,
        ),
    )
    fixes = [f.suggestedFix for f in ordered if f.suggestedFix != "NONE"]

    return Incident(
        id=f"INC-{index:03d}",
        start_ms=min(f.startMs for f in ordered),
        end_ms=max(f.endMs for f in ordered),
        category=lead.category,
        severity=_severity(ordered),
        confidence=_confidence(ordered, agents),
        finding_ids=[f.id for f in ordered],
        agents=sorted(agents),
        clauses=sorted({f.clauseId for f in ordered}),
        # The lead finding's fix, since it is the one the incident is named
        # for. Picking any other would repair something the title does not
        # mention.
        suggested_fix=lead.suggestedFix if lead.suggestedFix != "NONE" else (
            fixes[0] if fixes else "NONE"
        ),
        reasoning=_reasoning(ordered, agents),
        corroborated=len(agents) > 1,
    )


def correlate(findings: Iterable[Finding], duration_ms: int) -> list[Incident]:
    """Group findings into incidents.

    O(n log n): one sort, then a linear sweep. Each finding is compared only
    against the open cluster rather than against every other finding, which
    is what keeps a sixty-minute video with hundreds of findings from
    becoming the quadratic scan the obvious implementation produces.
    """
    items = list(findings)
    if not items:
        return []

    scoped = [f for f in items if is_file_scoped(f, duration_ms)]
    timed = sorted(
        (f for f in items if not is_file_scoped(f, duration_ms)),
        key=lambda f: (f.startMs, f.endMs),
    )

    clusters: list[list[Finding]] = []
    for finding in timed:
        placed = False
        # Only the most recent clusters can still be in range; the sweep is
        # ordered, so anything older is already too far behind to match.
        for cluster in reversed(clusters[-4:]):
            if all(
                _gap(existing, finding) <= PROXIMITY_MS and compatible(existing, finding)
                for existing in cluster
            ):
                cluster.append(finding)
                placed = True
                break
        if not placed:
            clusters.append([finding])

    # File-scoped findings become their own single-finding incidents. They
    # are real problems and belong on the report; they simply cannot be
    # correlated with a moment, because they do not describe one.
    groups = [*clusters, *([f] for f in scoped)]

    # Numbered after the final ordering, not before it. Assigning ids during
    # clustering and sorting afterwards produced a report reading INC-001,
    # INC-003, INC-002 — the file-scoped incidents were numbered last and
    # then sorted back into the middle. The identifier a reader cites should
    # count in the order they read.
    groups.sort(key=lambda g: (min(f.startMs for f in g), max(f.endMs for f in g)))
    return [_build(index + 1, group) for index, group in enumerate(groups)]


@dataclass
class IncidentGraph:
    """The correlated view, serialisable for the report and the UI."""

    incidents: list[Incident] = field(default_factory=list)

    @property
    def corroborated(self) -> list[Incident]:
        return [i for i in self.incidents if i.corroborated]

    def to_json(self) -> dict[str, Any]:
        return {
            "incidents": [i.to_json() for i in self.incidents],
            "total": len(self.incidents),
            "corroborated": len(self.corroborated),
        }


def build_graph(findings: Iterable[Finding], duration_ms: int) -> IncidentGraph:
    return IncidentGraph(incidents=correlate(findings, duration_ms))
