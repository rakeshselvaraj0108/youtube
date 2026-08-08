"""The remediation lifecycle, as a state machine that refuses bad histories.

The previous implementation exposed lifecycle through SSE events. Events are
a broadcast, not a record: they exist while a socket is open, they arrive in
whatever order the network allows, and nothing about them prevents a run from
claiming it went straight from RENDERING to VERIFIED. That claim is the exact
lie the whole verification loop exists to prevent — "we rendered it, so it
must be fixed" — and a system that can be *made* to say it will eventually
say it by accident.

So the lifecycle is a graph with explicit edges. RENDERING cannot reach
VERIFIED because there is no edge; the only path runs through structural
verification, re-analysis and comparison. Every transition is checked against
the graph before it is persisted, and an illegal one raises rather than being
silently coerced to something plausible.

Two naming notes, both deliberate:

`VERIFIED` here is the lifecycle state; `VERIFIED_SAFE` is the comparison's
verdict. They are different vocabularies — one describes where the process
got to, the other describes what it found — and `state_for_verdict` is the
single place they meet, so neither drifts into the other.

`FAILED` is terminal and reachable from everywhere. A crash is a legal thing
for a process to do at any point, and a state machine that cannot represent
it forces the caller to lie about what happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

State = Literal[
    "ANALYSIS_COMPLETE",
    "SIMULATION_READY",
    "SIMULATING",
    "SIMULATED",
    "REMEDIATION_REQUESTED",
    "RENDERING",
    "RENDERED",
    "STRUCTURAL_VERIFYING",
    "STRUCTURALLY_VALID",
    "REANALYSIS_QUEUED",
    "REANALYSING",
    "REANALYSIS_COMPLETE",
    "COMPARING",
    "VERIFIED",
    "PARTIALLY_REMEDIATED",
    "NEW_RISK_DETECTED",
    "NO_CHANGE",
    "INCONCLUSIVE",
    "FAILED",
]

# Where a remediation may go next. Anything not listed is impossible, and the
# omissions carry the meaning: RENDERING leads only to RENDERED, so no amount
# of optimism gets a render to a verdict without passing through the three
# stages that actually check it.
TRANSITIONS: dict[str, frozenset[str]] = {
    "ANALYSIS_COMPLETE": frozenset({"SIMULATION_READY", "REMEDIATION_REQUESTED"}),
    "SIMULATION_READY": frozenset({"SIMULATING", "REMEDIATION_REQUESTED"}),
    "SIMULATING": frozenset({"SIMULATED"}),
    # A reader may simulate repeatedly before committing; each pass returns to
    # SIMULATING rather than inventing a "RE_SIMULATING" state that would mean
    # the same thing.
    "SIMULATED": frozenset({"SIMULATING", "REMEDIATION_REQUESTED"}),
    "REMEDIATION_REQUESTED": frozenset({"RENDERING"}),
    "RENDERING": frozenset({"RENDERED"}),
    "RENDERED": frozenset({"STRUCTURAL_VERIFYING"}),
    "STRUCTURAL_VERIFYING": frozenset({"STRUCTURALLY_VALID"}),
    "STRUCTURALLY_VALID": frozenset({"REANALYSIS_QUEUED"}),
    "REANALYSIS_QUEUED": frozenset({"REANALYSING"}),
    # Re-analysis that does not complete is a real outcome, not an error: the
    # file exists and is structurally sound, and nothing is known about
    # whether it is safe. That is precisely INCONCLUSIVE.
    "REANALYSING": frozenset({"REANALYSIS_COMPLETE", "INCONCLUSIVE"}),
    "REANALYSIS_COMPLETE": frozenset({"COMPARING"}),
    "COMPARING": frozenset(
        {
            "VERIFIED",
            "PARTIALLY_REMEDIATED",
            "NEW_RISK_DETECTED",
            "NO_CHANGE",
            "INCONCLUSIVE",
        }
    ),
}

TERMINAL: frozenset[str] = frozenset(
    {
        "VERIFIED",
        "PARTIALLY_REMEDIATED",
        "NEW_RISK_DETECTED",
        "NO_CHANGE",
        "INCONCLUSIVE",
        "FAILED",
    }
)

ALL_STATES: frozenset[str] = frozenset(TRANSITIONS) | TERMINAL

# What the comparison's verdict means for the lifecycle. One mapping, in one
# place, so the two vocabularies can never disagree about the same run.
VERDICT_STATE: dict[str, State] = {
    "VERIFIED_SAFE": "VERIFIED",
    "PARTIALLY_REMEDIATED": "PARTIALLY_REMEDIATED",
    "NEW_RISK_DETECTED": "NEW_RISK_DETECTED",
    "NO_CHANGE": "NO_CHANGE",
    "INCONCLUSIVE": "INCONCLUSIVE",
    # A remediation that failed did not reach a verdict about the video; it
    # failed to produce one. That is FAILED, not a weaker kind of success.
    "REMEDIATION_FAILED": "FAILED",
}

# Which state a resumed remediation should re-enter, and what that costs.
# Only the work that actually depends on the interrupted step is redone: an
# interrupted comparison keeps the render and the re-analysis, because both
# already produced durable artifacts that the crash did not invalidate.
RESUME_FROM: dict[str, State] = {
    "REMEDIATION_REQUESTED": "REMEDIATION_REQUESTED",
    "RENDERING": "REMEDIATION_REQUESTED",
    "RENDERED": "RENDERED",
    "STRUCTURAL_VERIFYING": "RENDERED",
    "STRUCTURALLY_VALID": "STRUCTURALLY_VALID",
    "REANALYSIS_QUEUED": "STRUCTURALLY_VALID",
    "REANALYSING": "STRUCTURALLY_VALID",
    "REANALYSIS_COMPLETE": "REANALYSIS_COMPLETE",
    "COMPARING": "REANALYSIS_COMPLETE",
    "SIMULATING": "SIMULATION_READY",
}


class InvalidTransition(ValueError):
    """An edge that does not exist in the graph."""

    def __init__(self, current: str, target: str) -> None:
        self.current, self.target = current, target
        allowed = ", ".join(sorted(TRANSITIONS.get(current, frozenset()))) or "nothing"
        super().__init__(
            f"{current} cannot become {target}; from {current} the only legal "
            f"next states are {allowed}"
        )


def can_transition(current: str, target: str) -> bool:
    """FAILED is reachable from any non-terminal state — a process may crash
    at any point, and a machine that cannot say so forces a false record."""
    if current in TERMINAL:
        return False
    if target == "FAILED":
        return True
    return target in TRANSITIONS.get(current, frozenset())


def check(current: str, target: str) -> None:
    if not can_transition(current, target):
        raise InvalidTransition(current, target)


def is_terminal(state: str) -> bool:
    return state in TERMINAL


def state_for_verdict(verdict: str) -> State:
    return VERDICT_STATE.get(verdict, "INCONCLUSIVE")


def resume_state(interrupted: str) -> State | None:
    """Where a remediation interrupted in `interrupted` should pick up.

    None when there is nothing to resume — a terminal state is finished, and
    re-entering it would redo expensive work to reach the answer already on
    disk.
    """
    if interrupted in TERMINAL:
        return None
    return RESUME_FROM.get(interrupted)


def path_between(start: str, end: str) -> list[str] | None:
    """The shortest legal route, or None if there is not one.

    Used by the tests to assert the shape of the graph rather than its edge
    list — "there is no path from RENDERING to VERIFIED that skips
    comparison" is the property that matters, and it survives edits to the
    table that an edge-by-edge assertion would not.
    """
    if start == end:
        return [start]
    seen = {start}
    queue: list[list[str]] = [[start]]
    while queue:
        route = queue.pop(0)
        for nxt in sorted(TRANSITIONS.get(route[-1], frozenset())):
            if nxt in seen:
                continue
            if nxt == end:
                return [*route, nxt]
            seen.add(nxt)
            queue.append([*route, nxt])
    return None


@dataclass(frozen=True)
class Transition:
    """One durable step. `at` is set by the store, not by the caller."""

    from_state: str
    to_state: str
    at: str
    detail: str = ""
    error: str = ""

    def to_json(self) -> dict[str, object]:
        return {
            "from": self.from_state,
            "to": self.to_state,
            "at": self.at,
            "detail": self.detail,
            "error": self.error,
        }


def describe(state: str) -> str:
    """Plain English for the terminal and the deck."""
    return {
        "ANALYSIS_COMPLETE": "analysed, no remediation requested yet",
        "SIMULATION_READY": "scenarios available to simulate",
        "SIMULATING": "scoring scenarios",
        "SIMULATED": "scenarios scored, awaiting a choice",
        "REMEDIATION_REQUESTED": "queued to render",
        "RENDERING": "ffmpeg is writing the output",
        "RENDERED": "output written, not yet checked",
        "STRUCTURAL_VERIFYING": "checking the output against the edit list",
        "STRUCTURALLY_VALID": "output matches the edit list",
        "REANALYSIS_QUEUED": "waiting to re-analyse the rendered file",
        "REANALYSING": "running the pipeline against the rendered file",
        "REANALYSIS_COMPLETE": "rendered file analysed",
        "COMPARING": "matching findings and incidents across the two runs",
        "VERIFIED": "verified safe against post-analysis evidence",
        "PARTIALLY_REMEDIATED": "some findings resolved, others remain",
        "NEW_RISK_DETECTED": "the remediation introduced a serious finding",
        "NO_CHANGE": "nothing changed",
        "INCONCLUSIVE": "not enough evidence to conclude",
        "FAILED": "the remediation did not complete",
    }.get(state, state)


def unreachable_states() -> set[str]:
    """States no edge leads to. Should be exactly {ANALYSIS_COMPLETE}, the
    entry point — anything else is a state the machine can never enter, which
    means dead code somewhere is trying to set it."""
    reachable: set[str] = set()
    for targets in TRANSITIONS.values():
        reachable |= set(targets)
    reachable.add("FAILED")
    return set(ALL_STATES) - reachable


def validate_graph() -> list[str]:
    """Structural problems with the table itself, for the wiring test."""
    problems: list[str] = []
    for state, targets in TRANSITIONS.items():
        if state not in ALL_STATES:
            problems.append(f"{state} is not a declared state")
        for target in targets:
            if target not in ALL_STATES:
                problems.append(f"{state} -> {target}: {target} is not a state")
        if state in TERMINAL and targets:
            problems.append(f"{state} is terminal but has outgoing edges")
    for state in ALL_STATES - TERMINAL:
        if state not in TRANSITIONS:
            problems.append(f"{state} is non-terminal but has no outgoing edges")
    return problems


def ordered_states() -> list[str]:
    """The states in pipeline order, for display. Terminal verdicts last."""
    return [
        *[s for s in TRANSITIONS],
        *sorted(TERMINAL),
    ]


def reachable_from(state: str) -> Iterable[str]:
    return sorted(TRANSITIONS.get(state, frozenset()))
