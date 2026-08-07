"""Call budgets — a ceiling the run enforces on itself, and reports.

The decomposition plan states what a run will cost before it spends anything.
This is the other half: a ceiling it cannot exceed, and an honest account of
what was given up to stay inside it.

Three properties make this worth having rather than a naive counter.

**Shedding lowers coverage.** PREFLIGHT's central claim is that it reports
what it actually examined. A budget that silently examined 12 of 31 windows
while still reporting full coverage would break that claim far more
seriously than running out of money does. Every shed records how many
windows went unexamined, and the triad's coverage is derived from what it
reached — not from what it intended to reach.

**The adjudicator is reserved, never shed.** The stages are not equal. An
unexamined window is missing information; an *unruled charge* is worse than
missing information, because the AUDITOR alone is the unopposed-prosecutor
configuration this whole design exists to avoid. So the adjudicator's calls
are reserved up front and the auditor spends only what is left. Running out
mid-audit costs windows; it never ships accusations nobody ruled on.

**A budget that can be exceeded is not a budget.** `can_afford` is checked
before the call, never after, and the plan's estimates are upper bounds for
the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Shed:
    """One thing the run gave up, and what it cost in coverage."""

    stage: str
    reason: str
    windows_lost: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "reason": self.reason,
            "windowsLost": self.windows_lost,
        }


@dataclass
class CallBudget:
    """A ceiling on hosted model calls for one run.

    `ceiling=None` means unlimited, which is the default: a budget is opt-in
    via `--budget`, because silently capping a run someone expected to
    complete is its own kind of dishonesty.
    """

    ceiling: int | None = None
    spent: int = 0
    reserved: int = 0
    shed: list[Shed] = field(default_factory=list)

    @property
    def unlimited(self) -> bool:
        return self.ceiling is None

    @property
    def remaining(self) -> int:
        """Calls left to spend, ignoring anything held in reserve."""
        if self.ceiling is None:
            return 1 << 30
        return max(0, self.ceiling - self.spent - self.reserved)

    @property
    def exhausted(self) -> bool:
        return not self.unlimited and self.remaining <= 0

    def can_afford(self, calls: int = 1) -> bool:
        return self.unlimited or self.remaining >= calls

    def spend(self, calls: int = 1) -> None:
        self.spent += calls

    def reserve(self, calls: int) -> None:
        """Hold calls back for a downstream stage that must not be starved.

        Capped at **half** what is left, which is the non-obvious part. The
        adjudicator's worst case is one call per batch of every window, so a
        reserve sized for it can swallow a small budget whole — and then the
        audit never runs, raises no charges, and the reserve protects a stage
        with nothing to rule on. Reserving for the worst case starves the
        stage that produces the work.

        Half is the honest split: the ruling can always cover whatever the
        audit could afford to raise, because candidates cannot outnumber the
        windows examined and both stages batch at comparable sizes.
        """
        if self.unlimited:
            return
        available = max(0, self.ceiling - self.spent)
        self.reserved = min(max(0, calls), available // 2)

    def release(self) -> None:
        """Hand the reserve back once the stage it protected is next to run."""
        self.reserved = 0

    def record_shed(self, stage: str, reason: str, windows_lost: int = 0) -> None:
        self.shed.append(Shed(stage=stage, reason=reason, windows_lost=windows_lost))

    def to_json(self) -> dict[str, Any]:
        return {
            "ceiling": self.ceiling,
            "spent": self.spent,
            "shed": [s.to_json() for s in self.shed],
        }
