"""Call budgets, and the honesty properties that make them worth having.

A budget is easy; a budget that tells the truth about what it cost you is
the part worth testing. Three claims matter here, and each has a way of
being silently wrong:

  1. The ceiling actually holds. Checked before the call, never after.
  2. Shedding lowers coverage. A run that examined 12 of 31 windows and
     still reported full coverage would break this project's central claim
     far worse than running out of budget does.
  3. The adjudicator is never starved. An unruled charge is the
     unopposed-prosecutor output the triad exists to prevent, so its calls
     are reserved before the audit spends anything.
"""

from __future__ import annotations

import pytest

from preflight.agents.triad import ADJUDICATOR_BATCH, AUDITOR_BATCH
from preflight.budget import CallBudget


class TestCeiling:
    def test_an_unlimited_budget_affords_anything(self):
        budget = CallBudget()
        assert budget.unlimited
        assert budget.can_afford(10_000)
        assert not budget.exhausted

    def test_a_ceiling_is_not_exceeded(self):
        budget = CallBudget(ceiling=3)
        spent = 0
        while budget.can_afford(1):
            budget.spend(1)
            spent += 1
            assert spent <= 3, "spent past the ceiling"
        assert spent == 3
        assert budget.exhausted

    def test_remaining_never_goes_negative(self):
        budget = CallBudget(ceiling=2)
        budget.spend(5)
        assert budget.remaining == 0
        assert not budget.can_afford(1)

    def test_a_zero_ceiling_affords_nothing(self):
        assert not CallBudget(ceiling=0).can_afford(1)


class TestReserve:
    """The adjudicator's calls are held back before the audit runs."""

    def test_a_reserve_is_invisible_to_ordinary_spending(self):
        budget = CallBudget(ceiling=10)
        budget.reserve(4)
        assert budget.remaining == 6

    def test_releasing_hands_the_reserve_back(self):
        budget = CallBudget(ceiling=10)
        budget.reserve(4)
        for _ in range(6):
            budget.spend(1)
        assert not budget.can_afford(1)
        budget.release()
        assert budget.can_afford(4)

    def test_the_audit_cannot_spend_the_adjudicators_share(self):
        """The property that keeps unruled charges from ever shipping."""
        budget = CallBudget(ceiling=10)
        budget.reserve(3)
        while budget.can_afford(1):
            budget.spend(1)
        budget.release()
        assert budget.can_afford(3), "audit consumed the adjudicator's reserve"

    def test_an_oversized_reserve_never_starves_the_producing_stage(self):
        """The reserve is capped at half of what is left, and this is why.

        The adjudicator's worst case is a call per batch of every window, so
        a reserve sized for it swallows a small budget whole. The audit then
        never runs, raises no charges, and the reserve ends up protecting a
        stage with nothing to rule on — the run spends its entire budget on
        nothing. Half always leaves the audit room to produce work.
        """
        budget = CallBudget(ceiling=2)
        budget.reserve(50)
        assert budget.remaining >= 1, "reserve consumed the whole budget"
        assert budget.can_afford(1)
        budget.release()
        assert budget.can_afford(2)

    def test_the_reserve_never_exceeds_half_of_what_is_left(self):
        for ceiling in (1, 2, 4, 7, 10, 100):
            budget = CallBudget(ceiling=ceiling)
            budget.reserve(ceiling * 10)
            assert budget.reserved <= ceiling // 2

    def test_a_reserve_on_an_unlimited_budget_changes_nothing(self):
        budget = CallBudget()
        budget.reserve(100)
        assert budget.can_afford(1_000_000)


class TestShedIsRecorded:
    def test_a_shed_carries_its_reason_and_its_cost(self):
        budget = CallBudget(ceiling=1)
        budget.record_shed("auditor", "call budget reached", windows_lost=19)
        assert budget.shed[0].stage == "auditor"
        assert budget.shed[0].windows_lost == 19
        assert "budget" in budget.shed[0].reason

    def test_an_unshed_run_records_nothing(self):
        assert CallBudget(ceiling=100).shed == []

    def test_the_json_shape_carries_what_a_reader_needs(self):
        budget = CallBudget(ceiling=5)
        budget.spend(5)
        budget.record_shed("advocate", "no budget left", windows_lost=0)
        payload = budget.to_json()
        assert payload["ceiling"] == 5
        assert payload["spent"] == 5
        assert payload["shed"][0]["stage"] == "advocate"


class TestBudgetSizingAgainstTheRealBatches:
    """The reserve is computed from the triad's own batch constant. If that
    constant moves and the reserve arithmetic does not, the adjudicator gets
    starved again — quietly."""

    @pytest.mark.parametrize("windows", [1, 6, 7, 12, 31, 100])
    def test_the_reserve_covers_every_adjudicator_batch(self, windows):
        needed = -(-windows // ADJUDICATOR_BATCH)
        budget = CallBudget(ceiling=1000)
        budget.reserve(needed)
        budget.release()
        assert budget.can_afford(needed)

    def test_a_budget_matching_the_plan_estimate_sheds_nothing(self):
        """The plan's estimate is an upper bound, so a budget set to it must
        be sufficient by construction — otherwise the plan is lying."""
        from preflight.plan import build_plan

        plan = build_plan(600_000)
        budget = CallBudget(ceiling=plan.est_total_llm_calls)
        auditor_batches = -(-plan.chunk_count // AUDITOR_BATCH)
        adjudicator_batches = -(-plan.chunk_count // ADJUDICATOR_BATCH)

        budget.reserve(adjudicator_batches)
        for _ in range(auditor_batches):
            assert budget.can_afford(1), "plan under-estimated the audit"
            budget.spend(1)
        budget.release()
        for _ in range(adjudicator_batches):
            assert budget.can_afford(1), "plan under-estimated the ruling"
            budget.spend(1)
        assert budget.spent <= plan.est_total_llm_calls
