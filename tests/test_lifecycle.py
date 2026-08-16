"""The remediation state machine.

The claim under test is narrow and load-bearing: a remediation cannot reach a
verdict without passing through the steps that produce the evidence for one.
So the tests assert reachability properties rather than the edge table —
"there is no route from RENDERING to VERIFIED that skips comparison" survives
edits to the table that an edge-by-edge assertion would not.
"""

from __future__ import annotations

import pytest

from preflight import lifecycle


class TestTheGraphIsWellFormed:
    def test_the_table_has_no_structural_problems(self):
        assert lifecycle.validate_graph() == []

    def test_only_the_entry_point_is_unreachable(self):
        """Every state except the starting one must be enterable. A state
        nothing leads to is dead code somewhere trying to set it."""
        assert lifecycle.unreachable_states() == {"ANALYSIS_COMPLETE"}

    def test_terminal_states_have_no_exits(self):
        for state in lifecycle.TERMINAL:
            assert lifecycle.TRANSITIONS.get(state, frozenset()) == frozenset()


class TestImpossibleTransitions:
    """The brief's own example, and the reason the module exists."""

    def test_rendering_cannot_become_verified(self):
        assert not lifecycle.can_transition("RENDERING", "VERIFIED")
        with pytest.raises(lifecycle.InvalidTransition):
            lifecycle.check("RENDERING", "VERIFIED")

    def test_the_error_names_what_was_actually_allowed(self):
        with pytest.raises(lifecycle.InvalidTransition) as caught:
            lifecycle.check("RENDERING", "VERIFIED")
        assert "RENDERED" in str(caught.value)

    def test_a_render_cannot_skip_structural_verification(self):
        assert not lifecycle.can_transition("RENDERED", "REANALYSIS_QUEUED")

    def test_a_render_cannot_skip_reanalysis(self):
        assert not lifecycle.can_transition("STRUCTURALLY_VALID", "COMPARING")

    def test_reanalysis_cannot_reach_a_verdict_without_comparing(self):
        assert not lifecycle.can_transition("REANALYSIS_COMPLETE", "VERIFIED")

    def test_a_terminal_state_cannot_be_left(self):
        for state in lifecycle.TERMINAL:
            assert not lifecycle.can_transition(state, "RENDERING")
            assert not lifecycle.can_transition(state, "FAILED")


class TestEveryPathToAVerdictPassesThroughTheEvidence:
    """The property the whole graph exists to guarantee."""

    @pytest.mark.parametrize(
        "verdict",
        ["VERIFIED", "PARTIALLY_REMEDIATED", "NEW_RISK_DETECTED", "NO_CHANGE"],
    )
    def test_a_verdict_is_only_reachable_through_comparison(self, verdict):
        route = lifecycle.path_between("REMEDIATION_REQUESTED", verdict)
        assert route is not None, f"{verdict} is unreachable"
        assert "COMPARING" in route
        assert "STRUCTURALLY_VALID" in route
        assert "REANALYSIS_COMPLETE" in route

    def test_the_shortest_route_to_verified_is_the_full_loop(self):
        assert lifecycle.path_between("REMEDIATION_REQUESTED", "VERIFIED") == [
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
        ]

    def test_inconclusive_is_reachable_without_comparing(self):
        """A re-analysis that never completed is inconclusive, and must not
        have to pretend it compared anything to say so."""
        assert lifecycle.can_transition("REANALYSING", "INCONCLUSIVE")


class TestFailure:
    def test_failure_is_reachable_from_every_live_state(self):
        for state in lifecycle.ALL_STATES - lifecycle.TERMINAL:
            assert lifecycle.can_transition(state, "FAILED"), state

    def test_a_failed_remediation_is_not_a_weaker_success(self):
        assert lifecycle.state_for_verdict("REMEDIATION_FAILED") == "FAILED"


class TestVerdictMapping:
    def test_verified_safe_maps_to_the_verified_state(self):
        assert lifecycle.state_for_verdict("VERIFIED_SAFE") == "VERIFIED"

    def test_an_unknown_verdict_is_inconclusive_not_verified(self):
        """A verdict this module has not been taught about must degrade to the
        state that claims least, never to the one that claims most."""
        assert lifecycle.state_for_verdict("SOMETHING_NEW") == "INCONCLUSIVE"

    def test_every_comparison_verdict_has_a_state(self):
        from preflight.verify import Verdict
        import typing

        for verdict in typing.get_args(Verdict):
            assert verdict in lifecycle.VERDICT_STATE, verdict

    def test_every_mapped_state_is_terminal(self):
        for state in lifecycle.VERDICT_STATE.values():
            assert lifecycle.is_terminal(state), state


class TestResume:
    """Where an interrupted remediation picks up, and what that costs."""

    def test_an_interrupted_render_restarts_the_render(self):
        assert lifecycle.resume_state("RENDERING") == "REMEDIATION_REQUESTED"

    def test_an_interrupted_reanalysis_keeps_the_render(self):
        """The expensive thing already done is the render, and a crash during
        analysis did not invalidate it."""
        assert lifecycle.resume_state("REANALYSING") == "STRUCTURALLY_VALID"

    def test_an_interrupted_comparison_keeps_both_runs(self):
        assert lifecycle.resume_state("COMPARING") == "REANALYSIS_COMPLETE"

    def test_a_finished_remediation_has_nothing_to_resume(self):
        for state in lifecycle.TERMINAL:
            assert lifecycle.resume_state(state) is None

    def test_every_non_terminal_state_knows_where_it_resumes(self):
        """A state with no resume rule would strand a remediation after a
        crash — findable but not continuable, which is half a feature."""
        for state in lifecycle.ALL_STATES - lifecycle.TERMINAL:
            assert lifecycle.resume_state(state) is not None, state

    def test_a_resume_target_is_never_later_than_the_interruption(self):
        """Resuming forward would skip the work the crash interrupted."""
        order = list(lifecycle.TRANSITIONS)
        for state in lifecycle.ALL_STATES - lifecycle.TERMINAL:
            target = lifecycle.resume_state(state)
            assert order.index(str(target)) <= order.index(state), state
