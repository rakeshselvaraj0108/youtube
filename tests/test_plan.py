"""The decomposition plan.

A plan is only worth printing if it is right. These tests exist to stop it
becoming decorative: the window count is checked against the chunker that
actually builds the windows, and the call estimate against the batch
constants the triad actually batches by. If either drifts, the plan starts
lying and this suite fails instead of the demo.
"""

from __future__ import annotations

import pytest

from preflight.agents.triad import ADJUDICATOR_BATCH, ADVOCATE_BATCH, AUDITOR_BATCH
from preflight.chunking import build_windows, window_bounds
from preflight.plan import HIERARCHICAL_ABOVE_MS, build_plan


class TestWindowBoundsIsTheSingleSourceOfTruth:
    """`build_windows` and the plan must count windows the same way. They
    drifted apart the moment the arithmetic was written twice — which is the
    same failure mode as two prefix-to-category maps disagreeing, twice over
    in this project already."""

    @pytest.mark.parametrize(
        "duration_ms", [1, 999, 1_000, 29_999, 30_000, 30_001, 125_000, 3_600_000]
    )
    def test_the_plan_counts_exactly_what_the_chunker_builds(self, duration_ms):
        plan = build_plan(duration_ms)
        actual = build_windows(None, duration_ms)
        assert plan.chunk_count == len(actual)

    @pytest.mark.parametrize("duration_ms", [45_000, 300_000, 1_000_000])
    def test_bounds_match_the_windows_span_for_span(self, duration_ms):
        bounds = window_bounds(duration_ms)
        windows = build_windows(None, duration_ms)
        assert bounds == [(w.start_ms, w.end_ms) for w in windows]

    def test_a_zero_length_video_plans_no_work(self):
        plan = build_plan(0)
        assert plan.chunk_count == 0
        assert plan.est_total_llm_calls == 0

    def test_windows_cover_the_whole_timeline(self):
        bounds = window_bounds(125_000)
        assert bounds[0][0] == 0
        assert bounds[-1][1] == 125_000

    def test_a_custom_chunk_size_is_honoured_not_ignored(self):
        """A configuration override the plan silently dropped would make it
        describe a run that never happened."""
        plan = build_plan(300_000, chunk_ms=60_000, overlap_ms=10_000)
        actual = build_windows(None, 300_000, chunk_ms=60_000, overlap_ms=10_000)
        assert plan.chunk_count == len(actual)
        assert plan.chunk_ms == 60_000


class TestCallEstimateIsAnUpperBound:
    """The estimate must never be exceeded. A budget that can be overrun is
    not a budget, so every count here is the worst case: every window
    reaching every stage."""

    def test_auditor_batching_matches_the_triads_own_constant(self):
        plan = build_plan(600_000)
        assert plan.est_auditor_calls == -(-plan.chunk_count // AUDITOR_BATCH)

    def test_advocate_and_adjudicator_use_their_own_batch_sizes(self):
        plan = build_plan(600_000)
        assert plan.est_advocate_calls == -(-plan.chunk_count // ADVOCATE_BATCH)
        assert plan.est_adjudicator_calls == -(-plan.chunk_count // ADJUDICATOR_BATCH)

    def test_the_total_is_the_sum_of_its_parts(self):
        plan = build_plan(600_000)
        assert plan.est_total_llm_calls == (
            plan.est_auditor_calls
            + plan.est_advocate_calls
            + plan.est_adjudicator_calls
            + plan.est_embed_calls
        )

    def test_a_longer_video_never_costs_less(self):
        durations = [60_000, 300_000, 900_000, 2_400_000, 5_400_000]
        totals = [build_plan(d).est_total_llm_calls for d in durations]
        assert totals == sorted(totals)


class TestTiering:
    @pytest.mark.parametrize(
        "duration_ms,tier",
        [
            (40_000, "micro"),
            (300_000, "short"),
            (1_200_000, "standard"),
            (2_400_000, "long"),
            (5_400_000, "archive"),
        ],
    )
    def test_duration_selects_the_band(self, duration_ms, tier):
        assert build_plan(duration_ms).tier == tier

    def test_the_keyframe_budget_grows_with_duration(self):
        budgets = [
            build_plan(d).keyframe_budget
            for d in (40_000, 300_000, 1_200_000, 2_400_000, 5_400_000)
        ]
        assert budgets == sorted(budgets)

    def test_rollup_engages_only_past_the_threshold(self):
        assert not build_plan(HIERARCHICAL_ABOVE_MS).hierarchical
        assert build_plan(HIERARCHICAL_ABOVE_MS + 1).hierarchical
        assert build_plan(HIERARCHICAL_ABOVE_MS + 1).segment_ms is not None
        assert build_plan(HIERARCHICAL_ABOVE_MS).segment_ms is None


class TestPresentation:
    def test_describe_names_the_real_counts(self):
        plan = build_plan(300_000)
        rendered = " ".join(plan.describe())
        assert str(plan.chunk_count) in rendered
        assert str(plan.est_total_llm_calls) in rendered

    def test_the_plan_is_json_serialisable(self):
        import json

        json.dumps(build_plan(300_000).to_json())
