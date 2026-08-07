"""Adaptive frame sampling.

Two properties have to hold at once, and each is easy to satisfy alone by
breaking the other:

  * Frames concentrate where the picture moves. Otherwise the whole feature
    is uniform sampling with extra arithmetic.
  * No stretch of video is structurally invisible. A motionless slide
    covered in text has a near-zero frame difference and is exactly where a
    policy problem hides, so pure proportional allocation — the obvious
    implementation — is wrong.

Most of this file is the second property, because it is the one a
motion-weighted sampler naturally violates.
"""

from __future__ import annotations

import numpy as np
import pytest

from preflight.perception.sampling import (
    Allocation,
    allocate,
    nearest_frames,
)


class Frame:
    def __init__(self, index: int, ts_ms: int) -> None:
        self.index, self.ts_ms = index, ts_ms


def busy_then_still(samples: int = 100) -> np.ndarray:
    """First half active, second half frozen."""
    motion = np.zeros(samples, dtype=np.float64)
    motion[: samples // 2] = 30.0
    return motion


class TestBudgetIsRespected:
    @pytest.mark.parametrize("budget", [1, 2, 5, 8, 24, 60])
    def test_never_returns_more_than_the_budget(self, budget):
        result = allocate(600_000, budget, motion=busy_then_still())
        assert len(result.timestamps) <= budget

    def test_spends_the_budget_it_was_given(self):
        """Deduplication can under-spend; the top-up exists so the caller
        gets the frames it paid for."""
        result = allocate(600_000, 20, motion=busy_then_still())
        assert len(result.timestamps) == 20

    def test_a_zero_budget_samples_nothing(self):
        assert allocate(600_000, 0).timestamps == []

    def test_a_zero_length_video_samples_nothing(self):
        assert allocate(0, 10).timestamps == []

    def test_every_timestamp_lands_inside_the_video(self):
        for value in allocate(60_000, 12, motion=busy_then_still()).timestamps:
            assert 0 <= value < 60_000

    def test_timestamps_come_back_in_order(self):
        stamps = allocate(600_000, 15, motion=busy_then_still()).timestamps
        assert stamps == sorted(stamps)

    def test_no_two_frames_describe_the_same_moment(self):
        stamps = allocate(600_000, 20, motion=busy_then_still()).timestamps
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        assert all(gap > 0 for gap in gaps)


class TestMotionConcentratesTheBudget:
    def test_more_frames_land_in_the_moving_half(self):
        """The point of the feature."""
        result = allocate(100_000, 20, motion=busy_then_still())
        in_motion = sum(1 for t in result.timestamps if t < 50_000)
        assert in_motion > len(result.timestamps) / 2

    def test_a_short_burst_of_action_is_not_missed(self):
        """Ten percent of the runtime carrying all the movement should draw
        materially more than ten percent of the frames."""
        motion = np.zeros(100)
        motion[80:90] = 50.0
        result = allocate(100_000, 20, motion=motion)
        in_burst = sum(1 for t in result.timestamps if 80_000 <= t < 90_000)
        assert in_burst >= 3

    def test_the_busiest_moment_does_not_swallow_the_budget(self):
        """Top-N selection clusters every frame inside the single most
        active second — great detail about one moment, nothing about the
        rest. Inverse-CDF sampling is what avoids that."""
        motion = np.zeros(100)
        motion[50] = 1000.0
        motion[10:40] = 5.0
        result = allocate(100_000, 12, motion=motion)
        at_spike = sum(1 for t in result.timestamps if 49_000 <= t <= 51_500)
        assert at_spike <= 4, "the allocator piled onto one instant"


class TestStaticContentIsNeverStarved:
    """The property pure proportional allocation destroys."""

    def test_a_completely_static_video_is_still_sampled(self):
        """A motionless screencast is not an empty one — it may be a slide
        deck of policy-violating text."""
        result = allocate(600_000, 12, motion=np.zeros(200))
        assert len(result.timestamps) == 12

    def test_a_static_video_is_sampled_evenly(self):
        stamps = allocate(600_000, 10, motion=np.zeros(200)).timestamps
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        assert max(gaps) - min(gaps) < 20_000

    @pytest.mark.parametrize(
        "motion",
        [None, np.zeros(100), np.concatenate([np.zeros(50), np.full(50, 30.0)])],
        ids=["no-signal", "static", "busy-second-half"],
    )
    def test_the_whole_timeline_is_covered_not_just_the_start(self, motion):
        """The bug this exists to catch, which every gap-based assertion
        above missed: topping up produced more candidates than the budget
        and the list was then head-sliced, so a static clip put all twenty
        frames in the first third and looked at nothing after it. Uniform
        *gaps* stayed uniform, so nothing failed — the frames were evenly
        spaced across a third of the video.

        A screencast is the common case here, and it is exactly the shape
        of video where the last two thirds matter as much as the first.
        """
        duration = 100_000
        stamps = allocate(duration, 20, motion=motion).timestamps
        assert stamps, "no frames allocated"
        assert max(stamps) > duration * 0.8, (
            f"nothing sampled after {max(stamps) / duration:.0%} of the video"
        )
        deciles = {min(9, int(t / duration * 10)) for t in stamps}
        assert len(deciles) >= 8, f"only {len(deciles)}/10 deciles covered"

    def test_the_still_half_still_gets_coverage(self):
        """Motion-weighted alone would send zero frames here."""
        result = allocate(100_000, 20, motion=busy_then_still())
        in_still = sum(1 for t in result.timestamps if t >= 50_000)
        assert in_still >= 3, "the static half was starved"

    def test_the_uniform_floor_is_reported(self):
        result = allocate(600_000, 20, motion=busy_then_still())
        assert result.uniform_count > 0
        assert result.motion_count > 0

    def test_no_motion_signal_degrades_to_uniform(self):
        """Callers without a motion signal must still get sane coverage."""
        stamps = allocate(60_000, 10).timestamps
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        assert max(gaps) - min(gaps) < 2_000


class TestFlaggedSpansWin:
    def test_a_flagged_span_is_sampled(self):
        """Something was heard there, so a picture of it is corroboration
        the fusion layer can actually use."""
        result = allocate(
            100_000, 10, motion=np.zeros(100), flagged_spans=[(70_000, 72_000)]
        )
        assert any(69_000 <= t <= 73_000 for t in result.timestamps)

    def test_flagged_spans_survive_a_tiny_budget(self):
        result = allocate(
            100_000, 2, motion=busy_then_still(), flagged_spans=[(90_000, 92_000)]
        )
        assert any(89_000 <= t <= 93_000 for t in result.timestamps)

    def test_more_flagged_spans_than_budget_does_not_overflow(self):
        spans = [(i * 1000, i * 1000 + 500) for i in range(50)]
        result = allocate(100_000, 5, motion=None, flagged_spans=spans)
        assert len(result.timestamps) <= 5

    def test_an_empty_span_is_ignored(self):
        result = allocate(100_000, 5, flagged_spans=[(5000, 5000)])
        assert len(result.timestamps) == 5


class TestMappingOntoRealFrames:
    def test_each_timestamp_claims_its_nearest_frame(self):
        frames = [Frame(i, i * 1000) for i in range(20)]
        chosen = nearest_frames([3200, 7800], frames)
        assert [f.ts_ms for f in chosen] == [3000, 8000]

    def test_a_frame_is_never_sent_twice(self):
        """Two nearby timestamps must not both buy the same picture."""
        frames = [Frame(i, i * 10_000) for i in range(5)]
        chosen = nearest_frames([100, 200, 300], frames)
        assert len({f.ts_ms for f in chosen}) == len(chosen)

    def test_more_requests_than_frames_stops_cleanly(self):
        frames = [Frame(0, 0), Frame(1, 1000)]
        assert len(nearest_frames([0, 100, 200, 300], frames)) == 2

    def test_no_frames_yields_nothing(self):
        assert nearest_frames([1000], []) == []

    def test_results_are_in_timeline_order(self):
        frames = [Frame(i, i * 1000) for i in range(20)]
        chosen = nearest_frames([9000, 1000, 5000], frames)
        assert [f.ts_ms for f in chosen] == sorted(f.ts_ms for f in chosen)


class TestReporting:
    def test_the_allocation_explains_itself(self):
        result = allocate(600_000, 20, motion=busy_then_still())
        assert isinstance(result, Allocation)
        payload = result.to_json()
        assert payload["uniformCount"] + payload["motionCount"] <= 20
        assert 0.0 <= payload["motionShare"] <= 1.0
