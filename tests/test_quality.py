"""Picture quality, motion and thumbnail intelligence.

Every threshold in this module was set by measuring constructed video, and
the numbers in these tests are those measurements. They are recorded here so
a future change that shifts them fails loudly instead of quietly making the
labels mean something else.

Measured on testsrc2 at 320x180 (see the module docstring):

    case          sharpness   contrast   label
    sharp            1663.2      79.68   sharp
    boxblur=3          45.6      ~78      soft
    boxblur=6          15.2      70.83    very soft
    solid colour        0.0       0.00    featureless
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from preflight import ffmpeg
from preflight.perception import quality
from preflight.perception.quality import (
    MAX_SAMPLES,
    _sample_rate,
    analyse_motion,
    analyse_quality,
    blockiness,
    colorfulness,
    laplacian_variance,
    luma,
    pick_thumbnails,
)


def synthetic(n=6, h=18, w=32, fill=128) -> np.ndarray:
    return np.full((n, h, w, 3), fill, dtype=np.uint8)


class TestSampleBudget:
    """Cost bounded by count, not duration — the property that makes this
    safe to run on a thirty-minute upload."""

    @pytest.mark.parametrize(
        "duration_ms", [1_000, 60_000, 600_000, 1_800_000, 3_600_000]
    )
    def test_never_samples_more_than_the_budget(self, duration_ms):
        rate = _sample_rate(duration_ms)
        assert rate * (duration_ms / 1000) <= MAX_SAMPLES * 1.01

    def test_a_thirty_minute_file_costs_what_a_short_one_does(self):
        short = _sample_rate(120_000) * 120
        long = _sample_rate(1_800_000) * 1800
        assert abs(short - long) < 5

    def test_a_very_short_clip_is_not_sampled_absurdly_fast(self):
        assert _sample_rate(1_000) <= 4.0


class TestPureMetrics:
    """The maths, on arrays built by hand — no decode involved."""

    def test_a_flat_frame_has_no_laplacian_energy(self):
        assert laplacian_variance(luma(synthetic()))[0] == pytest.approx(0.0)

    def test_structure_raises_the_focus_measure(self):
        frames = synthetic()
        frames[:, ::2, :, :] = 255  # horizontal stripes
        assert laplacian_variance(luma(frames))[0] > 100

    def test_grey_is_not_colourful(self):
        assert colorfulness(synthetic(fill=128))[0] == pytest.approx(0.0, abs=1.0)

    def test_saturated_colour_scores_higher_than_grey(self):
        grey = synthetic(fill=128)
        red = synthetic()
        red[..., 0], red[..., 1], red[..., 2] = 255, 0, 0
        assert colorfulness(red)[0] > colorfulness(grey)[0]

    def test_a_flat_frame_has_no_blockiness(self):
        assert blockiness(luma(synthetic())) == pytest.approx(0.0)

    def test_blockiness_needs_enough_width_to_see_a_grid(self):
        """Below 24 columns there is no 8-pixel grid to measure against."""
        assert blockiness(luma(synthetic(w=16))) == 0.0

    def test_empty_input_never_raises(self):
        empty = np.empty((0, 4, 4, 3), dtype=np.uint8)
        assert laplacian_variance(luma(empty)).size == 0
        assert colorfulness(empty).size == 0
        report = analyse_quality(empty)
        assert report.frames_sampled == 0
        assert report.blur_label == "unknown"


class TestFeaturelessIsNotBlurry:
    """A solid title card reads zero sharpness. Calling that "very soft"
    accuses a deliberate design choice of being out of focus — measured on
    a solid blue clip, which reported 0.0 sharpness at 0.0 contrast."""

    def test_a_solid_frame_is_featureless_not_soft(self):
        assert analyse_quality(synthetic(fill=64)).blur_label == "featureless"

    def test_a_smooth_gradient_is_soft_not_featureless(self):
        """A gradient has plenty of contrast and almost no high-frequency
        detail, which is exactly what a soft picture looks like to a focus
        measure. The first version of this test used a hard two-tone split
        and expected "soft" — but a hard edge is the sharpest thing in the
        frame, so the code was right and the fixture was wrong."""
        ramp = np.linspace(0, 255, 32, dtype=np.uint8)
        frames = np.repeat(
            np.tile(ramp, (18, 1))[None, :, :, None], 3, axis=3
        ).astype(np.uint8)
        report = analyse_quality(frames)
        assert report.contrast > 8.0, "gradient should carry real contrast"
        assert report.blur_label in {"soft", "very soft"}


class TestMotionAndScenes:
    def test_a_static_sequence_has_no_motion(self):
        report = analyse_motion(synthetic(n=10), 10_000)
        assert report.motion_density == pytest.approx(0.0)
        assert report.still_pct == 1.0
        assert report.hard_cuts == []

    def test_a_single_frame_cannot_report_motion(self):
        assert analyse_motion(synthetic(n=1), 1000).scene_count == 0

    def test_an_abrupt_change_reads_as_a_hard_cut(self):
        frames = synthetic(n=10, fill=20)
        frames[5:] = 230
        report = analyse_motion(frames, 10_000)
        assert report.hard_cuts, "a full-frame change was not detected as a cut"
        assert report.scene_count >= 2

    def test_a_gradual_change_is_not_reported_as_a_hard_cut(self):
        """A dissolve raises the difference across several samples. Calling
        it a hard cut misreports the edit, and a fade misreported as a cut
        inflates the scene count."""
        frames = np.stack(
            [np.full((18, 32, 3), int(20 + i * 23), dtype=np.uint8) for i in range(10)]
        )
        report = analyse_motion(frames, 10_000)
        assert report.hard_cuts == []

    def test_scene_lengths_are_consistent(self):
        frames = synthetic(n=12, fill=20)
        frames[6:] = 230
        report = analyse_motion(frames, 12_000)
        assert report.shortest_scene_ms <= report.average_scene_ms
        assert report.average_scene_ms <= report.longest_scene_ms


class TestThumbnails:
    def test_the_most_colourful_frame_is_chosen_over_grey(self):
        frames = synthetic(n=5, fill=128)
        frames[3, ..., 0], frames[3, ..., 1], frames[3, ..., 2] = 255, 10, 10
        picked = pick_thumbnails(frames, 5000)
        assert picked.most_colorful_ms == 3000

    def test_the_sharpest_frame_is_chosen_over_flat_ones(self):
        frames = synthetic(n=5)
        frames[2, ::2, :, :] = 255
        assert pick_thumbnails(frames, 5000).sharpest_ms == 2000

    def test_no_frames_yields_no_candidates(self):
        picked = pick_thumbnails(np.empty((0, 4, 4, 3), dtype=np.uint8), 1000)
        assert picked.best_ms is None

    def test_candidates_land_inside_the_video(self):
        picked = pick_thumbnails(synthetic(n=8), 8000)
        for value in (picked.most_colorful_ms, picked.sharpest_ms, picked.best_ms):
            assert 0 <= value < 8000


@pytest.mark.skipif(not ffmpeg.available(), reason="ffmpeg is not installed")
class TestAgainstRealVideo:
    """The synthetic tests cannot catch a broken ffmpeg invocation."""

    def _build(self, tmp_path, name, vf=None, src=None):
        out = tmp_path / name
        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i",
            src or "testsrc2=size=320x180:rate=30:duration=3",
        ]
        if vf:
            command += ["-vf", vf]
        command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)]
        subprocess.run(command, check=True, capture_output=True)
        return out

    def test_a_real_file_decodes_into_frames(self, tmp_path):
        clip = self._build(tmp_path, "real.mp4")
        frames = quality.rgb_frames(clip, 3000)
        assert frames.shape[0] > 0
        assert frames.shape[3] == 3

    def test_blur_is_detected_on_real_video(self, tmp_path):
        """The measurement the thresholds came from: 1663 against 15."""
        sharp = quality.analyse_quality(
            quality.rgb_frames(self._build(tmp_path, "s.mp4"), 3000)
        )
        soft = quality.analyse_quality(
            quality.rgb_frames(self._build(tmp_path, "b.mp4", vf="boxblur=6"), 3000)
        )
        assert sharp.sharpness > soft.sharpness * 10
        assert sharp.blur_label == "sharp"
        assert soft.blur_label == "very soft"

    def test_a_solid_colour_clip_is_featureless(self, tmp_path):
        clip = self._build(
            tmp_path, "solid.mp4",
            src="color=c=blue:size=320x180:rate=30:duration=3",
        )
        report = quality.analyse_quality(quality.rgb_frames(clip, 3000))
        assert report.blur_label == "featureless"

    def test_heavier_compression_reads_as_more_blocky(self, tmp_path):
        """crf 51 against a default encode — measured 2.25 against 1.32."""
        good = tmp_path / "good.mp4"
        bad = tmp_path / "bad.mp4"
        for out, crf in ((good, "18"), (bad, "51")):
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=3",
                 "-c:v", "libx264", "-crf", crf, "-pix_fmt", "yuv420p", str(out)],
                check=True, capture_output=True,
            )
        clean = quality.analyse_quality(quality.rgb_frames(good, 3000))
        crushed = quality.analyse_quality(quality.rgb_frames(bad, 3000))
        assert crushed.blockiness > clean.blockiness

    def test_a_missing_file_returns_no_frames_rather_than_raising(self, tmp_path):
        assert quality.rgb_frames(tmp_path / "nope.mp4", 1000).shape[0] == 0
        assert quality.analyse(tmp_path / "nope.mp4", 1000) is None
