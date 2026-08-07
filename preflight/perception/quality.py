"""Picture quality, motion and thumbnail intelligence — from one decode.

The naive shape of this module is six ffmpeg passes: one for brightness, one
for blur, one for motion, one for colourfulness, one for the sharpest frame,
one for scene lengths. On a thirty-minute file that is six full decodes of
the same bytes to answer six questions about the same pixels.

This decodes once, at a fixed sample budget, and computes everything from
the frames it already has. Cost is bounded by `MAX_SAMPLES` rather than by
duration, so a thirty-minute podcast costs the same as a two-minute clip —
which is the property that makes it safe to run on every upload.

Every metric here is an **estimate of a perceptual property**, not a
measurement of a physical one, and the module says so in its own output.
Blur is Laplacian variance; that is a real, standard, well-behaved proxy,
and it is still a proxy. The thresholds separating "sharp" from "soft" were
set by measuring constructed video — a testsrc2 pattern against the same
pattern through boxblur — rather than chosen because they looked round.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from preflight import ffmpeg

# Sample budget. Bounded by count rather than rate so a 30-minute file costs
# what a 2-minute file costs: aggregate quality does not get more accurate
# with 4000 frames than with 240, and the difference is minutes of decode.
MAX_SAMPLES = 240

# Big enough for an 8x8 blockiness grid and a meaningful Laplacian; small
# enough that 240 frames is ~10MB rather than ~1GB.
SAMPLE_W = 160
SAMPLE_H = 90

# Measured, not guessed. See tests/test_quality.py: a testsrc2 pattern reads
# ~1800 Laplacian variance, the same pattern through boxblur=6 reads ~40.
BLUR_SOFT = 120.0
BLUR_VERY_SOFT = 40.0

# Fraction of pixels at the extremes before exposure is called clipped.
CLIP_LEVEL = 4
CLIP_FRACTION = 0.02

# Mean absolute frame difference, 0-255. Measured against constructed
# motion: a static frame reads ~0, testsrc2 reads ~25.
STILL_DIFF = 1.0
FAST_DIFF = 18.0

# A diff spike this many times the local median reads as a hard cut.
CUT_RATIO = 4.0
CUT_MIN_DIFF = 8.0


def _sample_rate(duration_ms: int) -> float:
    """Frames per second that yields at most MAX_SAMPLES over the file."""
    seconds = max(duration_ms, 1) / 1000
    if seconds <= 0:
        return 1.0
    rate = MAX_SAMPLES / seconds
    # The cap is the guarantee and wins outright. An earlier version also
    # imposed a 0.1fps floor so short scenes would not vanish, which quietly
    # broke the budget on anything over forty minutes — a one-hour file
    # sampled 360 frames against a 240 ceiling. A floor and a cap cannot
    # both be absolute, and the one worth keeping is the one that bounds
    # cost, because that is what makes this safe to run on every upload.
    return float(min(4.0, rate))


def rgb_frames(
    source: Path, duration_ms: int, *, width: int = SAMPLE_W, height: int = SAMPLE_H
) -> np.ndarray:
    """Evenly-spaced RGB frames, shape (n, height, width, 3), uint8.

    Piped straight out of ffmpeg with no intermediate files, the same way
    `signal._grayscale_frames` works — this is its colour sibling, kept
    separate because colourfulness and blockiness genuinely need the chroma
    that greyscale throws away.
    """
    fps = _sample_rate(duration_ms)
    command = [
        ffmpeg._resolve("ffmpeg"),
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        f"fps={fps},scale={width}:{height}",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]
    result = subprocess.run(command, capture_output=True)
    frame_bytes = width * height * 3
    if result.returncode != 0 or len(result.stdout) < frame_bytes:
        return np.empty((0, height, width, 3), dtype=np.uint8)

    usable = len(result.stdout) - (len(result.stdout) % frame_bytes)
    return (
        np.frombuffer(result.stdout[:usable], dtype=np.uint8)
        .reshape(-1, height, width, 3)
    )


def luma(frames: np.ndarray) -> np.ndarray:
    """Rec.709 luma, shape (n, h, w), float32."""
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return frames.astype(np.float32) @ weights


def laplacian_variance(gray: np.ndarray) -> np.ndarray:
    """Per-frame focus measure. Higher is sharper.

    The standard blur proxy: a sharp image has strong second derivatives,
    a soft one does not. Computed with an explicit 4-neighbour kernel rather
    than a library so the numbers are reproducible without scipy.
    """
    if gray.size == 0:
        return np.zeros(0, dtype=np.float32)
    centre = gray[:, 1:-1, 1:-1]
    lap = (
        gray[:, :-2, 1:-1]
        + gray[:, 2:, 1:-1]
        + gray[:, 1:-1, :-2]
        + gray[:, 1:-1, 2:]
        - 4.0 * centre
    )
    return lap.reshape(lap.shape[0], -1).var(axis=1)


def colorfulness(frames: np.ndarray) -> np.ndarray:
    """Hasler & Süsstrunk colourfulness, per frame.

    The published metric, not an invention: opponent-colour spread plus a
    weighted contribution from the mean. It is what picks a thumbnail that
    looks like something rather than a grey frame of someone blinking.
    """
    if frames.size == 0:
        return np.zeros(0, dtype=np.float32)
    values = frames.astype(np.float32)
    red, green, blue = values[..., 0], values[..., 1], values[..., 2]
    rg = red - green
    yb = 0.5 * (red + green) - blue
    flat = (rg.reshape(rg.shape[0], -1), yb.reshape(yb.shape[0], -1))
    std = np.sqrt(flat[0].std(axis=1) ** 2 + flat[1].std(axis=1) ** 2)
    mean = np.sqrt(flat[0].mean(axis=1) ** 2 + flat[1].mean(axis=1) ** 2)
    return std + 0.3 * mean


def blockiness(gray: np.ndarray) -> float:
    """Compression blocking, 0 upward.

    Codecs quantise on an 8x8 grid, so heavy compression leaves gradient
    energy concentrated at every eighth column. Comparing that against the
    energy elsewhere is what separates blocking from ordinary detail — an
    absolute gradient measure would just report "this image has edges".
    """
    if gray.shape[0] == 0 or gray.shape[2] < 24:
        return 0.0
    columns = np.abs(np.diff(gray, axis=2)).mean(axis=(0, 1))
    grid = np.zeros(columns.shape[0], dtype=bool)
    grid[7::8] = True
    if not grid.any() or grid.all():
        return 0.0
    on_grid = float(columns[grid].mean())
    off_grid = float(columns[~grid].mean())
    if off_grid <= 1e-6:
        return 0.0
    return max(0.0, on_grid / off_grid - 1.0)


def noise_estimate(gray: np.ndarray) -> float:
    """High-frequency residual in the flattest regions.

    Measured where the picture is smooth, because texture and noise look
    identical to a high-pass filter and only the flat areas can tell them
    apart.
    """
    if gray.shape[0] == 0 or gray.shape[1] < 4:
        return 0.0
    high_pass = gray[:, 1:-1, 1:-1] - 0.25 * (
        gray[:, :-2, 1:-1] + gray[:, 2:, 1:-1] + gray[:, 1:-1, :-2] + gray[:, 1:-1, 2:]
    )
    local = np.abs(high_pass).reshape(high_pass.shape[0], -1)
    # The lowest decile is the flat part of the picture.
    return float(np.percentile(local, 10, axis=1).mean())


@dataclass(frozen=True)
class QualityReport:
    """Estimates, labelled as such."""

    frames_sampled: int
    brightness: float          # 0-255 mean luma
    contrast: float            # luma standard deviation
    clipped_highlights: float  # fraction of pixels at ceiling
    clipped_shadows: float
    sharpness: float           # median Laplacian variance
    blur_label: str            # sharp | soft | very soft
    noise: float
    blockiness: float
    colorfulness: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MotionReport:
    motion_density: float      # mean absolute inter-frame difference
    fast_motion_pct: float
    still_pct: float
    camera_shake: float        # variance of the motion signal
    scene_count: int
    average_scene_ms: int
    longest_scene_ms: int
    shortest_scene_ms: int
    hard_cuts: list[int]
    gradual_transitions: list[int]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ThumbnailCandidates:
    """Frame timestamps worth offering as a poster."""

    most_colorful_ms: int | None
    sharpest_ms: int | None
    best_ms: int | None
    middle_ms: int | None
    last_ms: int | None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# Below this luma spread there is no detail for a focus measure to act on.
# Measured: a solid colour frame reads 0.0 sharpness and ~0 contrast, and
# calling that "very soft" accuses a deliberate title card of being out of
# focus. Featureless and blurry are different findings.
FEATURELESS_CONTRAST = 8.0


def _labelled_blur(sharpness: float, contrast: float) -> str:
    if contrast < FEATURELESS_CONTRAST:
        return "featureless"
    if sharpness < BLUR_VERY_SOFT:
        return "very soft"
    if sharpness < BLUR_SOFT:
        return "soft"
    return "sharp"


def analyse_quality(frames: np.ndarray) -> QualityReport:
    gray = luma(frames)
    if frames.shape[0] == 0:
        return QualityReport(0, 0, 0, 0, 0, 0, "unknown", 0, 0, 0)

    flat = gray.reshape(gray.shape[0], -1)
    sharpness = float(np.median(laplacian_variance(gray)))
    contrast = float(flat.std())
    return QualityReport(
        frames_sampled=int(frames.shape[0]),
        brightness=round(float(flat.mean()), 2),
        contrast=round(contrast, 2),
        clipped_highlights=round(float((gray >= 255 - CLIP_LEVEL).mean()), 4),
        clipped_shadows=round(float((gray <= CLIP_LEVEL).mean()), 4),
        sharpness=round(sharpness, 2),
        blur_label=_labelled_blur(sharpness, contrast),
        noise=round(noise_estimate(gray), 3),
        blockiness=round(blockiness(gray), 3),
        colorfulness=round(float(np.median(colorfulness(frames))), 2),
    )


def analyse_motion(frames: np.ndarray, duration_ms: int) -> MotionReport:
    gray = luma(frames)
    count = gray.shape[0]
    if count < 2:
        return MotionReport(0, 0, 0, 0, 0, 0, 0, 0, [], [])

    step_ms = duration_ms / count
    diffs = np.abs(np.diff(gray, axis=0)).reshape(count - 1, -1).mean(axis=1)

    median = float(np.median(diffs)) or 1e-6
    spikes = (diffs > median * CUT_RATIO) & (diffs > CUT_MIN_DIFF)
    cut_indices = np.flatnonzero(spikes)

    # A hard cut is one sample wide; a fade or dissolve raises the
    # difference across several consecutive samples. Separating them is the
    # difference between "12 cuts" and "12 cuts, 3 of which are dissolves".
    hard: list[int] = []
    gradual: list[int] = []
    for index in cut_indices:
        neighbours = spikes[max(0, index - 1): index + 2]
        (gradual if neighbours.sum() > 1 else hard).append(int(index * step_ms))
    gradual = sorted(set(gradual))

    boundaries = [0, *sorted(hard + gradual), duration_ms]
    lengths = [b - a for a, b in zip(boundaries, boundaries[1:]) if b > a]

    return MotionReport(
        motion_density=round(float(diffs.mean()), 3),
        fast_motion_pct=round(float((diffs > FAST_DIFF).mean()), 4),
        still_pct=round(float((diffs < STILL_DIFF).mean()), 4),
        # Shake is high-frequency variation in the motion signal itself: a
        # steady pan has high motion and low shake, a handheld shot has both.
        camera_shake=round(float(np.abs(np.diff(diffs)).mean()) if count > 2 else 0.0, 3),
        scene_count=len(lengths),
        average_scene_ms=int(sum(lengths) / len(lengths)) if lengths else duration_ms,
        longest_scene_ms=int(max(lengths)) if lengths else duration_ms,
        shortest_scene_ms=int(min(lengths)) if lengths else duration_ms,
        hard_cuts=hard,
        gradual_transitions=gradual,
    )


def pick_thumbnails(frames: np.ndarray, duration_ms: int) -> ThumbnailCandidates:
    """Frames worth offering as a poster.

    "Best" is sharpness times colourfulness, both normalised. A frame can be
    razor sharp and be a black screen; it can be vivid and be motion blur.
    The product is the only one of the three that is usually a picture of
    something.
    """
    count = frames.shape[0]
    if count == 0:
        return ThumbnailCandidates(None, None, None, None, None)

    step_ms = duration_ms / count
    gray = luma(frames)
    sharp = laplacian_variance(gray)
    colour = colorfulness(frames)

    def scale(values: np.ndarray) -> np.ndarray:
        spread = float(values.max() - values.min())
        return (values - values.min()) / spread if spread > 1e-6 else np.zeros_like(values)

    best = scale(sharp) * scale(colour)
    return ThumbnailCandidates(
        most_colorful_ms=int(int(np.argmax(colour)) * step_ms),
        sharpest_ms=int(int(np.argmax(sharp)) * step_ms),
        best_ms=int(int(np.argmax(best)) * step_ms),
        middle_ms=int(duration_ms // 2),
        last_ms=int(max(0, duration_ms - 1000)),
    )


@dataclass(frozen=True)
class MediaIntelligence:
    quality: QualityReport
    motion: MotionReport
    thumbnails: ThumbnailCandidates

    def to_json(self) -> dict[str, Any]:
        return {
            "quality": self.quality.to_json(),
            "motion": self.motion.to_json(),
            "thumbnails": self.thumbnails.to_json(),
        }


def analyse(source: Path, duration_ms: int) -> MediaIntelligence | None:
    """One decode, every metric. None when the file cannot be sampled."""
    frames = rgb_frames(Path(source), duration_ms)
    if frames.shape[0] == 0:
        return None
    return MediaIntelligence(
        quality=analyse_quality(frames),
        motion=analyse_motion(frames, duration_ms),
        thumbnails=pick_thumbnails(frames, duration_ms),
    )
