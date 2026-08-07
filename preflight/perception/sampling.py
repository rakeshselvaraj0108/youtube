"""Adaptive frame sampling — spend the vision budget where the picture moves.

Vision is the expensive agent: one hosted call per frame, and the budget is
small enough that which frames it sees decides what it can possibly find.
Uniform sampling spends that budget as though every second were equally
likely to contain something, which is only true of a video with no edits.

The motion signal this reads is free. `quality.analyse_motion` already
computes per-sample frame differences from a decode the run is paying for
anyway, so adaptive sampling costs arithmetic rather than another pass.

**The half that is not motion-weighted is the important half.** Pure
proportional allocation starves static content completely, and a motionless
slide covered in text is exactly where a policy problem hides — the frame
diff is near zero and the content is the whole point. So the budget splits:
a uniform floor that guarantees no stretch of video is unseen, and a
motion-weighted remainder that concentrates the rest where the picture is
actually changing. Neither half alone is defensible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Share of the budget spent on guaranteed uniform coverage. The remainder is
# allocated by motion. At 0.5 a completely static video still gets half its
# budget spread evenly — which is the correct answer, because a static video
# is not an empty one.
UNIFORM_SHARE = 0.5

# Motion below this reads as "nothing is happening" and is not competed for.
# Measured against constructed video in tests/test_quality.py: a static frame
# reads ~0 mean absolute difference, testsrc2 reads ~25.
QUIET_DIFF = 1.0


@dataclass(frozen=True)
class Allocation:
    """Where the budget went, and why — so a report can explain the choice."""

    timestamps: list[int]
    uniform_count: int
    motion_count: int
    flagged_count: int
    motion_share: float

    def to_json(self) -> dict[str, Any]:
        return {
            "timestamps": self.timestamps,
            "uniformCount": self.uniform_count,
            "motionCount": self.motion_count,
            "flaggedCount": self.flagged_count,
            "motionShare": self.motion_share,
        }


def _uniform(duration_ms: int, count: int) -> list[int]:
    """Evenly spaced timestamps, offset to the middle of each slot.

    Sampling at slot boundaries lands on cuts, where a frame is a
    half-dissolve of two shots and describes neither.
    """
    if count <= 0 or duration_ms <= 0:
        return []
    step = duration_ms / count
    return [int(step * (index + 0.5)) for index in range(count)]


def _motion_weighted(motion: np.ndarray, duration_ms: int, count: int) -> list[int]:
    """Timestamps drawn in proportion to motion energy.

    Uses a cumulative distribution rather than picking the top-N samples:
    top-N clusters every frame inside the single most active second, which
    describes one moment in great detail and the rest of the video not at
    all. Inverse-CDF sampling spreads them across the whole distribution
    while still favouring the busy parts.
    """
    if count <= 0 or motion.size == 0 or duration_ms <= 0:
        return []

    weights = np.clip(motion.astype(np.float64) - QUIET_DIFF, 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        # Nothing moved anywhere. Motion cannot rank what does not vary, so
        # this half of the budget falls back to even coverage rather than
        # piling onto sample zero by accident.
        return _uniform(duration_ms, count)

    cdf = np.cumsum(weights) / total
    step_ms = duration_ms / motion.size
    targets = (np.arange(count) + 0.5) / count
    indices = np.searchsorted(cdf, targets, side="left")
    indices = np.clip(indices, 0, motion.size - 1)
    return [int((index + 0.5) * step_ms) for index in indices]


def _dedupe(
    timestamps: list[int],
    duration_ms: int,
    spacing_ms: int,
    *,
    priority: list[int] | None = None,
) -> list[int]:
    """Collapse timestamps closer together than `spacing_ms`.

    Two frames 40ms apart are the same picture bought twice. But *which* of
    a close pair survives is not arbitrary, and the first version got it
    wrong by sorting and sweeping: a flagged span at 71s was silently
    dropped because a uniform frame happened to land at 70s, discarding the
    one timestamp another agent had specifically asked for.

    Priority timestamps are placed first and are never displaced; everything
    else fills the gaps around them.
    """
    def clamp(value: int) -> int:
        return max(0, min(int(value), max(0, duration_ms - 1)))

    kept: list[int] = []

    def offer(value: int) -> None:
        clamped = clamp(value)
        # Linear scan is fine and stays fine: the budget is tens of frames,
        # never thousands, so this is not the O(n^2) it would be on a
        # per-frame timeline.
        if all(abs(clamped - existing) >= spacing_ms for existing in kept):
            kept.append(clamped)

    for value in sorted(priority or []):
        offer(value)
    for value in sorted(timestamps):
        offer(value)

    return sorted(kept)


def _fill_gaps(timestamps: list[int], duration_ms: int, budget: int) -> list[int]:
    """Spend the leftover budget on the widest uncovered stretches.

    Two wrong versions came before this one, and both looked fine.

    Generating extra candidates and slicing `[:budget]` kept the earliest
    frames and abandoned the end of the video — a static clip put all twenty
    frames in the first third. Selecting evenly *by list index* then fixed
    the coverage but broke the spacing, because the candidate list is dense
    where the allocator concentrated and sparse elsewhere, so even index
    steps are uneven time steps.

    Filling the largest remaining gap, one frame at a time, is the version
    that preserves both: it never moves a frame the motion pass chose, and
    it puts every spare frame where the timeline is least covered.
    """
    kept = sorted(timestamps)
    while len(kept) < budget:
        edges = [0, *kept, duration_ms]
        widest = max(range(len(edges) - 1), key=lambda i: edges[i + 1] - edges[i])
        midpoint = (edges[widest] + edges[widest + 1]) // 2
        if midpoint in kept or not 0 <= midpoint < duration_ms:
            break
        kept.append(midpoint)
        kept.sort()
    return kept


def allocate(
    duration_ms: int,
    budget: int,
    *,
    motion: np.ndarray | None = None,
    flagged_spans: list[tuple[int, int]] | None = None,
    uniform_share: float = UNIFORM_SHARE,
) -> Allocation:
    """Timestamps worth spending the vision budget on.

    Priority order, highest first:

      1. Spans another modality already flagged. Something was heard there,
         so a picture of it is corroboration the fusion layer can use.
      2. A uniform floor, so no stretch of video is structurally invisible.
      3. Motion-weighted remainder, concentrated where the picture changes.
    """
    if budget <= 0 or duration_ms <= 0:
        return Allocation([], 0, 0, 0, 0.0)

    # Two frames closer than this describe the same moment.
    #
    # Sized well below the average gap on purpose. An earlier version used
    # duration/(budget*4), which is a quarter of the uniform spacing — wide
    # enough to dismantle the very concentration the motion pass exists to
    # create. A ten-second burst allocated ten frames had nine of them
    # merged away, and the feature silently reduced to uniform sampling.
    # This is a floor against buying the same picture twice, not a spacing
    # policy.
    spacing_ms = max(120, int(duration_ms / (budget * 40)))

    flagged: list[int] = []
    for start, end in flagged_spans or []:
        if end > start:
            flagged.append(int((start + end) // 2))
    flagged = _dedupe(flagged, duration_ms, spacing_ms)[:budget]

    remaining = budget - len(flagged)
    if remaining <= 0:
        return Allocation(sorted(flagged), 0, 0, len(flagged), 0.0)

    # Splitting the budget only makes sense when motion can tell one moment
    # from another. With no signal — or a flat one — both halves generate
    # the same uniform points, dedupe collapses them to half the coverage,
    # and the gap filler then patches the result into an uneven comb. One
    # uniform pass over the whole remaining budget is both simpler and the
    # correct answer for a video with nothing to discriminate on.
    usable = (
        motion is not None
        and motion.size > 0
        and float(np.clip(motion - QUIET_DIFF, 0.0, None).sum()) > 0.0
    )
    if not usable:
        uniform_count, motion_count = remaining, 0
        uniform = _uniform(duration_ms, remaining)
        weighted: list[int] = []
    else:
        uniform_count = max(1, int(round(remaining * uniform_share)))
        uniform = _uniform(duration_ms, uniform_count)
        motion_count = remaining - uniform_count
        weighted = _motion_weighted(motion, duration_ms, motion_count)

    merged = _dedupe(
        [*uniform, *weighted], duration_ms, spacing_ms, priority=flagged
    )

    # Deduplication can leave the budget under-spent. Spend the remainder on
    # the widest uncovered stretches rather than returning fewer frames than
    # the caller paid for.
    merged = _fill_gaps(merged, duration_ms, budget)[:budget]
    flagged_kept = sum(1 for value in merged if value in set(flagged))
    return Allocation(
        timestamps=merged,
        uniform_count=uniform_count,
        motion_count=motion_count,
        flagged_count=flagged_kept,
        motion_share=round(motion_count / budget, 3) if budget else 0.0,
    )


def nearest_frames(timestamps: list[int], frames: list) -> list:
    """Map chosen timestamps onto the keyframes that actually exist.

    The allocator reasons about an ideal timeline; only the extracted
    keyframes can be sent. Each timestamp claims its closest frame, and a
    frame already claimed is not sent twice.
    """
    if not frames:
        return []
    available = sorted(frames, key=lambda f: f.ts_ms)
    positions = np.array([f.ts_ms for f in available])
    taken: set[int] = set()
    chosen = []
    for value in timestamps:
        index = int(np.argmin(np.abs(positions - value)))
        # Walk outward to the nearest unclaimed frame rather than dropping
        # the request: the budget was allocated on purpose.
        if index in taken:
            offsets = np.argsort(np.abs(positions - value))
            index = next((int(i) for i in offsets if int(i) not in taken), -1)
            if index < 0:
                break
        taken.add(index)
        chosen.append(available[index])
    return sorted(chosen, key=lambda f: f.ts_ms)
