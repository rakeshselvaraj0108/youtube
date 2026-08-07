"""The decomposition plan — how a video will be taken apart, before any work.

Computed from the probe alone: no ffmpeg beyond the one that already ran, no
model, no network. It answers "what is this run going to cost me" before the
first API call rather than after the last one.

Every number here is derived from the constants the run actually uses —
`chunking.window_bounds` for the windows, the triad's own batch sizes for the
call counts. Nothing is restated. A plan that carries its own copy of the
stride arithmetic predicts one number while the pipeline does something else,
and a confidently wrong estimate is worse than no estimate at all;
`tests/test_plan.py` pins the two together.

The call counts are deliberately **upper bounds**. Windows containing neither
speech nor on-screen text never reach the AUDITOR, and only flagged candidates
reach the ADVOCATE and ADJUDICATOR — both of which can only push the real
number down. A budget that can be exceeded is not a budget.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from preflight.agents.triad import ADJUDICATOR_BATCH, ADVOCATE_BATCH, AUDITOR_BATCH
from preflight.chunking import window_bounds

# Duration bands. The keyframe budget is the part that genuinely has to move:
# 90 frames over 40 seconds is one every half-second, and over 90 minutes it is
# one per minute. The band names are what the report and the terminal show.
TIERS: list[tuple[int, str, int]] = [
    (120_000, "micro", 24),
    (600_000, "short", 45),
    (1_800_000, "standard", 60),
    (3_600_000, "long", 90),
]
ARCHIVE = ("archive", 120)

# Above this, a flat list of findings stops being readable and the report rolls
# up into segments instead.
HIERARCHICAL_ABOVE_MS = 1_800_000
SEGMENT_MS = 600_000

# One embedding call for the corpus, one for the query batch. Both are cached
# against the corpus digest, so this is what an uncached first run costs.
EMBED_CALLS = 2


@dataclass(frozen=True)
class DecompositionPlan:
    """What the run intends to do, in numbers a judge can check afterwards."""

    duration_ms: int
    tier: str
    chunk_ms: int
    overlap_ms: int
    chunk_count: int
    keyframe_budget: int
    est_auditor_calls: int
    est_advocate_calls: int
    est_adjudicator_calls: int
    est_embed_calls: int
    est_vision_calls: int
    est_total_llm_calls: int
    hierarchical: bool
    segment_ms: int | None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def describe(self) -> list[str]:
        """The plan as the terminal prints it."""
        seconds = self.duration_ms / 1000
        windows = "window" if self.chunk_count == 1 else "windows"
        lines = [
            f"temporal   {self.chunk_count} {windows} · "
            f"{self.chunk_ms // 1000}s window · {self.overlap_ms // 1000}s overlap",
            f"visual     up to {self.keyframe_budget} keyframes",
            f"cost       at most {self.est_total_llm_calls} LLM calls "
            f"({self.est_auditor_calls} auditor · {self.est_advocate_calls} advocate · "
            f"{self.est_adjudicator_calls} adjudicator · {self.est_embed_calls} embed · "
            f"{self.est_vision_calls} vision)",
        ]
        if self.hierarchical:
            lines.append(
                f"rollup     {-(-self.duration_ms // SEGMENT_MS)} segments · "
                f"{SEGMENT_MS // 60_000}min each"
            )
        lines.append(f"tier       {self.tier} · {seconds:.0f}s")
        return lines


def _tier_for(duration_ms: int) -> tuple[str, int]:
    for ceiling, name, frames in TIERS:
        if duration_ms < ceiling:
            return name, frames
    return ARCHIVE


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator) if denominator else 0


def build_plan(
    duration_ms: int, *, chunk_ms: int = 30_000, overlap_ms: int = 5_000
) -> DecompositionPlan:
    """Derive the plan for a video of this length.

    `chunk_ms`/`overlap_ms` come from the same `Settings` the run will use, so
    a configuration override is reflected here rather than silently ignored.
    """
    tier, keyframes = _tier_for(max(0, duration_ms))
    windows = len(window_bounds(duration_ms, chunk_ms, overlap_ms))

    auditor = _ceil_div(windows, AUDITOR_BATCH)
    # Worst case every window yields a candidate the ADVOCATE must defend and
    # the ADJUDICATOR must rule on. In practice the cascade is far narrower.
    advocate = _ceil_div(windows, ADVOCATE_BATCH)
    adjudicator = _ceil_div(windows, ADJUDICATOR_BATCH)
    embed = EMBED_CALLS if windows else 0

    # A05 issues one call per selected keyframe — it is gated, not batched,
    # whatever the module comment says. Leaving it out made the plan claim
    # "at most 5 calls" for a run whose vision layer alone can make 24, which
    # is not a loose estimate but a false upper bound: the report asserts
    # estimatedCalls >= actualCalls, and offline runs made 0 calls so nothing
    # ever contradicted it. The ceiling is the keyframe budget, because the
    # worst case is every extracted frame surviving the gate.
    # Gated on there being a timeline at all: a zero-length video yields no
    # frames, so budgeting frame calls for one is as wrong as omitting them.
    vision = keyframes if windows else 0

    return DecompositionPlan(
        duration_ms=duration_ms,
        tier=tier,
        chunk_ms=chunk_ms,
        overlap_ms=overlap_ms,
        chunk_count=windows,
        keyframe_budget=keyframes,
        est_auditor_calls=auditor,
        est_advocate_calls=advocate,
        est_adjudicator_calls=adjudicator,
        est_embed_calls=embed,
        est_vision_calls=vision,
        est_total_llm_calls=auditor + advocate + adjudicator + embed + vision,
        hierarchical=duration_ms > HIERARCHICAL_ABOVE_MS,
        segment_ms=SEGMENT_MS if duration_ms > HIERARCHICAL_ABOVE_MS else None,
    )
