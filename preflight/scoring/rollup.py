"""Segment rollup for long videos.

A 90-minute podcast with 200 findings is not usefully reported as a list of
200 findings. It is usefully reported as "segments 4 and 7 carry 82% of the
risk" — which is the same data, grouped so a creator knows where to go first.

Risk per segment reuses `readiness.finding_risk` rather than defining a
second severity weighting. A rollup that ranked segments by its own private
notion of severity could disagree with the headline score about which part of
the video is worst, and there is no reading of that disagreement where the
report is right.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from preflight.models import Finding
from preflight.scoring.readiness import finding_risk


@dataclass(frozen=True)
class Segment:
    index: int
    start_ms: int
    end_ms: int
    finding_count: int
    risk_share: float
    dominant_clause: str | None
    worst_severity: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "findingCount": self.finding_count,
            "riskShare": self.risk_share,
            "dominantClause": self.dominant_clause,
            "worstSeverity": self.worst_severity,
        }


def _overlaps(finding: Finding, start: int, end: int) -> bool:
    # Half-open on the right so a finding ending exactly at a boundary belongs
    # to the segment it played in, not the one that starts as it stops. A
    # zero-length finding still lands in exactly one segment.
    return finding.startMs < end and max(finding.endMs, finding.startMs + 1) > start


def rollup(
    findings: list[Finding], duration_ms: int, segment_ms: int = 600_000
) -> list[Segment]:
    """Group findings into fixed segments, ranked by share of total risk.

    A finding spanning a boundary counts in every segment it overlaps — it
    genuinely affects all of them. Shares are normalised across segments, so
    they sum to 1.0 and read as "this fraction of the problem is here"
    regardless of how much spanning occurred.
    """
    if duration_ms <= 0 or segment_ms <= 0:
        return []

    buckets: list[tuple[int, int, list[Finding]]] = []
    for index, start in enumerate(range(0, duration_ms, segment_ms)):
        end = min(start + segment_ms, duration_ms)
        buckets.append(
            (start, end, [f for f in findings if _overlaps(f, start, end)])
        )

    risks = [sum(finding_risk(f) for f in inside) for _, _, inside in buckets]
    total = sum(risks)

    segments = [
        Segment(
            index=index,
            start_ms=start,
            end_ms=end,
            finding_count=len(inside),
            risk_share=round(risk / total, 4) if total > 0 else 0.0,
            dominant_clause=(
                Counter(f.clauseId for f in inside).most_common(1)[0][0]
                if inside
                else None
            ),
            worst_severity=(
                max(inside, key=finding_risk).severity if inside else None
            ),
        )
        for index, ((start, end, inside), risk) in enumerate(zip(buckets, risks))
    ]
    return sorted(segments, key=lambda s: (-s.risk_share, s.index))


def concentration(segments: list[Segment], threshold: float = 0.8) -> list[Segment]:
    """The fewest segments accounting for `threshold` of the risk.

    This is the sentence the report wants: "two of nine segments carry 82% of
    the risk". Returns them in timeline order once chosen, because that is how
    someone scrubbing a timeline will visit them.
    """
    chosen: list[Segment] = []
    running = 0.0
    for segment in segments:
        if running >= threshold or segment.risk_share <= 0:
            break
        chosen.append(segment)
        running += segment.risk_share
    return sorted(chosen, key=lambda s: s.index)
