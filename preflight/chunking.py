"""Windowing the timeline.

30-second windows with 5 seconds of overlap. The overlap is not padding: a
sentence that begins at 0:29 and finishes at 0:33 must appear *whole* in at
least one window, or the adjudicator rules on half a sentence and either
invents the missing half or dismisses a real violation.

Overlap means the same violation can surface twice, so findings are later
deduplicated by clause plus temporal IoU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from preflight.ingest.frames import Keyframe
from preflight.perception.asr import Transcript


@dataclass
class Window:
    index: int
    start_ms: int
    end_ms: int
    transcript: str
    frames: list[Keyframe] = field(default_factory=list)
    ocr: list[str] = field(default_factory=list)

    @property
    def has_content(self) -> bool:
        """Whether this window is worth spending an LLM call on.

        Purely a presence check. Silence and frames with no readable text carry
        no evidence a policy clause could attach to, so they never reach the
        model. This is the only gate applied — similarity thresholds were
        measured and do not separate in-scope windows from clean ones.
        """
        return bool(self.transcript.strip()) or bool([t for t in self.ocr if t.strip()])

    def query(self) -> str:
        """Retrieval query: everything the window knows, concatenated."""
        parts = [self.transcript.strip()]
        parts.extend(t.strip() for t in self.ocr if t.strip())
        return " ".join(p for p in parts if p)

    def for_prompt(self) -> str:
        lines = [
            f"WINDOW {self.index} [{self.start_ms}ms - {self.end_ms}ms]",
            f"TRANSCRIPT: {self.transcript.strip() or '(no speech)'}",
        ]
        if self.ocr:
            joined = " | ".join(t.strip() for t in self.ocr if t.strip())
            if joined:
                lines.append(f"ON-SCREEN TEXT: {joined}")
        return "\n".join(lines)


def build_windows(
    transcript: Transcript | None,
    duration_ms: int,
    keyframes: list[Keyframe] | None = None,
    *,
    chunk_ms: int = 30_000,
    overlap_ms: int = 5_000,
) -> list[Window]:
    if duration_ms <= 0:
        return []

    step = max(1, chunk_ms - overlap_ms)
    keyframes = keyframes or []
    windows: list[Window] = []

    index = 0
    start = 0
    while start < duration_ms:
        end = min(start + chunk_ms, duration_ms)
        text = transcript.text_between(start, end) if transcript else ""
        frames = [f for f in keyframes if start <= f.ts_ms < end]
        windows.append(
            Window(
                index=index,
                start_ms=start,
                end_ms=end,
                transcript=text,
                frames=frames,
            )
        )
        index += 1
        if end >= duration_ms:
            break
        start += step

    return windows


def iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Temporal intersection over union."""
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    intersection = max(0, end - start)
    union = (a[1] - a[0]) + (b[1] - b[0]) - intersection
    return intersection / union if union > 0 else 0.0
