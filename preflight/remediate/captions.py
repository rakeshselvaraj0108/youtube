"""Caption emission.

Not every finding is fixed by a filter graph. The highest-severity
accessibility finding — no caption track — is repaired by writing a file, and
the run already holds everything needed: the speech agent produced word-level
timings, so generating captions costs nothing beyond formatting.

Emitting them here rather than telling the creator to go and make some is the
difference between a linter and a linter with `--fix`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from preflight.perception.asr import Transcript, Word

MAX_LINE_CHARS = 42
MAX_CUE_MS = 6_000
MAX_CUE_WORDS = 14
GAP_SPLIT_MS = 700


@dataclass
class Cue:
    start_ms: int
    end_ms: int
    text: str


def _stamp(ms: int, sep: str = ".") -> str:
    ms = max(0, ms)
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{sep}{millis:03d}"


def build_cues(transcript: Transcript) -> list[Cue]:
    """Group words into readable cues.

    Split on three signals, whichever comes first: a pause long enough to be a
    sentence boundary, a cue growing past comfortable reading length, or the
    duration ceiling. Captions that run longer than about six seconds are read
    twice and then ignored.
    """
    cues: list[Cue] = []
    current: list[Word] = []

    def flush() -> None:
        if not current:
            return
        text = " ".join(w.w for w in current).strip()
        if text:
            cues.append(Cue(current[0].start_ms, current[-1].end_ms, text))
        current.clear()

    for word in transcript.words:
        if current:
            gap = word.start_ms - current[-1].end_ms
            span = word.end_ms - current[0].start_ms
            length = sum(len(w.w) + 1 for w in current)
            if (
                gap > GAP_SPLIT_MS
                or span > MAX_CUE_MS
                or len(current) >= MAX_CUE_WORDS
                or length > MAX_LINE_CHARS * 2
            ):
                flush()
        current.append(word)
    flush()

    # Never let two cues overlap — players render the collision as flicker.
    for earlier, later in zip(cues, cues[1:]):
        if earlier.end_ms > later.start_ms:
            earlier.end_ms = later.start_ms
    return [c for c in cues if c.end_ms > c.start_ms]


def _wrap(text: str) -> str:
    if len(text) <= MAX_LINE_CHARS:
        return text
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) > MAX_LINE_CHARS and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return "\n".join(lines[:2]) if len(lines) <= 2 else "\n".join(lines[:2])


def to_vtt(transcript: Transcript) -> str:
    cues = build_cues(transcript)
    out = ["WEBVTT", ""]
    for index, cue in enumerate(cues, start=1):
        out.append(str(index))
        out.append(f"{_stamp(cue.start_ms)} --> {_stamp(cue.end_ms)}")
        out.append(_wrap(cue.text))
        out.append("")
    return "\n".join(out)


def to_srt(transcript: Transcript) -> str:
    cues = build_cues(transcript)
    out: list[str] = []
    for index, cue in enumerate(cues, start=1):
        out.append(str(index))
        out.append(f"{_stamp(cue.start_ms, ',')} --> {_stamp(cue.end_ms, ',')}")
        out.append(_wrap(cue.text))
        out.append("")
    return "\n".join(out)


def write_captions(transcript: Transcript, destination: Path) -> Path | None:
    """Write a .vtt beside the remediated video. None when there is no speech."""
    if not transcript.words:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(to_vtt(transcript), encoding="utf-8")
    return destination
