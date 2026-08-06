"""Thin, honest wrappers around ffmpeg and ffprobe.

Everything the engine does to media routes through here so there is exactly one
place that knows how the binaries are invoked, and exactly one error type to
catch when they are missing.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


class FfmpegMissing(RuntimeError):
    """ffmpeg or ffprobe is not on PATH."""


class FfmpegFailed(RuntimeError):
    """A command ran and returned non-zero."""

    def __init__(self, command: list[str], returncode: int, stderr: str) -> None:
        self.command = command
        self.returncode = returncode
        self.stderr = stderr
        tail = stderr.strip().splitlines()[-6:]
        super().__init__(
            f"{command[0]} exited {returncode}\n  " + "\n  ".join(tail)
        )


def _resolve(binary: str) -> str:
    found = shutil.which(binary)
    if not found:
        raise FfmpegMissing(
            f"{binary} was not found on PATH. Install ffmpeg and re-run.\n"
            "  Windows:  winget install Gyan.FFmpeg\n"
            "  macOS:    brew install ffmpeg\n"
            "  Debian:   sudo apt install ffmpeg"
        )
    return found


def available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def version() -> str | None:
    if not available():
        return None
    out = subprocess.run(
        [_resolve("ffmpeg"), "-version"], capture_output=True, text=True, check=False
    ).stdout
    first = out.splitlines()[0] if out else ""
    return first.split(" Copyright")[0].strip() or None


def run(args: list[str], *, capture_stderr: bool = False) -> subprocess.CompletedProcess[str]:
    """Run ffmpeg with the given arguments after `ffmpeg -hide_banner`."""
    command = [_resolve("ffmpeg"), "-hide_banner", "-nostdin", *args]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise FfmpegFailed(command, result.returncode, result.stderr)
    return result if capture_stderr else result


def probe(path: Path) -> dict[str, Any]:
    """`ffprobe -show_format -show_streams` as parsed JSON."""
    command = [
        _resolve("ffprobe"),
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise FfmpegFailed(command, result.returncode, result.stderr)
    return json.loads(result.stdout)


_SSIM_ALL = re.compile(r"All:\s*([\d.]+)")


def quality_delta(original: Path, fixed: Path, *, timeout: int = 900) -> float | None:
    """Mean SSIM between two videos — how much a re-encode actually changed.

    "The remediation changed 0.4% of frames, SSIM 0.998" answers the question
    a creator actually has about an automated fix: will this wreck my video?
    None when the comparison itself fails (mismatched resolution, a source
    that no longer exists) rather than raising — a failed quality check
    should not undo a render that otherwise succeeded and verified correctly
    on duration.

    Only worth computing when the video stream was actually re-encoded. An
    audio-only edit stream-copies the video (`-c:v copy`), which makes it
    byte-identical, and SSIM would trivially read 1.0 for real work spent
    decoding two copies of the same frames.
    """
    command = [
        _resolve("ffmpeg"),
        "-hide_banner",
        "-nostdin",
        "-i",
        str(fixed),
        "-i",
        str(original),
        "-lavfi",
        "[0:v][1:v]ssim",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    match = _SSIM_ALL.search(result.stderr)
    return float(match.group(1)) if match else None


def parse_fraction(value: str | None, default: float = 0.0) -> float:
    """Parse ffprobe's `num/den` rationals.

    `r_frame_rate` arrives as "30000/1001", not 29.97 and certainly not 30.
    Assuming 30 here would put every visual finding's timestamp out by 0.1% —
    small, but it compounds across an 18-minute file and is exactly the sort of
    detail that makes a timestamp untrustworthy.
    """
    if not value:
        return default
    if "/" in value:
        num, _, den = value.partition("/")
        try:
            numerator, denominator = float(num), float(den)
        except ValueError:
            return default
        return numerator / denominator if denominator else default
    try:
        return float(value)
    except ValueError:
        return default
