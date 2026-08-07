"""Never decode the same frame twice.

This property is invisible when it breaks. The pipeline keeps working, every
finding stays correct, and the run simply costs more — so it degrades run by
run as agents are added, and nothing fails. It degraded exactly that way
here: `_grayscale_frames` carried a comment claiming it was shared, and a
measured run showed `fps=10,scale=64:36` decoded twice back to back for two
views of identical pixels.

Instrumenting a real run is the only way to test it. Counting call sites in
the source cannot see that two different agents reached the same extractor
with the same arguments.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from preflight import cas, ffmpeg
from preflight.config import Settings
from preflight.ingest import audio as audio_io
from preflight.perception import signal as sig

pytestmark = pytest.mark.skipif(not ffmpeg.available(), reason="ffmpeg is not installed")


def _describe(command) -> str:
    """A stable name for one ffmpeg invocation: the filter graph is what
    makes two passes the same work or different work."""
    if not isinstance(command, (list, tuple)) or not command:
        return "?"
    exe = str(command[0]).lower()
    if "ffprobe" in exe:
        return "ffprobe"
    if "ffmpeg" not in exe:
        return exe
    for index, token in enumerate(command):
        if token in ("-vf", "-af", "-filter_complex") and index + 1 < len(command):
            return f"ffmpeg[{command[index + 1]}]"
    return "ffmpeg[copy]"


@pytest.fixture
def counted_run(monkeypatch):
    """Every subprocess launch during the block, by filter graph."""
    counts: dict[str, int] = {}
    real = subprocess.run

    def spy(command, *args, **kwargs):
        counts[_describe(command)] = counts.get(_describe(command), 0) + 1
        return real(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy)
    return counts


@pytest.fixture
def clip(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("reuse") / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=3",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(out),
        ],
        check=True, capture_output=True,
    )
    return out


class TestFrameCache:
    def setup_method(self):
        sig.clear_frame_cache()

    def test_the_same_extraction_decodes_once(self, clip, counted_run):
        """Two views of identical pixels, one decode. This is the exact
        pair that was costing a duplicate: black-frame luminance and
        frozen-frame differences, both at the same rate and size."""
        sig.luminance_series(clip, fps=10)
        sig.frame_diff_series(clip, fps=10)
        greyscale = [k for k in counted_run if "format=gray" in k]
        assert sum(counted_run[k] for k in greyscale) == 1
        assert sig.frame_cache_stats()["hits"] == 1

    def test_a_different_rate_is_a_different_decode(self, clip, counted_run):
        """The cache must not answer a question it was not asked. Flash
        detection needs 30fps and freeze detection needs 10; returning the
        10fps frames for both would silently halve the flash agent's
        temporal resolution."""
        sig.luminance_series(clip, fps=10)
        sig.luminance_series(clip, fps=30)
        greyscale = [k for k in counted_run if "format=gray" in k]
        assert sum(counted_run[k] for k in greyscale) == 2

    def test_an_edited_file_is_not_served_from_cache(self, clip, tmp_path):
        """Keyed on mtime and size, so a file rewritten in place between
        calls is a miss. A stale hit here would analyse the previous cut."""
        first = sig.luminance_series(clip, fps=10)
        replacement = tmp_path / "clip.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=c=white:size=320x180:rate=30:duration=3",
             "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
             str(replacement)],
            check=True, capture_output=True,
        )
        replacement.replace(clip)
        second = sig.luminance_series(clip, fps=10)
        assert second.mean() != pytest.approx(first.mean(), abs=1.0)

    def test_the_cache_is_bounded(self, clip):
        """An unbounded cache holds every frame of every file the process
        ever touched — a leak that only shows on a long-running server."""
        for fps in range(2, 12):
            sig.luminance_series(clip, fps=fps)
        assert len(sig._FRAME_CACHE) <= sig._FRAME_CACHE_LIMIT

    def test_cached_frames_cannot_be_mutated_by_a_consumer(self, clip):
        """Shared by reference, so one agent normalising in place would
        corrupt the next agent's view of the same decode."""
        frames = sig._grayscale_frames(clip, 10, 64, 36)
        with pytest.raises(ValueError):
            frames[0][0] = 1.0


class TestLoudnessCache:
    def setup_method(self):
        audio_io.clear_loudness_cache()

    def test_loudness_is_measured_once_per_file(self, clip, counted_run):
        """R128 is a full decode of the audio and was the single most
        expensive duplicate in the pipeline — A04 and the acoustic tier both
        asked for it."""
        first = audio_io.loudness(clip)
        second = audio_io.loudness(clip)
        passes = [k for k in counted_run if "loudnorm" in k and "print_format" in k]
        assert sum(counted_run[k] for k in passes) == 1
        assert first == second

    def test_a_file_without_audio_is_not_retried(self, tmp_path, counted_run):
        """A silent file will not grow an audio stream mid-run. Retrying the
        doomed pass for every caller is the same waste as repeating a
        successful one."""
        silent = tmp_path / "silent.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=c=black:size=64x64:rate=10:duration=1",
             "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
             str(silent)],
            check=True, capture_output=True,
        )
        audio_io.loudness(silent)
        audio_io.loudness(silent)
        passes = [k for k in counted_run if "loudnorm" in k]
        assert sum(counted_run[k] for k in passes) <= 1


class TestWholePipelineHasNoDuplicatePasses:
    """The regression guard. Adding an agent that re-extracts something an
    existing agent already produced fails here rather than quietly making
    every run slower."""

    def test_no_filter_graph_runs_twice(self, clip, counted_run):
        sig.clear_frame_cache()
        audio_io.clear_loudness_cache()

        from preflight.pipeline import run_perception

        store = cas.Store(Path(tempfile.mkdtemp()))
        run_perception(clip, store, settings=Settings.load(offline=True))

        repeated = {
            name: count
            for name, count in counted_run.items()
            # Keyframe extraction legitimately runs twice: a scene-cut pass,
            # then a uniform fallback when the first selected nothing. They
            # are different filter graphs, so they do not collide here.
            if count > 1 and name.startswith("ffmpeg[")
        }
        assert repeated == {}, f"duplicate ffmpeg passes: {repeated}"
