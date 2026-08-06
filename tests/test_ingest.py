"""Ingest — probing, extraction, and the reproducibility guarantee.

These tests need ffmpeg. They synthesise their own fixture rather than
depending on a committed binary, so a clean clone can run them.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from preflight import cas, ffmpeg
from preflight.ingest import frames as frames_mod
from preflight.ingest.frames import frames_in_span, nearest_frame
from preflight.ingest.pipeline import ingest
from preflight.ingest.probe import UnsupportedInput, probe_video

pytestmark = pytest.mark.skipif(
    not ffmpeg.available(), reason="ffmpeg is not installed"
)

SCENE_ONE_MS = 4000
SCENE_TWO_MS = 8000


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    """A 12s clip with two hard scene cuts at 4s and 8s, plus a tone."""
    out = tmp_path_factory.mktemp("media") / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=4",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
            "-f", "lavfi", "-i", "color=c=darkred:size=320x180:rate=30:duration=4",
            "-f", "lavfi", "-i", "smptebars=size=320x180:rate=30:duration=4",
            "-filter_complex", "[0:v][2:v][3:v]concat=n=3:v=1:a=0[v]",
            "-map", "[v]", "-map", "1:a",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


class TestProbe:
    def test_reads_real_stream_data(self, clip):
        meta = probe_video(clip)
        assert meta.width == 320
        assert meta.height == 180
        assert meta.durationMs == pytest.approx(12_000, abs=200)
        assert meta.sizeBytes > 0
        assert meta.audioCodec == "AAC"
        assert meta.sampleRate > 0

    def test_fps_is_parsed_as_a_fraction_not_assumed(self, clip):
        # 30/1 here, but the point is that it is parsed. Assuming 30 would put
        # every visual timestamp out by 0.1% on 30000/1001 material.
        assert probe_video(clip).fps == pytest.approx(30.0, abs=0.01)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            probe_video(tmp_path / "nope.mp4")

    def test_audio_only_input_is_rejected(self, tmp_path):
        wav = tmp_path / "tone.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=1", str(wav)],
            check=True, capture_output=True,
        )
        with pytest.raises(UnsupportedInput):
            probe_video(wav)


class TestFractionParsing:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("30000/1001", 29.97002997),
            ("30/1", 30.0),
            ("25", 25.0),
            ("0/0", 0.0),
            (None, 0.0),
            ("garbage", 0.0),
        ],
    )
    def test_parses_or_falls_back(self, value, expected):
        assert ffmpeg.parse_fraction(value) == pytest.approx(expected, rel=1e-6)


class TestIngest:
    def test_first_run_extracts_everything(self, clip, tmp_path):
        result = ingest(clip, cas.Store(tmp_path))

        assert result.cached is False
        assert result.asr_wav.is_file()
        assert result.fingerprint_wav.is_file()
        assert result.poster.is_file()
        assert result.poster.stat().st_size > 0
        assert len(result.keyframes) >= 2

    def test_keyframes_land_on_the_real_scene_cuts(self, clip, tmp_path):
        """The pts_time mapping is what makes a visual finding placeable."""
        result = ingest(clip, cas.Store(tmp_path))
        stamps = [f.ts_ms for f in result.keyframes]

        assert any(abs(ts - SCENE_ONE_MS) < 400 for ts in stamps), stamps
        assert any(abs(ts - SCENE_TWO_MS) < 400 for ts in stamps), stamps

    def test_keyframe_timestamps_are_monotonic(self, clip, tmp_path):
        result = ingest(clip, cas.Store(tmp_path))
        stamps = [f.ts_ms for f in result.keyframes]
        assert stamps == sorted(stamps)

    def test_second_run_hits_cache_and_does_no_work(self, clip, tmp_path):
        """The reproducibility guarantee, mechanically verified."""
        store = cas.Store(tmp_path)
        first = ingest(clip, store)
        second = ingest(clip, store)

        assert first.cached is False
        assert second.cached is True
        assert second.video_hash == first.video_hash
        assert second.meta == first.meta
        assert [f.ts_ms for f in second.keyframes] == [f.ts_ms for f in first.keyframes]
        assert "0 ffmpeg invocations" in " ".join(second.log)

    def test_cache_hit_is_substantially_faster(self, clip, tmp_path):
        store = cas.Store(tmp_path)
        first = ingest(clip, store)
        second = ingest(clip, store)
        assert second.elapsed_ms < first.elapsed_ms

    def test_partial_entry_is_treated_as_a_miss(self, clip, tmp_path):
        store = cas.Store(tmp_path)
        result = ingest(clip, store)
        # Simulate a crash between extraction and commit.
        store.entry("v", result.video_hash).marker.unlink()
        assert ingest(clip, store).cached is False

    def test_frame_cap_is_respected(self, clip, tmp_path):
        result = ingest(clip, cas.Store(tmp_path), max_frames=1, scene_threshold=0.05)
        assert len(result.keyframes) <= 1


@pytest.fixture(scope="module")
def cutless_clip(tmp_path_factory) -> Path:
    """8s of a single locked-off shot — no cuts anywhere in it.

    A single-take talking head, a screencast and a static-camera piece all
    look like this to a scene detector, and all three are ordinary YouTube
    formats. This fixture exists because that case took the whole run down.
    """
    out = tmp_path_factory.mktemp("media") / "cutless.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=steelblue:size=320x180:rate=30:duration=8",
            "-f", "lavfi", "-i", "sine=frequency=330:duration=8",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


class TestCutlessVideo:
    """A video with no scene cuts must still be analysable.

    ffmpeg 9 fails the extraction command outright when the filter chain
    passes zero frames to the encoder — "Could not open encoder before EOF" —
    rather than writing nothing and exiting clean. That turned an ordinary
    format into a crash, and even without the crash, zero keyframes leaves
    vision and OCR with nothing to look at: 35% of the analysis surface dark
    for exactly the videos least likely to have a problem visible at a cut.
    """

    def test_a_cutless_video_still_yields_keyframes(self, cutless_clip, tmp_path):
        frames = frames_mod.extract_keyframes(
            cutless_clip, tmp_path / "frames", duration_ms=8000
        )
        assert frames

    def test_those_frames_are_declared_as_uniformly_sampled(self, cutless_clip, tmp_path):
        """A frame at a cut and a frame at a clock tick are not equally
        informative, and the report says which it has."""
        frames = frames_mod.extract_keyframes(
            cutless_clip, tmp_path / "frames", duration_ms=8000
        )
        assert frames_mod.sampled_uniformly(frames) is True

    def test_they_spread_across_the_runtime(self, cutless_clip, tmp_path):
        frames = frames_mod.extract_keyframes(
            cutless_clip, tmp_path / "frames", duration_ms=8000
        )
        assert frames[0].ts_ms < 2000
        assert frames[-1].ts_ms > 5000

    def test_a_cut_bearing_video_still_uses_scene_detection(self, clip, tmp_path):
        frames = frames_mod.extract_keyframes(
            clip, tmp_path / "frames", duration_ms=12_000
        )
        assert frames_mod.sampled_uniformly(frames) is False

    def test_the_full_ingest_of_a_cutless_video_succeeds(self, cutless_clip, tmp_path):
        result = ingest(cutless_clip, cas.Store(tmp_path))
        assert result.keyframes
        assert any("uniformly" in line for line in result.log)


class TestFrameLookup:
    def test_nearest_frame_of_empty_list_is_none(self):
        assert nearest_frame([], 1000) is None

    def test_nearest_frame_picks_the_closest(self, clip, tmp_path):
        frames = ingest(clip, cas.Store(tmp_path)).keyframes
        picked = nearest_frame(frames, SCENE_ONE_MS)
        assert picked is not None
        assert abs(picked.ts_ms - SCENE_ONE_MS) < 400

    def test_span_lookup_falls_back_to_nearest(self, clip, tmp_path):
        """A finding in a gap between cuts still gets visual evidence."""
        frames = ingest(clip, cas.Store(tmp_path)).keyframes
        assert frames_in_span(frames, 11_500, 11_900) != []

    def test_data_uri_is_a_jpeg(self, clip, tmp_path):
        frames = ingest(clip, cas.Store(tmp_path)).keyframes
        uri = frames[0].data_uri()
        assert uri.startswith("data:image/jpeg;base64,")
        assert len(uri) > 100
