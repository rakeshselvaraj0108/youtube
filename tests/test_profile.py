"""The media profile — one probe, one source of truth.

Split deliberately. The derivations (bit depth, chroma, HDR class, rotation)
are unit-tested against hand-built payloads, because that is the only way to
cover a Dolby Vision side-data block or a rotation matrix without shipping
sample files for every format in existence. The parse as a whole is then
tested against real ffprobe output, so the two cannot drift apart.
"""

from __future__ import annotations

import pytest

from preflight import ffmpeg
from preflight.ingest.profile import (
    bit_depth_of,
    build_profile,
    chroma_subsampling_of,
    hdr_class_of,
    is_variable_frame_rate,
    profile_video,
    rotation_of,
)


class TestBitDepth:
    @pytest.mark.parametrize(
        "pix_fmt,expected",
        [
            ("yuv420p", 8),
            ("yuv422p", 8),
            ("yuv444p", 8),
            ("yuv420p10le", 10),
            ("yuv422p10le", 10),
            ("yuv420p12le", 12),
            ("gbrp16le", 16),
            ("rgb24", 8),
            ("gray", 8),
        ],
    )
    def test_depth_is_read_from_the_format_name(self, pix_fmt, expected):
        assert bit_depth_of(pix_fmt) == expected

    def test_the_420_in_yuv420p_is_not_mistaken_for_depth(self):
        """The obvious way to write this returns 420."""
        assert bit_depth_of("yuv420p") == 8

    def test_a_stated_depth_wins_over_the_inferred_one(self):
        assert bit_depth_of("yuv420p", bits_per_raw=10) == 10

    def test_an_unknown_format_admits_it(self):
        assert bit_depth_of(None) is None
        assert bit_depth_of("something_else") is None


class TestChromaSubsampling:
    @pytest.mark.parametrize(
        "pix_fmt,expected",
        [
            ("yuv420p", "4:2:0"),
            ("yuv422p10le", "4:2:2"),
            ("yuv444p", "4:4:4"),
            ("gray", "4:0:0"),
            ("rgb24", "4:4:4"),
        ],
    )
    def test_subsampling_is_read_from_the_format_name(self, pix_fmt, expected):
        assert chroma_subsampling_of(pix_fmt) == expected

    def test_an_unknown_format_admits_it(self):
        assert chroma_subsampling_of(None) is None


class TestHdrClass:
    def test_pq_is_hdr10(self):
        assert hdr_class_of("smpte2084", None) == "HDR10"

    def test_hlg_is_named_separately(self):
        assert hdr_class_of("arib-std-b67", None) == "HLG"

    def test_ordinary_content_is_sdr(self):
        assert hdr_class_of("bt709", None) == "SDR"
        assert hdr_class_of(None, None) == "SDR"

    def test_dolby_vision_is_read_from_side_data_not_the_transfer(self):
        """A DV stream carries a base layer whose transfer is often plain
        PQ. Reading the transfer alone reports HDR10 for a Dolby Vision
        file — the right answer to the wrong question."""
        side = [{"side_data_type": "DOVI configuration record"}]
        assert hdr_class_of("smpte2084", side) == "Dolby Vision"


class TestRotation:
    """Phone footage is stored landscape with a rotation flag. Anything
    reasoning about orientation from width and height alone gets it
    backwards for most vertical video ever shot."""

    @pytest.mark.parametrize("stored,expected", [(-90, 90), (90, 270), (180, 180), (0, 0)])
    def test_display_matrix_rotation_is_normalised(self, stored, expected):
        stream = {"side_data_list": [{"rotation": stored}]}
        assert rotation_of(stream) == expected

    def test_the_legacy_rotate_tag_still_works(self):
        assert rotation_of({"tags": {"rotate": "90"}}) == 90

    def test_no_rotation_information_is_zero(self):
        assert rotation_of({}) == 0

    def test_orientation_accounts_for_rotation(self):
        """1920x1080 with a 90 degree flag is a portrait video."""
        data = {
            "format": {"duration": "5.0", "size": "1000", "nb_streams": 1},
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "side_data_list": [{"rotation": -90}],
                }
            ],
        }
        video = build_profile(data).video
        assert video is not None
        assert video.rotation == 90
        assert video.orientation == "portrait"


class TestVariableFrameRate:
    def test_matching_rates_are_constant(self):
        stream = {"r_frame_rate": "30/1", "avg_frame_rate": "30/1"}
        assert is_variable_frame_rate(stream) is False

    def test_ntsc_rounding_is_not_variable(self):
        """30000/1001 against 29.97 is the same rate written twice."""
        stream = {"r_frame_rate": "30000/1001", "avg_frame_rate": "29.97"}
        assert is_variable_frame_rate(stream) is False

    def test_a_real_divergence_is_variable(self):
        stream = {"r_frame_rate": "60/1", "avg_frame_rate": "24/1"}
        assert is_variable_frame_rate(stream) is True

    def test_missing_rates_admit_they_cannot_say(self):
        """None is an answer a boolean cannot give."""
        assert is_variable_frame_rate({}) is None


class TestAbsenceIsNotZero:
    """Real files omit most optional fields. Reporting a default as a fact
    is inventing something about the file."""

    def test_unstated_colour_fields_stay_none(self):
        data = {
            "format": {"duration": "1.0", "size": "10", "nb_streams": 1},
            "streams": [
                {"index": 0, "codec_type": "video", "codec_name": "h264",
                 "width": 640, "height": 480}
            ],
        }
        video = build_profile(data).video
        assert video is not None
        assert video.color_space is None
        assert video.color_primaries is None
        assert video.color_transfer is None

    def test_an_undetermined_language_is_not_a_language(self):
        """'und' is the container saying it was never set."""
        data = {
            "format": {"duration": "1.0", "size": "10", "nb_streams": 1},
            "streams": [
                {"index": 0, "codec_type": "audio", "codec_name": "aac",
                 "tags": {"language": "und"}}
            ],
        }
        assert build_profile(data).audio[0].language is None

    def test_a_video_with_no_streams_does_not_crash(self):
        profile = build_profile({"format": {}, "streams": []})
        assert profile.video is None
        assert not profile.has_audio
        assert not profile.has_captions


class TestMultipleStreams:
    def test_every_audio_track_is_reported(self):
        data = {
            "format": {"duration": "1.0", "size": "10", "nb_streams": 3},
            "streams": [
                {"index": 0, "codec_type": "video", "codec_name": "h264",
                 "width": 1, "height": 1},
                {"index": 1, "codec_type": "audio", "codec_name": "aac",
                 "channels": 2, "tags": {"language": "eng"},
                 "disposition": {"default": 1}},
                {"index": 2, "codec_type": "audio", "codec_name": "ac3",
                 "channels": 6, "tags": {"language": "fra"}},
            ],
        }
        profile = build_profile(data)
        assert [a.language for a in profile.audio] == ["eng", "fra"]
        assert [a.channels for a in profile.audio] == [2, 6]
        assert profile.audio[0].default and not profile.audio[1].default

    def test_subtitle_dispositions_are_read(self):
        data = {
            "format": {"duration": "1.0", "size": "10", "nb_streams": 1},
            "streams": [
                {"index": 0, "codec_type": "subtitle", "codec_name": "mov_text",
                 "tags": {"language": "eng"},
                 "disposition": {"forced": 1, "hearing_impaired": 1}},
            ],
        }
        subtitle = build_profile(data).subtitles[0]
        assert subtitle.forced and subtitle.hearing_impaired
        assert not subtitle.default
        assert build_profile(data).has_captions


@pytest.mark.skipif(not ffmpeg.available(), reason="ffmpeg is not installed")
class TestAgainstRealOutput:
    """The unit tests above build their own payloads, so they cannot catch a
    field ffprobe renamed. This one reads a real file."""

    def test_a_real_clip_profiles_completely(self):
        from pathlib import Path

        clip = Path("data/corpus/clips/g001.mp4")
        if not clip.is_file():
            pytest.skip("corpus clip not present")

        profile = profile_video(clip)
        assert profile.container.duration_ms > 0
        assert profile.container.size_bytes > 0
        assert profile.video is not None
        assert profile.video.codec
        assert profile.video.width > 0 and profile.video.height > 0
        assert profile.video.bit_depth == 8
        assert profile.video.chroma_subsampling == "4:2:0"
        assert profile.video.hdr == "SDR"
        assert profile.has_audio

    def test_the_profile_serialises(self):
        import json
        from pathlib import Path

        clip = Path("data/corpus/clips/g001.mp4")
        if not clip.is_file():
            pytest.skip("corpus clip not present")
        json.dumps(profile_video(clip).to_json())
