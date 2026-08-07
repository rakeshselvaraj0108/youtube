"""The technical profile of a media file — one probe, one source of truth.

Every agent that wanted to know the frame rate, the colour space or whether
there was a second audio track used to reach for ffprobe itself. That is
three or four extra process launches per run for information that does not
change, and worse, four places that each parse the same JSON slightly
differently and disagree at the edges.

This is that parse, done once. `ingest` builds it and hands it down; nothing
downstream needs to probe again.

Two principles run through the whole module.

**Absence is not zero.** Real files omit most of the optional fields — the
demo clip carries no `color_space`, `color_primaries` or `color_transfer` at
all, because SDR H.264 usually does not bother. A profile that reported
"colour space: unknown" as "BT.601" would be inventing a fact about the
file. Every optional field is `None` when the container did not say, and
`None` means *not stated*, never *not present*.

**Derived facts show their working.** Bit depth, chroma subsampling and HDR
class are not fields in the container; they are read out of the pixel format
and the transfer characteristics. Each derivation is a named function with
its own test, because "10-bit" appearing in a report is a claim, and a claim
needs somewhere to point when it is wrong.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from preflight import ffmpeg

# Pixel formats carry depth and subsampling in their name. Parsing the name
# is not elegant, but the alternative is a table of every format ffmpeg
# supports, which goes stale the first time one is added.
_SUBSAMPLING = {"420": "4:2:0", "422": "4:2:2", "444": "4:4:4", "410": "4:1:0", "411": "4:1:1"}

# Transfer characteristics that mean HDR. Everything else is SDR.
_HDR_TRANSFER = {
    "smpte2084": "HDR10",       # PQ
    "arib-std-b67": "HLG",      # Hybrid Log-Gamma
}

_INTERLACED_FIELD_ORDERS = {"tt", "bb", "tb", "bt", "interlaced"}


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bit_depth_of(pix_fmt: str | None, bits_per_raw: Any = None) -> int | None:
    """Bits per component, from the pixel format name.

    `yuv420p10le` is ten-bit; a bare `yuv420p` is eight. ffprobe sometimes
    also reports `bits_per_raw_sample`, which is preferred when present
    because it is stated rather than inferred.
    """
    stated = _int(bits_per_raw)
    if stated:
        return stated
    if not pix_fmt:
        return None
    digits = ""
    for char in pix_fmt:
        if char.isdigit():
            digits += char
        elif digits and char in "lb":  # le / be suffix terminates the depth
            break
        elif digits and len(digits) >= 3:
            digits = ""  # the 420 in yuv420p is subsampling, not depth
    # Formats name depth only when it is not 8: yuv420p vs yuv420p10le.
    for marker in ("p16", "p14", "p12", "p10", "p9"):
        if marker in pix_fmt:
            return int(marker[1:])
    return 8 if pix_fmt.startswith(("yuv", "gbr", "rgb", "bgr", "gray")) else None


def chroma_subsampling_of(pix_fmt: str | None) -> str | None:
    """4:2:0, 4:2:2, 4:4:4 — read out of the format name."""
    if not pix_fmt:
        return None
    for token, label in _SUBSAMPLING.items():
        if token in pix_fmt:
            return label
    if pix_fmt.startswith("gray"):
        return "4:0:0"
    if pix_fmt.startswith(("rgb", "bgr", "gbr")):
        return "4:4:4"
    return None


def hdr_class_of(color_transfer: str | None, side_data: list[dict] | None) -> str:
    """SDR, HDR10, HLG or Dolby Vision.

    Dolby Vision is detected from a side-data block rather than the transfer
    function, because a DV stream carries a base layer whose transfer is
    often ordinary PQ — reading the transfer alone reports HDR10 for a file
    that is really Dolby Vision.
    """
    for block in side_data or []:
        kind = str(block.get("side_data_type", "")).lower()
        if "dovi" in kind or "dolby vision" in kind:
            return "Dolby Vision"
    return _HDR_TRANSFER.get((color_transfer or "").lower(), "SDR")


def rotation_of(stream: dict[str, Any]) -> int:
    """Playback rotation in degrees, normalised to 0/90/180/270.

    Phone footage is routinely stored landscape with a rotation flag, so a
    portrait video reports 1920x1080 with a 90° rotation. Anything that
    reasons about orientation from width and height alone gets it backwards
    for most vertical video ever shot.
    """
    for block in stream.get("side_data_list", []) or []:
        if "rotation" in block:
            degrees = _int(block.get("rotation"))
            if degrees is not None:
                return int(-degrees % 360)
    tag = _int((stream.get("tags") or {}).get("rotate"))
    return int(tag % 360) if tag is not None else 0


def is_variable_frame_rate(stream: dict[str, Any]) -> bool | None:
    """Whether the stream is VFR, as far as the container admits.

    `r_frame_rate` is the base rate and `avg_frame_rate` the measured
    average; they diverge when frames are not evenly spaced. Screen
    recordings and phone captures are routinely VFR, and a VFR source is why
    a cut placed at a computed timestamp can land on the wrong frame.

    None when there is not enough information to say — an honest answer that
    a boolean cannot give.
    """
    base = ffmpeg.parse_fraction(stream.get("r_frame_rate"))
    average = ffmpeg.parse_fraction(stream.get("avg_frame_rate"))
    if not base or not average:
        return None
    # Tolerance covers rounding in the container's rationals rather than
    # real variation: 30000/1001 against 29.97 is the same rate.
    return abs(base - average) / base > 0.01


@dataclass(frozen=True)
class Container:
    format_name: str
    format_long_name: str
    duration_ms: int
    size_bytes: int
    bitrate_bps: int | None
    created_at: str | None
    encoder: str | None
    stream_count: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoStream:
    index: int
    codec: str
    profile: str | None
    level: int | None
    width: int
    height: int
    display_aspect_ratio: str | None
    sample_aspect_ratio: str | None
    pix_fmt: str | None
    bit_depth: int | None
    chroma_subsampling: str | None
    color_space: str | None
    color_primaries: str | None
    color_transfer: str | None
    color_range: str | None
    hdr: str
    frame_rate: float
    avg_frame_rate: float
    variable_frame_rate: bool | None
    scan_type: str | None
    interlaced: bool
    rotation: int
    bitrate_bps: int | None
    frame_count: int | None

    @property
    def orientation(self) -> str:
        """What the viewer sees, after rotation is applied."""
        width, height = self.width, self.height
        if self.rotation in (90, 270):
            width, height = height, width
        if width == height:
            return "square"
        return "landscape" if width > height else "portrait"

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "orientation": self.orientation}


@dataclass(frozen=True)
class AudioStream:
    index: int
    codec: str
    profile: str | None
    channels: int | None
    channel_layout: str | None
    sample_rate: int | None
    bitrate_bps: int | None
    language: str | None
    duration_ms: int | None
    default: bool

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SubtitleStream:
    index: int
    codec: str
    language: str | None
    forced: bool
    default: bool
    hearing_impaired: bool

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MediaProfile:
    """Everything the container knows, parsed once."""

    container: Container
    video: VideoStream | None
    audio: list[AudioStream] = field(default_factory=list)
    subtitles: list[SubtitleStream] = field(default_factory=list)

    @property
    def has_audio(self) -> bool:
        return bool(self.audio)

    @property
    def has_captions(self) -> bool:
        """An embedded subtitle track. Not the same as a caption *file*
        sitting next to the video, which A02 checks separately."""
        return bool(self.subtitles)

    def to_json(self) -> dict[str, Any]:
        return {
            "container": self.container.to_json(),
            "video": self.video.to_json() if self.video else None,
            "audio": [a.to_json() for a in self.audio],
            "subtitles": [s.to_json() for s in self.subtitles],
        }


def _streams(data: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [s for s in data.get("streams", []) if s.get("codec_type") == kind]


def _disposition(stream: dict[str, Any], flag: str) -> bool:
    return bool((stream.get("disposition") or {}).get(flag))


def _language(stream: dict[str, Any]) -> str | None:
    tag = (stream.get("tags") or {}).get("language")
    # "und" is the container's way of saying it was never set; reporting it
    # as a language would be worse than admitting it is unknown.
    return None if not tag or tag.lower() in {"und", "unknown"} else str(tag)


def build_profile(data: dict[str, Any]) -> MediaProfile:
    """Parse one ffprobe payload into the profile. Pure — no subprocess."""
    fmt = data.get("format", {})
    tags = {str(k).lower(): v for k, v in (fmt.get("tags") or {}).items()}

    container = Container(
        format_name=str(fmt.get("format_name", "")),
        format_long_name=str(fmt.get("format_long_name", "")),
        duration_ms=int(round((_float(fmt.get("duration")) or 0.0) * 1000)),
        size_bytes=_int(fmt.get("size")) or 0,
        bitrate_bps=_int(fmt.get("bit_rate")),
        created_at=tags.get("creation_time"),
        encoder=tags.get("encoder") or tags.get("writing_library"),
        stream_count=_int(fmt.get("nb_streams")) or len(data.get("streams", [])),
    )

    video_streams = _streams(data, "video")
    video: VideoStream | None = None
    if video_streams:
        raw = video_streams[0]
        pix_fmt = raw.get("pix_fmt")
        field_order = raw.get("field_order")
        video = VideoStream(
            index=_int(raw.get("index")) or 0,
            codec=str(raw.get("codec_name", "")),
            profile=raw.get("profile"),
            level=_int(raw.get("level")),
            width=_int(raw.get("width")) or 0,
            height=_int(raw.get("height")) or 0,
            display_aspect_ratio=raw.get("display_aspect_ratio"),
            sample_aspect_ratio=raw.get("sample_aspect_ratio"),
            pix_fmt=pix_fmt,
            bit_depth=bit_depth_of(pix_fmt, raw.get("bits_per_raw_sample")),
            chroma_subsampling=chroma_subsampling_of(pix_fmt),
            color_space=raw.get("color_space"),
            color_primaries=raw.get("color_primaries"),
            color_transfer=raw.get("color_transfer"),
            color_range=raw.get("color_range"),
            hdr=hdr_class_of(raw.get("color_transfer"), raw.get("side_data_list")),
            frame_rate=round(ffmpeg.parse_fraction(raw.get("r_frame_rate")), 3),
            avg_frame_rate=round(ffmpeg.parse_fraction(raw.get("avg_frame_rate")), 3),
            variable_frame_rate=is_variable_frame_rate(raw),
            scan_type=field_order,
            interlaced=str(field_order or "").lower() in _INTERLACED_FIELD_ORDERS,
            rotation=rotation_of(raw),
            bitrate_bps=_int(raw.get("bit_rate")),
            frame_count=_int(raw.get("nb_frames")),
        )

    audio = [
        AudioStream(
            index=_int(raw.get("index")) or 0,
            codec=str(raw.get("codec_name", "")),
            profile=raw.get("profile"),
            channels=_int(raw.get("channels")),
            channel_layout=raw.get("channel_layout"),
            sample_rate=_int(raw.get("sample_rate")),
            bitrate_bps=_int(raw.get("bit_rate")),
            language=_language(raw),
            duration_ms=(
                int(round(_float(raw.get("duration")) * 1000))
                if _float(raw.get("duration")) is not None
                else None
            ),
            default=_disposition(raw, "default"),
        )
        for raw in _streams(data, "audio")
    ]

    subtitles = [
        SubtitleStream(
            index=_int(raw.get("index")) or 0,
            codec=str(raw.get("codec_name", "")),
            language=_language(raw),
            forced=_disposition(raw, "forced"),
            default=_disposition(raw, "default"),
            hearing_impaired=_disposition(raw, "hearing_impaired"),
        )
        for raw in _streams(data, "subtitle")
    ]

    return MediaProfile(
        container=container, video=video, audio=audio, subtitles=subtitles
    )


def profile_video(path: Path) -> MediaProfile:
    """Probe a file and parse it. The one place a container is inspected."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    return build_profile(ffmpeg.probe(path))
