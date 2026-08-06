"""Stream probing — populates `report.video`."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from preflight import ffmpeg


@dataclass
class VideoMeta:
    """Mirrors the TypeScript `VideoMeta` in src/types/analysis.ts."""

    filename: str
    durationMs: int
    width: int
    height: int
    fps: float
    sizeBytes: int
    audioCodec: str
    sampleRate: int
    posterUrl: str
    srcUrl: str
    posterDataUri: str | None = None

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        if data["posterDataUri"] is None:
            del data["posterDataUri"]
        return data


class UnsupportedInput(ValueError):
    """The file has no decodable video stream."""


# Smallest plausible media file. Anything under this is a truncated download or
# a stray text file, and saying so beats forwarding ffprobe's exit code.
MIN_BYTES = 1024

# Container signatures, used ONLY to explain a probe failure — never to gate a
# file that ffprobe would have accepted. A magic-byte allowlist as a
# precondition rejects valid containers it has not heard of, which is a worse
# failure than a vague error message. ffprobe decides; this table explains.
_MAGIC: list[tuple[bytes, str]] = [
    (b"\x1aE\xdf\xa3", "Matroska/WebM"),
    (b"RIFF", "AVI"),
    (b"OggS", "Ogg"),
    (b"FLV\x01", "FLV"),
    (b"\x00\x00\x01\xba", "MPEG-PS"),
    (b"0&\xb2u", "ASF/WMV"),
]


def container_hint(head: bytes) -> str | None:
    """Name the container a file's first bytes suggest, or None."""
    if len(head) >= 8 and head[4:8] == b"ftyp":
        return "ISO-BMFF (MP4/MOV)"
    for signature, name in _MAGIC:
        if head.startswith(signature):
            return name
    if head[:1] == b"\x47":
        return "MPEG-TS"
    return None


def _diagnose(path: Path) -> str:
    """Explain why ffprobe refused a file, in the user's terms."""
    size = path.stat().st_size
    if size < MIN_BYTES:
        unit = "byte" if size == 1 else "bytes"
        return (
            f"{path.name} is {size} {unit} — truncated or empty, not a video. "
            "Check the download completed."
        )
    with path.open("rb") as handle:
        head = handle.read(16)
    hint = container_hint(head)
    if hint is None:
        printable = sum(32 <= b < 127 or b in (9, 10, 13) for b in head)
        looks_textual = printable == len(head) and len(head) > 0
        return (
            f"{path.name} is not a recognised media container"
            + (" — it looks like a text file with a video extension." if looks_textual
               else " — no known container signature in its first bytes.")
        )
    return (
        f"{path.name} looks like {hint} but ffprobe could not decode it — "
        "the file is most likely corrupt or truncated."
    )


def _stream(streams: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    for stream in streams:
        if stream.get("codec_type") == kind:
            return stream
    return None


def probe_video(path: Path) -> VideoMeta:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")

    try:
        data = ffmpeg.probe(path)
    except ffmpeg.FfmpegFailed as exc:
        # ffprobe's own stderr names an ffprobe.EXE path and an exit code,
        # which tells a creator nothing about their file.
        raise UnsupportedInput(_diagnose(path)) from exc
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    video = _stream(streams, "video")
    if video is None:
        raise UnsupportedInput(f"{path.name} contains no video stream")
    audio = _stream(streams, "audio")

    # Prefer the container duration; fall back to the video stream's own.
    duration_s = float(fmt.get("duration") or video.get("duration") or 0.0)

    # avg_frame_rate reflects the whole file including VFR; r_frame_rate is the
    # base rate. Prefer the average and fall back, never assume 30.
    fps = ffmpeg.parse_fraction(video.get("avg_frame_rate")) or ffmpeg.parse_fraction(
        video.get("r_frame_rate")
    )

    size = int(fmt.get("size") or path.stat().st_size)

    return VideoMeta(
        filename=path.name,
        durationMs=int(round(duration_s * 1000)),
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=round(fps, 3),
        sizeBytes=size,
        audioCodec=(audio or {}).get("codec_name", "none").upper(),
        sampleRate=int((audio or {}).get("sample_rate") or 0),
        # Relative paths, resolved next to report.html. The CLI writes the
        # renders alongside the report so a judge can open one file.
        posterUrl=f"./{path.stem}.poster.jpg",
        srcUrl=f"./{path.name}",
    )


def has_audio(path: Path) -> bool:
    return _stream(ffmpeg.probe(Path(path)).get("streams", []), "audio") is not None
