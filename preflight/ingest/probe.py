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


def _stream(streams: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    for stream in streams:
        if stream.get("codec_type") == kind:
            return stream
    return None


def probe_video(path: Path) -> VideoMeta:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")

    data = ffmpeg.probe(path)
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
