"""Ingest orchestration — probe, extract, cache.

Everything here is keyed by the hash of the video bytes, so a second run on an
unchanged file does no ffmpeg work at all.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from preflight import cas, ffmpeg
from preflight.ingest import audio as audio_mod
from preflight.ingest import frames as frames_mod
from preflight.ingest.probe import VideoMeta, probe_video
from preflight.ingest.profile import MediaProfile, build_profile


@dataclass
class Ingested:
    video_hash: str
    meta: VideoMeta
    asr_wav: Path
    fingerprint_wav: Path
    poster: Path
    keyframes: list[frames_mod.Keyframe]
    cached: bool
    elapsed_ms: int
    log: list[str] = field(default_factory=list)
    # The full technical profile — codec, colour, HDR class, every audio and
    # subtitle track. Parsed once here and cached with the rest of the
    # entry, so an agent that needs the frame rate or a second audio track
    # reads it from the run rather than launching ffprobe again. None only
    # for entries cached before this existed.
    profile: MediaProfile | None = None

    @property
    def coverage(self) -> float:
        return 1.0


def _cached_profile(entry) -> MediaProfile | None:
    """Rebuild the profile from the cached raw ffprobe payload, or None.

    Reparsed rather than restored, so a cache hit gets today's derivations
    of yesterday's probe.
    """
    try:
        return build_profile(entry.read_json("probe.json"))
    except Exception:  # noqa: BLE001 - an old or damaged entry is just a miss
        return None


def ingest(
    source: Path,
    store: cas.Store,
    *,
    max_frames: int = frames_mod.MAX_FRAMES,
    scene_threshold: float = frames_mod.SCENE_THRESHOLD,
) -> Ingested:
    source = Path(source)
    started = time.perf_counter()
    log: list[str] = []

    video_hash = cas.hash_file(source)
    entry = store.entry("v", video_hash)

    asr_wav = entry.path(audio_mod.ASR_WAV)
    fp_wav = entry.path(audio_mod.FINGERPRINT_WAV)
    poster = entry.path("poster.jpg")
    frame_dir = entry.path("frames")

    if entry.exists:
        meta = VideoMeta(**entry.read_json("meta.json"))
        index = entry.read_json("frames.json")
        keyframes = [
            frames_mod.Keyframe(
                index=item["index"],
                ts_ms=item["ts_ms"],
                path=frame_dir / item["file"],
            )
            for item in index
        ]
        log.append(f"cache hit {cas.prefixed(video_hash)[:18]} — 0 ffmpeg invocations")
        return Ingested(
            video_hash=video_hash,
            meta=meta,
            asr_wav=asr_wav,
            fingerprint_wav=fp_wav,
            poster=poster,
            keyframes=keyframes,
            cached=True,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            log=log,
            # Entries written before the profile existed have no profile.json,
            # and re-probing to backfill would break the property that a cache
            # hit costs zero ffmpeg invocations. None is the honest answer.
            profile=_cached_profile(entry),
        )

    # Miss. Build the entry, then commit — a crash mid-extraction leaves no
    # `.ready` marker, so the partial directory is treated as a miss next time.
    entry.discard()
    entry.root.mkdir(parents=True, exist_ok=True)

    # One ffprobe call feeds both. `probe_video` produces the compact
    # VideoMeta the report renders; `build_profile` reads the same payload
    # into the full technical profile. Probing twice for two views of the
    # same bytes is the duplication this profile exists to remove.
    raw = ffmpeg.probe(source)
    meta = probe_video(source, data=raw)
    profile = build_profile(raw)
    log.append(
        f"probed {meta.width}x{meta.height} @ {meta.fps:g}fps · "
        f"{meta.durationMs / 1000:.1f}s · {meta.audioCodec}"
    )
    if profile.video is not None:
        detail = [
            f"{profile.video.codec}",
            f"{profile.video.bit_depth or '?'}-bit",
            profile.video.chroma_subsampling or "?",
            profile.video.hdr,
        ]
        if profile.video.rotation:
            detail.append(f"rotated {profile.video.rotation}°")
        if profile.video.variable_frame_rate:
            detail.append("VFR")
        log.append(
            f"profile: {' · '.join(detail)} · "
            f"{len(profile.audio)} audio, {len(profile.subtitles)} subtitle track(s)"
        )

    if meta.sampleRate:
        audio_mod.extract_for_asr(source, asr_wav)
        audio_mod.extract_for_fingerprint(source, fp_wav)
        log.append("extracted 16kHz mono (ASR) and 44.1kHz stereo (fingerprint)")
    else:
        log.append("no audio stream — speech and copyright layers will report SKIPPED")

    frames_mod.extract_poster(source, poster, meta.durationMs)
    keyframes = frames_mod.extract_keyframes(
        source,
        frame_dir,
        threshold=scene_threshold,
        max_frames=max_frames,
        duration_ms=meta.durationMs,
    )
    if frames_mod.sampled_uniformly(keyframes):
        # Worth saying out loud. A single-take talking head has no cuts to
        # detect, and a reader should know the frames came from a clock rather
        # than from the picture changing.
        log.append(
            f"no scene cuts above {scene_threshold} — "
            f"sampled {len(keyframes)} keyframes uniformly"
        )
    else:
        log.append(
            f"scene detection: {len(keyframes)} keyframes at threshold {scene_threshold}"
        )

    entry.write_json("meta.json", meta.to_json())
    # The RAW ffprobe payload, not the parsed profile. Caching the parse
    # would freeze it: every later improvement to how a field is derived
    # would apply only to files probed after the change, and a cached entry
    # would keep reporting the old answer forever. The bytes ffprobe emitted
    # are the fact; the profile is an interpretation of them.
    entry.write_json("probe.json", raw)
    entry.write_json(
        "frames.json",
        [{"index": f.index, "ts_ms": f.ts_ms, "file": f.path.name} for f in keyframes],
    )
    entry.commit()

    return Ingested(
        video_hash=video_hash,
        meta=meta,
        asr_wav=asr_wav,
        fingerprint_wav=fp_wav,
        poster=poster,
        keyframes=keyframes,
        cached=False,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        log=log,
        profile=profile,
    )
