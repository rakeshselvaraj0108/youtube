"""Raw audio and luminance sampling, on numpy alone.

Deliberately no librosa. Everything here is RMS, FFT and differencing, all of
which numpy does directly — and every dependency that is not required is one
more way a clean clone fails on a stranger's machine.
"""

from __future__ import annotations

import subprocess
import wave
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from preflight import ffmpeg


@dataclass
class Audio:
    """Decoded PCM, shaped (channels, samples), float32 in [-1, 1]."""

    samples: np.ndarray
    sample_rate: int

    @property
    def channels(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration_ms(self) -> int:
        return int(self.samples.shape[1] / self.sample_rate * 1000)

    @property
    def mono(self) -> np.ndarray:
        return self.samples.mean(axis=0)


def read_wav(path: Path) -> Audio:
    """Read a PCM wav with the stdlib, normalised to float32."""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        raise ValueError(f"unsupported sample width: {width} bytes")

    data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if width == 1:  # 8-bit wav is unsigned, centred on 128
        data = (data - 128.0) / 128.0
    else:
        data /= float(np.iinfo(dtype).max)

    if channels > 1:
        usable = (data.size // channels) * channels
        data = data[:usable].reshape(-1, channels).T
    else:
        data = data.reshape(1, -1)

    return Audio(samples=np.ascontiguousarray(data), sample_rate=rate)


def frame_signal(signal: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """Split into overlapping frames without copying more than necessary."""
    if signal.size < frame:
        return np.empty((0, frame), dtype=signal.dtype)
    count = 1 + (signal.size - frame) // hop
    strides = (signal.strides[0] * hop, signal.strides[0])
    return np.lib.stride_tricks.as_strided(
        signal, shape=(count, frame), strides=strides, writeable=False
    )


def rms_envelope(signal: np.ndarray, sample_rate: int, window_ms: int = 100) -> np.ndarray:
    frame = max(1, int(sample_rate * window_ms / 1000))
    frames = frame_signal(signal, frame, frame)
    if frames.size == 0:
        return np.array([float(np.sqrt(np.mean(signal**2)))]) if signal.size else np.array([])
    return np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))


def spectral_flatness(signal: np.ndarray, sample_rate: int, window_ms: int = 100) -> np.ndarray:
    """Geometric mean over arithmetic mean of the power spectrum.

    Near 1 for noise, near 0 for tonal content. Sustained low flatness under
    speech is the signature of a music bed — which is a licensing question even
    when no fingerprint matches.
    """
    frame = max(256, int(sample_rate * window_ms / 1000))
    frames = frame_signal(signal, frame, frame)
    if frames.size == 0:
        return np.array([])

    windowed = frames * np.hanning(frame)
    power = np.abs(np.fft.rfft(windowed, axis=1)) ** 2
    power = np.maximum(power, 1e-12)

    geometric = np.exp(np.log(power).mean(axis=1))
    arithmetic = power.mean(axis=1)
    return geometric / np.maximum(arithmetic, 1e-12)


# One decode of one (file, rate, size) is reusable by every consumer of it.
# Small on purpose: three entries covers the accessibility agent's two rates
# plus one spare, and each entry is a few megabytes of downscaled greyscale.
# An unbounded cache here would hold every frame of every file the process
# ever touched, which is the memory leak this bound exists to prevent.
_FRAME_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_FRAME_CACHE_LIMIT = 3

# Observable so a test can prove the reuse rather than assume it.
_CACHE_STATS = {"hits": 0, "misses": 0}


def _cache_key(source: Path, fps: int, width: int, height: int) -> tuple:
    """Identity of a decode. Includes mtime and size so an edited file in
    the same path is a miss rather than a stale hit."""
    try:
        stat = Path(source).stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        stamp = (0, 0)
    return (str(source), fps, width, height, *stamp)


def clear_frame_cache() -> None:
    _FRAME_CACHE.clear()
    _CACHE_STATS.update(hits=0, misses=0)


def frame_cache_stats() -> dict[str, int]:
    return dict(_CACHE_STATS)


def _grayscale_frames(
    source: Path, fps: int, width: int, height: int
) -> np.ndarray:
    """Downscaled greyscale frames at a fixed rate, shape (n, width*height).

    Scene-cut keyframes are far too sparse for this kind of analysis — a
    strobe or a freeze lives entirely between two cuts. This pipes downscaled
    frames straight out of ffmpeg, so an 18-minute file costs about 24MB of
    transfer and no intermediate files.

    Memoised, because it was not shared in practice even though the comment
    claimed it was: the accessibility agent asks for frozen-frame differences
    and black-frame luminance at the same rate and size, and measurement of a
    real run showed `fps=10,scale=64:36` decoded twice, back to back, for two
    views of identical pixels. The returned array is read-only so a consumer
    cannot mutate the copy the next one will receive.
    """
    key = _cache_key(source, fps, width, height)
    cached = _FRAME_CACHE.get(key)
    if cached is not None:
        _FRAME_CACHE.move_to_end(key)
        _CACHE_STATS["hits"] += 1
        return cached
    _CACHE_STATS["misses"] += 1

    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        f"fps={fps},scale={width}:{height},format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        raise ffmpeg.FfmpegFailed(command, result.returncode, result.stderr.decode(errors="replace"))

    frame_bytes = width * height
    usable = (len(result.stdout) // frame_bytes) * frame_bytes
    if usable == 0:
        return np.zeros((0, frame_bytes), dtype=np.float32)

    frames = np.frombuffer(result.stdout[:usable], dtype=np.uint8).reshape(-1, frame_bytes)
    frames = frames.astype(np.float32)
    # Shared by reference from here on, so it must not be writable — one
    # consumer normalising in place would silently corrupt the next one's
    # view of the same decode.
    frames.flags.writeable = False

    _FRAME_CACHE[key] = frames
    while len(_FRAME_CACHE) > _FRAME_CACHE_LIMIT:
        _FRAME_CACHE.popitem(last=False)   # evict least recently used
    return frames


def luminance_series(source: Path, fps: int = 10, width: int = 64, height: int = 36) -> np.ndarray:
    """Mean luminance per frame, sampled at a fixed rate."""
    frames = _grayscale_frames(source, fps, width, height)
    if frames.shape[0] == 0:
        return np.array([])
    return frames.mean(axis=1)


def frame_diff_series(source: Path, fps: int = 10, width: int = 64, height: int = 36) -> np.ndarray:
    """Mean absolute difference between each frame and the one before it.

    Matching MEAN luminance is not the same claim as matching PICTURE — two
    different frames of a static scene can share an average brightness by
    coincidence while genuinely changing content, and a single scalar per
    frame cannot tell the difference. This compares the full downscaled
    frame to its predecessor, so a real freeze (near-zero difference,
    sustained) is distinguishable from a static-average scene that is
    actually still changing (low mean-luminance variance, but real
    frame-to-frame difference from motion, grain, or a slow pan).

    Length is one shorter than the frame count — there is no predecessor for
    the first frame — and the caller's timestamps should account for that.
    """
    frames = _grayscale_frames(source, fps, width, height)
    if frames.shape[0] < 2:
        return np.array([])
    return np.abs(np.diff(frames, axis=0)).mean(axis=1)


def spans_where(mask: np.ndarray, step_ms: float, min_ms: int) -> list[tuple[int, int]]:
    """Contiguous True runs in `mask`, as (start_ms, end_ms), filtered by length."""
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    spans: list[tuple[int, int]] = []
    for start, stop in zip(edges[0::2], edges[1::2]):
        start_ms = int(start * step_ms)
        end_ms = int(stop * step_ms)
        if end_ms - start_ms >= min_ms:
            spans.append((start_ms, end_ms))
    return spans
