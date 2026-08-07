"""Audio extraction — two variants, for two different jobs."""

from __future__ import annotations

from pathlib import Path

from preflight import ffmpeg

ASR_WAV = "audio.wav"
FINGERPRINT_WAV = "audio_fp.wav"


def extract_for_asr(source: Path, destination: Path) -> Path:
    """Mono 16 kHz PCM, loudness-normalised — what every ASR model wants.

    Normalisation matters here: a quiet passage transcribed against a loud one
    loses words, and lost words are lost evidence spans.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg.run(
        [
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
            str(destination),
        ]
    )
    return destination


def extract_for_fingerprint(source: Path, destination: Path) -> Path:
    """Untouched stereo 44.1 kHz.

    Deliberately NOT loudness-normalised. loudnorm applies dynamic gain, which
    perturbs the spectral peaks Chromaprint hashes — a normalised track
    fingerprints to something the reference database has never seen, and the
    copyright layer silently reports no match on audio that would in fact be
    claimed. This is the single easiest way to make the copyright dimension
    quietly wrong, so the two extractions stay separate files.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg.run(
        [
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            str(destination),
        ]
    )
    return destination


# EBU R128 is a full decode of the audio, and two agents want the same
# numbers from the same file in the same run: A04's loudness finding and the
# acoustic tier's quality block both called this, and a measured run showed
# the pass executing twice back to back — the single most expensive
# duplicate in the pipeline at 1.8s of a 12s run.
_LOUDNESS_CACHE: dict[tuple, dict[str, float] | None] = {}
_LOUDNESS_LIMIT = 4


def _loudness_key(source: Path) -> tuple:
    try:
        stat = Path(source).stat()
        return (str(source), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (str(source), 0, 0)


def clear_loudness_cache() -> None:
    _LOUDNESS_CACHE.clear()


def loudness(source: Path) -> dict[str, float] | None:
    """EBU R128 measurement from a single analysis pass.

    Returns integrated loudness (LUFS), true peak (dBTP) and loudness range.
    None when the file has no audio stream.

    Memoised per (path, mtime, size). Measuring the same unchanged file
    twice cannot produce two answers, so the second pass is pure cost.
    """
    import json
    import re

    cache_key = _loudness_key(source)
    if cache_key in _LOUDNESS_CACHE:
        return _LOUDNESS_CACHE[cache_key]

    def remember(value: dict[str, float] | None) -> dict[str, float] | None:
        # A failure is cached too. A file with no audio stream will not grow
        # one mid-run, and retrying the same doomed pass for every agent
        # that asks is the same waste as repeating a successful one.
        if len(_LOUDNESS_CACHE) >= _LOUDNESS_LIMIT:
            _LOUDNESS_CACHE.pop(next(iter(_LOUDNESS_CACHE)))
        _LOUDNESS_CACHE[cache_key] = value
        return value

    try:
        result = ffmpeg.run(
            [
                "-i",
                str(source),
                "-af",
                "loudnorm=I=-14:TP=-1.0:LRA=11:print_format=json",
                "-f",
                "null",
                "-",
            ],
            capture_stderr=True,
        )
    except ffmpeg.FfmpegFailed:
        return remember(None)

    # loudnorm prints its JSON block to stderr, after the usual banner noise.
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", result.stderr, re.DOTALL)
    if not match:
        return remember(None)
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return remember(None)

    def _num(key: str) -> float:
        try:
            return float(payload.get(key, "nan"))
        except (TypeError, ValueError):
            return float("nan")

    return remember({
        "integrated_lufs": _num("input_i"),
        "true_peak_dbtp": _num("input_tp"),
        "loudness_range": _num("input_lra"),
        "threshold": _num("input_thresh"),
    })
