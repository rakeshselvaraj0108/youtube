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


def loudness(source: Path) -> dict[str, float] | None:
    """EBU R128 measurement from a single analysis pass.

    Returns integrated loudness (LUFS), true peak (dBTP) and loudness range.
    None when the file has no audio stream.
    """
    import json
    import re

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
        return None

    # loudnorm prints its JSON block to stderr, after the usual banner noise.
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", result.stderr, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    def _num(key: str) -> float:
        try:
            return float(payload.get(key, "nan"))
        except (TypeError, ValueError):
            return float("nan")

    return {
        "integrated_lufs": _num("input_i"),
        "true_peak_dbtp": _num("input_tp"),
        "loudness_range": _num("input_lra"),
        "threshold": _num("input_thresh"),
    }
