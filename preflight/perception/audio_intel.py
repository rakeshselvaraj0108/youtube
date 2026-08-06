"""A04 — AUDIO INTELLIGENCE.

Non-speech acoustic analysis. Real signal processing on real samples, in numpy
alone: no TensorFlow, no PyTorch, no downloaded checkpoint. Every measurement
here is reproducible to the sample from the file on disk.

The honest boundary, stated once and enforced throughout: DSP resolves that an
impulsive broadband transient with a six-millisecond attack occurred at 5:32.
It does not resolve that the transient was a gunshot. A gunshot, a slammed
door, a dropped microphone and a balloon pop are acoustically similar at this
level of analysis, and most implementations label it anyway.

A04 emits `IMPULSIVE_TRANSIENT` with the attack time, crest factor and
bandwidth that justify it, and marks the label unresolved. A confident wrong
label is worse than an honest gap, because the fusion layer will weigh whatever
it is handed — and a `gunshot` at 0.9 that was a door would corrupt the score
in a way an unresolved transient never can.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from preflight.ingest import audio as audio_io
from preflight.models import AgentResult
from preflight.perception import signal as sig

AGENT_ID = "audio_intel"
AGENT_NAME = "Audio Intelligence"
TAXONOMY_DIR = Path("data/audio")

# Analysis frame. 46ms at 44.1kHz — long enough for a stable spectrum, short
# enough that a 20ms attack is not smeared across the whole frame.
FRAME = 2048
HOP = 512

# Segment thresholds, taken from measured values on signals of known type
# rather than chosen and then defended. Reference readings (median flatness,
# envelope modulation):
#
#   silence      1.000   0.0Hz     tone         0.000  18.6Hz
#   music        0.000   2.0Hz     speech       0.562   7.9Hz
#   white noise  0.562   0.8Hz     applause     0.563   1.3Hz
#
# Flatness alone cannot separate speech from noise — both read 0.562. Envelope
# modulation separates them cleanly, which is why it is checked first.
SILENCE_RMS = 0.003
SPEECH_MODULATION_HZ = (2.0, 8.0)  # syllable rate
MUSIC_FLATNESS = 0.05
SPEECH_MIN_FLATNESS = 0.20  # speech has formants; music does not reach this
NOISE_FLATNESS = 0.45

# Tempo needs an onset envelope that actually varies. Measured coefficient of
# variation: sustained tone 0.74, white noise 0.10, music 3.36. Below this,
# autocorrelation finds structure in what is essentially a flat line and
# returns a confident BPM for a held note.
TEMPO_MIN_FLUX_CV = 1.0

# Applause: broadband and impulsive, but NOT syllable-modulated. Measured CV
# — applause 0.43, white noise 0.10, speech 0.36 — so flux variance alone does
# not separate applause from speech; excluding the speech modulation band does.
APPLAUSE_MIN_FLATNESS = 0.40
APPLAUSE_FLUX_CV = (0.25, 1.5)
APPLAUSE_MIN_DENSITY = 0.30

# Hum needs frequency resolution the analysis frame does not have. At 2048
# samples and 22kHz the bins are 10.8Hz wide, putting 50Hz and 60Hz one bin
# apart — indistinguishable, and the detector reported 50Hz for a 60Hz signal.
# 16384 samples gives 1.35Hz bins and eight bins of separation.
HUM_FRAME = 16384

# Impulsive transient. A crest factor above 8 with a sub-20ms attack is the
# signature shared by every percussive broadband event.
TRANSIENT_CREST = 8.0
TRANSIENT_ATTACK_MS = 20.0
TRANSIENT_MIN_GAP_MS = 120

# Mains hum sits at 50Hz or 60Hz with harmonics. A peak this far above the
# local spectral neighbourhood is not programme material.
HUM_PROMINENCE_DB = 12.0

MIN_SEGMENT_MS = 700

CONFIDENCE_BANDS = [(0.98, "VERY_HIGH"), (0.90, "HIGH"), (0.80, "MEDIUM")]


def band(confidence: float) -> str:
    for floor, name in CONFIDENCE_BANDS:
        if confidence >= floor:
            return name
    return "LOW"


@dataclass
class AudioEvent:
    """One acoustic observation. Never a verdict."""

    type: str
    start_ms: int
    end_ms: int
    confidence: float
    resolved: bool = True
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.type,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "confidence": round(self.confidence, 3),
            "band": band(self.confidence),
        }
        if not self.resolved:
            # The label is the acoustic class, not the source. Saying so in the
            # payload stops a downstream reader treating it as identification.
            payload["label_resolved"] = False
            payload["requires"] = "audio.classify"
        if self.evidence:
            payload["evidence"] = self.evidence
        return payload


@dataclass
class Segment:
    kind: str  # silence | speech | music | ambient | noise
    start_ms: int
    end_ms: int
    confidence: float

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "confidence": round(self.confidence, 3),
        }


# ------------------------------------------------------------------ #
# Spectral front end                                                  #
# ------------------------------------------------------------------ #


@dataclass
class Spectra:
    magnitude: np.ndarray  # frames x bins
    freqs: np.ndarray
    times_ms: np.ndarray
    rms: np.ndarray
    flatness: np.ndarray
    centroid: np.ndarray
    flux: np.ndarray
    sample_rate: int


def analyse_spectra(mono: np.ndarray, sample_rate: int) -> Spectra:
    """One STFT pass. Everything else is derived from it."""
    frames = sig.frame_signal(mono, FRAME, HOP)
    if frames.size == 0:
        empty = np.array([])
        return Spectra(
            np.zeros((0, 0)), empty, empty, empty, empty, empty, empty, sample_rate
        )

    window = np.hanning(FRAME)
    magnitude = np.abs(np.fft.rfft(frames * window, axis=1))
    freqs = np.fft.rfftfreq(FRAME, 1 / sample_rate)
    times_ms = np.arange(frames.shape[0]) * HOP / sample_rate * 1000

    power = np.maximum(magnitude**2, 1e-12)
    geometric = np.exp(np.log(power).mean(axis=1))
    arithmetic = power.mean(axis=1)
    flatness = geometric / np.maximum(arithmetic, 1e-12)

    total = np.maximum(magnitude.sum(axis=1), 1e-12)
    centroid = (magnitude * freqs).sum(axis=1) / total

    # Positive spectral difference: energy appearing, not energy leaving.
    # Onsets are additions, and the negative half is decay.
    difference = np.diff(magnitude, axis=0, prepend=magnitude[:1])
    flux = np.maximum(difference, 0).sum(axis=1)

    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))

    return Spectra(magnitude, freqs, times_ms, rms, flatness, centroid, flux, sample_rate)


# ------------------------------------------------------------------ #
# Segmentation                                                        #
# ------------------------------------------------------------------ #


def _envelope_modulation_hz(rms: np.ndarray, hop_rate: float) -> float:
    """Dominant modulation rate of the loudness envelope.

    Speech modulates at the syllable rate, roughly 2-8Hz. Music and steady
    ambience do not. This is the single most discriminative cheap feature for
    telling speech from a music bed.
    """
    if rms.size < 8:
        return 0.0
    centred = rms - rms.mean()
    if not np.any(centred):
        return 0.0
    spectrum = np.abs(np.fft.rfft(centred * np.hanning(centred.size)))
    freqs = np.fft.rfftfreq(centred.size, 1 / hop_rate)
    usable = (freqs > 0.5) & (freqs < 20)
    if not usable.any():
        return 0.0
    return float(freqs[usable][int(np.argmax(spectrum[usable]))])


def segment(spectra: Spectra, window_ms: int = 1000) -> list[Segment]:
    """Speech / music / ambient / silence / noise timeline.

    Classified per window from three measured features, then merged. The
    remediation compiler consumes this directly: REPLACE_AUDIO needs to know
    where a bed actually starts and stops, not roughly.
    """
    if spectra.rms.size == 0:
        return []

    hop_rate = spectra.sample_rate / HOP
    per_window = max(2, int(window_ms / 1000 * hop_rate))
    raw: list[Segment] = []

    for start in range(0, spectra.rms.size, per_window):
        stop = min(start + per_window, spectra.rms.size)
        if stop - start < 2:
            break

        rms = spectra.rms[start:stop]
        flatness = float(np.median(spectra.flatness[start:stop]))
        modulation = _envelope_modulation_hz(rms, hop_rate)
        level = float(np.median(rms))

        start_ms = int(spectra.times_ms[start])
        end_ms = int(spectra.times_ms[stop - 1])

        speech_rate = SPEECH_MODULATION_HZ[0] <= modulation <= SPEECH_MODULATION_HZ[1]

        # Order matters. Music is checked before speech because a steady beat
        # lands inside the syllable band — measured at 2.0Hz for 120bpm, which
        # is exactly the bottom of the speech range. Flatness separates them
        # decisively: tonal material reads 0.00, speech reads 0.56.
        if level < SILENCE_RMS:
            kind, confidence = "silence", 0.97
        elif flatness < MUSIC_FLATNESS:
            kind, confidence = "music", 0.86
        elif speech_rate and flatness > SPEECH_MIN_FLATNESS:
            kind, confidence = "speech", 0.88
        elif flatness > NOISE_FLATNESS:
            kind, confidence = "noise", 0.85
        else:
            kind, confidence = "ambient", 0.80

        raw.append(Segment(kind, start_ms, end_ms, confidence))

    return _merge_segments(raw)


def _merge_segments(segments: list[Segment]) -> list[Segment]:
    """Adjacent windows of one kind are one segment.

    Also absorbs a lone dissenting window between two agreeing neighbours: a
    single second classified `ambient` inside thirty seconds of music is a
    quiet bar, not a change of material.
    """
    if not segments:
        return []

    smoothed = list(segments)
    for i in range(1, len(smoothed) - 1):
        if smoothed[i - 1].kind == smoothed[i + 1].kind != smoothed[i].kind:
            smoothed[i] = Segment(
                smoothed[i - 1].kind, smoothed[i].start_ms, smoothed[i].end_ms, 0.75
            )

    merged: list[Segment] = [smoothed[0]]
    for current in smoothed[1:]:
        last = merged[-1]
        if current.kind == last.kind:
            merged[-1] = Segment(
                last.kind, last.start_ms, current.end_ms,
                min(last.confidence, current.confidence),
            )
        else:
            merged.append(current)

    return [s for s in merged if s.end_ms - s.start_ms >= MIN_SEGMENT_MS] or merged[:1]


# ------------------------------------------------------------------ #
# Transients — the honest boundary                                    #
# ------------------------------------------------------------------ #


def detect_transients(
    mono: np.ndarray, spectra: Spectra, sample_rate: int
) -> list[AudioEvent]:
    """Impulsive broadband events, characterised but NOT identified.

    Attack time, crest factor and bandwidth are measured from the waveform.
    What produced them is left to a classifier tier, because nothing in these
    three numbers separates a gunshot from a door.
    """
    if spectra.flux.size < 4:
        return []

    flux = spectra.flux
    threshold = float(np.median(flux) + 3.0 * np.std(flux))
    if threshold <= 0:
        return []

    events: list[AudioEvent] = []
    last_ms = -TRANSIENT_MIN_GAP_MS

    for index in np.flatnonzero(flux > threshold):
        onset_ms = int(spectra.times_ms[index])
        if onset_ms - last_ms < TRANSIENT_MIN_GAP_MS:
            continue

        start = int(onset_ms / 1000 * sample_rate)
        stop = min(len(mono), start + int(0.25 * sample_rate))
        window = mono[max(0, start - int(0.02 * sample_rate)) : stop]
        if window.size < 64:
            continue

        peak = float(np.abs(window).max())
        rms = float(np.sqrt((window.astype(np.float64) ** 2).mean()))
        if rms <= 1e-9:
            continue
        crest = peak / rms

        attack_ms = _attack_time_ms(window, sample_rate, peak)
        if crest < TRANSIENT_CREST or attack_ms > TRANSIENT_ATTACK_MS:
            continue

        spread = float(spectra.flatness[index])
        decay_ms = _decay_time_ms(window, sample_rate, peak)

        # Confidence in the MEASUREMENT, not in any identification. A sharper
        # attack and a higher crest factor make it more certainly impulsive;
        # they say nothing about the source.
        confidence = min(
            0.94,
            0.70
            + min(0.12, (crest - TRANSIENT_CREST) / 40)
            + min(0.12, (TRANSIENT_ATTACK_MS - attack_ms) / 100),
        )

        events.append(
            AudioEvent(
                type="IMPULSIVE_TRANSIENT",
                start_ms=onset_ms,
                end_ms=onset_ms + int(decay_ms),
                confidence=confidence,
                resolved=False,
                evidence={
                    "attack_ms": round(attack_ms, 1),
                    "decay_ms": round(decay_ms, 1),
                    "crest_factor": round(crest, 1),
                    "spectral_flatness": round(spread, 4),
                    "broadband": bool(spread > 0.2),
                    "note": (
                        "acoustic class only — a gunshot, a door slam and a "
                        "dropped microphone are not separable at this level"
                    ),
                },
            )
        )
        last_ms = onset_ms

    return events


def _attack_time_ms(window: np.ndarray, sample_rate: int, peak: float) -> float:
    """10% to 90% rise time of the envelope."""
    envelope = np.abs(window)
    low = np.flatnonzero(envelope >= 0.1 * peak)
    high = np.flatnonzero(envelope >= 0.9 * peak)
    if low.size == 0 or high.size == 0:
        return TRANSIENT_ATTACK_MS * 2
    return max(0.0, (high[0] - low[0]) / sample_rate * 1000)


def _decay_time_ms(window: np.ndarray, sample_rate: int, peak: float) -> float:
    """Time from peak until the envelope falls below a tenth of it."""
    envelope = np.abs(window)
    apex = int(np.argmax(envelope))
    below = np.flatnonzero(envelope[apex:] < 0.1 * peak)
    if below.size == 0:
        return (len(window) - apex) / sample_rate * 1000
    return below[0] / sample_rate * 1000


# ------------------------------------------------------------------ #
# Applause, hum, tempo, noise floor                                   #
# ------------------------------------------------------------------ #


def detect_applause(spectra: Spectra) -> list[AudioEvent]:
    """Dense aperiodic broadband impulses.

    Applause is genuinely separable from music (flatness 0.56 vs 0.00) and
    from steady noise (flux variance 0.43 vs 0.10). It is NOT separable from
    speech on either of those — both read 0.56 flatness and similar variance —
    so the syllable band is excluded explicitly.

    Laughter is left to the classifier tier for the same reason: it overlaps
    speech in every feature available here.
    """
    if spectra.flux.size < 20:
        return []

    mean_flux = float(np.mean(spectra.flux))
    if mean_flux <= 1e-12:
        return []
    flux_cv = float(np.std(spectra.flux) / mean_flux)
    if not APPLAUSE_FLUX_CV[0] <= flux_cv <= APPLAUSE_FLUX_CV[1]:
        return []

    hop_rate = spectra.sample_rate / HOP
    if SPEECH_MODULATION_HZ[0] <= _envelope_modulation_hz(spectra.rms, hop_rate) <= (
        SPEECH_MODULATION_HZ[1]
    ):
        return []

    # A percentile threshold rather than mean-plus-sigma: dense impulse trains
    # raise their own mean, so an adaptive threshold rises with the very thing
    # it is meant to detect and finds nothing.
    window = max(4, int(hop_rate))
    threshold = float(np.percentile(spectra.flux, 60))
    onsets = (spectra.flux > threshold).astype(np.float64)
    density = np.convolve(onsets, np.ones(window) / window, mode="same")

    dense = (density > APPLAUSE_MIN_DENSITY) & (spectra.flatness > APPLAUSE_MIN_FLATNESS)
    spans = sig.spans_where(dense, 1000 / hop_rate, 1200)

    return [
        AudioEvent(
            type="APPLAUSE",
            start_ms=start,
            end_ms=end,
            confidence=0.83,
            evidence={
                "onset_density": round(float(density.max()), 3),
                "flux_cv": round(flux_cv, 3),
            },
        )
        for start, end in spans
    ]


def detect_hum(spectra: Spectra, mono: np.ndarray | None = None) -> list[AudioEvent]:
    """Mains hum at 50Hz or 60Hz, plus harmonics.

    Uses its own long FFT rather than the analysis spectra. At the 2048-sample
    analysis frame the bins are 10.8Hz wide, which puts 50Hz and 60Hz one bin
    apart — the detector reported 50Hz for a pure 60Hz signal, because the 60Hz
    peak sat inside 50Hz's own exclusion window. 16384 samples gives 1.35Hz
    bins and eight bins of separation.

    A narrowband peak well above its own spectral neighbourhood is not
    programme material: no instrument sits at exactly 50Hz for a whole file.
    """
    if mono is None or mono.size < HUM_FRAME:
        return []

    # Average several long frames: hum is stationary, so averaging lifts it out
    # of whatever else is happening.
    frames = sig.frame_signal(mono, HUM_FRAME, HUM_FRAME // 2)
    if frames.size == 0:
        return []
    window = np.hanning(HUM_FRAME)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)).mean(axis=0)
    freqs = np.fft.rfftfreq(HUM_FRAME, 1 / spectra.sample_rate)

    def prominence_db(target: float) -> float | None:
        if target >= freqs[-1]:
            return None
        index = int(np.argmin(np.abs(freqs - target)))
        # Exclude only the peak itself and its immediate skirt; at 1.35Hz bins
        # a +/-3 bin guard is about 4Hz, far narrower than the 10Hz gap.
        low, high = max(0, index - 30), min(len(spectrum), index + 31)
        neighbourhood = np.concatenate(
            [spectrum[low : max(low, index - 3)], spectrum[index + 4 : high]]
        )
        if neighbourhood.size == 0:
            return None
        floor = float(np.median(neighbourhood))
        if floor <= 1e-12:
            return None
        return float(20 * np.log10(spectrum[index] / floor))

    candidates: list[tuple[float, float, int]] = []
    for mains in (50.0, 60.0):
        fundamental = prominence_db(mains)
        if fundamental is None or fundamental < HUM_PROMINENCE_DB:
            continue
        harmonics = sum(
            1
            for h in (2, 3)
            if (p := prominence_db(mains * h)) is not None and p >= 6
        )
        candidates.append((fundamental, mains, harmonics))

    if not candidates:
        return []

    # 50 and 60 are mutually exclusive. Take the stronger rather than the first
    # tested, which is what produced the wrong answer before.
    fundamental, mains, harmonics = max(candidates)
    return [
        AudioEvent(
            type="MAINS_HUM",
            start_ms=0,
            end_ms=int(spectra.times_ms[-1]) if spectra.times_ms.size else 0,
            confidence=0.90,
            evidence={
                "frequency_hz": mains,
                "prominence_db": round(fundamental, 1),
                "harmonics_detected": harmonics,
            },
        )
    ]


def estimate_tempo(spectra: Spectra) -> float | None:
    """BPM from autocorrelation of the onset envelope."""
    if spectra.flux.size < 32:
        return None

    hop_rate = spectra.sample_rate / HOP
    envelope = spectra.flux - spectra.flux.mean()
    if not np.any(envelope):
        return None

    # A held note has an onset envelope that barely moves. Autocorrelating it
    # finds structure in what is essentially a flat line and returns a
    # confident BPM for a sustained tone — measured at 86 BPM for a pure sine.
    mean_flux = float(np.mean(spectra.flux))
    if mean_flux <= 1e-12:
        return None
    if float(np.std(spectra.flux) / mean_flux) < TEMPO_MIN_FLUX_CV:
        return None

    correlation = np.correlate(envelope, envelope, mode="full")[len(envelope) - 1 :]
    if correlation[0] <= 0:
        return None
    correlation = correlation / correlation[0]

    # 60-200 BPM covers essentially all programme music.
    low = max(1, int(hop_rate * 60 / 200))
    high = min(len(correlation) - 1, int(hop_rate * 60 / 60))
    if high <= low:
        return None

    lag = low + int(np.argmax(correlation[low:high]))
    if correlation[lag] < 0.15:
        return None
    return round(60.0 * hop_rate / lag, 1)


def noise_floor_db(spectra: Spectra) -> float | None:
    """Tenth percentile of the RMS envelope — the level between the content."""
    if spectra.rms.size == 0:
        return None
    floor = float(np.percentile(spectra.rms, 10))
    return round(20 * np.log10(max(floor, 1e-9)), 1)


# ------------------------------------------------------------------ #
# Taxonomy                                                            #
# ------------------------------------------------------------------ #


class AudioTaxonomy:
    """What each tier can resolve. Loaded once, never indexed."""

    def __init__(self, directory: Path = TAXONOMY_DIR) -> None:
        self.directory = Path(directory)
        self.dsp: set[str] = set()
        self.classifier: set[str] = set()
        self.loaded: list[str] = []
        self._load()

    def _load(self) -> None:
        if not self.directory.is_dir():
            return
        for path in sorted(self.directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.loaded.append(path.name)
            for entry in payload.get("labels", []):
                target = self.dsp if entry["tier"] == "dsp" else self.classifier
                target.add(entry["label"])

    @property
    def unresolvable(self) -> list[str]:
        return sorted(self.classifier)


# ------------------------------------------------------------------ #
# Agent entry point                                                   #
# ------------------------------------------------------------------ #


def analyse(
    fingerprint_wav: Path,
    source: Path | None = None,
    *,
    taxonomy: AudioTaxonomy | None = None,
) -> tuple[AgentResult, list[AudioEvent], list[Segment]]:
    started = time.perf_counter()
    taxonomy = taxonomy or AudioTaxonomy()

    if not Path(fingerprint_wav).is_file():
        return (
            AgentResult.skipped(AGENT_ID, AGENT_NAME, "no audio track to analyse"),
            [], [],
        )

    try:
        audio = sig.read_wav(Path(fingerprint_wav))
    except (ValueError, OSError, EOFError) as exc:
        # The specification's failure contract, verbatim.
        result = AgentResult.failed(AGENT_ID, AGENT_NAME, "Corrupted Audio Track")
        result.artifacts = {"detail": str(exc)}
        return result, [], []

    mono = audio.mono
    spectra = analyse_spectra(mono, audio.sample_rate)

    events: list[AudioEvent] = []
    events.extend(detect_transients(mono, spectra, audio.sample_rate))
    events.extend(detect_applause(spectra))
    events.extend(detect_hum(spectra, mono))
    segments = segment(spectra)

    for seg in segments:
        if seg.kind in {"music", "silence"}:
            events.append(
                AudioEvent(
                    type="MUSIC_SEGMENT" if seg.kind == "music" else "SILENCE",
                    start_ms=seg.start_ms,
                    end_ms=seg.end_ms,
                    confidence=seg.confidence,
                )
            )

    events.sort(key=lambda e: (e.start_ms, e.type))

    quality: dict[str, Any] = {
        "noise_floor_db": noise_floor_db(spectra),
        "channels": audio.channels,
        "sample_rate": audio.sample_rate,
    }
    tempo = estimate_tempo(spectra)
    if tempo:
        quality["tempo_bpm"] = tempo
    if source is not None:
        measured = audio_io.loudness(Path(source))
        if measured:
            quality["lufs"] = round(measured["integrated_lufs"], 1)
            quality["true_peak_dbtp"] = round(measured["true_peak_dbtp"], 1)
            quality["loudness_range"] = round(measured["loudness_range"], 1)

    kinds = ", ".join(
        f"{k} {sum(1 for s in segments if s.kind == k)}"
        for k in sorted({s.kind for s in segments})
    ) or "none"
    unresolved = sum(1 for e in events if not e.resolved)

    log = [
        f"{len(taxonomy.dsp)} labels resolvable by DSP, "
        f"{len(taxonomy.classifier)} need audio.classify",
        f"segmented into {len(segments)}: {kinds}",
        f"{len(events)} acoustic event(s)"
        + (f", {unresolved} characterised but unidentified" if unresolved else ""),
    ]

    return (
        AgentResult(
            agent_id=AGENT_ID,
            name=AGENT_NAME,
            status="OK",
            coverage=1.0,
            artifacts={
                "events": [e.to_json() for e in events],
                "segments": [s.to_json() for s in segments],
                "quality": quality,
                "unresolvable_without_classifier": taxonomy.unresolvable,
            },
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            log=log,
        ),
        events,
        segments,
    )


def to_json(
    events: Iterable[AudioEvent],
    segments: Iterable[Segment] = (),
    quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The A04 output contract. JSON only — never a paragraph."""
    return {
        "events": [event.to_json() for event in events],
        "segments": [seg.to_json() for seg in segments],
        "quality": quality or {},
    }
