"""A4 — Audio quality.

Entirely deterministic. Every finding here is a measurement, not a judgement,
which is why they carry confidence near 1.0 and an empty defence: you cannot
argue with an LUFS reading.

The one worth pointing at is AUDIO-CHANNEL — dead-mic detection. A recording
where one channel is 20dB below the other is a real, common and expensive
mistake that nobody notices until the video is live and half the audience is
hearing silence.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from preflight.ingest import audio as audio_io
from preflight.models import Adversarial, AgentResult, Evidence, Finding, PolicyRef
from preflight.perception import signal as sig

AGENT_ID = "audio"
AGENT_NAME = "Audio Agent"

TARGET_LUFS = -14.0
LUFS_TOLERANCE = 2.0
CLIP_THRESHOLD = 0.999
DEAD_AIR_RMS = 0.002
DEAD_AIR_MIN_MS = 3_000
CHANNEL_DELTA_DB = 12.0
MUSIC_FLATNESS = 0.12
MUSIC_MIN_MS = 8_000

WINDOW_MS = 100


def _clause(clause_id: str, title: str, section: str, text: str) -> PolicyRef:
    return PolicyRef(clauseId=clause_id, title=title, section=section, text=text)


def _measured(charge: str, rationale: str, confidence: float) -> Adversarial:
    return Adversarial(charge=charge, rationale=rationale, confidence=confidence)


def analyse(
    fingerprint_wav: Path,
    source: Path,
    duration_ms: int,
) -> AgentResult:
    started = time.perf_counter()
    log: list[str] = []
    findings: list[Finding] = []

    if not Path(fingerprint_wav).is_file():
        return AgentResult.skipped(AGENT_ID, AGENT_NAME, "no audio stream to analyse")

    try:
        audio = sig.read_wav(Path(fingerprint_wav))
    except (ValueError, OSError, EOFError) as exc:
        return AgentResult.failed(AGENT_ID, AGENT_NAME, f"could not decode audio: {exc}")

    mono = audio.mono
    findings.extend(_loudness_findings(source, duration_ms, log))
    findings.extend(_clipping_findings(audio, duration_ms, log))
    findings.extend(_dead_air_findings(mono, audio.sample_rate, duration_ms, log))
    findings.extend(_channel_findings(audio, duration_ms, log))
    findings.extend(_music_bed_findings(mono, audio.sample_rate, log))

    if not findings:
        log.append("no audio defects measured")

    return AgentResult(
        agent_id=AGENT_ID,
        name=AGENT_NAME,
        status="OK",
        findings=findings,
        artifacts={"channels": audio.channels, "sample_rate": audio.sample_rate},
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        log=log,
    )


def _loudness_findings(source: Path, duration_ms: int, log: list[str]) -> list[Finding]:
    measured = audio_io.loudness(source)
    if not measured:
        log.append("loudness measurement unavailable")
        return []

    lufs = measured["integrated_lufs"]
    peak = measured["true_peak_dbtp"]
    if not np.isfinite(lufs):
        return []

    log.append(f"EBU R128: {lufs:.1f} LUFS · true peak {peak:.1f} dBTP")
    delta = lufs - TARGET_LUFS
    if abs(delta) <= LUFS_TOLERANCE:
        return []

    direction = "above" if delta > 0 else "below"
    return [
        Finding(
            id="a_loud",
            clauseId="AUD-01",
            category="Audio Delivery",
            title=f"Integrated loudness {abs(delta):.1f} LU {direction} target",
            description=(
                f"Measured {lufs:.1f} LUFS against a {TARGET_LUFS:.0f} LUFS target; "
                + (
                    "playback will attenuate and flatten dynamics."
                    if delta > 0
                    else "the upload will sound quiet next to normalised content."
                )
            ),
            startMs=0,
            # File-scoped, like the caption finding — a proper span rather
            # than the (0, 0) point this measured before, which made this
            # finding invisible to `_is_file_scoped` (divides by a zero-width
            # span) and to every band in `build_risk_bands` (a (0, 0) span
            # overlaps nothing, including band 0, whose exclusion test is
            # `endMs <= start`).
            endMs=max(duration_ms, 1),
            severity="LOW" if abs(delta) < 6 else "MEDIUM",
            confidence=0.95,
            modalities={"audio": 0.95},
            evidence=Evidence(transcript="[file-scoped measurement · EBU R128]"),
            # No suggestedFix. The mechanical repair is a full-file loudnorm
            # pass, and none of the five remediation op kinds (MUTE, BLEEP,
            # BLUR_REGION, REPLACE_AUDIO, CUT) express "adjust the overall
            # level" — REPLACE_AUDIO swaps in a different track entirely,
            # which would delete the narration this finding is measuring.
            # A sixth op kind is a real, separately-scoped feature spanning
            # this file, edl.py, codegen.py, the TS types, the JSON schema
            # and the UI's op maps — not something to bolt on as a side
            # effect of fixing this finding's span.
            policy=_clause(
                "AUD-01",
                "Loudness normalisation",
                "PREFLIGHT audio ruleset § 3.1",
                "Playback normalises loud uploads downward. Delivering above target does "
                "not increase perceived volume; it only reduces headroom and dynamic "
                "range after normalisation.",
            ),
            adversarial=_measured(
                f"EBU R128 integrated loudness {lufs:.1f} LUFS, true peak {peak:.1f} dBTP.",
                "Measured value, not a classification. Correctable with a single "
                "loudnorm pass.",
                0.95,
            ),
        )
    ]


def _clipping_findings(audio: sig.Audio, duration_ms: int, log: list[str]) -> list[Finding]:
    clipped = int(np.count_nonzero(np.abs(audio.samples) >= CLIP_THRESHOLD))
    total = int(audio.samples.size)
    if total == 0:
        return []
    ratio = clipped / total
    if clipped == 0:
        return []

    log.append(f"clipping: {clipped:,} samples ({ratio * 100:.3f}%)")
    if ratio < 1e-5:
        return []

    return [
        Finding(
            id="a_clip",
            clauseId="AUD-02",
            category="Audio Delivery",
            title=f"{clipped:,} clipped samples",
            description=(
                f"{ratio * 100:.3f}% of samples sit at full scale — audible distortion "
                "that cannot be repaired after the fact."
            ),
            startMs=0,
            endMs=max(duration_ms, 1),
            severity="MEDIUM" if ratio > 1e-4 else "LOW",
            confidence=0.97,
            modalities={"audio": 0.97},
            evidence=Evidence(transcript="[file-scoped measurement · sample peak]"),
            policy=_clause(
                "AUD-02",
                "Clipping",
                "PREFLIGHT audio ruleset § 3.2",
                "Samples driven to full scale distort irreversibly. Reduce gain before "
                "export rather than limiting after.",
            ),
            adversarial=_measured(
                f"{clipped:,} of {total:,} samples at or beyond {CLIP_THRESHOLD} full scale.",
                "Sample-level measurement. No interpretation involved.",
                0.97,
            ),
        )
    ]


def _dead_air_findings(
    mono: np.ndarray, sample_rate: int, duration_ms: int, log: list[str]
) -> list[Finding]:
    envelope = sig.rms_envelope(mono, sample_rate, WINDOW_MS)
    if envelope.size == 0:
        return []

    spans = sig.spans_where(envelope < DEAD_AIR_RMS, WINDOW_MS, DEAD_AIR_MIN_MS)
    # Trailing silence is normal — an outro fade is not a defect.
    spans = [s for s in spans if duration_ms == 0 or s[1] < duration_ms - 2_000]
    if not spans:
        return []

    log.append(f"dead air: {len(spans)} span(s) over {DEAD_AIR_MIN_MS / 1000:.0f}s")
    findings: list[Finding] = []
    for index, (start, end) in enumerate(spans[:5]):
        seconds = (end - start) / 1000
        findings.append(
            Finding(
                id=f"a_dead_{index}",
                clauseId="AUD-03",
                category="Audio Delivery",
                title=f"{seconds:.1f}s of dead air",
                description="Sustained near-silence mid-programme. Viewers read this as a "
                "playback fault and drop off.",
                startMs=start,
                endMs=end,
                severity="LOW" if seconds < 6 else "MEDIUM",
                confidence=0.9,
                modalities={"audio": 0.9},
                evidence=Evidence(transcript=f"[RMS below {DEAD_AIR_RMS} for {seconds:.1f}s]"),
                policy=_clause(
                    "AUD-03",
                    "Dead air",
                    "PREFLIGHT audio ruleset § 3.3",
                    "Extended silence inside a programme is indistinguishable from a "
                    "playback failure and drives immediate drop-off.",
                ),
                adversarial=_measured(
                    f"RMS below {DEAD_AIR_RMS} continuously for {seconds:.1f}s.",
                    "Envelope measurement over a 100ms window.",
                    0.9,
                ),
                suggestedFix="CUT",
            )
        )
    return findings


def _channel_findings(audio: sig.Audio, duration_ms: int, log: list[str]) -> list[Finding]:
    """Dead-mic detection.

    One channel far quieter than the other means a microphone that was not
    recording. It is invisible on headphones if you only wear one, survives the
    entire edit, and is discovered by the audience.
    """
    if audio.channels < 2:
        return []

    levels = np.sqrt((audio.samples.astype(np.float64) ** 2).mean(axis=1))
    levels = np.maximum(levels, 1e-9)
    db = 20 * np.log10(levels)
    delta = float(db.max() - db.min())
    log.append(f"channel balance: {delta:.1f} dB spread")

    if delta < CHANNEL_DELTA_DB:
        return []

    quiet = int(np.argmin(db))
    return [
        Finding(
            id="a_channel",
            clauseId="AUD-04",
            category="Audio Delivery",
            title=f"Channel {quiet} is {delta:.0f} dB below the other",
            description="Consistent with a microphone that was not recording. Half the "
            "audience hears silence.",
            startMs=0,
            endMs=max(duration_ms, 1),
            severity="HIGH",
            confidence=0.92,
            modalities={"audio": 0.92},
            evidence=Evidence(
                transcript=f"[per-channel RMS: {', '.join(f'{v:.1f} dB' for v in db)}]"
            ),
            policy=_clause(
                "AUD-04",
                "Channel balance",
                "PREFLIGHT audio ruleset § 3.4",
                "A large sustained level difference between channels usually indicates a "
                "dead microphone rather than an intentional mix.",
            ),
            adversarial=_measured(
                f"Per-channel RMS differs by {delta:.1f} dB across the whole programme.",
                "Measured across the full duration, so this is not a transient pan.",
                0.92,
            ),
        )
    ]


def _music_bed_findings(mono: np.ndarray, sample_rate: int, log: list[str]) -> list[Finding]:
    """Music presence without a fingerprint database.

    This never claims a match and never claims safety. It says a bed is present
    and licensing needs checking — which is the honest limit of what spectral
    flatness alone can tell you.
    """
    flatness = sig.spectral_flatness(mono, sample_rate, WINDOW_MS)
    if flatness.size == 0:
        return []

    spans = sig.spans_where(flatness < MUSIC_FLATNESS, WINDOW_MS, MUSIC_MIN_MS)
    if not spans:
        return []

    total = sum(end - start for start, end in spans)
    log.append(f"tonal/music energy in {total / 1000:.0f}s across {len(spans)} span(s)")

    findings: list[Finding] = []
    for index, (start, end) in enumerate(spans[:5]):
        seconds = (end - start) / 1000
        findings.append(
            Finding(
                id=f"a_music_{index}",
                clauseId="COPY-01",
                category="Copyright",
                title=f"Sustained music bed, {seconds:.0f}s",
                description="Tonal energy consistent with a music bed. Verify licensing — "
                "this check cannot identify the recording.",
                startMs=start,
                endMs=end,
                severity="MEDIUM",
                confidence=0.55,
                modalities={"audio": 0.55},
                evidence=Evidence(
                    transcript=f"[spectral flatness below {MUSIC_FLATNESS} for {seconds:.0f}s]"
                ),
                policy=_clause(
                    "COPY-01",
                    "Third-party content and Content ID",
                    "Copyright policy — Content ID",
                    "Uploading a commercially released recording without a licence permits "
                    "the rights holder to claim the video. This detector reports presence "
                    "only; absence of a public fingerprint match does not prove safety.",
                ),
                adversarial=Adversarial(
                    charge=f"Spectral flatness below {MUSIC_FLATNESS} for {seconds:.0f}s "
                    "continuously, which is characteristic of music rather than speech.",
                    rationale="Reported as MUSIC_BED_PRESENT, never as a match and never as "
                    "safe. Identifying the recording requires a fingerprint lookup.",
                    confidence=0.55,
                    defense="Tonal energy can also come from room tone, a drone, or a "
                    "sustained sound effect.",
                    defense_strength=0.4,
                ),
                suggestedFix="REPLACE_AUDIO",
            )
        )
    return findings
