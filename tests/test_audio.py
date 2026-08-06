"""A04's deterministic tier — loudness, clipping, channel balance, dead air,
music-bed presence. Every finding here is a measurement, and every measurement
is checked against a signal built to have exactly the property being tested.

This module had zero direct test coverage until three of its five finding
kinds were found to carry a `(0, 0)` span — which is not the same defect as
having no span, and is worse: `report/build.py`'s file-scope detection divides
by the span width, and `build_risk_bands` excludes any finding whose span
overlaps no band, which a zero-width span at t=0 manages for every band
including the first. All three findings were silently invisible on the risk
timeline and never recognised as file-scoped, and nothing caught it because
nothing here was testing the agent directly — only the corpus and bench
integration tests exercised it, and neither asserts on span shape.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from preflight import ffmpeg
from preflight.perception import audio, signal as sig

pytestmark = pytest.mark.skipif(not ffmpeg.available(), reason="ffmpeg is not installed")

RATE = 44_100


def write_wav(path: Path, data: np.ndarray, rate: int = RATE) -> Path:
    """data shaped (channels, samples), float in [-1, 1]."""
    channels, _ = data.shape
    interleaved = (np.clip(data.T, -1, 1) * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(interleaved)
    return path


def tone(freq: float, duration_s: float, amplitude: float = 0.1) -> np.ndarray:
    t = np.linspace(0, duration_s, int(RATE * duration_s), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t)


def stereo(mono: np.ndarray) -> np.ndarray:
    return np.vstack([mono, mono])


class TestFileScopedFindingsCarryARealSpan:
    """The regression this file exists to pin. Every file-scoped finding must
    span (0, duration_ms), matching the pattern `_caption_finding` and
    `_dead_air_findings` already used correctly — not (0, 0), which looks
    superficially like 'the whole file' but is a zero-width span at t=0."""

    def test_loudness_finding_spans_the_full_duration(self, tmp_path):
        # -6 LU into a hot signal, well outside the +-2 tolerance.
        wav = write_wav(tmp_path / "hot.wav", stereo(tone(440, 3.0, amplitude=0.5)))
        result = audio.analyse(wav, wav, duration_ms=3000)
        loud = next(f for f in result.findings if f.clauseId == "AUD-01")
        assert loud.startMs == 0
        assert loud.endMs == 3000

    def test_clipping_finding_spans_the_full_duration(self, tmp_path):
        clipped = np.full(RATE, 0.9999, dtype=np.float64)
        clipped[::2] = -0.9999
        wav = write_wav(tmp_path / "clip.wav", stereo(clipped))
        result = audio.analyse(wav, wav, duration_ms=1000)
        clip = next(f for f in result.findings if f.clauseId == "AUD-02")
        assert clip.startMs == 0
        assert clip.endMs == 1000

    def test_channel_finding_spans_the_full_duration(self, tmp_path):
        loud = tone(440, 2.0, amplitude=0.3)
        silent = np.zeros_like(loud)
        wav = write_wav(tmp_path / "dead.wav", np.vstack([loud, silent]))
        result = audio.analyse(wav, wav, duration_ms=2000)
        channel = next(f for f in result.findings if f.clauseId == "AUD-04")
        assert channel.startMs == 0
        assert channel.endMs == 2000

    def test_a_zero_width_span_is_never_produced(self, tmp_path):
        """The specific shape of the old bug, asserted directly so it cannot
        reappear in a different finding kind without this failing."""
        loud = tone(440, 2.0, amplitude=0.3)
        silent = np.zeros_like(loud)
        wav = write_wav(tmp_path / "mixed.wav", np.vstack([loud, silent]))
        result = audio.analyse(wav, wav, duration_ms=2000)
        for finding in result.findings:
            assert finding.endMs > finding.startMs, finding.clauseId


class TestLoudness:
    def test_a_signal_far_above_target_is_flagged(self, tmp_path):
        wav = write_wav(tmp_path / "hot.wav", stereo(tone(440, 2.0, amplitude=0.5)))
        result = audio.analyse(wav, wav, duration_ms=2000)
        assert any(f.clauseId == "AUD-01" for f in result.findings)

    def test_a_signal_near_target_is_not_flagged(self, tmp_path):
        # Measured directly rather than guessed: a 0.22-amplitude 440Hz tone
        # renders at -13.85 LUFS, comfortably inside the +-2 LU tolerance
        # around the -14 target. (0.1 amplitude, tried first, measured -20.75
        # — nowhere near target, and would have made this test assert nothing
        # useful about the "near target" case it exists to cover.)
        wav = write_wav(tmp_path / "target.wav", stereo(tone(440, 2.0, amplitude=0.22)))
        result = audio.analyse(wav, wav, duration_ms=2000)
        assert not any(f.clauseId == "AUD-01" for f in result.findings)

    def test_direction_is_named_correctly(self, tmp_path):
        wav = write_wav(tmp_path / "hot.wav", stereo(tone(440, 2.0, amplitude=0.5)))
        result = audio.analyse(wav, wav, duration_ms=2000)
        loud = next(f for f in result.findings if f.clauseId == "AUD-01")
        assert "above" in loud.title

    def test_no_suggested_fix_is_an_explicit_choice_not_an_oversight(self, tmp_path):
        """None of the five remediation op kinds express 'adjust the overall
        level' — REPLACE_AUDIO swaps in a different track, which would delete
        the narration this finding measures. Pinned so a future op-kind
        addition is a deliberate decision, not a silent side effect."""
        wav = write_wav(tmp_path / "hot.wav", stereo(tone(440, 2.0, amplitude=0.5)))
        result = audio.analyse(wav, wav, duration_ms=2000)
        loud = next(f for f in result.findings if f.clauseId == "AUD-01")
        assert loud.suggestedFix == "NONE"


class TestClipping:
    def test_sustained_clipping_is_flagged(self, tmp_path):
        clipped = np.full(RATE, 0.9999, dtype=np.float64)
        clipped[::2] = -0.9999
        wav = write_wav(tmp_path / "clip.wav", stereo(clipped))
        result = audio.analyse(wav, wav, duration_ms=1000)
        assert any(f.clauseId == "AUD-02" for f in result.findings)

    def test_clean_audio_has_no_clipping_finding(self, tmp_path):
        wav = write_wav(tmp_path / "clean.wav", stereo(tone(440, 1.0, amplitude=0.1)))
        result = audio.analyse(wav, wav, duration_ms=1000)
        assert not any(f.clauseId == "AUD-02" for f in result.findings)


class TestChannelBalance:
    def test_a_dead_channel_is_flagged(self, tmp_path):
        loud = tone(440, 2.0, amplitude=0.3)
        silent = np.zeros_like(loud)
        wav = write_wav(tmp_path / "dead.wav", np.vstack([loud, silent]))
        result = audio.analyse(wav, wav, duration_ms=2000)
        assert any(f.clauseId == "AUD-04" for f in result.findings)

    def test_balanced_stereo_is_not_flagged(self, tmp_path):
        tone_signal = tone(440, 2.0, amplitude=0.3)
        wav = write_wav(tmp_path / "balanced.wav", stereo(tone_signal))
        result = audio.analyse(wav, wav, duration_ms=2000)
        assert not any(f.clauseId == "AUD-04" for f in result.findings)

    def test_mono_audio_is_never_flagged(self, tmp_path):
        """A single channel has nothing to be unbalanced against."""
        wav = write_wav(
            tmp_path / "mono.wav", tone(440, 2.0, amplitude=0.3)[np.newaxis, :]
        )
        result = audio.analyse(wav, wav, duration_ms=2000)
        assert not any(f.clauseId == "AUD-04" for f in result.findings)


class TestPhaseCorrelation:
    """Out-of-phase stereo collapses toward silence on any playback that sums
    to mono - a phone speaker, a laptop, most of a video platform's audience
    most of the time. Every threshold here was measured on constructed
    signals, not chosen and defended: identical channels give +1.0, fully
    inverted channels give -1.0, and independent stereo content sits within a
    few thousandths of 0.0."""

    def test_identical_channels_are_perfectly_correlated(self, tmp_path):
        signal = tone(440, 2.0, amplitude=0.3)
        wav = write_wav(tmp_path / "mono_safe.wav", stereo(signal))
        result = audio.analyse(wav, wav, duration_ms=2000)
        assert not any(f.clauseId == "AUD-05" for f in result.findings)

    def test_inverted_right_channel_is_flagged(self, tmp_path):
        signal = tone(440, 2.0, amplitude=0.3)
        wav = write_wav(tmp_path / "inverted.wav", np.vstack([signal, -signal]))
        result = audio.analyse(wav, wav, duration_ms=2000)
        phase = next(f for f in result.findings if f.clauseId == "AUD-05")
        assert phase.severity == "HIGH"
        assert "out of phase" in phase.title.lower()

    def test_ordinary_stereo_content_is_not_flagged(self, tmp_path):
        """Shared dialogue plus independent per-channel room noise — not
        identical channels, but genuinely safe stereo. Two FULLY independent
        random signals were tried first and measured 0.003 correlation,
        landing this test in the exact 'weak correlation' band it meant to
        prove was safe: real stereo content almost always shares something
        between channels (dialogue, room tone, a partially-panned mix), and
        total independence is the degenerate case AUD-05 actually exists to
        flag, not a stand-in for an ordinary recording."""
        rng = np.random.default_rng(20260806)
        n = int(RATE * 2)
        shared = rng.normal(0, 0.15, n)
        left = shared + rng.normal(0, 0.08, n)
        right = shared + rng.normal(0, 0.08, n)
        wav = write_wav(tmp_path / "ordinary_stereo.wav", np.vstack([left, right]))
        result = audio.analyse(wav, wav, duration_ms=2000)
        assert not any(f.clauseId == "AUD-05" for f in result.findings)

    def test_mono_audio_has_no_phase_relationship_to_measure(self, tmp_path):
        wav = write_wav(
            tmp_path / "mono.wav", tone(440, 2.0, amplitude=0.3)[np.newaxis, :]
        )
        result = audio.analyse(wav, wav, duration_ms=2000)
        assert not any(f.clauseId == "AUD-05" for f in result.findings)

    def test_a_dead_channel_has_no_phase_relationship_to_measure(self, tmp_path):
        """Silence carries no phase information — AUD-04 is the correct
        finding for a dead channel, not AUD-05."""
        loud = tone(440, 2.0, amplitude=0.3)
        silent = np.zeros_like(loud)
        wav = write_wav(tmp_path / "dead.wav", np.vstack([loud, silent]))
        result = audio.analyse(wav, wav, duration_ms=2000)
        assert not any(f.clauseId == "AUD-05" for f in result.findings)

    def test_finding_spans_the_full_duration(self, tmp_path):
        signal = tone(440, 2.0, amplitude=0.3)
        wav = write_wav(tmp_path / "inverted.wav", np.vstack([signal, -signal]))
        result = audio.analyse(wav, wav, duration_ms=2000)
        phase = next(f for f in result.findings if f.clauseId == "AUD-05")
        assert phase.startMs == 0
        assert phase.endMs == 2000

    def test_phase_correlation_function_matches_measured_calibration(self):
        """The four reference readings this module's thresholds were set
        from, pinned directly against the function rather than only through
        the agent."""
        signal = tone(440, 2.0, amplitude=0.3)

        class FakeAudio:
            channels = 2
            samples = np.vstack([signal, signal])

        identical = audio.phase_correlation(FakeAudio())
        assert identical == pytest.approx(1.0, abs=0.01)

        FakeAudio.samples = np.vstack([signal, -signal])
        inverted = audio.phase_correlation(FakeAudio())
        assert inverted == pytest.approx(-1.0, abs=0.01)


class TestAgentContract:
    def test_missing_audio_file_is_skipped(self, tmp_path):
        missing = tmp_path / "nope.wav"
        result = audio.analyse(missing, missing, duration_ms=1000)
        assert result.status == "SKIPPED"

    def test_clean_audio_produces_no_findings_and_says_so(self, tmp_path):
        wav = write_wav(tmp_path / "clean.wav", stereo(tone(440, 1.0, amplitude=0.1)))
        result = audio.analyse(wav, wav, duration_ms=1000)
        if not result.findings:
            assert any("no audio defects" in line for line in result.log)

    def test_never_emits_a_verdict(self, tmp_path):
        wav = write_wav(tmp_path / "hot.wav", stereo(tone(440, 2.0, amplitude=0.5)))
        result = audio.analyse(wav, wav, duration_ms=2000)
        blob = str([f.to_json() for f in result.findings]).upper()
        for forbidden in ("VIOLATION", "UNSAFE", "DEMONETIZ", "LIMITED ADS"):
            assert forbidden not in blob
