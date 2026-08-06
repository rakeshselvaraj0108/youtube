"""A04 — audio intelligence.

Every test builds a signal with known properties and asserts the analyser
measures what was constructed. A click train IS impulsive; a sine IS tonal;
50Hz hum IS at 50Hz. Same discipline as the synthetic video corpus: ground
truth by construction rather than by annotation.

The tests that matter most are the negative ones. Any onset detector fires on
a click; the question is whether it also fires on speech, on a sustained note,
and on silence — and whether the module resists labelling a transient as a
gunshot when nothing in the waveform justifies it.
"""

from __future__ import annotations

import numpy as np
import pytest

from preflight.perception.audio_intel import (
    TRANSIENT_CREST,
    AudioEvent,
    AudioTaxonomy,
    analyse_ducking,
    analyse_spectra,
    band,
    detect_applause,
    detect_hum,
    detect_transients,
    estimate_tempo,
    noise_floor_db,
    segment,
    to_json,
)

RATE = 22_050
TAXONOMY = AudioTaxonomy()
RNG = np.random.default_rng(20260805)


def seconds(n: float) -> int:
    return int(RATE * n)


def sine(freq: float, duration: float, amplitude: float = 0.3) -> np.ndarray:
    t = np.linspace(0, duration, seconds(duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t)


def noise(duration: float, amplitude: float = 0.1) -> np.ndarray:
    return amplitude * RNG.normal(0, 1, seconds(duration)).astype(np.float64)


def silence(duration: float) -> np.ndarray:
    return np.zeros(seconds(duration))


def click(at: float, total: float, amplitude: float = 0.95) -> np.ndarray:
    """An impulse with a near-instant attack and a short exponential decay."""
    out = np.zeros(seconds(total))
    start = seconds(at)
    length = seconds(0.05)
    decay = np.exp(-np.linspace(0, 12, length))
    burst = amplitude * decay * RNG.normal(0, 1, length)
    out[start : start + length] += burst[: max(0, len(out) - start)]
    return out


def speech_like(duration: float, amplitude: float = 0.3, rate: float = 3.5) -> np.ndarray:
    """Word-shaped noise bursts with genuine gaps between them.

    An earlier version of this fixture was continuous noise under a `0.5 +
    0.5·|sin|` tremolo. It modulated at the syllable rate but never fell
    silent, giving it an envelope depth of 0.14 — closer to a wobbling
    ventilation fan than to a person talking. Speech stops between words, and
    a fixture that does not stop cannot exercise the feature the segmenter now
    keys on.
    """
    out = np.zeros(seconds(duration))
    step = 1.0 / rate
    position = 0.05
    while position + 0.18 < duration:
        start = seconds(position)
        length = seconds(0.18)
        out[start : start + length] += (
            amplitude * RNG.normal(0, 1, length) * np.hanning(length)
        )
        position += step
    return out


def music_like(duration: float, bpm: float = 120) -> np.ndarray:
    """Sustained tones plus a beat at a known tempo."""
    tone = sine(220, duration, 0.25) + sine(330, duration, 0.18)
    beat = np.zeros(seconds(duration))
    interval = 60.0 / bpm
    for index in range(int(duration / interval)):
        start = seconds(index * interval)
        length = seconds(0.03)
        if start + length < len(beat):
            beat[start : start + length] += 0.4 * np.exp(-np.linspace(0, 8, length))
    return tone + beat


class TestSpectralFrontEnd:
    def test_a_tone_is_far_from_flat(self):
        spectra = analyse_spectra(sine(440, 2.0), RATE)
        assert float(np.median(spectra.flatness)) < 0.05

    def test_white_noise_is_flat(self):
        spectra = analyse_spectra(noise(2.0), RATE)
        assert float(np.median(spectra.flatness)) > 0.3

    def test_centroid_tracks_pitch(self):
        low = analyse_spectra(sine(200, 1.5), RATE)
        high = analyse_spectra(sine(4000, 1.5), RATE)
        assert float(np.median(high.centroid)) > float(np.median(low.centroid)) * 5

    def test_empty_input_does_not_raise(self):
        spectra = analyse_spectra(np.array([]), RATE)
        assert spectra.rms.size == 0
        assert segment(spectra) == []
        assert detect_transients(np.array([]), spectra, RATE) == []


class TestTransients:
    """The honest boundary: characterise, do not identify."""

    def test_detects_a_constructed_click(self):
        signal = noise(3.0, 0.02) + click(1.5, 3.0)
        events = detect_transients(signal, analyse_spectra(signal, RATE), RATE)
        assert events
        assert abs(events[0].start_ms - 1500) < 200

    def test_never_claims_to_know_the_source(self):
        """A gunshot, a slammed door and a dropped microphone are acoustically
        similar here. Labelling one would be a confident guess."""
        signal = noise(2.0, 0.02) + click(1.0, 2.0)
        event = detect_transients(signal, analyse_spectra(signal, RATE), RATE)[0]
        assert event.type == "IMPULSIVE_TRANSIENT"
        assert event.resolved is False
        assert event.to_json()["requires"] == "audio.classify"
        for guess in ("GUNSHOT", "EXPLOSION", "GLASS", "SCREAM"):
            assert guess not in event.type.upper()

    def test_reports_the_measurements_that_justify_it(self):
        signal = noise(2.0, 0.02) + click(1.0, 2.0)
        evidence = detect_transients(signal, analyse_spectra(signal, RATE), RATE)[0].evidence
        assert evidence["crest_factor"] >= TRANSIENT_CREST
        assert evidence["attack_ms"] <= 20.0
        assert "decay_ms" in evidence

    def test_does_not_fire_on_a_sustained_tone(self):
        """A note has an onset but no impulsive character."""
        signal = sine(440, 3.0)
        assert detect_transients(signal, analyse_spectra(signal, RATE), RATE) == []

    def test_does_not_fire_on_steady_noise(self):
        signal = noise(3.0, 0.2)
        assert detect_transients(signal, analyse_spectra(signal, RATE), RATE) == []

    def test_does_not_fire_on_silence(self):
        signal = silence(3.0)
        assert detect_transients(signal, analyse_spectra(signal, RATE), RATE) == []

    def test_does_not_fire_on_speech_like_modulation(self):
        """Speech has onsets at every syllable and must not read as gunfire."""
        signal = speech_like(4.0)
        assert detect_transients(signal, analyse_spectra(signal, RATE), RATE) == []

    def test_close_clicks_are_not_double_counted(self):
        signal = noise(3.0, 0.02) + click(1.0, 3.0) + click(1.02, 3.0)
        events = detect_transients(signal, analyse_spectra(signal, RATE), RATE)
        assert len(events) == 1

    def test_separate_clicks_are_separate_events(self):
        signal = noise(4.0, 0.02) + click(1.0, 4.0) + click(2.5, 4.0)
        events = detect_transients(signal, analyse_spectra(signal, RATE), RATE)
        assert len(events) == 2


class TestHum:
    def test_detects_fifty_hertz_hum(self):
        signal = speech_like(3.0) + sine(50, 3.0, 0.05)
        events = detect_hum(analyse_spectra(signal, RATE), signal)
        assert events
        assert events[0].evidence["frequency_hz"] == 50.0

    def test_detects_sixty_hertz_hum(self):
        signal = speech_like(3.0) + sine(60, 3.0, 0.05)
        events = detect_hum(analyse_spectra(signal, RATE), signal)
        assert events
        assert events[0].evidence["frequency_hz"] == 60.0

    def test_clean_audio_has_no_hum(self):
        assert detect_hum(analyse_spectra(speech_like(3.0), RATE)) == []

    def test_musical_bass_is_not_hum(self):
        """No instrument sits at exactly 50Hz for a whole file, which is what
        separates hum from a bass line."""
        assert detect_hum(analyse_spectra(music_like(3.0), RATE)) == []


class TestSegmentation:
    def test_silence_is_segmented_as_silence(self):
        segments = segment(analyse_spectra(silence(4.0), RATE))
        assert segments
        assert all(s.kind == "silence" for s in segments)

    def test_speech_like_modulation_reads_as_speech(self):
        segments = segment(analyse_spectra(speech_like(6.0), RATE))
        assert any(s.kind == "speech" for s in segments)

    def test_tonal_material_reads_as_music(self):
        segments = segment(analyse_spectra(music_like(6.0), RATE))
        assert any(s.kind == "music" for s in segments)

    def test_broadband_noise_reads_as_noise(self):
        segments = segment(analyse_spectra(noise(5.0, 0.3), RATE))
        assert any(s.kind == "noise" for s in segments)

    def test_a_transition_produces_two_segments(self):
        signal = np.concatenate([silence(3.0), music_like(4.0)])
        kinds = [s.kind for s in segment(analyse_spectra(signal, RATE))]
        assert "silence" in kinds
        assert len(set(kinds)) >= 2

    def test_segments_tile_forward_without_overlap(self):
        signal = np.concatenate([speech_like(3.0), silence(2.0), music_like(3.0)])
        segments = segment(analyse_spectra(signal, RATE))
        for earlier, later in zip(segments, segments[1:]):
            assert later.start_ms >= earlier.start_ms

    def test_a_lone_dissenting_window_is_absorbed(self):
        """One second reading `ambient` inside thirty of music is a quiet bar,
        not a change of material."""
        signal = music_like(10.0)
        segments = segment(analyse_spectra(signal, RATE))
        assert len(segments) <= 4


class TestTempo:
    @pytest.mark.parametrize("bpm", [90, 120, 150])
    def test_recovers_a_constructed_tempo(self, bpm):
        estimate = estimate_tempo(analyse_spectra(music_like(8.0, bpm), RATE))
        assert estimate is not None
        # Half and double time are the classic ambiguity and both are correct
        # readings of the same beat grid.
        assert any(abs(estimate - bpm * k) < bpm * 0.12 for k in (0.5, 1, 2))

    def test_silence_has_no_tempo(self):
        assert estimate_tempo(analyse_spectra(silence(4.0), RATE)) is None

    def test_a_sustained_tone_has_no_tempo(self):
        assert estimate_tempo(analyse_spectra(sine(440, 6.0), RATE)) is None


class TestApplause:
    def test_dense_aperiodic_impulses_read_as_applause(self):
        signal = noise(4.0, 0.05)
        for start in np.arange(0.5, 3.5, 0.045):
            signal = signal + click(float(start), 4.0, amplitude=0.35)
        events = detect_applause(analyse_spectra(signal, RATE))
        assert events

    def test_music_is_not_applause(self):
        assert detect_applause(analyse_spectra(music_like(6.0), RATE)) == []

    def test_silence_is_not_applause(self):
        assert detect_applause(analyse_spectra(silence(4.0), RATE)) == []


class TestDucking:
    """Was the bed placed by an editor, or was it in the room?

    Every fixture here is a bed at a known level with speech over a known
    attenuation of it, so the assertion is that the estimator recovers the
    number that was constructed.
    """

    @staticmethod
    def analyse(signal: np.ndarray):
        spectra = analyse_spectra(signal, RATE)
        return analyse_ducking(spectra, segment(spectra))

    def test_recovers_a_constructed_duck(self):
        bed = music_like(4.0)
        ducked = music_like(4.0) * (10 ** (-9 / 20))
        result = self.analyse(np.concatenate([bed, ducked + speech_like(4.0)]))
        assert result is not None
        assert abs(result.duck_db - 9.0) < 2.0
        assert result.deliberate_bed is True

    def test_a_bed_that_holds_its_level_is_not_deliberate(self):
        """Music at a constant level through dialogue is more likely in the
        room than on a timeline — a café, a car radio, a busker in shot."""
        bed = music_like(4.0)
        result = self.analyse(np.concatenate([bed, music_like(4.0) + speech_like(4.0)]))
        assert result is not None
        assert abs(result.duck_db) < 3.0
        assert result.deliberate_bed is False

    def test_speech_with_no_music_yields_nothing(self):
        assert self.analyse(speech_like(6.0)) is None

    def test_music_with_no_speech_yields_nothing(self):
        assert self.analyse(music_like(6.0)) is None

    def test_silence_yields_nothing(self):
        assert self.analyse(silence(5.0)) is None

    def test_reports_the_frame_counts_behind_the_number(self):
        bed = music_like(4.0)
        ducked = music_like(4.0) * (10 ** (-9 / 20))
        result = self.analyse(np.concatenate([bed, ducked + speech_like(4.0)]))
        payload = result.to_json()
        assert payload["speech_frames"] >= 10
        assert payload["music_frames"] >= 10

    def test_a_deliberate_bed_only_ever_raises_copyright_priority(self):
        """There is no reading of this measurement that makes music safer, and
        the payload must not be usable to argue one."""
        bed = music_like(4.0)
        ducked = music_like(4.0) * (10 ** (-9 / 20))
        result = self.analyse(np.concatenate([bed, ducked + speech_like(4.0)]))
        assert result.to_json()["raises_copyright_priority"] is True


class TestSpeechOverAMusicBed:
    """The commonest configuration in real video, and the one that broke.

    Spectral flatness was the speech gate. Narration over a bed measures 0.046
    — below the 0.05 music gate — so a presenter talking over their own intro
    was classified as music, and every downstream consumer of speech spans was
    blind to it.
    """

    def test_narration_over_a_bed_is_speech_not_music(self):
        signal = speech_like(6.0) + music_like(6.0) * (10 ** (-9 / 20))
        kinds = {s.kind for s in segment(analyse_spectra(signal, RATE))}
        assert "speech" in kinds

    def test_the_bed_alone_is_still_music(self):
        kinds = {s.kind for s in segment(analyse_spectra(music_like(6.0), RATE))}
        assert kinds == {"music"}

    def test_a_loud_bed_does_not_swallow_the_speech(self):
        signal = speech_like(6.0) + music_like(6.0)
        kinds = {s.kind for s in segment(analyse_spectra(signal, RATE))}
        assert "speech" in kinds


class TestQualityMetrics:
    def test_noise_floor_is_lower_for_quieter_audio(self):
        quiet = noise_floor_db(analyse_spectra(noise(3.0, 0.005), RATE))
        loud = noise_floor_db(analyse_spectra(noise(3.0, 0.3), RATE))
        assert quiet < loud

    def test_confidence_bands(self):
        assert band(0.99) == "VERY_HIGH"
        assert band(0.93) == "HIGH"
        assert band(0.84) == "MEDIUM"
        assert band(0.55) == "LOW"


class TestTaxonomy:
    def test_loads_every_file(self):
        assert len(TAXONOMY.loaded) >= 8
        assert len(TAXONOMY.dsp) >= 20

    def test_states_what_needs_a_classifier(self):
        """The gap is declared, not hidden. A reader can see that birdsong is
        unavailable rather than assuming it was checked and absent."""
        assert "gunshot" in TAXONOMY.classifier
        assert "rain" in TAXONOMY.classifier
        assert "speaker_count" in TAXONOMY.classifier

    def test_no_label_is_claimed_by_both_tiers(self):
        assert not (TAXONOMY.dsp & TAXONOMY.classifier)

    def test_speaker_count_is_not_guessed(self):
        """Estimating it from energy statistics produces a number that looks
        authoritative and is guesswork."""
        assert "speaker_count" not in TAXONOMY.dsp


class TestOutputContract:
    def test_output_is_json_only(self):
        payload = to_json([AudioEvent("SILENCE", 0, 1000, 0.9)], [], {"lufs": -14.0})
        assert set(payload) == {"events", "segments", "quality"}

    def test_never_emits_a_verdict(self):
        """'Never output Unsafe. Never output Policy Violation. Never output
        Copyright Claim.' Those belong to later agents."""
        signal = noise(3.0, 0.02) + click(1.0, 3.0)
        events = detect_transients(signal, analyse_spectra(signal, RATE), RATE)
        events += detect_hum(analyse_spectra(signal, RATE), signal)
        blob = str(to_json(events)).upper()
        for forbidden in ("UNSAFE", "POLICY VIOLATION", "LIMITED ADS", "COPYRIGHT CLAIM"):
            assert forbidden not in blob

    def test_every_event_is_timestamped_and_scored(self):
        signal = noise(3.0, 0.02) + click(1.2, 3.0)
        for event in detect_transients(signal, analyse_spectra(signal, RATE), RATE):
            assert event.end_ms >= event.start_ms
            assert 0.0 < event.confidence <= 1.0
            assert event.to_json()["band"] in {"VERY_HIGH", "HIGH", "MEDIUM", "LOW"}

    def test_an_unresolved_event_says_so_in_its_payload(self):
        payload = AudioEvent("IMPULSIVE_TRANSIENT", 0, 10, 0.9, resolved=False).to_json()
        assert payload["label_resolved"] is False
        assert payload["requires"] == "audio.classify"

    def test_a_resolved_event_carries_no_such_marker(self):
        payload = AudioEvent("SILENCE", 0, 10, 0.9).to_json()
        assert "label_resolved" not in payload

