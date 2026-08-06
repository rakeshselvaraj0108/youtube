"""Perception agents — signal maths, deterministic findings, degradation."""

from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from preflight import ffmpeg
from preflight.models import AgentResult, Evidence
from preflight.perception import metadata, signal as sig
from preflight.perception.accessibility import (
    FLASH_DELTA,
    SAMPLE_FPS,
    flash_risk,
)
from preflight.perception.asr import Segment, Transcript, Word, speech_rate_wpm
from preflight.pipeline import SURFACE_WEIGHT, compute_coverage


def write_wav(path: Path, data: np.ndarray, rate: int = 44_100) -> Path:
    """data shaped (channels, samples), float in [-1, 1]."""
    channels, _ = data.shape
    interleaved = (np.clip(data.T, -1, 1) * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(interleaved)
    return path


class TestWavIO:
    def test_round_trips_stereo(self, tmp_path):
        rate = 8000
        t = np.linspace(0, 1, rate, endpoint=False)
        data = np.vstack([np.sin(2 * np.pi * 220 * t), np.sin(2 * np.pi * 440 * t)])
        audio = sig.read_wav(write_wav(tmp_path / "s.wav", data, rate))

        assert audio.channels == 2
        assert audio.sample_rate == rate
        assert audio.duration_ms == pytest.approx(1000, abs=5)
        assert np.abs(audio.samples[0] - data[0]).max() < 0.001

    def test_mono_is_the_channel_mean(self, tmp_path):
        data = np.vstack([np.full(1000, 0.5), np.full(1000, -0.5)])
        audio = sig.read_wav(write_wav(tmp_path / "m.wav", data, 8000))
        assert np.abs(audio.mono).max() < 0.001


class TestSpectralFlatness:
    """Near 1 for noise, near 0 for tones. This is what separates a music bed
    from speech without a fingerprint database."""

    def test_tone_is_far_from_flat(self):
        rate = 22_050
        t = np.linspace(0, 2, rate * 2, endpoint=False)
        flatness = sig.spectral_flatness(np.sin(2 * np.pi * 440 * t), rate)
        assert flatness.size > 0
        assert float(np.median(flatness)) < 0.05

    def test_white_noise_is_flat(self):
        rate = 22_050
        rng = np.random.default_rng(0)
        flatness = sig.spectral_flatness(rng.normal(0, 0.2, rate * 2), rate)
        assert float(np.median(flatness)) > 0.3

    def test_empty_input_is_empty_output(self):
        assert sig.spectral_flatness(np.array([]), 44_100).size == 0


class TestRmsAndSpans:
    def test_rms_tracks_amplitude(self):
        rate = 8000
        quiet = np.full(rate, 0.001)
        loud = np.full(rate, 0.5)
        envelope = sig.rms_envelope(np.concatenate([quiet, loud]), rate, 100)
        assert envelope[:5].max() < 0.01
        assert envelope[-5:].min() > 0.4

    def test_spans_where_finds_runs_over_the_minimum(self):
        mask = np.array([False] * 5 + [True] * 40 + [False] * 5)
        spans = sig.spans_where(mask, step_ms=100, min_ms=3000)
        assert len(spans) == 1
        assert spans[0] == (500, 4500)

    def test_spans_shorter_than_the_minimum_are_dropped(self):
        mask = np.array([False] * 5 + [True] * 4 + [False] * 5)
        assert sig.spans_where(mask, step_ms=100, min_ms=3000) == []

    def test_no_spans_in_an_empty_mask(self):
        assert sig.spans_where(np.array([]), 100, 1000) == []


class TestFlashRisk:
    """Photosensitive flash detection — the standout accessibility check."""

    def _strobe(self, flashes_per_second: int, seconds: int = 3) -> np.ndarray:
        """Luminance alternating at the requested rate, sampled at SAMPLE_FPS."""
        samples = SAMPLE_FPS * seconds
        series = np.zeros(samples, dtype=np.float32)
        period = max(1, SAMPLE_FPS // max(flashes_per_second, 1))
        for i in range(samples):
            series[i] = 255.0 if (i // period) % 2 == 0 else 0.0
        return series

    def test_static_footage_is_low_risk(self):
        result = flash_risk(np.full(100, 128.0), SAMPLE_FPS)
        assert result["max_flashes_per_second"] == 0
        assert result["risk"] == "LOW"

    def test_gentle_gradient_is_low_risk(self):
        result = flash_risk(np.linspace(0, 255, 200), SAMPLE_FPS)
        assert result["risk"] == "LOW"

    def test_fast_strobe_is_high_risk(self):
        result = flash_risk(self._strobe(5), SAMPLE_FPS)
        assert result["max_flashes_per_second"] >= 3
        assert result["risk"] == "HIGH"

    def test_below_threshold_swings_do_not_count(self):
        """A flash is a swing over 10% of full range. Smaller is not a flash."""
        series = np.array([128.0, 128.0 + FLASH_DELTA * 0.5] * 30)
        assert flash_risk(series, SAMPLE_FPS)["risk"] == "LOW"

    def test_reports_when_the_worst_moment_happens(self):
        calm = np.full(SAMPLE_FPS * 3, 100.0)
        result = flash_risk(np.concatenate([calm, self._strobe(5, 3)]), SAMPLE_FPS)
        assert result["risk"] == "HIGH"
        assert result["worst_ts_ms"] >= 2500

    def test_degenerate_input_does_not_crash(self):
        assert flash_risk(np.array([]), SAMPLE_FPS)["risk"] == "LOW"
        assert flash_risk(np.array([1.0]), SAMPLE_FPS)["risk"] == "LOW"


class TestSpeechRate:
    def _words(self, count: int, over_ms: int) -> list[Word]:
        step = over_ms // max(count, 1)
        return [Word(w="x", start_ms=i * step, end_ms=i * step + 50, conf=0.9) for i in range(count)]

    def test_no_words_is_no_samples(self):
        assert speech_rate_wpm([]) == []

    def test_measures_words_per_minute(self):
        # 90 words in 30s == 180 wpm
        samples = speech_rate_wpm(self._words(90, 30_000), window_ms=30_000)
        assert samples
        assert samples[0][1] == pytest.approx(180, abs=6)


class TestTranscript:
    def _transcript(self) -> Transcript:
        words = [
            Word(w="the", start_ms=1000, end_ms=1200, conf=0.99),
            Word(w="anchor", start_ms=1200, end_ms=1700, conf=0.98),
            Word(w="pulled", start_ms=1700, end_ms=2100, conf=0.97),
        ]
        return Transcript(
            language="en",
            duration_ms=5000,
            words=words,
            segments=[Segment(start_ms=1000, end_ms=2100, text="the anchor pulled")],
        )

    def test_text_between_selects_overlapping_words(self):
        assert self._transcript().text_between(1150, 1800) == "the anchor pulled"

    def test_snap_to_words_widens_to_boundaries(self):
        """Clipping mid-syllable is audible, so audio ops expand to whole words."""
        start, end = self._transcript().snap_to_words(1300, 1900)
        assert start == 1200
        assert end == 2100

    def test_snap_leaves_a_span_with_no_words_alone(self):
        assert self._transcript().snap_to_words(4000, 4500) == (4000, 4500)

    def test_json_round_trip(self):
        original = self._transcript()
        restored = Transcript.from_json(original.to_json())
        assert restored.words == original.words
        assert restored.text == original.text


class TestMetadataAgent:
    def _sidecar(self, **kwargs) -> metadata.Sidecar:
        base = {
            "title": "A perfectly reasonable title about climbing",
            "description": "x" * 400,
            "tags": ["climbing", "alpine"],
            "category": "Travel",
        }
        base.update(kwargs)
        return metadata.Sidecar(**base)

    def _transcript(self, text: str) -> Transcript:
        return Transcript(
            language="en",
            duration_ms=60_000,
            words=[],
            segments=[Segment(start_ms=0, end_ms=60_000, text=text)],
        )

    def test_clean_metadata_produces_no_findings(self, tmp_path):
        result = metadata.analyse(
            tmp_path / "v.mp4", 60_000, self._transcript("just a normal video"), self._sidecar()
        )
        assert result.status == "OK"
        assert result.findings == []

    def test_spoken_sponsorship_without_disclosure_is_flagged(self, tmp_path):
        result = metadata.analyse(
            tmp_path / "v.mp4",
            60_000,
            self._transcript("this video is sponsored by a gear company"),
            self._sidecar(),
        )
        ids = {f.id for f in result.findings}
        assert "m_disclosure" in ids
        finding = next(f for f in result.findings if f.id == "m_disclosure")
        assert finding.severity == "HIGH"

    def test_disclosure_in_the_description_clears_it(self, tmp_path):
        result = metadata.analyse(
            tmp_path / "v.mp4",
            60_000,
            self._transcript("this video is sponsored by a gear company"),
            self._sidecar(description="Includes paid promotion. " + "x" * 400),
        )
        assert "m_disclosure" not in {f.id for f in result.findings}

    def test_affiliate_link_alone_is_enough_to_flag(self, tmp_path):
        result = metadata.analyse(
            tmp_path / "v.mp4",
            60_000,
            self._transcript("nothing sponsored here at all"),
            self._sidecar(description="Gear: amzn.to/3xKp2Qw " + "x" * 400),
        )
        assert "m_disclosure" in {f.id for f in result.findings}

    def test_flags_thin_description_long_title_and_tag_stuffing(self, tmp_path):
        result = metadata.analyse(
            tmp_path / "v.mp4",
            60_000,
            None,
            self._sidecar(
                title="T" * 90,
                description="short",
                tags=[f"tag{i}" for i in range(30)],
            ),
        )
        ids = {f.id for f in result.findings}
        assert {"m_desc", "m_title_len", "m_tags"} <= ids

    def test_uppercase_title_is_flagged(self, tmp_path):
        result = metadata.analyse(
            tmp_path / "v.mp4", 60_000, None, self._sidecar(title="THE FULL UNCUT ASCENT")
        )
        assert "m_title_caps" in {f.id for f in result.findings}

    def test_missing_sidecar_skips_rather_than_crashing(self, tmp_path):
        result = metadata.analyse(tmp_path / "absent.mp4", 60_000, None)
        assert result.status == "SKIPPED"
        assert result.coverage == 0.0
        assert "sidecar" in (result.error or "")

    def test_sidecar_loads_from_disk(self, tmp_path):
        video = tmp_path / "clip.mp4"
        (tmp_path / "clip.meta.json").write_text(
            json.dumps({"title": "Hello", "description": "d", "tags": ["a"]}),
            encoding="utf-8",
        )
        loaded = metadata.Sidecar.load(video)
        assert loaded is not None
        assert loaded.title == "Hello"

    def test_malformed_sidecar_is_ignored_not_fatal(self, tmp_path):
        video = tmp_path / "clip.mp4"
        (tmp_path / "clip.meta.json").write_text("{not json", encoding="utf-8")
        assert metadata.Sidecar.load(video) is None


class TestEvidence:
    def test_marking_locates_the_span(self):
        evidence = Evidence.marking("the anchor pulled clean out", "anchor pulled")
        start, end = evidence.highlightSpan
        assert evidence.transcript[start:end] == "anchor pulled"

    def test_missing_needle_yields_an_empty_span(self):
        assert Evidence.marking("abc", "zzz").highlightSpan == (0, 0)


class TestCoverage:
    def _agent(self, agent_id: str, coverage: float) -> AgentResult:
        return AgentResult(agent_id=agent_id, name=agent_id, coverage=coverage)

    def test_all_agents_at_full_coverage_is_one(self):
        agents = [self._agent(a, 1.0) for a, w in SURFACE_WEIGHT.items() if w > 0]
        assert compute_coverage(agents) == pytest.approx(1.0)

    def test_a_degraded_agent_costs_its_own_weight(self):
        agents = [self._agent(a, 1.0) for a, w in SURFACE_WEIGHT.items() if w > 0]
        agents = [self._agent(a.agent_id, 0.42 if a.agent_id == "vision" else 1.0) for a in agents]
        expected = 1.0 - SURFACE_WEIGHT["vision"] * (1 - 0.42)
        assert compute_coverage(agents) == pytest.approx(expected, abs=1e-6)

    def test_an_agent_that_never_ran_counts_as_zero_coverage(self):
        """Dropping absent agents would let a run that skipped vision entirely
        still claim full coverage."""
        agents = [self._agent(a, 1.0) for a, w in SURFACE_WEIGHT.items() if w > 0 and a != "vision"]
        assert compute_coverage(agents) == pytest.approx(1.0 - SURFACE_WEIGHT["vision"], abs=1e-6)

    def test_no_agents_at_all_is_zero(self):
        assert compute_coverage([]) == pytest.approx(0.0)

    def test_weight_table_holds_no_agent_run_perception_cannot_produce(self):
        """`test_all_agents_at_full_coverage_is_one` above is arithmetic on a
        SYNTHETIC agent list built directly from SURFACE_WEIGHT's own keys —
        it would pass even if SURFACE_WEIGHT listed an agent id nothing ever
        produces, because it never asks whether that id is real. That is
        exactly what happened: "remedy" and "report" carried weight while
        `run_perception` never populated either, so real coverage was capped
        at 97% forever, on every run, regardless of what actually succeeded.

        The real ids come from `TOPOLOGY`, which lists every stage across the
        full pipeline including the two that run from other commands
        (`preflight fix` for remedy, `_emit` for report). Excluding those two
        explicitly, this asserts SURFACE_WEIGHT's keys are exactly the
        run_perception-producible ones — no more, no less.
        """
        from preflight.pipeline import TOPOLOGY

        outside_run_perception = {"remedy", "report"}
        producible = set(TOPOLOGY) - outside_run_perception
        assert set(SURFACE_WEIGHT) == producible

    def test_coverage_weights_sum_to_exactly_one(self):
        assert sum(SURFACE_WEIGHT.values()) == pytest.approx(1.0, abs=1e-9)


# Stages the pipeline composes from other agents' output rather than
# invoking as a stage of their own.
COMPOSED_STAGES = {"orchestrator", "score", "remedy", "report"}


@pytest.fixture(scope="module")
def wired_run(tmp_path_factory):
    """One real end-to-end run of the perception pipeline.

    Offline, always. A suite that reaches a hosted model is slow,
    non-deterministic and spends the user's quota to assert something about
    wiring — and it hangs on a rate limit, which is how this was found.
    """
    from preflight import cas
    from preflight.config import Settings
    from preflight.pipeline import run_perception

    media = tmp_path_factory.mktemp("media")
    clip = media / "wired.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=3",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(clip),
        ],
        check=True,
        capture_output=True,
    )
    return run_perception(
        clip,
        cas.Store(media / "cas"),
        skip_speech=True,
        settings=Settings.load(offline=True),
    )


@pytest.mark.skipif(not ffmpeg.available(), reason="ffmpeg is not installed")
class TestEveryWeightedAgentActuallyRuns:
    """A weight in SURFACE_WEIGHT is a promise that the agent runs.

    Vision and OCR carried 35% of the analysis surface between them while
    `run_perception` called neither. Coverage reported that honestly — the
    pipeline was not lying about what it saw — but a third of the surface was
    dark on every run, and nothing in the suite noticed, because both modules
    had their own passing tests and no caller.

    A unit test cannot catch that shape of bug. Only running the thing can.
    """

    def test_every_weighted_agent_appears_in_the_result(self, wired_run):
        result = wired_run
        ran = {agent.agent_id for agent in result.agents}
        for agent_id, weight in SURFACE_WEIGHT.items():
            if weight == 0.0 or agent_id in COMPOSED_STAGES:
                continue
            assert agent_id in ran, (
                f"{agent_id} carries {weight:.0%} of the analysis surface "
                "but the pipeline never ran it"
            )

    def test_no_agent_reports_an_uncaught_crash(self, wired_run):
        for agent in wired_run.agents:
            assert "Traceback" not in (agent.error or "")

    def test_the_acoustic_tier_reaches_the_audio_agent(self, wired_run):
        """A04's second tier folds into the same node rather than inventing a
        thirteenth agent the roster does not declare."""
        audio_agent = wired_run.agent("audio")
        assert audio_agent is not None
        if audio_agent.status == "OK":
            assert "segments" in audio_agent.artifacts

    def test_an_absent_optional_provider_skips_rather_than_failing(self, wired_run):
        """Offline with no key is the default experience for anyone cloning
        this repo. Nothing in that run should read as broken."""
        for agent in wired_run.agents:
            assert agent.status != "FAILED", f"{agent.agent_id}: {agent.error}"

    def test_the_report_carries_a_strategy_when_one_was_requested(self, wired_run):
        """`--strategy` reached `preflight fix` when it was built; it did not
        reach `preflight check`, the command someone actually runs to see a
        report, until the two were wired together. This builds the same
        report `check --format json` writes and validates it against the
        real schema — the same contract the UI reads."""
        import json as json_mod
        from pathlib import Path as PathAlias

        from preflight.report.build import build_report

        bundle = build_report(wired_run, strategy="conservative")
        assert bundle.report["remediation"]["strategy"] == "conservative"
        assert isinstance(bundle.report["remediation"]["log"], list)

        schema_path = PathAlias("schema/analysis-report.schema.json")
        if schema_path.is_file():
            import jsonschema

            jsonschema.validate(
                bundle.report, json_mod.loads(schema_path.read_text(encoding="utf-8"))
            )

    def test_the_report_omits_strategy_when_none_was_requested(self, wired_run):
        """Backward compatible: no strategy means the field is simply absent,
        not present-and-null — every existing consumer of this report shape
        that never heard of strategies keeps working unchanged."""
        from preflight.report.build import build_report

        bundle = build_report(wired_run)
        assert "strategy" not in bundle.report["remediation"]
        assert "log" in bundle.report["remediation"]

    def test_coverage_is_a_real_number_between_zero_and_one(self, wired_run):
        assert 0.0 <= wired_run.coverage <= 1.0
