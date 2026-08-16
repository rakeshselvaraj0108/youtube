"""Observability that only reports what something counted.

The failure mode being guarded against is not a wrong number — it is a
*plausible* number. A dashboard showing "CPU 34%" beside real figures teaches
a reader that everything on the page is the same kind of value, and once one
is invented the rest stop being evidence. So the tests here mostly assert that
unmeasured things say NOT MEASURED.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

import pytest

from preflight import cas, ffmpeg, telemetry


@dataclass
class FakeAgent:
    agent_id: str
    calls: int = 0
    coverage: float = 1.0
    artifacts: dict = field(default_factory=dict)


@dataclass
class FakeIngested:
    keyframes: list = field(default_factory=list)
    cached: bool = False


@dataclass
class FakeResult:
    agents: list = field(default_factory=list)
    ingested: FakeIngested = field(default_factory=FakeIngested)


class TestCountersAreIncrementedAtTheSource:
    def test_running_ffmpeg_counts_a_process(self, tmp_path):
        if not ffmpeg.available():
            pytest.skip("ffmpeg is required")
        before = telemetry.snapshot().get(telemetry.FFMPEG_RUNS, 0)
        ffmpeg.run(
            ["-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.1",
             str(tmp_path / "x.mp4")]
        )
        assert telemetry.snapshot()[telemetry.FFMPEG_RUNS] == before + 1

    def test_probing_counts_an_ffprobe(self, tmp_path):
        if not ffmpeg.available():
            pytest.skip("ffmpeg is required")
        media = tmp_path / "x.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.1", str(media)],
            check=True, capture_output=True,
        )
        before = telemetry.snapshot().get(telemetry.FFPROBE_RUNS, 0)
        ffmpeg.probe(media)
        assert telemetry.snapshot()[telemetry.FFPROBE_RUNS] == before + 1

    def test_a_cache_miss_is_counted(self, tmp_path):
        store = cas.Store(tmp_path / "cache")
        before = telemetry.snapshot().get(telemetry.CACHE_MISSES, 0)
        assert not store.entry("v", "nothing-here").exists
        assert telemetry.snapshot()[telemetry.CACHE_MISSES] == before + 1

    def test_a_cache_hit_is_counted(self, tmp_path):
        store = cas.Store(tmp_path / "cache")
        entry = store.entry("v", "something")
        entry.commit()
        before = telemetry.snapshot().get(telemetry.CACHE_HITS, 0)
        assert entry.exists
        assert telemetry.snapshot()[telemetry.CACHE_HITS] == before + 1


class TestPhasesAreDifferencedNotZeroed:
    def test_a_phase_records_elapsed_time(self):
        recorder = telemetry.Recorder()
        with recorder.phase("work"):
            sum(range(200_000))
        assert recorder.phase_ms("work") is not None
        assert recorder.phase_ms("work") >= 0

    def test_a_phase_attributes_only_its_own_counters(self):
        recorder = telemetry.Recorder()
        telemetry.count("testOnly", 5)
        with recorder.phase("inside"):
            telemetry.count("testOnly", 3)
        phase = next(p for p in recorder.phases if p.name == "inside")
        assert phase.counters["testOnly"] == 3

    def test_counters_are_never_reset_by_a_phase(self):
        """Another thread may be analysing concurrently. Zeroing a
        process-wide counter for one phase would subtract its work."""
        telemetry.count("concurrentProbe", 7)
        before = telemetry.snapshot()["concurrentProbe"]
        recorder = telemetry.Recorder()
        with recorder.phase("x"):
            pass
        assert telemetry.snapshot()["concurrentProbe"] == before

    def test_an_absent_phase_is_none_not_zero(self):
        """Zero milliseconds and "this never ran" are different facts."""
        assert telemetry.Recorder().phase_ms("never-happened") is None

    def test_a_phase_that_raises_is_still_timed(self):
        recorder = telemetry.Recorder()
        with pytest.raises(ValueError):
            with recorder.phase("doomed"):
                raise ValueError("boom")
        assert recorder.phase_ms("doomed") is not None


class TestNothingIsEstimated:
    def test_cpu_is_reported_as_not_measured(self):
        """Nothing in this engine samples CPU. Saying so is the whole point of
        the field existing."""
        assert telemetry.Recorder().to_json()["cpuPercent"] == telemetry.NOT_MEASURED

    def test_queue_depth_is_reported_as_not_measured(self):
        assert telemetry.Recorder().to_json()["queueDepth"] == telemetry.NOT_MEASURED

    def test_peak_rss_is_a_number_or_the_sentinel_never_a_guess(self):
        value = telemetry.Recorder().to_json()["peakRssBytes"]
        assert value == telemetry.NOT_MEASURED or (
            isinstance(value, int) and value > 0
        )

    def test_peak_rss_is_actually_measured_where_the_platform_allows(self):
        """This project runs on Windows and CI runs on Linux; both can report
        it. A silent fallback to NOT MEASURED on a platform that *can* measure
        would be an unnoticed downgrade, so assert the real path works here."""
        import sys

        if sys.platform not in ("win32", "linux", "darwin"):
            pytest.skip("no supported peak-RSS source")
        peak = telemetry.peak_rss_bytes()
        assert peak is not None and peak > 1024 * 1024


class TestRunObservation:
    def test_real_per_agent_figures_are_taken_from_the_run(self):
        recorder = telemetry.Recorder()
        recorder.observe_run(
            "analysis",
            FakeResult(
                agents=[
                    FakeAgent("vision", calls=4, coverage=0.62),
                    FakeAgent("speech", calls=2, coverage=0.95),
                ],
                ingested=FakeIngested(keyframes=[1, 2, 3, 4, 5], cached=True),
            ),
        )
        out = recorder.to_json()
        assert out["analysis.framesSampled"] == 5
        assert out["analysis.agentCalls"] == 6
        assert out["analysis.calls.vision"] == 4
        assert out["analysis.ingestCached"] is True
        assert out["coverageByAgent"]["analysis"] == {"vision": 0.62, "speech": 0.95}

    def test_two_runs_are_recorded_separately(self):
        """The original and the re-analysis are different measurements and
        must not be merged into one figure."""
        recorder = telemetry.Recorder()
        recorder.observe_run(
            "analysis", FakeResult(ingested=FakeIngested(keyframes=[1, 2, 3]))
        )
        recorder.observe_run(
            "reanalysis", FakeResult(ingested=FakeIngested(keyframes=[1]))
        )
        out = recorder.to_json()
        assert out["analysis.framesSampled"] == 3
        assert out["reanalysis.framesSampled"] == 1

    def test_an_agent_that_made_no_calls_is_not_listed_as_making_one(self):
        recorder = telemetry.Recorder()
        recorder.observe_run("analysis", FakeResult(agents=[FakeAgent("idle")]))
        assert "analysis.calls.idle" not in recorder.to_json()


class TestPayload:
    def test_the_payload_is_json(self):
        import json

        recorder = telemetry.Recorder()
        with recorder.phase("a"):
            pass
        json.dumps(recorder.to_json())

    def test_ffmpeg_process_count_is_the_sum_of_both_binaries(self):
        recorder = telemetry.Recorder()
        telemetry.count(telemetry.FFMPEG_RUNS, 3)
        telemetry.count(telemetry.FFPROBE_RUNS, 2)
        out = recorder.to_json()
        assert out["ffmpegRuns"] == 3
        assert out["ffprobeRuns"] == 2
        assert out["ffmpegProcesses"] == 5

    def test_the_recorder_measures_its_own_window_not_the_process(self):
        """A long-lived server accumulates counters across every run. A
        per-verification figure must be the delta, or every report after the
        first would claim work it did not do."""
        telemetry.count(telemetry.FFMPEG_RUNS, 100)
        recorder = telemetry.Recorder()
        telemetry.count(telemetry.FFMPEG_RUNS, 2)
        assert recorder.to_json()["ffmpegRuns"] == 2
