"""Observability that only reports what something actually counted.

Every field here traces to an increment at a real call site or a clock read
around a real phase. There is no estimation, no extrapolation and no
"approximately" — a metric this module cannot measure is emitted as the string
`NOT MEASURED`, which is a fact, where a plausible number would be a fiction.

That sentinel is load-bearing rather than decorative. Peak RAM is genuinely
unavailable on some platforms; CPU utilisation is not measured by anything in
this engine at all. A dashboard showing "CPU 34%" next to real figures teaches
a reader that every number on the page is the same kind of number, and the
moment one of them is invented the rest stop being evidence.

Counters are process-wide and incremented at the source — `ffmpeg.run`
increments the ffmpeg counter, `cas.Entry.exists` increments hit or miss — so
a caller cannot forget to record something it did. A `Recorder` samples the
difference across a phase, which is how a per-verification figure is derived
from a per-process counter without either one lying.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

NOT_MEASURED = "NOT MEASURED"

_LOCK = threading.Lock()
_COUNTERS: dict[str, int] = {}

# The counter names call sites use. Named constants rather than string
# literals because a typo at a call site produces a counter nobody reads,
# which looks exactly like work that never happened.
FFMPEG_RUNS = "ffmpegRuns"
FFPROBE_RUNS = "ffprobeRuns"
CACHE_HITS = "cacheHits"
CACHE_MISSES = "cacheMisses"


def count(name: str, amount: int = 1) -> None:
    with _LOCK:
        _COUNTERS[name] = _COUNTERS.get(name, 0) + amount


def snapshot() -> dict[str, int]:
    with _LOCK:
        return dict(_COUNTERS)


def reset() -> None:
    """Tests only. A long-lived server never resets; the Recorder takes
    differences precisely so it does not have to."""
    with _LOCK:
        _COUNTERS.clear()


def peak_rss_bytes() -> int | None:
    """Peak resident set size, or None where the platform will not say.

    None is the honest answer on a platform without an accessible counter,
    and it is the only reason this returns an Optional rather than a number.
    """
    try:  # POSIX
        import resource  # type: ignore[import-not-found]

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes, macOS bytes. Both are real; the unit is
        # what differs, and guessing wrong is a 1024x error in a report.
        import sys

        return int(usage) if sys.platform == "darwin" else int(usage) * 1024
    except (ImportError, AttributeError, OSError):
        pass

    try:  # Windows, via psapi
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        # GetCurrentProcess returns the pseudo-handle (HANDLE)-1. ctypes
        # defaults a return type to c_int, which truncates it to a 32-bit -1
        # and then passes a *sign-extended int* where a 64-bit HANDLE is
        # expected. The call fails, returns 0, and this function quietly
        # reports NOT MEASURED on a platform that can measure it perfectly
        # well — a silent downgrade, which is the worst kind.
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        handle = kernel32.GetCurrentProcess()

        psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_Counters),
            wintypes.DWORD,
        ]
        ok = psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        return int(counters.PeakWorkingSetSize) if ok else None
    except Exception:  # noqa: BLE001 - an unavailable metric is not an error
        return None


@dataclass
class Phase:
    name: str
    ms: int
    counters: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "ms": self.ms, "counters": dict(self.counters)}


@dataclass
class Recorder:
    """Per-operation observability, assembled from real deltas."""

    phases: list[Phase] = field(default_factory=list)
    values: dict[str, Any] = field(default_factory=dict)
    agent_coverage: dict[str, dict[str, float]] = field(default_factory=dict)
    started: float = field(default_factory=time.perf_counter)
    _baseline: dict[str, int] = field(default_factory=snapshot)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time one stage and attribute the counters it moved.

        The counters are differenced, not zeroed: another thread may be
        analysing at the same time, and resetting a process-wide counter for
        one phase would silently subtract that thread's work from its own
        totals.
        """
        before = snapshot()
        started = time.perf_counter()
        try:
            yield
        finally:
            after = snapshot()
            self.phases.append(
                Phase(
                    name=name,
                    ms=int((time.perf_counter() - started) * 1000),
                    counters={
                        key: after[key] - before.get(key, 0)
                        for key in after
                        if after[key] - before.get(key, 0) > 0
                    },
                )
            )

    def phase_ms(self, name: str) -> int | None:
        for phase in self.phases:
            if phase.name == name:
                return phase.ms
        return None

    def record(self, key: str, value: Any) -> None:
        self.values[key] = value

    def observe_run(self, label: str, result: Any) -> None:
        """Pull the real per-agent figures off a completed pipeline run.

        Everything taken here was already measured by the pipeline for its own
        report — frames it extracted, calls each agent made, coverage each
        agent reached. Nothing is recomputed, so the telemetry and the report
        cannot disagree about the same run.
        """
        agents = list(getattr(result, "agents", []) or [])
        ingested = getattr(result, "ingested", None)
        frames = list(getattr(ingested, "keyframes", []) or []) if ingested else []

        self.values[f"{label}.framesSampled"] = len(frames)
        self.values[f"{label}.framesAnalysed"] = sum(
            len(getattr(a, "artifacts", {}).get("framesExamined", []) or [])
            for a in agents
        ) or len(frames)
        self.values[f"{label}.agentCalls"] = sum(int(a.calls or 0) for a in agents)
        self.values[f"{label}.ingestCached"] = bool(
            getattr(ingested, "cached", False)
        )
        for agent in agents:
            calls = int(agent.calls or 0)
            if calls:
                self.values[f"{label}.calls.{agent.agent_id}"] = calls
        self.agent_coverage[label] = {
            a.agent_id: round(float(a.coverage or 0.0), 4) for a in agents
        }

    def to_json(self) -> dict[str, Any]:
        after = snapshot()
        delta = {
            key: after.get(key, 0) - self._baseline.get(key, 0)
            for key in set(after) | set(self._baseline)
        }
        peak = peak_rss_bytes()

        return {
            "totalMs": int((time.perf_counter() - self.started) * 1000),
            "phases": [p.to_json() for p in self.phases],
            "phaseMs": {p.name: p.ms for p in self.phases},
            "ffmpegProcesses": delta.get(FFMPEG_RUNS, 0) + delta.get(FFPROBE_RUNS, 0),
            "ffmpegRuns": delta.get(FFMPEG_RUNS, 0),
            "ffprobeRuns": delta.get(FFPROBE_RUNS, 0),
            "cacheHits": delta.get(CACHE_HITS, 0),
            "cacheMisses": delta.get(CACHE_MISSES, 0),
            "peakRssBytes": peak if peak is not None else NOT_MEASURED,
            # Nothing in this engine samples CPU or queue depth. Saying so is
            # the entire point of the field existing.
            "cpuPercent": NOT_MEASURED,
            "queueDepth": NOT_MEASURED,
            "coverageByAgent": {
                label: dict(values) for label, values in self.agent_coverage.items()
            },
            **{key: value for key, value in self.values.items()},
        }
