"""Rate governance, circuit breaking and budget accounting.

**One bucket per vendor, shared by every capability that vendor serves.**

That is the non-obvious part and it is worth stating plainly: ASR, chat,
embeddings, reranking, vision and OCR all hit the same NVIDIA rate limit. Six
independent limiters at 30 RPM each is 180 RPM against a ceiling near 40 — an
instant 429 storm that looks like a vendor problem and is actually an
architecture problem.

The circuit breaker is the live-demo insurance. When a vendor starts failing,
its capabilities demote a tier for the cooldown rather than retrying into a
wall: ASR moves to local, vision moves to null and reports SKIPPED, and the run
finishes with reduced coverage that the report states honestly.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Literal

from preflight.providers.base import BudgetExhausted, CircuitOpen

BreakerState = Literal["CLOSED", "OPEN", "HALF_OPEN"]

# Consecutive failures across the whole vendor before the breaker opens.
#
# Was 5. Measured live: a single `_call()` retries up to `MAX_ATTEMPTS`
# times internally before the breaker is ever consulted again, so at 5 the
# breaker could not trip until the *first* call had already exhausted its
# own retry budget — it was protecting nothing. A vendor genuinely
# unreachable for a whole run cost 25.6 minutes on one policy-retrieval
# stage alone, almost all of it retries against a host that was never going
# to answer. 3 still requires real repeated evidence — one blip does not
# trip it — while letting the breaker actually intervene during the second
# failing call instead of after it.
FAILURE_THRESHOLD = 3
COOLDOWN_S = 60.0
BACKOFF_CAP_S = 30.0


@dataclass
class Ledger:
    """Real usage, for the telemetry strip and the certificate.

    The call counter the UI renders must be this number. A constant that looks
    plausible is worse than no counter at all — it is a claim you cannot back.
    """

    vendor: str
    budget: int
    calls: int = 0
    cached: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    quota_units: int = 0
    retries: int = 0
    by_capability: dict[str, dict[str, int]] = field(default_factory=dict)
    latencies_ms: list[int] = field(default_factory=list)
    circuit_events: int = 0

    def record(
        self,
        capability: str,
        *,
        calls: int = 1,
        tokens_in: int = 0,
        tokens_out: int = 0,
        units: int = 0,
        latency_ms: int = 0,
    ) -> None:
        self.calls += calls
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.quota_units += units
        if latency_ms:
            self.latencies_ms.append(latency_ms)

        bucket = self.by_capability.setdefault(
            capability, {"calls": 0, "tokens": 0, "units": 0}
        )
        bucket["calls"] += calls
        bucket["tokens"] += tokens_in + tokens_out
        bucket["units"] += units

    def record_cached(self, capability: str) -> None:
        """A cache hit is not a call. Counting it as one overstates usage."""
        self.cached += 1
        self.by_capability.setdefault(
            capability, {"calls": 0, "tokens": 0, "units": 0}
        )

    def check(self, cost: int = 1) -> None:
        if self.budget and self.calls + cost > self.budget:
            raise BudgetExhausted(self.vendor, self.calls, self.budget)

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.calls) if self.budget else 10**9

    @property
    def p95_latency_ms(self) -> int:
        if not self.latencies_ms:
            return 0
        ordered = sorted(self.latencies_ms)
        index = min(len(ordered) - 1, int(len(ordered) * 0.95))
        return ordered[index]

    def to_json(self) -> dict[str, object]:
        return {
            "calls": self.calls,
            "cacheHits": self.cached,
            "tokensIn": self.tokens_in,
            "tokensOut": self.tokens_out,
            "quotaUnits": self.quota_units,
            "retries": self.retries,
            "p95LatencyMs": self.p95_latency_ms,
            "circuitEvents": self.circuit_events,
            "budget": self.budget,
            "byCapability": self.by_capability,
        }


class CircuitBreaker:
    """CLOSED → (N failures) → OPEN → (cooldown) → HALF_OPEN → (1 ok) → CLOSED."""

    def __init__(
        self,
        vendor: str,
        threshold: int = FAILURE_THRESHOLD,
        cooldown_s: float = COOLDOWN_S,
        clock=time.monotonic,
    ) -> None:
        self.vendor = vendor
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self.state: BreakerState = "CLOSED"
        self.failures = 0
        self.opened_at = 0.0
        self.trips = 0

    @property
    def remaining_s(self) -> float:
        if self.state != "OPEN":
            return 0.0
        return max(0.0, self.cooldown_s - (self._clock() - self.opened_at))

    def check(self) -> None:
        if self.state == "OPEN":
            if self.remaining_s <= 0:
                self.state = "HALF_OPEN"
            else:
                raise CircuitOpen(self.vendor, f"{self.remaining_s:.0f}s remaining")

    def record(self, ok: bool) -> None:
        if ok:
            self.state = "CLOSED"
            self.failures = 0
            return
        self.failures += 1
        if self.failures >= self.threshold and self.state != "OPEN":
            self.trip()

    def trip(self) -> None:
        """Open immediately. Used for 402 and other terminal conditions."""
        self.state = "OPEN"
        self.opened_at = self._clock()
        self.trips += 1


class VendorGovernor:
    """One rate bucket, one breaker and one ledger for a whole vendor."""

    def __init__(
        self,
        vendor: str,
        rpm: int = 30,
        call_budget: int = 0,
        clock=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        self.vendor = vendor
        self.rpm = max(1, rpm)
        self.interval = 60.0 / self.rpm
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_at = 0.0
        self.ledger = Ledger(vendor, call_budget)
        self.breaker = CircuitBreaker(vendor, clock=clock)

    def acquire(self, capability: str, cost: int = 1) -> float:
        """Block until a slot is free. Returns seconds waited.

        Spacing calls evenly rather than allowing a burst is deliberate: a
        token bucket that permits 30 immediate calls then starves is far more
        likely to trip a vendor's own burst detection than one that paces.
        """
        self.breaker.check()
        self.ledger.check(cost)

        with self._lock:
            now = self._clock()
            wait = max(0.0, self._next_at - now)
            self._next_at = max(now, self._next_at) + self.interval

        if wait > 0:
            self._sleep(wait)
        return wait

    def backoff(self, attempt: int, cap: float = BACKOFF_CAP_S) -> float:
        """Exponential with jitter. Jitter matters when several capabilities
        retry at once — without it they resynchronise and hit together."""
        delay = min(2**attempt + random.random(), cap)
        self.ledger.retries += 1
        self._sleep(delay)
        return delay

    def on_success(self, capability: str, **usage) -> None:
        self.breaker.record(True)
        self.ledger.record(capability, **usage)

    def on_failure(self, terminal: bool = False) -> None:
        if terminal:
            self.breaker.trip()
            self.ledger.circuit_events += 1
        else:
            before = self.breaker.state
            self.breaker.record(False)
            if before != "OPEN" and self.breaker.state == "OPEN":
                self.ledger.circuit_events += 1


# One governor per vendor, created once. Every provider for a vendor shares it.
GOVERNORS: dict[str, VendorGovernor] = {}


def governor(vendor: str, *, rpm: int = 30, call_budget: int = 0) -> VendorGovernor:
    if vendor not in GOVERNORS:
        GOVERNORS[vendor] = VendorGovernor(vendor, rpm=rpm, call_budget=call_budget)
    return GOVERNORS[vendor]


def reset_governors() -> None:
    """Test helper. Never called in a run."""
    GOVERNORS.clear()


def usage_report() -> dict[str, dict[str, object]]:
    return {vendor: gov.ledger.to_json() for vendor, gov in GOVERNORS.items()}
