"""The provider protocol.

A capability is served by an ordered chain of providers, best to worst, ending
in a null. Three properties follow, and they are the whole design:

1. A missing key demotes a tier. It never breaks a feature.
2. The null provider is a real implementation. It returns Unavailable with a
   reason. It never fabricates a result, never returns an empty success, and
   never silently drops work.
3. Every report states which provider served each capability and whether that
   was the preferred tier or a fallback.

The third point closes a loop: PREFLIGHT exists to make video analysis
auditable. A tool that demands auditability of others while hiding its own
provenance is inconsistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

TierLabel = str  # "hosted" | "local" | "null"


@dataclass
class Unavailable:
    """This provider cannot serve the request, and why."""

    reason: str
    provider: str
    retryable: bool = False

    ok = False

    def __bool__(self) -> bool:
        return False


@dataclass
class Served:
    """A result, with the provenance needed to reproduce or challenge it."""

    value: Any
    provider: str  # "nvidia:meta/llama-3.3-70b-instruct"
    tier: int = 0  # 0 = preferred
    calls: int = 1
    tokens: int = 0
    latency_ms: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    ok = True

    def __bool__(self) -> bool:
        return True


Result = Served | Unavailable


@runtime_checkable
class Provider(Protocol):
    id: str
    capability: str
    tier_label: TierLabel

    def available(self) -> tuple[bool, str]:
        """Cheap, offline readiness check.

        Called at resolution time for every provider in every chain, so it must
        never make a network call. Checking a key's presence and shape, or
        whether a binary is on PATH, is the right kind of work here.
        """
        ...

    def healthcheck(self) -> tuple[bool, str, int]:
        """One real round-trip. Only ever called by `preflight doctor`."""
        ...

    def invoke(self, **kwargs: Any) -> Result:
        ...


class BaseProvider:
    """Shared defaults. Subclasses override `available` and `invoke`."""

    id: str = "base"
    capability: str = ""
    tier_label: TierLabel = "null"
    model: str = ""

    def __init__(self, capability: str) -> None:
        self.capability = capability

    @property
    def label(self) -> str:
        return f"{self.id}:{self.model}" if self.model else self.id

    def available(self) -> tuple[bool, str]:
        return False, "not implemented"

    def healthcheck(self) -> tuple[bool, str, int]:
        ok, reason = self.available()
        return ok, reason, 0

    def invoke(self, **kwargs: Any) -> Result:
        return Unavailable("not implemented", self.id)


class NullProvider(BaseProvider):
    """The end of every chain.

    Deliberately a real object rather than a None the caller has to guard. An
    agent that receives Unavailable reports SKIPPED with the reason; an agent
    that receives None writes `if provider:` everywhere and eventually forgets
    once.
    """

    id = "null"
    tier_label = "null"

    def __init__(self, capability: str, reason: str = "no provider available") -> None:
        super().__init__(capability)
        self.reason = reason

    def available(self) -> tuple[bool, str]:
        # Always "available" in the sense that it always answers. It simply
        # answers that nothing can serve this.
        return True, self.reason

    def healthcheck(self) -> tuple[bool, str, int]:
        return False, self.reason, 0

    def invoke(self, **kwargs: Any) -> Result:
        return Unavailable(self.reason, "null")


class CircuitOpen(RuntimeError):
    """The vendor's breaker is open; do not attempt the call."""

    def __init__(self, vendor: str, detail: str) -> None:
        self.vendor = vendor
        super().__init__(f"{vendor} circuit open — {detail}")


class BudgetExhausted(RuntimeError):
    """The vendor's call or quota budget for this run is spent."""

    def __init__(self, vendor: str, used: int, budget: int) -> None:
        self.vendor, self.used, self.budget = vendor, used, budget
        super().__init__(f"{vendor} budget exhausted: {used}/{budget}")
