"""The capability catalogue and resolver.

Agents ask for capabilities. They never construct a client, never read an
environment variable, and never learn a vendor's name. Which provider serves a
capability is a resolution decision made once at startup and recorded in the
certificate.

The plan is printed at the top of every run. That is a small thing that tells a
user exactly what they are about to get — including which tiers are degraded
and why — before anything has run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from preflight.providers.base import (
    NullProvider,
    Provider,
    Result,
    Unavailable,
)
from preflight.providers.governor import GOVERNORS, usage_report
from preflight.providers.local import (
    LocalMiniLM,
    LocalTesseract,
    LocalWhisper,
    NoLocalReranker,
    NumpyVectorStore,
)
from preflight.providers.nvidia import (
    NvidiaChat,
    NvidiaEmbed,
    NvidiaRerank,
    NvidiaVision,
)
from preflight.providers.qdrant import QdrantVectorStore
from preflight.providers.secrets import Secret, load_secrets

# Capability names. Referenced by agents; never a vendor string.
CHAT_REASONING = "chat.reasoning"
CHAT_EXTRACTION = "chat.extraction"
ASR_TRANSCRIBE = "asr.transcribe"
EMBED_TEXT = "embed.text"
RERANK_TEXT = "rerank.text"
VECTOR_SEARCH = "vector.search"
VISION_DESCRIBE = "vision.describe"
OCR_IMAGE = "ocr.image"

# Model preferences, best first. NVIDIA's catalogue rotates, so a slug that
# 404s must fall through to the next preference rather than failing the run.
MODEL_PREFERENCES: dict[str, list[str]] = {
    CHAT_REASONING: [
        "nvidia/llama-3.3-nemotron-super-49b-v1",
        "meta/llama-3.3-70b-instruct",
        "qwen/qwen2.5-72b-instruct",
    ],
    CHAT_EXTRACTION: [
        "meta/llama-3.3-70b-instruct",
        "nvidia/llama-3.3-nemotron-super-49b-v1",
    ],
    EMBED_TEXT: ["nvidia/nv-embedqa-e5-v5"],
    RERANK_TEXT: ["nvidia/nv-rerankqa-mistral-4b-v3"],
    VISION_DESCRIBE: ["meta/llama-3.2-90b-vision-instruct"],
}


@dataclass
class Resolution:
    capability: str
    provider: Provider
    tier: int
    tier_label: str
    reason: str
    degraded: bool
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return getattr(self.provider, "label", self.provider.id)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider": self.provider.id,
            "tier": (
                "preferred" if self.tier == 0
                else "unavailable" if self.tier_label == "null"
                else "fallback"
            ),
        }
        model = getattr(self.provider, "model", "")
        if model:
            payload["model"] = model
        if self.degraded:
            payload["reason"] = self.reason
        return payload


class Registry:
    """Resolves every capability once, then hands out providers."""

    def __init__(
        self,
        secrets: dict[str, Secret] | None = None,
        *,
        offline: bool = False,
        qdrant_url: str | None = None,
    ) -> None:
        self.secrets = secrets if secrets is not None else load_secrets()
        self.offline = offline
        self.qdrant_url = qdrant_url or os.environ.get("QDRANT_URL", "")
        self.plan: dict[str, Resolution] = {}
        for capability, chain in self._chains().items():
            self.plan[capability] = self._resolve(capability, chain)

    # ---------------------------------------------------------------- #

    def _nvidia(self, capability: str, cls, index: int = 0):
        preferences = MODEL_PREFERENCES.get(capability, [])
        model = preferences[index] if index < len(preferences) else ""
        return cls(capability, model, self.secrets["NVIDIA_API_KEY"])

    def _chains(self) -> dict[str, list[Provider]]:
        """Ordered provider chains, best to worst. Configuration, not code."""
        qdrant = QdrantVectorStore(
            VECTOR_SEARCH,
            url=self.qdrant_url or "",
            secret=self.secrets.get("QDRANT_API_KEY"),
        )
        return {
            CHAT_REASONING: [
                self._nvidia(CHAT_REASONING, NvidiaChat, 0),
                self._nvidia(CHAT_REASONING, NvidiaChat, 1),
            ],
            CHAT_EXTRACTION: [
                self._nvidia(CHAT_EXTRACTION, NvidiaChat, 0),
                self._nvidia(CHAT_EXTRACTION, NvidiaChat, 1),
            ],
            # Local is the DEFAULT here, not a fallback. Zero-key operation is
            # the property that makes this run on a stranger's machine.
            ASR_TRANSCRIBE: [LocalWhisper(ASR_TRANSCRIBE)],
            EMBED_TEXT: [
                self._nvidia(EMBED_TEXT, NvidiaEmbed, 0),
                LocalMiniLM(EMBED_TEXT),
            ],
            RERANK_TEXT: [
                self._nvidia(RERANK_TEXT, NvidiaRerank, 0),
                NoLocalReranker(RERANK_TEXT),
            ],
            VECTOR_SEARCH: [qdrant, NumpyVectorStore(VECTOR_SEARCH)],
            VISION_DESCRIBE: [self._nvidia(VISION_DESCRIBE, NvidiaVision, 0)],
            OCR_IMAGE: [LocalTesseract(OCR_IMAGE)],
        }

    def _resolve(self, capability: str, chain: list[Provider]) -> Resolution:
        notes: list[str] = []
        for tier, provider in enumerate(chain):
            if self.offline and provider.tier_label == "hosted":
                notes.append(f"{provider.id}: skipped (offline)")
                continue
            try:
                ok, why = provider.available()
            except Exception as exc:  # noqa: BLE001 - a probe must never crash a run
                notes.append(f"{provider.id}: probe failed ({exc})")
                continue
            if ok:
                return Resolution(
                    capability=capability,
                    provider=provider,
                    tier=tier,
                    tier_label=provider.tier_label,
                    reason=why,
                    degraded=tier > 0,
                    notes=notes,
                )
            notes.append(f"{provider.id}: {why}")

        reason = "; ".join(notes) or "no provider available"
        return Resolution(
            capability=capability,
            provider=NullProvider(capability, reason),
            tier=99,
            tier_label="null",
            reason=reason,
            degraded=True,
            notes=notes,
        )

    # ---------------------------------------------------------------- #

    def get(self, capability: str) -> Provider:
        resolution = self.plan.get(capability)
        if resolution is None:
            return NullProvider(capability, f"unknown capability: {capability}")
        return resolution.provider

    def invoke(self, capability: str, **kwargs: Any) -> Result:
        """Call a capability. Never raises — an unavailable tier returns
        Unavailable and the agent reports SKIPPED with the reason."""
        provider = self.get(capability)
        try:
            return provider.invoke(**kwargs)
        except Exception as exc:  # noqa: BLE001
            return Unavailable(f"{type(exc).__name__}: {exc}", provider.id)

    def is_degraded(self, capability: str) -> bool:
        resolution = self.plan.get(capability)
        return resolution.degraded if resolution else True

    @property
    def degraded_capabilities(self) -> list[str]:
        return sorted(c for c, r in self.plan.items() if r.degraded)

    def summary(self) -> tuple[int, int, int]:
        preferred = sum(1 for r in self.plan.values() if r.tier == 0)
        unavailable = sum(1 for r in self.plan.values() if r.tier_label == "null")
        fallback = len(self.plan) - preferred - unavailable
        return preferred, fallback, unavailable

    def provenance(self) -> dict[str, Any]:
        """What goes into the certificate."""
        return {
            "capabilities": {c: r.to_json() for c, r in self.plan.items()},
            "degradedCapabilities": self.degraded_capabilities,
            "vendorUsage": usage_report(),
            "offlineMode": self.offline,
            "credentials": [s.to_json() for s in self.secrets.values()],
            "circuitState": {
                vendor: gov.breaker.state for vendor, gov in GOVERNORS.items()
            },
        }
