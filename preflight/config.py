"""Configuration.

Model IDs live here and are resolved at runtime, never hardcoded at a call
site. NVIDIA's catalogue rotates — entries get renamed and retired — so the
resolved IDs are logged into the certificate, which is what lets a report be
interpreted a year later.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env reader.

    A dependency on python-dotenv to parse KEY=VALUE would be one more install
    between a judge and a working demo.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        # Real environment wins over the file, so CI can override.
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


@dataclass
class Models:
    """Resolved model identifiers. Overridable by environment."""

    auditor: str = "meta/llama-3.3-70b-instruct"
    advocate: str = "nvidia/llama-3.3-nemotron-super-49b-v1"
    adjudicator: str = "nvidia/llama-3.3-nemotron-super-49b-v1"
    embed: str = "nvidia/nv-embedqa-e5-v5"
    asr_local: str = "base.en"

    @classmethod
    def from_env(cls) -> "Models":
        return cls(
            auditor=os.getenv("PREFLIGHT_MODEL_AUDITOR", cls.auditor),
            advocate=os.getenv("PREFLIGHT_MODEL_ADVOCATE", cls.advocate),
            adjudicator=os.getenv("PREFLIGHT_MODEL_ADJUDICATOR", cls.adjudicator),
            embed=os.getenv("PREFLIGHT_MODEL_EMBED", cls.embed),
            asr_local=os.getenv("PREFLIGHT_MODEL_ASR", cls.asr_local),
        )

    def to_json(self) -> dict[str, str]:
        return {
            "auditor": self.auditor,
            "advocate": self.advocate,
            "adjudicator": self.adjudicator,
            "embed": self.embed,
            "asr": self.asr_local,
        }


@dataclass
class Settings:
    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    rpm: int = 30
    offline: bool = False
    cache_dir: Path = Path(".preflight/cache")
    policy_dir: Path = Path("data/policy")
    models: Models = field(default_factory=Models)

    # Chunking. 30s windows with 5s overlap: a sentence beginning at 29s and
    # finishing at 33s must appear whole in at least one window, or the
    # adjudicator rules on half a sentence.
    chunk_ms: int = 30_000
    overlap_ms: int = 5_000

    # How long to wait on one hosted call before giving up and retrying.
    #
    # This was 120s, chosen when nothing had measured the service. A single
    # 64-token request against the free tier was then timed at 108s — twelve
    # seconds of headroom, on the smallest call the system can make. A real
    # AUDITOR batch carries eight windows plus their clause text and is far
    # larger, so the original value would time out, retry, and time out again
    # on exactly the calls that matter, while a trivial request passed and
    # made the configuration look sound.
    http_timeout_s: int = 300

    @property
    def online(self) -> bool:
        """True when a hosted model may actually be called."""
        return bool(self.api_key) and not self.offline

    @classmethod
    def load(cls, *, offline: bool | None = None) -> "Settings":
        _load_dotenv()
        env_offline = os.getenv("PREFLIGHT_OFFLINE", "0").strip() in {"1", "true", "yes"}
        return cls(
            api_key=(os.getenv("NVIDIA_API_KEY") or "").strip() or None,
            base_url=os.getenv("NVIDIA_BASE_URL", DEFAULT_BASE_URL).strip(),
            rpm=int(os.getenv("PREFLIGHT_RPM", "30")),
            offline=env_offline if offline is None else offline,
            cache_dir=Path(os.getenv("PREFLIGHT_CACHE_DIR", ".preflight/cache")),
            policy_dir=Path(os.getenv("PREFLIGHT_POLICY_DIR", "data/policy")),
            models=Models.from_env(),
            http_timeout_s=int(os.getenv("PREFLIGHT_HTTP_TIMEOUT", "300")),
        )

    def describe_mode(self) -> str:
        if self.offline:
            return "offline — local models only, no network"
        if not self.api_key:
            return "no API key — local models only, reduced policy coverage"
        return f"online — {self.base_url}, {self.rpm} rpm cap"
