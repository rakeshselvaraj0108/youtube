"""The capability layer: secrets, governance, resolution."""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from preflight.providers.base import (
    BudgetExhausted,
    CircuitOpen,
    NullProvider,
    Served,
    Unavailable,
)
from preflight.providers.governor import (
    CircuitBreaker,
    Ledger,
    VendorGovernor,
    governor,
    reset_governors,
)
from preflight.providers.registry import (
    ASR_TRANSCRIBE,
    CHAT_REASONING,
    VECTOR_SEARCH,
    Registry,
)
from preflight.providers.secrets import (
    Secret,
    fingerprint,
    load_secrets,
    redact,
)

FAKE_KEY = "nvapi-abcdefghijklmnopqrstuvwxyz0123456789ABCD"  # pragma: allowlist secret


@pytest.fixture(autouse=True)
def clean_governors():
    reset_governors()
    yield
    reset_governors()


@pytest.fixture
def no_env(monkeypatch, tmp_path):
    """No credentials from any source, including the repo's own .env."""
    for name in ("NVIDIA_API_KEY", "ACOUSTID_API_KEY", "HUGGINGFACE_TOKEN",
                 "QDRANT_API_KEY", "QDRANT_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestRedaction:
    """A key must not survive into any artifact. This is the one test whose
    failure is a security incident rather than a bug."""

    def test_redacts_a_key_in_free_text(self):
        assert FAKE_KEY not in redact(f"Authorization: Bearer {FAKE_KEY}")
        assert "REDACTED" in redact(f"Bearer {FAKE_KEY}")

    # Every one of these is invented. The pragma exempts them from the
    # pre-commit credential scan, which is deliberately strict enough to match
    # them — an exemption visible in the diff beats a looser pattern that
    # would also miss the real thing.
    @pytest.mark.parametrize(
        "key",
        [
            "nvapi-" + "a" * 30,  # pragma: allowlist secret
            "hf_" + "b" * 30,  # pragma: allowlist secret
            "sk-proj-" + "c" * 30,  # pragma: allowlist secret
            "sk-" + "d" * 30,  # pragma: allowlist secret
            "AIza" + "E" * 32,  # pragma: allowlist secret
            "gsk_" + "f" * 30,  # pragma: allowlist secret
        ],
    )
    def test_redacts_every_known_vendor_shape(self, key):
        assert key not in redact(f"the key is {key} ok")

    def test_survives_embedding_in_json(self):
        blob = json.dumps({"error": f"401 for {FAKE_KEY}", "nested": [FAKE_KEY]})
        assert FAKE_KEY not in redact(blob)

    def test_fingerprint_never_reveals_the_whole_key(self):
        printed = fingerprint(FAKE_KEY)
        assert FAKE_KEY not in printed
        assert printed.startswith("nvapi-abc")
        assert str(len(FAKE_KEY)) in printed

    def test_fingerprint_handles_absent_and_short(self):
        assert fingerprint(None) == "—"
        assert "abc" not in fingerprint("abcdefg")[3:]

    def test_logging_filter_scrubs_records(self, caplog):
        from preflight.providers.secrets import RedactFilter

        logger = logging.getLogger("preflight.test.redaction")
        logger.addFilter(RedactFilter())
        with caplog.at_level(logging.INFO, logger=logger.name):
            logger.info("calling with %s", FAKE_KEY)
        assert FAKE_KEY not in caplog.text

    def test_secret_serialisation_omits_the_value(self):
        secret = Secret("NVIDIA_API_KEY", FAKE_KEY, "env", True)
        payload = json.dumps(secret.to_json())
        assert FAKE_KEY not in payload
        assert payload.count("REDACTED") == 0  # fingerprint, not redaction
        assert "nvapi-abc" in payload  # the shape is fine to show


class TestSecretResolution:
    def test_absent_when_nothing_is_configured(self, no_env):
        secrets = load_secrets()
        assert secrets["NVIDIA_API_KEY"].source == "absent"
        assert secrets["NVIDIA_API_KEY"].present is False

    def test_environment_is_read(self, no_env, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", FAKE_KEY)
        secret = load_secrets()["NVIDIA_API_KEY"]
        assert secret.source == "env"
        assert secret.usable

    def test_flag_overrides_environment(self, no_env, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", FAKE_KEY)
        secret = load_secrets({"NVIDIA_API_KEY": "nvapi-" + "z" * 30})["NVIDIA_API_KEY"]
        assert secret.source == "flag"

    def test_dotenv_is_read_when_environment_is_empty(self, no_env):
        (no_env / ".env").write_text(f"NVIDIA_API_KEY={FAKE_KEY}\n", encoding="utf-8")
        secret = load_secrets()["NVIDIA_API_KEY"]
        assert secret.source == "dotenv"
        assert secret.usable

    def test_malformed_key_is_rejected_before_any_call(self, no_env, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "not-a-key")
        secret = load_secrets()["NVIDIA_API_KEY"]
        assert secret.present
        assert secret.shape_ok is False
        assert secret.usable is False

    def test_wrong_vendor_key_is_named(self, no_env, monkeypatch):
        """A confusing 401 three minutes in becomes a clear message up front."""
        monkeypatch.setenv("NVIDIA_API_KEY", "sk-proj-" + "a" * 40)
        secret = load_secrets()["NVIDIA_API_KEY"]
        assert secret.shape_ok is False
        assert "OpenAI" in (secret.problem or "")
        assert "build.nvidia.com" in (secret.problem or "")

    def test_problem_message_never_contains_the_value(self, no_env, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "sk-proj-" + "a" * 40)
        secret = load_secrets()["NVIDIA_API_KEY"]
        assert "sk-proj-aaaa" not in (secret.problem or "")


class TestSharedBucket:
    """Six capabilities, one vendor, one limit."""

    def test_one_governor_per_vendor(self):
        first = governor("nvidia", rpm=30)
        second = governor("nvidia", rpm=30)
        assert first is second

    def test_every_nvidia_capability_shares_the_bucket(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", FAKE_KEY)
        registry = Registry(load_secrets())
        buckets = {
            id(resolution.provider.gov)
            for resolution in registry.plan.values()
            if getattr(resolution.provider, "id", "") == "nvidia"
        }
        # Six independent limiters at 30 RPM each is 180 RPM against a ~40 RPM
        # ceiling. There must be exactly one.
        assert len(buckets) == 1

    def test_acquire_paces_calls(self):
        elapsed = []
        gov = VendorGovernor(
            "test", rpm=60, clock=lambda: sum(elapsed), sleep=elapsed.append
        )
        for _ in range(4):
            gov.acquire("cap")
        # 60 rpm is one per second; three waits after the first call.
        assert sum(elapsed) == pytest.approx(3.0, abs=0.01)

    def test_budget_is_enforced_before_the_call(self):
        gov = VendorGovernor("test", rpm=600, call_budget=2, sleep=lambda _: None)
        gov.acquire("cap")
        gov.ledger.record("cap")
        gov.acquire("cap")
        gov.ledger.record("cap")
        with pytest.raises(BudgetExhausted):
            gov.acquire("cap")


class TestCircuitBreaker:
    def test_opens_after_the_threshold(self):
        breaker = CircuitBreaker("test", threshold=5)
        for _ in range(4):
            breaker.record(False)
        assert breaker.state == "CLOSED"
        breaker.record(False)
        assert breaker.state == "OPEN"

    def test_open_circuit_refuses_calls(self):
        breaker = CircuitBreaker("test", threshold=1)
        breaker.record(False)
        with pytest.raises(CircuitOpen):
            breaker.check()

    def test_half_opens_after_cooldown_then_closes_on_success(self):
        now = [0.0]
        breaker = CircuitBreaker("test", threshold=1, cooldown_s=60, clock=lambda: now[0])
        breaker.record(False)
        assert breaker.state == "OPEN"

        now[0] = 61.0
        breaker.check()
        assert breaker.state == "HALF_OPEN"

        breaker.record(True)
        assert breaker.state == "CLOSED"

    def test_success_resets_the_failure_count(self):
        breaker = CircuitBreaker("test", threshold=3)
        breaker.record(False)
        breaker.record(False)
        breaker.record(True)
        breaker.record(False)
        assert breaker.state == "CLOSED"

    def test_trip_opens_immediately(self):
        """402 and auth rejection do not wait for a threshold."""
        breaker = CircuitBreaker("test", threshold=5)
        breaker.trip()
        assert breaker.state == "OPEN"


class TestLedger:
    def test_records_usage_per_capability(self):
        ledger = Ledger("nvidia", budget=0)
        ledger.record("chat.reasoning", tokens_in=100, tokens_out=50, latency_ms=200)
        ledger.record("embed.text", tokens_in=20, latency_ms=80)
        assert ledger.calls == 2
        assert ledger.tokens_in == 120
        assert ledger.by_capability["chat.reasoning"]["tokens"] == 150

    def test_cache_hits_are_not_calls(self):
        """Counting a cache hit as a call overstates usage, and the telemetry
        strip has to show a true number."""
        ledger = Ledger("nvidia", budget=0)
        ledger.record_cached("chat.reasoning")
        assert ledger.calls == 0
        assert ledger.cached == 1

    def test_p95_latency(self):
        ledger = Ledger("nvidia", budget=0)
        for ms in range(1, 101):
            ledger.record("cap", latency_ms=ms)
        assert 94 <= ledger.p95_latency_ms <= 96

    def test_json_has_no_secrets(self):
        payload = json.dumps(Ledger("nvidia", budget=200).to_json())
        assert "nvapi" not in payload


class TestNullProvider:
    def test_answers_rather_than_raising(self):
        result = NullProvider("vision.describe", "no key").invoke(prompt="x")
        assert isinstance(result, Unavailable)
        assert not result
        assert "no key" in result.reason

    def test_is_falsy_so_callers_can_branch_plainly(self):
        assert not Unavailable("nope", "null")
        assert Served(value=1, provider="p")

    def test_never_fabricates_a_result(self):
        result = NullProvider("chat.reasoning").invoke(system="s", user="u")
        assert not hasattr(result, "value")


class TestRegistry:
    def test_resolves_every_capability(self, no_env):
        registry = Registry(load_secrets())
        assert len(registry.plan) == 8
        for resolution in registry.plan.values():
            assert resolution.provider is not None

    def test_no_key_still_serves_speech_locally(self, no_env):
        """Zero-key operation is the default path, not a fallback."""
        registry = Registry(load_secrets())
        assert registry.plan[ASR_TRANSCRIBE].provider.id == "local"

    def test_no_key_demotes_chat_to_null_without_raising(self, no_env):
        registry = Registry(load_secrets())
        resolution = registry.plan[CHAT_REASONING]
        assert resolution.tier_label == "null"
        assert resolution.degraded
        result = registry.invoke(CHAT_REASONING, system="s", user="u")
        assert isinstance(result, Unavailable)

    def test_offline_selects_no_hosted_provider_even_with_a_valid_key(
        self, no_env, monkeypatch
    ):
        monkeypatch.setenv("NVIDIA_API_KEY", FAKE_KEY)
        registry = Registry(load_secrets(), offline=True)
        hosted = [
            capability
            for capability, resolution in registry.plan.items()
            if resolution.tier_label == "hosted"
        ]
        assert hosted == []

    def test_vector_search_falls_back_to_numpy(self, no_env):
        registry = Registry(load_secrets())
        provider = registry.get(VECTOR_SEARCH)
        assert provider.id == "local"

        matrix = np.eye(4, dtype=np.float32)
        result = registry.invoke(VECTOR_SEARCH, matrix=matrix, query=matrix[2], top_k=2)
        assert result
        assert result.value[0][0] == 2

    def test_invoke_never_raises(self, no_env):
        """A provider that throws must not take the run with it."""

        class Exploding(NullProvider):
            def invoke(self, **kwargs):
                raise RuntimeError("boom")

        registry = Registry(load_secrets())
        registry.plan[CHAT_REASONING].provider = Exploding(CHAT_REASONING)
        result = registry.invoke(CHAT_REASONING)
        assert isinstance(result, Unavailable)
        assert "boom" in result.reason

    def test_unknown_capability_returns_null(self, no_env):
        registry = Registry(load_secrets())
        assert registry.get("does.not.exist").id == "null"


class TestProvenance:
    def test_records_every_capability_with_tier(self, no_env):
        provenance = Registry(load_secrets()).provenance()
        assert set(provenance["capabilities"]) == set(Registry(load_secrets()).plan)
        for entry in provenance["capabilities"].values():
            assert entry["tier"] in {"preferred", "fallback", "unavailable"}

    def test_lists_degraded_capabilities(self, no_env):
        provenance = Registry(load_secrets()).provenance()
        assert CHAT_REASONING in provenance["degradedCapabilities"]

    def test_contains_no_secret_values(self, no_env, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", FAKE_KEY)
        payload = json.dumps(Registry(load_secrets()).provenance())
        assert FAKE_KEY not in payload

    def test_reports_offline_mode(self, no_env):
        assert Registry(load_secrets(), offline=True).provenance()["offlineMode"] is True
