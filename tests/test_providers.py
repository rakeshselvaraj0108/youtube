"""The capability layer: secrets, governance, resolution."""

from __future__ import annotations

import json
import time
import logging
from pathlib import Path

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
    VISION_DESCRIBE,
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


class TestVisionRepairsProseIntoJson:
    """The measured failure: seven of eight frames lost because the model
    answered a JSON request with a paragraph, leaving vision at 1% coverage
    against its 22% share of the analysis surface.

    The subtle part is why a plain retry is not the fix. Temperature is 0,
    so re-sending the identical request returns the identical paragraph. The
    second attempt has to say something the first did not — these tests pin
    that it does, because a repair that quietly degenerates into a retry
    would look identical in every log and recover nothing.
    """

    def _provider(self, monkeypatch, replies: list[str]):
        from preflight.providers.nvidia import NvidiaVision
        from preflight.providers.secrets import Secret

        provider = NvidiaVision(
            "vision.describe",
            "test/vision-model",
            Secret(
                name="NVIDIA_API_KEY",
                value="nvapi-TESTONLY0123456789abcdef",  # pragma: allowlist secret
                source="env",
                shape_ok=True,
            ),
        )
        sent: list[list[dict]] = []

        def fake_call(path, payload):
            sent.append(payload["messages"])
            content = replies[min(len(sent) - 1, len(replies) - 1)]
            return Served(
                value={"choices": [{"message": {"content": content}}]},
                provider="nvidia",
                tier=0,
                calls=1,
            )

        monkeypatch.setattr(provider, "_call", fake_call)
        return provider, sent

    def test_json_on_the_first_try_makes_one_call(self, monkeypatch):
        provider, sent = self._provider(monkeypatch, ['{"observations": []}'])
        result = provider.invoke(prompt="p", image_b64="x")
        assert result.ok
        assert len(sent) == 1

    def test_prose_then_json_recovers_the_frame(self, monkeypatch):
        provider, sent = self._provider(
            monkeypatch,
            ["The image presents a gradient background that transitions from",
             '{"observations": [{"label": "person"}]}'],
        )
        result = provider.invoke(prompt="p", image_b64="x")
        assert result.ok, "a recoverable frame was still lost"
        assert result.value["observations"][0]["label"] == "person"
        assert len(sent) == 2

    def test_the_repair_turn_is_not_a_repeat_of_the_first_request(self, monkeypatch):
        """The assertion that separates a repair from a retry."""
        provider, sent = self._provider(
            monkeypatch, ["prose about a gradient", '{"observations": []}']
        )
        provider.invoke(prompt="p", image_b64="x")
        first, second = sent
        assert second != first
        assert len(second) > len(first)
        assert second[-1]["role"] == "user"
        assert "JSON" in second[-1]["content"]

    def test_the_model_is_shown_its_own_answer(self, monkeypatch):
        """"Return JSON" alone is the instruction it already ignored, so the
        correction names what actually went wrong."""
        provider, sent = self._provider(
            monkeypatch, ["a paragraph about colour", '{"observations": []}']
        )
        provider.invoke(prompt="p", image_b64="x")
        echoed = [m for m in sent[1] if m["role"] == "assistant"]
        assert echoed and "paragraph about colour" in echoed[0]["content"]

    def test_two_paragraphs_give_up_rather_than_looping(self, monkeypatch):
        provider, sent = self._provider(monkeypatch, ["prose one", "prose two"])
        result = provider.invoke(prompt="p", image_b64="x")
        assert not result.ok
        assert "after repair" in result.reason
        assert len(sent) == 2, "repair must not loop"

    def test_a_failed_repair_stays_retryable_for_a_transient_cause(self, monkeypatch):
        provider, _ = self._provider(monkeypatch, ["prose", "prose"])
        assert provider.invoke(prompt="p", image_b64="x").retryable

    def test_a_fenced_json_reply_needs_no_repair(self, monkeypatch):
        """`extract_json` already strips fences — repairing that would spend
        a second call to reach the answer it already had."""
        provider, sent = self._provider(
            monkeypatch, ['```json\n{"observations": []}\n```']
        )
        assert provider.invoke(prompt="p", image_b64="x").ok
        assert len(sent) == 1


class TestRetryableIsHonoured:
    """`retryable` was set in three places and read in none.

    Vision-language models intermittently answer a JSON request with prose.
    `nvidia.py` labels that unparseable-but-retryable, correctly — and
    nothing retried. On a live run seven of eight frames were lost that way
    and the vision agent finished at 1% coverage against its 22% share of
    the analysis surface. It reported DEGRADED with an honest reason, which
    is precisely why it read as a slow day rather than a bug.
    """

    class Flaky:
        """Fails retryably once, then succeeds."""

        id = "stub"

        def __init__(self, fail_times: int = 1, retryable: bool = True) -> None:
            self.calls = 0
            self.fail_times = fail_times
            self.retryable = retryable

        def invoke(self, **kwargs):
            self.calls += 1
            if self.calls <= self.fail_times:
                return Unavailable("prose instead of JSON", "stub", self.retryable)
            return Served(value={"ok": True}, provider="stub", tier=0, calls=1)

    def _registry_with(self, monkeypatch, provider):
        registry = Registry(load_secrets())
        monkeypatch.setattr(registry, "get", lambda capability: provider)
        return registry

    def test_a_retryable_failure_is_retried_once_and_recovers(self, no_env, monkeypatch):
        provider = self.Flaky(fail_times=1)
        result = self._registry_with(monkeypatch, provider).invoke(VISION_DESCRIBE)
        assert provider.calls == 2
        assert result.ok

    def test_a_non_retryable_failure_is_not_retried(self, no_env, monkeypatch):
        """A missing binary will not become present on a second try."""
        provider = self.Flaky(fail_times=1, retryable=False)
        result = self._registry_with(monkeypatch, provider).invoke(VISION_DESCRIBE)
        assert provider.calls == 1
        assert not result.ok

    def test_retrying_is_bounded(self, no_env, monkeypatch):
        """A second prose answer means the model will not produce JSON for
        this frame; a third call at ~80s buys nothing."""
        provider = self.Flaky(fail_times=99)
        result = self._registry_with(monkeypatch, provider).invoke(VISION_DESCRIBE)
        assert provider.calls == 2
        assert not result.ok

    def test_a_first_time_success_makes_no_second_call(self, no_env, monkeypatch):
        provider = self.Flaky(fail_times=0)
        self._registry_with(monkeypatch, provider).invoke(VISION_DESCRIBE)
        assert provider.calls == 1


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


class TestRequestDeadline:
    """A slow-trickling response must not hold the pipeline open forever.

    `urllib`'s `timeout=` governs one socket operation, not the exchange. A
    server that dribbles a byte every few seconds resets it on every chunk,
    so a nominal 120s ceiling becomes unbounded. Observed live: one vision
    request held a real analysis for over eight minutes with no error raised
    and nothing to distinguish it from a model that was merely slow. The
    whole run simply stopped.

    The deadline below is what makes a long-video run bounded regardless of
    what the far end does.
    """

    def _trickling_server(self):
        """A server that sends headers, then bytes forever, very slowly."""
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                # No Content-Length: chunked-ish stream we never finish.
                self.end_headers()
                try:
                    while True:
                        self.wfile.write(b" ")
                        self.wfile.flush()
                        time.sleep(0.05)
                except Exception:
                    pass

            def log_message(self, *a):  # silence
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server

    def test_a_trickling_response_hits_the_deadline(self, monkeypatch):
        from preflight.providers import nvidia as nv

        server = self._trickling_server()
        try:
            monkeypatch.setattr(nv, "REQUEST_DEADLINE_S", 2)
            provider = nv.NvidiaChat(
                CHAT_REASONING, "m", Secret("NVIDIA_API_KEY", "nvapi-" + "x" * 24, "env", True)
            )
            provider.base_url = f"http://127.0.0.1:{server.server_port}"

            started = time.monotonic()
            with pytest.raises(TimeoutError):
                provider._post("/chat/completions", {"model": "m"})
            elapsed = time.monotonic() - started

            # Bounded by the deadline, not by the far end's patience.
            assert elapsed < 10, f"deadline not enforced: took {elapsed:.1f}s"
        finally:
            server.shutdown()

    def test_the_deadline_surfaces_as_a_retryable_failure_not_a_crash(
        self, monkeypatch
    ):
        """A stuck call must degrade the modality, never take the run down."""
        from preflight.providers import nvidia as nv

        server = self._trickling_server()
        try:
            monkeypatch.setattr(nv, "REQUEST_DEADLINE_S", 1)
            monkeypatch.setattr(nv, "MAX_ATTEMPTS", 2)
            provider = nv.NvidiaChat(
                CHAT_REASONING, "m", Secret("NVIDIA_API_KEY", "nvapi-" + "x" * 24, "env", True)
            )
            provider.base_url = f"http://127.0.0.1:{server.server_port}"

            result = provider._call("/chat/completions", {"model": "m"})
            assert not result
            assert result.retryable
            assert "unreachable" in result.reason
        finally:
            server.shutdown()

    def test_a_prompt_response_is_unaffected(self, monkeypatch):
        """The deadline must not penalise a server that answers normally."""
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        payload = json.dumps(
            {"choices": [{"message": {"content": "ok"}}], "usage": {}}
        ).encode()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            from preflight.providers import nvidia as nv

            provider = nv.NvidiaChat(
                CHAT_REASONING, "m", Secret("NVIDIA_API_KEY", "nvapi-" + "x" * 24, "env", True)
            )
            provider.base_url = f"http://127.0.0.1:{server.server_port}"
            raw = provider._post("/chat/completions", {"model": "m"})
            assert raw["choices"][0]["message"]["content"] == "ok"
        finally:
            server.shutdown()


class TestUnreachableVendorFailsFast:
    """A vendor down for the whole run must not cost the whole run.

    Measured live: one policy-retrieval stage against a genuinely
    unreachable NVIDIA endpoint cost 1536.7s (25.6 minutes) on a video whose
    entire runtime should have been a few minutes — almost all of it retries
    against a host that was never going to answer, because the circuit
    breaker's failure threshold could not be reached until a call had
    already exhausted its own five-attempt, 180s-each budget. The breaker
    was protecting nothing: by the time it could trip, the damage for that
    call was already done.
    """

    def _unreachable_server(self):
        """A listening socket that refuses every connection outright —
        the fast transport failure, not the slow-trickle one covered by
        TestRequestDeadline above."""
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()  # nothing is listening on this port now
        return port

    def test_one_call_against_a_dead_host_is_bounded_well_under_the_old_cost(
        self, monkeypatch
    ):
        from preflight.providers import nvidia as nv

        monkeypatch.setattr(nv, "REQUEST_DEADLINE_S", 1)
        monkeypatch.setattr(nv, "TRANSPORT_MAX_ATTEMPTS", 2)
        port = self._unreachable_server()
        provider = nv.NvidiaChat(
            CHAT_REASONING, "m", Secret("NVIDIA_API_KEY", "nvapi-" + "x" * 24, "env", True)
        )
        provider.base_url = f"http://127.0.0.1:{port}"

        started = time.monotonic()
        result = provider._call("/chat/completions", {"model": "m"})
        elapsed = time.monotonic() - started

        assert not result
        assert result.retryable
        # Two attempts, not five — this is the actual fix under test.
        assert elapsed < 10, f"took {elapsed:.1f}s for what should be ~2 fast attempts"

    def test_the_transport_retry_budget_is_smaller_than_the_http_status_budget(self):
        """HTTP 429/5xx means the vendor is reachable and answered "try
        again" — retrying the full budget there is correct, because the
        same endpoint moments later often succeeds. A transport failure
        means nothing answered at all, and gets a smaller budget on
        purpose."""
        from preflight.providers import nvidia as nv

        assert nv.TRANSPORT_MAX_ATTEMPTS < nv.MAX_ATTEMPTS

    def test_the_breaker_trips_before_a_single_calls_retry_budget_used_to_end(self):
        """The threshold that made the breaker protect nothing: it could
        not open until a call's own retries were already exhausted. Now it
        must be reachable within one transport retry budget's worth of
        failures, so the breaker can actually cut off a bad run early."""
        from preflight.providers import governor as gov
        from preflight.providers import nvidia as nv

        assert gov.FAILURE_THRESHOLD <= nv.TRANSPORT_MAX_ATTEMPTS + 1

    def test_worst_case_before_the_run_is_protected_beats_the_old_regression(self):
        """The exact number that mattered live: 25.6 minutes on one stage.
        The new worst case, before the breaker takes over for the rest of
        the run, must be a small fraction of that."""
        from preflight.providers import governor as gov
        from preflight.providers import nvidia as nv

        worst_case_s = (gov.FAILURE_THRESHOLD / nv.TRANSPORT_MAX_ATTEMPTS) * (
            nv.TRANSPORT_MAX_ATTEMPTS * nv.REQUEST_DEADLINE_S
        )
        observed_regression_s = 1536.7
        assert worst_case_s < observed_regression_s / 4


class TestHfCacheResolvesTheRunningUsersHome:
    """A model baked into a Docker image at build time (as root, HOME=/root)
    and served at runtime as a different unprivileged user (HOME=/home/x)
    silently reports "not cached": `_hf_cached` trusts `Path.home()`, which
    resolves per-process, not per-image. Nothing raised — capability
    resolution just fell back to the hosted tier, which reads as "working"
    right up until someone checks which tier actually served a request. The
    real fix is a Dockerfile fix (copy the cache into the runtime user's own
    HOME, not just chown /app); this locks in the function's actual
    contract so the failure mode is at least covered by something."""

    def test_a_cache_under_a_different_home_is_invisible(self, tmp_path, monkeypatch):
        from preflight.providers.local import _hf_cached

        other_home = tmp_path / "root"
        cache = other_home / ".cache" / "huggingface" / "hub"
        cache.mkdir(parents=True)
        (cache / "models--Systran--faster-whisper-base.en").mkdir()

        monkeypatch.setattr(Path, "home", lambda: tmp_path / "not-root")
        assert _hf_cached("faster-whisper-base.en") is False

    def test_a_cache_under_the_running_users_home_is_found(self, tmp_path, monkeypatch):
        from preflight.providers.local import _hf_cached

        cache = tmp_path / ".cache" / "huggingface" / "hub"
        cache.mkdir(parents=True)
        (cache / "models--Systran--faster-whisper-base.en").mkdir()

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert _hf_cached("faster-whisper-base.en") is True
