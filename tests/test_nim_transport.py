"""`NimClient`'s HTTP layer must not be able to stall a run for 25 minutes.

This is the code path that actually produced the regression: a live audit
of a real 87-second video spent 1536.7s (25.6 minutes) — 76% of the run —
on the policy/triad stage, entirely retries against an NVIDIA endpoint that
never finished answering. `preflight/providers/nvidia.py` was fixed for the
same failure class first, but `NimClient` here is a second, independent
HTTP implementation that predates the provider registry and shares no code
with it — the fix there did nothing for this one, confirmed by a second
live run landing at 1539.2s, statistically identical to the first.

These tests exercise the actual bug: a socket that never returns a full
response, and one that refuses the connection outright.
"""

from __future__ import annotations

import http.server
import socket
import threading
import time

import pytest

from preflight import cas
from preflight.agents.nim import NimClient, NimUnavailable
from preflight.config import Settings


def settings(base_url: str) -> Settings:
    return Settings(api_key="nvapi-" + "x" * 24, offline=False, base_url=base_url, rpm=6000)


class TestTricklingResponseIsBounded:
    """The read-side half of the bug: `read()` blocks until EOF, so a
    response that dribbles bytes never lets the deadline check run."""

    def _trickling_server(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                try:
                    while True:
                        self.wfile.write(b" ")
                        self.wfile.flush()
                        time.sleep(0.05)
                except Exception:
                    pass

            def log_message(self, *a):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    def test_a_trickling_response_hits_the_deadline_not_a_hang(
        self, monkeypatch, tmp_path
    ):
        import preflight.agents.nim as nim_mod

        server = self._trickling_server()
        try:
            monkeypatch.setattr(nim_mod, "REQUEST_DEADLINE_S", 1)
            monkeypatch.setattr(nim_mod, "TRANSPORT_MAX_ATTEMPTS", 1)
            client = NimClient(
                settings(f"http://127.0.0.1:{server.server_port}"),
                cas.Store(tmp_path / "cache"),
            )

            started = time.monotonic()
            with pytest.raises(NimUnavailable):
                client._post_with_retries("/chat/completions", {"model": "m"})
            elapsed = time.monotonic() - started

            assert elapsed < 10, f"took {elapsed:.1f}s — the deadline did not bound it"
        finally:
            server.shutdown()

    def test_a_normal_response_is_unaffected(self, tmp_path):
        import json as json_mod

        payload = json_mod.dumps(
            {"choices": [{"message": {"content": "{}"}}], "usage": {}}
        ).encode()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            client = NimClient(
                settings(f"http://127.0.0.1:{server.server_port}"),
                cas.Store(tmp_path / "cache"),
            )
            raw = client._post_with_retries("/chat/completions", {"model": "m"})
            assert raw["choices"][0]["message"]["content"] == "{}"
        finally:
            server.shutdown()


class TestUnreachableHostFailsFast:
    """The retry-budget half of the bug: five attempts at up to 300s each
    against a host that was never going to answer."""

    def _dead_port(self) -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        return port

    def test_one_call_against_a_dead_host_is_bounded(self, monkeypatch, tmp_path):
        import preflight.agents.nim as nim_mod

        monkeypatch.setattr(nim_mod, "REQUEST_DEADLINE_S", 1)
        monkeypatch.setattr(nim_mod, "TRANSPORT_MAX_ATTEMPTS", 2)
        client = NimClient(
            settings(f"http://127.0.0.1:{self._dead_port()}"),
            cas.Store(tmp_path / "cache"),
        )

        started = time.monotonic()
        with pytest.raises(NimUnavailable):
            client._post_with_retries("/chat/completions", {"model": "m"})
        elapsed = time.monotonic() - started

        # Two fast-failing attempts plus jittered backoff, not five.
        assert elapsed < 10, f"took {elapsed:.1f}s for what should be ~2 fast attempts"

    def test_the_transport_budget_is_smaller_than_the_http_status_budget(self):
        """HTTP 429/5xx means the vendor answered and said try again — the
        full budget is correct there. A transport failure means nothing
        answered, and gets a smaller one on purpose."""
        import preflight.agents.nim as nim_mod

        assert nim_mod.TRANSPORT_MAX_ATTEMPTS < nim_mod.MAX_ATTEMPTS

    def test_the_error_names_how_many_attempts_actually_happened(
        self, monkeypatch, tmp_path
    ):
        """Not a hardcoded MAX_ATTEMPTS in the message when the loop exited
        early on the transport-failure branch — the reader should be able
        to tell a fast transport failure from an exhausted HTTP-retry one."""
        import preflight.agents.nim as nim_mod

        monkeypatch.setattr(nim_mod, "REQUEST_DEADLINE_S", 1)
        monkeypatch.setattr(nim_mod, "TRANSPORT_MAX_ATTEMPTS", 2)
        client = NimClient(
            settings(f"http://127.0.0.1:{self._dead_port()}"),
            cas.Store(tmp_path / "cache"),
        )
        with pytest.raises(NimUnavailable) as caught:
            client._post_with_retries("/chat/completions", {"model": "m"})
        assert "2 attempts" in str(caught.value)

    def test_worst_case_beats_the_observed_regression_by_a_wide_margin(self):
        """The number that mattered live: 1536.7s. The new worst case for
        one call must be a small fraction of it."""
        import preflight.agents.nim as nim_mod

        worst_case_s = nim_mod.TRANSPORT_MAX_ATTEMPTS * nim_mod.REQUEST_DEADLINE_S
        observed_regression_s = 1536.7
        assert worst_case_s < observed_regression_s / 5


class TestCachingStillWorks:
    """The fix must not disturb the caching contract: a failed call is
    never committed, and a successful one still short-circuits a repeat."""

    def _served_once(self, tmp_path):
        import json as json_mod

        calls = {"n": 0}
        payload = json_mod.dumps(
            {"choices": [{"message": {"content": '{"ok": true}'}}], "usage": {}}
        ).encode()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                calls["n"] += 1
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, calls

    def test_a_repeat_prompt_hits_the_cache_not_the_network(self, tmp_path):
        server, calls = self._served_once(tmp_path)
        try:
            client = NimClient(
                settings(f"http://127.0.0.1:{server.server_port}"),
                cas.Store(tmp_path / "cache"),
            )
            first = client.chat_json(model="m", system="s", user="u")
            second = client.chat_json(model="m", system="s", user="u")
            assert first == second == {"ok": True}
            assert calls["n"] == 1
        finally:
            server.shutdown()
