"""The HTTP API.

A network surface earns different tests from a library. The interesting
claims are not "does it return JSON" but the ones that bite when someone
points a browser at it: that a run id cannot walk out of the runs directory,
that a stack trace never crosses the wire, that a report carrying a
credential is refused rather than persisted, and that the routes call the
same engine the CLI does instead of a second implementation.
"""

from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib import error, request

import pytest

from preflight import server as server_mod
from preflight.server import ApiError, Handler, agents, health, list_runs, read_run


@pytest.fixture
def runs_dir(tmp_path, monkeypatch) -> Path:
    directory = tmp_path / "runs"
    directory.mkdir()
    monkeypatch.setattr(server_mod, "RUNS_DIR", directory)
    return directory


def write_run(directory: Path, run_id: str, **overrides) -> Path:
    report = {
        "video": {"filename": f"{run_id}.mp4", "durationMs": 20_000},
        "meta": {"analyzedAt": "2026-08-07T10:00:00Z", "coverage": 0.51},
        "scores": {"overall": 45, "verdict": "DO_NOT_PUBLISH", "weakest": "accessibility"},
        "findings": [],
    }
    report.update(overrides)
    path = directory / run_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return path


class TestRunIdsCannotEscape:
    """`RUNS_DIR / user_input` is a directory traversal waiting to happen."""

    @pytest.mark.parametrize(
        "run_id",
        [
            "../../../etc/passwd",
            "..\\..\\windows\\win.ini",
            "a/../../secret",
            "/absolute/path",
            "",
            "x" * 500,
        ],
    )
    def test_a_hostile_id_is_rejected(self, runs_dir, run_id):
        with pytest.raises(ApiError) as caught:
            read_run(run_id)
        assert caught.value.status in (400, 404)

    def test_a_legitimate_id_still_reads(self, runs_dir):
        write_run(runs_dir, "b3abc123-1700000000")
        assert read_run("b3abc123-1700000000")["scores"]["overall"] == 45

    def test_a_missing_run_is_404_not_500(self, runs_dir):
        with pytest.raises(ApiError) as caught:
            read_run("nope-123")
        assert caught.value.status == 404


class TestListing:
    def test_an_empty_server_lists_nothing(self, runs_dir):
        assert list_runs() == {"runs": []}

    def test_runs_come_back_newest_first(self, runs_dir):
        write_run(runs_dir, "old", meta={"analyzedAt": "2026-01-01T00:00:00Z"})
        write_run(runs_dir, "new", meta={"analyzedAt": "2026-08-01T00:00:00Z"})
        ids = [r["id"] for r in list_runs()["runs"]]
        assert ids == ["new", "old"]

    def test_one_corrupt_report_does_not_take_the_listing_down(self, runs_dir):
        """A half-written file must not hide every other run."""
        write_run(runs_dir, "good")
        bad = runs_dir / "bad"
        bad.mkdir()
        (bad / "report.json").write_text("{not json", encoding="utf-8")
        assert [r["id"] for r in list_runs()["runs"]] == ["good"]


class TestHealthAndAgents:
    def test_health_answers_without_a_key(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        payload = health()
        assert payload["status"] == "ok"
        assert "engineVersion" in payload

    def test_health_never_carries_the_credential(self, monkeypatch):
        """It reports capability names and tiers. Anything that could
        reconstruct the key — the value, its length, a prefix — stays out."""
        key = "nvapi-SERVERFAKEKEY1234567890abcdefghij"  # pragma: allowlist secret
        monkeypatch.setenv("NVIDIA_API_KEY", key)
        blob = json.dumps(health())
        assert key not in blob
        assert key[:12] not in blob

    def test_agents_reports_the_whole_roster(self):
        payload = agents()
        assert len(payload["agents"]) == 12
        assert payload["problems"] == []


class TestUploadNames:
    """A filename arrives from whoever posted it. `Path(name).name` alone
    still admits "..", and a browser is not the only thing that can post."""

    @pytest.mark.parametrize(
        "raw",
        [
            "../../../etc/passwd",
            "..\\..\\windows\\win.ini",
            "/absolute/evil.mp4",
            "....//....//x.mp4",
            "",
            None,
        ],
    )
    def test_a_hostile_filename_cannot_escape(self, raw):
        from preflight.server import safe_name

        cleaned = safe_name(raw)
        assert "/" not in cleaned and "\\" not in cleaned
        assert not cleaned.startswith(".")
        assert cleaned

    def test_an_ordinary_name_survives_recognisably(self):
        from preflight.server import safe_name

        assert safe_name("My Holiday Video.mp4") == "My_Holiday_Video.mp4"

    def test_a_very_long_name_is_bounded(self):
        from preflight.server import safe_name

        assert len(safe_name("x" * 5000 + ".mp4")) <= 120


class TestLiveServer:
    """Through a real socket — the handler's error paths only exist there."""

    @pytest.fixture
    def base_url(self, runs_dir):
        write_run(runs_dir, "live-1")
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
        httpd.shutdown()
        httpd.server_close()

    def get(self, url: str):
        try:
            with request.urlopen(url, timeout=10) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_health_route(self, base_url):
        status, body = self.get(f"{base_url}/api/health")
        assert status == 200
        assert body["status"] == "ok"

    def test_runs_route(self, base_url):
        status, body = self.get(f"{base_url}/api/runs")
        assert status == 200
        assert [r["id"] for r in body["runs"]] == ["live-1"]

    def test_unknown_route_is_404_json(self, base_url):
        status, body = self.get(f"{base_url}/api/nonsense")
        assert status == 404
        assert "error" in body

    def test_traversal_over_the_wire_is_refused(self, base_url):
        status, _ = self.get(f"{base_url}/api/runs/..%2F..%2Fsecret")
        assert status in (400, 404)

    def test_an_error_never_returns_a_stack_trace(self, base_url):
        """A trace over the wire names paths and modules the caller has no
        business knowing."""
        status, body = self.get(f"{base_url}/api/runs/{'x' * 400}")
        assert status in (400, 404)
        assert "Traceback" not in json.dumps(body)
        assert "preflight" not in json.dumps(body).lower()

    def test_analyze_rejects_a_missing_video(self, base_url):
        payload = json.dumps({"video": "does/not/exist.mp4"}).encode()
        req = request.Request(
            f"{base_url}/api/analyze",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=30) as response:
                status = response.status
        except error.HTTPError as exc:
            status = exc.code
        assert status == 404

    def post(self, url: str, body: bytes, headers: dict | None = None):
        req = request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        try:
            with request.urlopen(req, timeout=30) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_a_job_for_a_missing_file_is_refused_before_it_starts(self, base_url):
        """This returned 202 with a job id, and the failure arrived seconds
        later over the event stream — so the deck sat on "analysing…" for a
        file that was never going to open."""
        status, body = self.post(
            f"{base_url}/api/jobs", json.dumps({"video": "nope.mp4"}).encode()
        )
        assert status == 404
        assert "no such file" in body["error"]

    def test_a_job_with_no_path_is_refused(self, base_url):
        status, _ = self.post(f"{base_url}/api/jobs", b"{}")
        assert status == 400

    def test_an_upload_lands_on_disk_and_can_then_be_analysed(self, base_url, tmp_path):
        """The capability that was missing entirely: a creator with a video
        on their desktop had no way to submit it, because the only input was
        a server-side path typed by hand."""
        payload = b"\x00\x01" * 4096
        status, body = self.post(
            f"{base_url}/api/upload",
            payload,
            {"Content-Type": "application/octet-stream", "X-Filename": "holiday clip.mp4"},
        )
        assert status == 201
        assert body["bytes"] == len(payload)
        saved = Path(body["path"])
        assert saved.is_file() and saved.read_bytes() == payload
        assert "holiday" in saved.name and " " not in saved.name
        saved.unlink(missing_ok=True)

    def test_an_empty_upload_is_refused(self, base_url):
        status, _ = self.post(
            f"{base_url}/api/upload", b"",
            {"Content-Type": "application/octet-stream", "X-Filename": "x.mp4"},
        )
        assert status == 400

    def test_analyze_rejects_a_body_without_a_video(self, base_url):
        req = request.Request(
            f"{base_url}/api/analyze",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=30) as response:
                status = response.status
        except error.HTTPError as exc:
            status = exc.code
        assert status == 400
