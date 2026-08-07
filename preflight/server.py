"""HTTP API — the surface the Command Deck talks to.

Built on `http.server` rather than FastAPI on purpose. This project ships
numpy and typer and nothing else heavy; adding a web framework and an ASGI
server to serve six JSON routes would be the largest dependency in the tree,
for a surface that fits in one file. The stdlib server is threaded, which is
all the concurrency six routes need when the expensive work is a subprocess.

Routes:

    GET  /api/health          engine, ffmpeg, capability plan
    GET  /api/agents          the roster, as `preflight agents` prints it
    GET  /api/runs            every report on disk, newest first
    GET  /api/runs/{id}       one full report
    POST /api/analyze         run the pipeline on a video, return the report

Two rules govern everything here.

**The engine is not reimplemented.** Every route calls the same
`run_perception` / `build_report` the CLI calls. A server with its own idea
of how analysis works is a second implementation to keep in sync, and it
would drift the moment either side changed.

**Nothing leaves this process that a credential could ride out on.** Reports
go through the same redaction sweep `preflight check` uses before writing,
and `/api/health` reports capability *names and tiers*, never the secrets
that resolved them.
"""

from __future__ import annotations

import json
import re
import threading
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from preflight import __version__, cas, ffmpeg
from preflight.agents.roster import load_roster
from preflight.budget import CallBudget
from preflight.config import Settings
from preflight.pipeline import run_perception
from preflight.plan import build_plan
from preflight.report.build import build_report

# Reports are written here by the server and listed from here. The CLI's own
# --out directories are separate; a run triggered from the deck lands in one
# predictable place so the deck can find it again.
RUNS_DIR = Path(".preflight/runs")

# A run id is a hash prefix plus a timestamp. Anchored and character-limited
# so a crafted id cannot walk out of RUNS_DIR — `Path(RUNS_DIR) / user_input`
# is a directory traversal waiting to happen otherwise.
RUN_ID = re.compile(r"^[A-Za-z0-9_.-]{1,120}$")

MAX_BODY_BYTES = 1 << 20


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message
        super().__init__(message)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def health() -> dict[str, Any]:
    settings = Settings.load()
    try:
        from preflight.providers.registry import Registry

        registry = Registry(offline=settings.offline)
        preferred, fallback, unavailable = registry.summary()
        capabilities = {
            name: {"tier": r.tier_label, "degraded": r.degraded}
            for name, r in registry.plan.items()
        }
    except Exception:  # noqa: BLE001 - health must answer even when resolution fails
        preferred = fallback = unavailable = 0
        capabilities = {}

    return {
        "status": "ok",
        "engineVersion": __version__,
        "ffmpeg": ffmpeg.version(),
        "ffmpegAvailable": ffmpeg.available(),
        # Whether a key is configured — never the key, never its length.
        "online": settings.online,
        "capabilities": capabilities,
        "capabilitySummary": {
            "preferred": preferred,
            "fallback": fallback,
            "unavailable": unavailable,
        },
        "time": _now(),
    }


def agents() -> dict[str, Any]:
    roster = load_roster()
    return {
        "digest": roster.digest,
        "agents": [
            {
                "id": spec.agent_id,
                "codename": spec.codename,
                "kind": spec.kind,
                "capability": spec.model_capability,
                "dependsOn": list(spec.parents),
                "implemented": spec.implemented,
            }
            for spec in roster.ordered
        ],
        "problems": roster.validate(),
    }


def _summarise(report: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "id": run_id,
        "filename": report.get("video", {}).get("filename", "?"),
        "durationMs": report.get("video", {}).get("durationMs", 0),
        "overall": report.get("scores", {}).get("overall"),
        "verdict": report.get("scores", {}).get("verdict"),
        "weakest": report.get("scores", {}).get("weakest"),
        "findingCount": len(report.get("findings", [])),
        "coverage": report.get("meta", {}).get("coverage"),
        "analyzedAt": report.get("meta", {}).get("analyzedAt"),
    }


def list_runs() -> dict[str, Any]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for path in RUNS_DIR.glob("*/report.json"):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A half-written or hand-edited report must not take the listing
            # down with it — the other runs are still readable.
            continue
        rows.append(_summarise(report, path.parent.name))
    rows.sort(key=lambda r: r.get("analyzedAt") or "", reverse=True)
    return {"runs": rows}


def read_run(run_id: str) -> dict[str, Any]:
    if not RUN_ID.match(run_id):
        raise ApiError(400, "malformed run id")
    path = RUNS_DIR / run_id / "report.json"
    # Resolve and confirm containment: the regex already forbids separators,
    # and this proves it rather than trusting it.
    try:
        resolved = path.resolve()
        resolved.relative_to(RUNS_DIR.resolve())
    except (OSError, ValueError) as exc:
        raise ApiError(400, "run id escapes the runs directory") from exc
    if not resolved.is_file():
        raise ApiError(404, f"no run {run_id}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the real pipeline. Same code path as `preflight check`."""
    raw = str(payload.get("video", "")).strip()
    if not raw:
        raise ApiError(400, "body must carry a 'video' path")

    video = Path(raw)
    if not video.is_file():
        raise ApiError(404, f"no such file: {video}")
    if not ffmpeg.available():
        raise ApiError(503, "ffmpeg and ffprobe are required")

    offline = bool(payload.get("offline", False))
    ceiling = payload.get("budget")
    settings = Settings.load(offline=True) if offline else Settings.load()
    budget = CallBudget(ceiling=int(ceiling) if ceiling else None)

    result = run_perception(video, cas.Store(settings.cache_dir), settings=settings, budget=budget)
    bundle = build_report(
        result,
        policy_version=result.corpus.version if result.corpus else "unknown",
        embed_media=True,
        strategy=str(payload.get("strategy") or "") or None,
        chunk_ms=settings.chunk_ms,
        overlap_ms=settings.overlap_ms,
    )

    run_id = f"{cas.prefixed(result.ingested.video_hash)[3:15]}-{int(datetime.now().timestamp())}"
    directory = RUNS_DIR / run_id
    directory.mkdir(parents=True, exist_ok=True)

    # Same redaction gate the CLI applies before anything becomes durable.
    text = json.dumps(bundle.report, indent=2, ensure_ascii=False)
    key = (settings.api_key or "").strip()
    if len(key) >= 12 and key in text:
        raise ApiError(500, "refusing to persist a report containing the API key")
    (directory / "report.json").write_text(text, encoding="utf-8")

    return {"id": run_id, "report": bundle.report}


def plan_for(payload: dict[str, Any]) -> dict[str, Any]:
    """The decomposition plan alone — cheap, and answers "what will this cost"
    without spending anything."""
    from preflight.ingest.probe import probe_video

    raw = str(payload.get("video", "")).strip()
    if not raw:
        raise ApiError(400, "body must carry a 'video' path")
    video = Path(raw)
    if not video.is_file():
        raise ApiError(404, f"no such file: {video}")

    settings = Settings.load()
    meta = probe_video(video)
    plan = build_plan(
        meta.durationMs, chunk_ms=settings.chunk_ms, overlap_ms=settings.overlap_ms
    )
    return {"video": meta.to_json(), "plan": plan.to_json(), "describe": plan.describe()}


class Handler(BaseHTTPRequestHandler):
    server_version = f"preflight/{__version__}"

    # Serialises analysis. The pipeline shells out to ffmpeg and holds a rate
    # limiter; two concurrent runs would fight over both and interleave their
    # progress into one terminal. Health and listing stay responsive because
    # the server is threaded — only this lock is contended.
    _analysis_lock = threading.Lock()

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        print(f"  {self.address_string()} {fmt % args}", flush=True)

    # ---- plumbing ---------------------------------------------------- #

    def _cors(self) -> None:
        # The deck runs on a different origin in dev (Vite on 5173). Reads are
        # public; this server binds loopback by default and holds no session,
        # so there is no cookie for a hostile page to ride.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _send(self, status: int, body: dict[str, Any]) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self._cors()
        self.end_headers()
        self.wfile.write(raw)

    def _fail(self, exc: Exception) -> None:
        if isinstance(exc, ApiError):
            self._send(exc.status, {"error": exc.message})
            return
        # Log the trace locally, return the type only. A stack trace over the
        # wire names paths and modules the caller has no business knowing.
        traceback.print_exc()
        self._send(500, {"error": f"{type(exc).__name__}"})

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise ApiError(413, "body too large")
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except ValueError as exc:
            raise ApiError(400, "body is not valid JSON") from exc

    # ---- routes ------------------------------------------------------- #

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        try:
            if path in ("/api/health", "/api"):
                self._send(200, health())
            elif path == "/api/agents":
                self._send(200, agents())
            elif path == "/api/runs":
                self._send(200, list_runs())
            elif path.startswith("/api/runs/"):
                self._send(200, read_run(path[len("/api/runs/"):]))
            else:
                self._send(404, {"error": f"no route {path}"})
        except Exception as exc:  # noqa: BLE001
            self._fail(exc)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        try:
            if path == "/api/plan":
                self._send(200, plan_for(self._body()))
            elif path == "/api/analyze":
                payload = self._body()
                if not self._analysis_lock.acquire(blocking=False):
                    raise ApiError(409, "an analysis is already running")
                try:
                    self._send(200, analyze(payload))
                finally:
                    self._analysis_lock.release()
            else:
                self._send(404, {"error": f"no route {path}"})
        except Exception as exc:  # noqa: BLE001
            self._fail(exc)


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    print(f"  PREFLIGHT API {__version__}  http://{host}:{port}/api/health")
    print(f"  runs -> {RUNS_DIR.resolve()}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        httpd.server_close()
