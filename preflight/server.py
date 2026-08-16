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
import cgi
import mimetypes
import os
import re
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from preflight import (
    __version__,
    cas,
    certificate as certificate_mod,
    evidence as evidence_mod,
    ffmpeg,
    lifecycle,
    lineage,
    telemetry,
)
from preflight.agents.roster import load_roster
from preflight.budget import CallBudget
from preflight.config import Settings
from preflight.pipeline import run_perception
from preflight.plan import build_plan
from preflight.report.build import build_report
from preflight.verify import compare, prediction_outcome

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


class Job:
    """One analysis, running on its own thread, with a live event feed.

    The synchronous `/api/analyze` answered only when the whole run was
    finished. For an offline run that is three seconds and fine; online it is
    minutes of silence, and a deck that shows nothing for two minutes is
    indistinguishable from a deck that has hung.

    Events are buffered rather than broadcast: a client that connects late,
    reloads, or drops for a moment gets the whole run from the beginning
    instead of joining midway with a half-populated graph.
    """

    def __init__(self, job_id: str, payload: dict[str, Any]) -> None:
        self.id = job_id
        self.payload = payload
        self.events: list[dict[str, Any]] = []
        self.done = threading.Event()
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self._lock = threading.Lock()

    def emit(self, event: dict[str, Any]) -> None:
        with self._lock:
            event = {"seq": len(self.events), **event}
            self.events.append(event)

    def since(self, index: int) -> list[dict[str, Any]]:
        with self._lock:
            return self.events[index:]

    def run(self) -> None:
        try:
            worker = apply_fix if self.payload.get("_fix") else analyze
            self.result = worker(self.payload, on_event=self.emit)
            # A fix result carries an output path, not a run id. Reading
            # `result["id"]` unconditionally turns a successful render into a
            # KeyError reported as a failed job.
            self.emit(
                {
                    "type": "run.complete",
                    **{k: v for k, v in (self.result or {}).items() if k != "report"},
                }
            )
        except ApiError as exc:
            self.error = exc.message
            self.emit({"type": "run.error", "error": exc.message})
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.error = type(exc).__name__
            self.emit({"type": "run.error", "error": type(exc).__name__})
        finally:
            self.done.set()


# Bounded so a long-lived server does not accumulate every run it ever did.
JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
MAX_JOBS = 32


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


def run_media(run_id: str) -> Path:
    """Resolve the measured input/output artifact for a persisted run.

    The report intentionally stores a portable relative media URL for the
    standalone HTML artifact. The API deck is a different origin, so it must
    resolve media from the durable lineage record rather than guessing a path
    from the filename supplied by a browser.
    """
    if not RUN_ID.match(run_id):
        raise ApiError(400, "malformed run id")
    node = lineage.Lineage().run(run_id)
    if node is None:
        raise ApiError(404, f"no recorded media for run {run_id}")
    path = Path(node.video_path)
    if not path.is_file():
        raise ApiError(404, "recorded media is no longer available")
    return path


def resolve_video_argument(raw: str) -> Path:
    """Resolve either a local path or this server's authenticated-by-id media
    URL. The latter lets the dashboard's central Apply control reuse the same
    real remediation endpoint after loading a persisted run.
    """
    value = str(raw).strip()
    match = re.search(r"/api/runs/([A-Za-z0-9_.-]+)/media(?:\?.*)?$", value)
    if match:
        return run_media(match.group(1))
    return Path(value)


# Uploads land here. Separate from RUNS_DIR because these are inputs, not
# results, and a creator clearing old reports should not delete the video
# they are still working on.
UPLOAD_DIR = Path(".preflight/uploads")

# Matches the CLI's own ceiling. A cap enforced while streaming rather than
# after is the difference between refusing a 20GB upload and dying on it.
MAX_UPLOAD_BYTES = 8 * 1024**3
UPLOAD_CHUNK = 1 << 20


def safe_name(raw: str) -> str:
    """A filename that cannot escape the upload directory.

    `Path(name).name` alone still admits "..", and a browser is not the only
    thing that can post here. Anything that is not a plain component of a
    name is replaced rather than trusted.
    """
    stem = Path(str(raw or "upload")).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", stem).strip("._") or "upload"
    return cleaned[:120]


def receive_upload(handler: Any) -> dict[str, Any]:
    """Stream an uploaded video to disk.

    Never buffered in memory. A two-hour 4K file read into RAM before being
    written is how a laptop running this locally dies, and the request that
    kills it looks identical to a working one until it does.
    """
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        raise ApiError(400, "upload has no body")
    if length > MAX_UPLOAD_BYTES:
        raise ApiError(413, f"upload exceeds {MAX_UPLOAD_BYTES // 1024**3} GB")

    content_type = handler.headers.get("Content-Type", "")
    multipart = content_type.lower().startswith("multipart/form-data")
    original_name = "upload.mp4"
    source: Any = handler.rfile
    if multipart:
        form = cgi.FieldStorage(
            fp=handler.rfile,
            headers=handler.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": str(length),
            },
        )
        field = form["file"] if "file" in form else None
        if field is None or not getattr(field, "file", None):
            raise ApiError(400, "multipart upload must contain a file field")
        original_name = str(getattr(field, "filename", "") or "upload.mp4")
        source = field.file
    elif handler.headers.get("X-Filename"):
        # Backwards-compatible for old API clients; the browser no longer
        # sends untrusted names through HTTP headers.
        original_name = str(handler.headers.get("X-Filename"))

    suffix = Path(original_name).suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
        raise ApiError(415, "unsupported video format")
    name = Path(original_name).name
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    video_id = f"vid_{uuid.uuid4().hex}"
    destination = UPLOAD_DIR / f"{video_id}{suffix}"

    written = 0
    try:
        with destination.open("wb") as out:
            while multipart or written < length:
                chunk = source.read(UPLOAD_CHUNK if multipart else min(UPLOAD_CHUNK, length - written))
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise ApiError(413, "upload exceeded the size limit mid-stream")
                out.write(chunk)
    except ApiError:
        destination.unlink(missing_ok=True)
        raise
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise ApiError(500, f"could not write upload: {exc}") from exc

    if not multipart and written < length:
        destination.unlink(missing_ok=True)
        raise ApiError(400, "upload ended early")

    metadata = {
        "videoId": video_id,
        "originalFilename": name,
        "storageFilename": destination.name,
        "sizeBytes": written,
        "createdAt": _now(),
    }
    (UPLOAD_DIR / f"{video_id}.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    return {"id": video_id, "path": str(destination), "name": name, "bytes": written}


def start_job(payload: dict[str, Any]) -> Job:
    job_id = f"job-{int(datetime.now().timestamp() * 1000)}"
    job = Job(job_id, payload)
    with JOBS_LOCK:
        if len(JOBS) >= MAX_JOBS:
            for stale in sorted(JOBS)[: len(JOBS) - MAX_JOBS + 1]:
                JOBS.pop(stale, None)
        JOBS[job_id] = job
    threading.Thread(target=job.run, daemon=True).start()
    return job


def get_job(job_id: str) -> Job:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise ApiError(404, f"no job {job_id}")
    return job


def analyze(
    payload: dict[str, Any], on_event: Any = None
) -> dict[str, Any]:
    """Run the real pipeline. Same code path as `preflight check`."""
    raw = str(payload.get("video", "")).strip()
    if not raw:
        raise ApiError(400, "body must carry a 'video' path")

    video = resolve_video_argument(raw)
    if not video.is_file():
        raise ApiError(404, f"no such file: {video}")
    if not ffmpeg.available():
        raise ApiError(503, "ffmpeg and ffprobe are required")

    offline = bool(payload.get("offline", False))
    ceiling = payload.get("budget")
    settings = Settings.load(offline=True) if offline else Settings.load()
    budget = CallBudget(ceiling=int(ceiling) if ceiling else None)

    result = run_perception(
        video,
        cas.Store(settings.cache_dir),
        settings=settings,
        budget=budget,
        on_event=on_event,
    )
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

    # Reports on disk are the UI's source of truth; the lineage record is the
    # durable index that lets a later remediation name exactly which analysis
    # and bytes it derives from. Record both from the same measured result,
    # never from request data supplied by the browser.
    graph = lineage.Lineage()
    artifact = graph.record_artifact(video, duration_ms=result.ingested.meta.durationMs)
    graph.record_run(
        run_id,
        bundle.report,
        role="ORIGINAL",
        video_path=str(video),
        video_hash=cas.prefixed(result.ingested.video_hash),
        report_path=str(directory / "report.json"),
        artifact_id=artifact.artifact_id,
    )
    simulation = bundle.report.get("simulation")
    if isinstance(simulation, dict):
        graph.record_simulation(run_id, simulation)

    return {"id": run_id, "report": bundle.report}


def find_resumable(graph: lineage.Lineage, video: Path) -> lineage.RemediationRecord | None:
    """The newest remediation a previous process left unfinished for this file.

    Matched on the source path, because a restart has no remediation id to
    offer — a reader points at the same video again and expects the system to
    know it was already halfway through rather than silently starting over.
    """
    open_records = [
        record
        for record in graph.interrupted()
        if Path(record.source_path) == video
    ]
    return open_records[-1] if open_records else None


def reusable_artifact(
    graph: lineage.Lineage, record: lineage.RemediationRecord
) -> Path | None:
    """The rendered file from an earlier attempt, only if it is still itself.

    Three conditions, all required: the row names an artifact, the file is
    there, and its bytes still hash to the recorded digest. Trusting a path
    because a row mentions it would make persistence a correctness
    *regression* — the ephemeral version at least never reused a stale file.
    """
    if not record.artifact_id:
        return None
    artifact = graph.artifact(record.artifact_id)
    if artifact is None or not artifact.still_matches():
        return None
    return Path(artifact.path)


def apply_fix(payload: dict[str, Any], on_event: Any = None) -> dict[str, Any]:
    """Compile the remediation, render it, then prove it worked.

    Every step is bracketed by a persisted lifecycle transition, and the order
    of the two matters: the record moves to RENDERING *before* ffmpeg starts
    and to RENDERED *after* the file lands, so a process killed mid-render
    leaves a row that says RENDERING — which is true — rather than no row at
    all. A lifecycle that only becomes durable on success can only ever record
    successes, and the failures are the interesting part.

    Renders atomically — to a temp file whose extension stays last so
    ffmpeg's muxer can infer the container, verified on duration, then
    promoted. A half-written .safe.mp4 sitting next to the original is worse
    than no output, because it looks like a finished render.
    """
    from preflight.ingest.probe import UnsupportedInput, probe_video
    from preflight.remediate.codegen import build_program
    from preflight.remediate.edl import InvalidEDL, compile_edl

    raw = str(payload.get("video", "")).strip()
    if not raw:
        raise ApiError(400, "body must carry a 'video' path")
    video = resolve_video_argument(raw)
    if not video.is_file():
        raise ApiError(404, f"no such file: {video}")
    if not ffmpeg.available():
        raise ApiError(503, "ffmpeg and ffprobe are required")

    def emit(stage: str, detail: str = "", **extra: Any) -> None:
        if on_event:
            on_event({"type": "fix.progress", "stage": stage, "detail": detail, **extra})

    def step(state: str, stage: str, detail: str = "", **extra: Any) -> None:
        """Emit a progress event that names the persisted lifecycle state.

        The state travels with the event so the deck renders the machine the
        engine is actually in, rather than a label the frontend inferred from
        a stage word. Those two drift the moment either side is edited, and a
        progress display that disagrees with the stored history is worse than
        no progress display — it is a second, unbacked account of the run.
        """
        emit(stage, detail, state=state, **extra)

    offline = bool(payload.get("offline", False))
    settings = Settings.load(offline=True) if offline else Settings.load()
    recorder = telemetry.Recorder()

    emit("analysing", "re-reading findings for this video")
    with recorder.phase("analysis"):
        result = run_perception(video, cas.Store(settings.cache_dir), settings=settings)
    recorder.observe_run("analysis", result)

    # Establish the original run before requesting a remediation. The lineage
    # record must exist before ffmpeg starts, otherwise an interrupted render
    # would be indistinguishable from a remediation that was never requested.
    before_bundle = build_report(
        result,
        policy_version=result.corpus.version if result.corpus else "unknown",
        embed_media=False,
        strategy=str(payload.get("strategy") or "") or None,
        chunk_ms=settings.chunk_ms,
        overlap_ms=settings.overlap_ms,
    )
    before_report = before_bundle.report
    source_run_id = (
        f"fix-{cas.prefixed(result.ingested.video_hash)[3:15]}-{time.time_ns()}"
    )
    source_dir = RUNS_DIR / source_run_id
    source_dir.mkdir(parents=True, exist_ok=True)
    source_report_path = source_dir / "report.json"
    source_report_path.write_text(
        json.dumps(before_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    graph = lineage.Lineage()
    source_artifact = graph.record_artifact(
        video, duration_ms=result.ingested.meta.durationMs
    )
    graph.record_run(
        source_run_id,
        before_report,
        role="ORIGINAL",
        video_path=str(video),
        video_hash=cas.prefixed(result.ingested.video_hash),
        report_path=str(source_report_path),
        artifact_id=source_artifact.artifact_id,
    )
    simulation_id = graph.record_simulation(
        source_run_id, before_report.get("simulation", {})
    )

    step("REMEDIATION_REQUESTED", "compiling", "lowering findings to an edit list")
    try:
        edl = compile_edl(
            result.findings,
            str(video),
            result.ingested.meta.durationMs,
            result.transcript,
            strategy=str(payload.get("strategy") or "") or None,
        )
    except InvalidEDL as exc:
        raise ApiError(400, f"remediation could not be compiled: {exc}") from exc

    if not edl.ops:
        emit("done", "nothing to repair")
        return {
            "rendered": False,
            "reason": "no remediable findings",
            "ops": 0,
            "sourceRunId": source_run_id,
        }

    # What this edit actually targets, taken from the compiled operations
    # rather than from every finding in the report. A remediation that names
    # findings it never acts on cannot later be audited against what it did.
    targeted = sorted({op.finding_id for op in edl.ops if op.finding_id})
    incidents_before = before_report.get("incidents") or []
    targeted_incidents = sorted(
        {
            str(incident.get("id"))
            for incident in incidents_before
            if set(incident.get("findingIds", [])) & set(targeted)
        }
    )

    destination = video.with_name(f"{video.stem}.safe{video.suffix}")
    program = build_program(edl, video, destination)

    resumed_from: str | None = None
    reused = None
    existing = find_resumable(graph, video) if payload.get("resume", True) else None
    if existing is not None:
        reused = reusable_artifact(graph, existing)
        resumed_from = existing.state
        emit(
            "resuming",
            f"{existing.remediation_id} was interrupted during {existing.state}",
            remediationId=existing.remediation_id,
            interruptedAt=existing.state,
        )
        remediation = graph.resume(existing.remediation_id)
        remediation_id = remediation.remediation_id
    else:
        remediation = graph.open_remediation(
            source_run_id,
            source_path=str(video),
            simulation_id=simulation_id,
            finding_ids=targeted,
            incident_ids=targeted_incidents,
        )
        remediation_id = remediation.remediation_id

    out_meta = None
    if reused is not None and remediation.state == "STRUCTURALLY_VALID":
        # The earlier attempt got as far as a render that still hashes to what
        # was recorded, so re-rendering identical bytes would spend minutes to
        # produce the file already sitting there. The re-analysis below is
        # *not* skipped — that is the part that proves anything.
        destination = reused
        output_artifact = graph.artifact(remediation.artifact_id or "")
        elapsed_ms = 0
        step("STRUCTURALLY_VALID", "rendered", f"reusing verified {destination.name}", reused=True)
    else:
        graph.transition(
            remediation_id,
            "RENDERING",
            detail="ffmpeg render started",
            # The operations, not the whole EDL envelope. `RemediationRecord`
            # exposes `.ops` as a list of operation dicts, and storing
            # `edl.to_json()` here put a `{source, durationMs, ops, warnings}`
            # wrapper in the column — iterating it yielded key strings, so
            # every consumer of `.ops` got characters instead of operations.
            edl_json=json.dumps(
                [op.to_json() for op in edl.ops], separators=(",", ":")
            ),
        )

        # Extension last so the muxer can still infer the container — a suffix
        # like ".mp4.tmp1234" is one ffmpeg refuses outright.
        staged = destination.with_name(
            f"{destination.stem}.tmp{os.getpid()}{destination.suffix}"
        )
        step("RENDERING", "rendering", f"{len(edl.ops)} operation(s)", ops=len(edl.ops))
        started = time.perf_counter()
        with recorder.phase("render"):
            try:
                ffmpeg.run(program.command[1:-1] + [staged.as_posix()])
            except ffmpeg.FfmpegFailed as exc:
                staged.unlink(missing_ok=True)
                graph.fail(remediation_id, f"render failed: {exc}")
                raise ApiError(500, f"render failed: {exc}") from exc
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        if not staged.is_file() or staged.stat().st_size == 0:
            staged.unlink(missing_ok=True)
            graph.fail(remediation_id, "render produced no output")
            raise ApiError(500, "render produced no output")

        # Verified against the file, not against ffmpeg's exit code: a
        # truncated render from a killed process still exits clean.
        step("STRUCTURAL_VERIFYING", "verifying", "checking duration against the edit list")
        graph.transition(remediation_id, "RENDERED", detail="ffmpeg output written")
        graph.transition(
            remediation_id, "STRUCTURAL_VERIFYING", detail="probing rendered output"
        )
        with recorder.phase("structural"):
            try:
                out_meta = probe_video(staged)
                cut_ms = sum(op.duration_ms for op in edl.ops if op.op == "CUT")
                expected = result.ingested.meta.durationMs - cut_ms
                drift = abs(out_meta.durationMs - expected)
                recorder.record("structuralDriftMs", drift)
                recorder.record("outputDurationMs", out_meta.durationMs)
                recorder.record("expectedDurationMs", expected)
                if drift > 1500:
                    staged.unlink(missing_ok=True)
                    raise ApiError(
                        500,
                        f"verification failed: output is {out_meta.durationMs}ms, "
                        f"expected {expected}ms — nothing was written",
                    )
            except ApiError as exc:
                graph.fail(remediation_id, exc.message)
                raise
            except (ffmpeg.FfmpegFailed, UnsupportedInput) as exc:
                staged.unlink(missing_ok=True)
                graph.fail(
                    remediation_id, f"structural verification failed: {exc}"
                )
                raise ApiError(500, f"verification failed: {exc}") from exc

        staged.replace(destination)
        output_artifact = graph.record_artifact(
            destination, duration_ms=out_meta.durationMs
        )
        graph.transition(
            remediation_id,
            "STRUCTURALLY_VALID",
            detail="duration matches edit list",
            artifact_id=output_artifact.artifact_id,
            output_path=str(destination),
        )
        step("STRUCTURALLY_VALID", "rendered", f"wrote {destination.name}")

    recorder.record("renderMs", elapsed_ms)
    graph.transition(
        remediation_id, "REANALYSIS_QUEUED", detail="render queued for analysis"
    )
    graph.transition(
        remediation_id, "REANALYSING", detail="rendered artifact analysis started"
    )

    # A successful render is not a successful remediation. ffmpeg exiting
    # zero proves a file was written; only the same pipeline finding fewer
    # problems in the output proves the fix worked. So the output goes back
    # through the real analysis — no second scorer, no deleting findings
    # from the original report to simulate success.
    step("REANALYSING", "reanalysing", "running the pipeline against the rendered file")
    verified: dict[str, Any] = {}
    after = None
    try:
        # Bounded so a verification pass on a long video terminates.
        # This is only honest because `compare` refuses to call a finding
        # resolved when the modality that would have seen it fell below the
        # coverage floor — otherwise making re-analysis cheaper would make
        # success more likely, which is the worst incentive to build in.
        #
        # The rendered file gets its own perception pass. Reusing the
        # original's vision results — tempting, since they are already cached
        # and the hashes differ only because the bytes do — would compare the
        # file against itself and invalidate every verdict below it.
        with recorder.phase("reanalysis"):
            after = run_perception(
                destination,
                cas.Store(settings.cache_dir),
                settings=settings,
                budget=CallBudget(ceiling=int(payload.get("verifyBudget", 12))),
            )
            after_bundle = build_report(
                after,
                policy_version=after.corpus.version if after.corpus else "unknown",
                embed_media=False,
                chunk_ms=settings.chunk_ms,
                overlap_ms=settings.overlap_ms,
            )
        recorder.observe_run("reanalysis", after)
        reanalysis_ok = True
    except Exception as exc:  # noqa: BLE001 - a failed re-analysis is a state
        step("INCONCLUSIVE", "reanalysis_failed", type(exc).__name__)
        after_bundle = None
        reanalysis_ok = False

    verification_run_id: str | None = None
    if reanalysis_ok and after_bundle is not None:
        verification_run_id = f"verify-{cas.prefixed(after.ingested.video_hash)[3:15]}-{time.time_ns()}"
        verification_dir = RUNS_DIR / verification_run_id
        verification_dir.mkdir(parents=True, exist_ok=True)
        verification_path = verification_dir / "report.json"
        verification_path.write_text(
            json.dumps(after_bundle.report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        graph.record_run(
            verification_run_id,
            after_bundle.report,
            role="VERIFICATION",
            parent_run_id=source_run_id,
            video_path=str(destination),
            video_hash=cas.prefixed(after.ingested.video_hash),
            report_path=str(verification_path),
            artifact_id=output_artifact.artifact_id,
        )
        graph.transition(remediation_id, "REANALYSIS_COMPLETE", detail="rendered artifact analysed")
        graph.transition(remediation_id, "COMPARING", detail="matching original and rendered findings")

    step("COMPARING", "comparing", "matching findings and incidents across the two runs")
    coverage = (
        {a.agent_id: a.coverage for a in after.agents}
        if reanalysis_ok and after is not None
        else {}
    )
    with recorder.phase("comparison"):
        comparison = compare(
            before_report["findings"],
            after_bundle.report["findings"] if after_bundle else [],
            edl.ops,
            original_score=int(before_report["scores"]["overall"]),
            remediated_score=(
                int(after_bundle.report["scores"]["overall"]) if after_bundle else 0
            ),
            structural_ok=True,
            reanalysis_ok=reanalysis_ok,
            coverage=coverage,
            original_incidents=incidents_before,
            remediated_incidents=(
                (after_bundle.report.get("incidents") or []) if after_bundle else []
            ),
        )
    verified = comparison.to_json()

    # Predicted against actual. Both scores come from the same scorer over the
    # same two artifacts, so they are comparable by construction.
    #
    # The *resolved counts* are not, unless the scenario describes the edit
    # that actually ran. The compiler picks its own balanced operation set,
    # which routinely differs from the highest-scoring scenario — here it
    # muted one span and bleeped another, where the best scenario bleeped one.
    # Feeding those two counts to `prediction_outcome` compares a prediction
    # about one edit against the result of a different one, and reports
    # UNDERESTIMATED for a simulation that was never wrong. So the count
    # comparison is made only when the sets match exactly.
    simulation_block = before_report.get("simulation") or {}
    scenarios = simulation_block.get("scenarios", [])
    targeted_set = set(targeted)

    exact = next(
        (
            s
            for s in scenarios
            if set(s.get("removedFindingIds") or []) == targeted_set
        ),
        None,
    )
    headline = next(
        (s for s in scenarios if s.get("name") == simulation_block.get("best")), None
    )
    scenario = exact or headline

    predicted = int(scenario["overall"]) if scenario else None
    predicted_resolved = (
        len(exact.get("removedFindingIds") or []) if exact else None
    )

    verified["predictedScore"] = predicted
    verified["predictedScenario"] = scenario.get("name") if scenario else None
    # Stated plainly, because a reader comparing 43 against 42 is entitled to
    # know whether the 43 was a forecast of this edit or of another one.
    verified["predictionIsForThisEdit"] = exact is not None
    verified["predictionOutcome"] = (
        prediction_outcome(
            predicted,
            comparison.remediated_score,
            predicted_resolved=predicted_resolved,
            actual_resolved=len(comparison.resolved) if exact else None,
        )
        if reanalysis_ok
        else "INCONCLUSIVE"
    )
    if scenario is not None and exact is None:
        verified["notes"].append(
            f"Predicted score is from scenario '{scenario.get('name')}', which "
            "is not the operation set the compiler rendered; only the scores "
            "are compared, not the resolved counts."
        )

    # Before / after stills, pulled from the two files that actually exist.
    # The after frame comes out of the rendered artifact or it does not exist;
    # relabelling the original would be the exact claim this loop disproves.
    step("COMPARING", "evidence", "extracting before/after frames from both artifacts")
    with recorder.phase("evidence"):
        pairs = evidence_mod.build_pairs(
            comparison.changes,
            before_report["findings"],
            after_bundle.report["findings"] if after_bundle else [],
            original_path=video,
            remediated_path=destination if destination.is_file() else None,
            ops=list(edl.ops),
            remediation_id=remediation_id,
            original_run_id=source_run_id,
            verification_run_id=verification_run_id,
            out_dir=RUNS_DIR / source_run_id / "evidence" / remediation_id,
            coverage=coverage,
            incidents=incidents_before,
        )
    verified["evidence"] = evidence_mod.summarise(pairs)
    measurements = recorder.to_json()

    verification_id = graph.record_verification(
        remediation_id,
        original_run_id=source_run_id,
        verification_run_id=verification_run_id,
        comparison=verified,
        telemetry=measurements,
    )
    if reanalysis_ok:
        graph.transition(
            remediation_id,
            lifecycle.state_for_verdict(verified["verdict"]),
            detail="comparison completed",
            verification_run_id=verification_run_id,
            verification_id=verification_id,
            verdict=verified["verdict"],
        )
    else:
        graph.transition(
            remediation_id,
            "INCONCLUSIVE",
            detail="rendered artifact could not be re-analysed",
            verification_id=verification_id,
            verdict=verified["verdict"],
        )

    # The verification certificate. Distinct from the release certificate a
    # report carries: that one attests to an *analysis*, this one attests to a
    # *remediation* — both artifact hashes, both run ids, predicted against
    # actual, and the coverage the second run actually reached. Issued from
    # the stored verification's timestamp rather than the clock, so re-issuing
    # it for the same verification reproduces the same document and the same
    # hash. A hash that changed on every read would certify nothing.
    stored = graph.verification(verification_id) or {}
    body = certificate_mod.build(
        remediation=graph.remediation(remediation_id) or remediation,
        verification_id=verification_id,
        comparison=verified,
        original_report=before_report,
        remediated_report=after_bundle.report if after_bundle else None,
        original_artifact=source_artifact,
        remediated_artifact=output_artifact,
        coverage=coverage,
        telemetry=measurements,
        issued_at=str(stored.get("created_at", "")),
        evidence=verified["evidence"],
    )
    certificate_id = graph.record_certificate(
        verification_id, body, certificate_mod.digest(body)
    )
    sealed = certificate_mod.seal(body, certificate_id)

    emit(
        "verified",
        verified["verdict"],
        verdict=verified["verdict"],
        remediationId=remediation_id,
        certificateId=certificate_id,
    )
    return {
        "rendered": True,
        "output": str(destination),
        "ops": len(edl.ops),
        "renderMs": elapsed_ms,
        "videoStreamCopied": program.video_stream_copied,
        "command": program.pretty(),
        "verification": verified,
        "certificate": sealed,
        "evidence": [pair.to_json(embed=True) for pair in pairs],
        "telemetry": measurements,
        "sourceRunId": source_run_id,
        "remediationId": remediation_id,
        "verificationId": verification_id,
        "certificateId": certificate_id,
        "lifecycle": (graph.remediation(remediation_id) or remediation).to_json(),
        **({"resumedFrom": resumed_from} if resumed_from else {}),
        # The rendered file is shown as an "after" only when it was actually
        # re-analysed. A successful render proves bytes were written; it does
        # not prove the policy findings changed.
        **({"afterReport": after_bundle.report} if after_bundle else {}),
    }


def read_lineage(run_id: str) -> dict[str, Any]:
    """One original run and everything derived from it."""
    if not RUN_ID.match(run_id):
        raise ApiError(400, "malformed run id")
    graph = lineage.Lineage().graph(run_id)
    if graph.root is None:
        raise ApiError(404, f"no lineage for {run_id}")
    return graph.to_json()


def list_remediations() -> dict[str, Any]:
    """Every remediation on disk, including the ones a crash left open.

    This is what makes a restart informative rather than lossy. The deck asks
    once at startup and can say "REM-0001 was interrupted during REANALYSING"
    instead of behaving as though the operation never happened.
    """
    store = lineage.Lineage()
    records = store.remediations()
    return {
        "remediations": [r.to_json() for r in records],
        "interrupted": [
            {
                "remediationId": r.remediation_id,
                "state": r.state,
                "describe": r.describe(),
                "resumesAt": r.resume_state(),
            }
            for r in records
            if r.interrupted
        ],
        "stats": store.stats(),
    }


def read_remediation(remediation_id: str) -> dict[str, Any]:
    """One remediation, resolved through to its certificate.

    The whole chain in one response, because that is the navigation the deck
    needs: verdict → incident → finding → evidence → remediation → run.
    """
    if not RUN_ID.match(remediation_id):
        raise ApiError(400, "malformed remediation id")
    store = lineage.Lineage()
    record = store.remediation(remediation_id)
    if record is None:
        raise ApiError(404, f"no remediation {remediation_id}")

    out = record.to_json()
    out["describe"] = record.describe()
    out["resumesAt"] = record.resume_state()
    if record.verification_id:
        verification = store.verification(record.verification_id) or {}
        out["verification"] = verification.get("comparison")
        out["telemetry"] = verification.get("telemetry")
        certificate = store.certificate_for(record.verification_id)
        if certificate:
            payload = dict(certificate["payload"])
            payload.setdefault("certificateId", certificate["certificate_id"])
            payload.setdefault("certificateHash", certificate["certificate_hash"])
            out["certificate"] = payload
            # Recomputed on read, not trusted from the row. A certificate that
            # was edited after issue stops verifying, and says so here.
            out["certificateIntegrity"] = (
                "VALID" if certificate_mod.verify_integrity(payload) else "MISMATCH"
            )
    return out


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

    def _stream(self, job: Job) -> None:
        """Server-sent events for one job, replayed from the beginning.

        Written by hand rather than pulled from a framework: SSE is a
        content type, a blank-line delimiter and a flush, and this is all of
        it. The heartbeat matters — an idle proxy will close a connection
        that says nothing for a minute, and an agent that takes two minutes
        is normal here.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self._cors()
        self.end_headers()

        cursor = 0
        try:
            while True:
                for event in job.since(cursor):
                    cursor += 1
                    self.wfile.write(
                        f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()
                    )
                    self.wfile.flush()
                if job.done.is_set() and cursor >= len(job.events):
                    break
                # Comment frame: keeps the connection warm without being
                # delivered to the EventSource `message` handler.
                self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
                job.done.wait(timeout=0.4)
        except (BrokenPipeError, ConnectionResetError):
            # The client navigated away mid-run. The analysis keeps going and
            # its report still lands on disk.
            pass

    def _media(self, path: Path) -> None:
        """Send media with byte ranges so the browser can seek without loading
        the entire source video. The path comes only from lineage, never from
        a URL component.
        """
        size = path.stat().st_size
        start, end = 0, max(0, size - 1)
        status = 200
        header = self.headers.get("Range", "")
        if header.startswith("bytes="):
            try:
                raw_start, raw_end = header[6:].split("-", 1)
                start = int(raw_start) if raw_start else 0
                end = int(raw_end) if raw_end else end
                if start < 0 or end < start or start >= size:
                    raise ValueError
                end = min(end, size - 1)
                status = 206
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self._cors()
                self.end_headers()
                return
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self._cors()
        self.end_headers()
        with path.open("rb") as media:
            media.seek(start)
            remaining = length
            while remaining:
                chunk = media.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        try:
            if path in ("/api/health", "/api"):
                self._send(200, health())
            elif path == "/api/agents":
                self._send(200, agents())
            elif path == "/api/runs":
                self._send(200, list_runs())
            elif path.startswith("/api/runs/") and path.endswith("/media"):
                run_id = path[len("/api/runs/"):-len("/media")].rstrip("/")
                self._media(run_media(run_id))
            elif path.startswith("/api/events/"):
                self._stream(get_job(path[len("/api/events/"):]))
            elif path.startswith("/api/jobs/"):
                job = get_job(path[len("/api/jobs/"):])
                self._send(
                    200,
                    {
                        "id": job.id,
                        "done": job.done.is_set(),
                        "error": job.error,
                        "events": job.events,
                        "result": job.result,
                    },
                )
            elif path == "/api/remediations":
                self._send(200, list_remediations())
            elif path.startswith("/api/remediations/"):
                self._send(200, read_remediation(path[len("/api/remediations/"):]))
            elif path.startswith("/api/lineage/"):
                self._send(200, read_lineage(path[len("/api/lineage/"):]))
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
                # Synchronous, for scripts and the CLI-shaped caller that
                # just wants a report back.
                payload = self._body()
                if not self._analysis_lock.acquire(blocking=False):
                    raise ApiError(409, "an analysis is already running")
                try:
                    self._send(200, analyze(payload))
                finally:
                    self._analysis_lock.release()
            elif path == "/api/fix":
                payload = self._body()
                candidate = str(payload.get("video", "")).strip()
                if not candidate:
                    raise ApiError(400, "body must carry a 'video' path")
                if not Path(candidate).is_file():
                    raise ApiError(404, f"no such file: {candidate}")
                job = start_job({**payload, "_fix": True})
                self._send(202, {"id": job.id, "events": f"/api/events/{job.id}"})
            elif path == "/api/upload":
                self._send(201, receive_upload(self))
            elif path == "/api/jobs":
                # Asynchronous, for the deck: returns immediately with an id
                # to stream, so twelve agents can be watched rather than
                # waited on.
                payload = self._body()
                # Validated here, before a job exists. Deferring this to the
                # worker meant a typo'd path returned 202 and a job id, and
                # the failure arrived seconds later over the event stream —
                # so the deck showed "analysing…" for a file that was never
                # going to open.
                candidate = str(payload.get("video", "")).strip()
                if not candidate:
                    raise ApiError(400, "body must carry a 'video' path")
                if not Path(candidate).is_file():
                    raise ApiError(404, f"no such file: {candidate}")
                job = start_job(payload)
                self._send(202, {"id": job.id, "events": f"/api/events/{job.id}"})
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
