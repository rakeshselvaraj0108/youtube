"""PREFLIGHT command line.

Exit codes:
    0  pass
    1  findings exceed the configured threshold
    2  input or configuration error
    3  upstream unavailable and no fallback permitted
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from preflight import __version__, cas, ffmpeg
from preflight import bench as bench_mod
from preflight.ingest.pipeline import ingest
from preflight.ingest.probe import UnsupportedInput, probe_video
from preflight.models import SEVERITY_RANK
from preflight.pipeline import SURFACE_WEIGHT, run_perception
from preflight.agents.nim import NimClient
from preflight.agents.roster import load_roster
from preflight.archive import Archive
from preflight.config import Settings
from preflight.drift import detect, write_snapshot
from preflight.policy.corpus import load_corpus
from preflight.providers.doctor import run_doctor
from preflight.providers.registry import Registry
from preflight.report.build import build_report, validate
from preflight.report.html import BundleMissing, emit_html
from preflight.report.html import emit_fixture as emit_fixture_file
from preflight.report.sarif import build_certificate, build_sarif
from preflight.remediate.captions import write_captions
from preflight.remediate.codegen import build_program, write_fix_script
from preflight.remediate.edl import InvalidEDL, compile_edl
from preflight.scoring.readiness import SUB_SCORE_ORDER

# Verdicts that let CI through.
PASSING = {"READY_TO_PUBLISH", "PUBLISH_WITH_FIXES"}

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Static analysis and CI for video. Ship your video like you ship your code.",
)
console = Console()

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_INPUT = 2
EXIT_UPSTREAM = 3


def _timecode(ms: int) -> str:
    total = ms // 1000
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _human_bytes(size: int) -> str:
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GB"
    if size >= 1024**2:
        return f"{size / 1024**2:.1f} MB"
    return f"{size / 1024:.0f} KB"


@app.command()
def version() -> None:
    """Print versions of PREFLIGHT and its media toolchain."""
    console.print(f"[bold]PREFLIGHT[/bold] {__version__}")
    console.print(f"hash      {cas.HASH_NAME}")
    console.print(f"ffmpeg    {ffmpeg.version() or '[red]not found[/red]'}")


@app.command()
def probe(
    video: Path = typer.Argument(..., help="Video file to inspect"),
    cache_dir: Path = typer.Option(
        Path(".preflight/cache"), "--cache-dir", help="Content-addressed store root"
    ),
    max_frames: int = typer.Option(90, "--max-frames", help="Keyframe cap"),
    scene: float = typer.Option(0.35, "--scene", help="Scene-change threshold"),
) -> None:
    """Probe a video and populate the artifact store.

    Run it twice: the second run reports a cache hit and performs no ffmpeg
    work at all. That property is what makes the report hash a reproducibility
    proof rather than a decoration.
    """
    if not ffmpeg.available():
        console.print("[red]ffmpeg and ffprobe are required.[/red]")
        console.print("  Windows:  winget install Gyan.FFmpeg")
        raise typer.Exit(EXIT_UPSTREAM)

    store = cas.Store(cache_dir)

    try:
        result = ingest(video, store, max_frames=max_frames, scene_threshold=scene)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_INPUT) from exc
    except UnsupportedInput as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_INPUT) from exc
    except ffmpeg.FfmpegFailed as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_INPUT) from exc

    meta = result.meta
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim", width=14)
    table.add_column()
    table.add_row("file", meta.filename)
    table.add_row("duration", _timecode(meta.durationMs))
    table.add_row("resolution", f"{meta.width}x{meta.height}")
    table.add_row("fps", f"{meta.fps:g}")
    table.add_row("size", _human_bytes(meta.sizeBytes))
    table.add_row("audio", f"{meta.audioCodec} @ {meta.sampleRate} Hz" if meta.sampleRate else "none")
    table.add_row("video hash", cas.prefixed(result.video_hash)[:26] + "…")
    table.add_row("keyframes", str(len(result.keyframes)))
    table.add_row("cache", "[green]HIT[/green]" if result.cached else "[yellow]MISS[/yellow]")
    table.add_row("elapsed", f"{result.elapsed_ms} ms")

    console.print()
    console.print(table)
    console.print()
    for line in result.log:
        console.print(f"  [dim]·[/dim] {line}")
    console.print()

    if result.keyframes:
        first, last = result.keyframes[0], result.keyframes[-1]
        console.print(
            f"  keyframe span {_timecode(first.ts_ms)} → {_timecode(last.ts_ms)}",
            style="dim",
        )
        console.print()

    raise typer.Exit(EXIT_OK)


@app.command()
def check(
    video: Path = typer.Argument(..., help="Video file to analyse"),
    cache_dir: Path = typer.Option(Path(".preflight/cache"), "--cache-dir"),
    asr_model: str = typer.Option("base.en", "--asr-model", help="faster-whisper model"),
    no_speech: bool = typer.Option(False, "--no-speech", help="Skip transcription"),
    html: bool = typer.Option(False, "--html", help="Emit a self-contained report.html"),
    fmt: str = typer.Option(
        "", "--format", help="Comma-separated: json,sarif,certificate,html,all"
    ),
    out: Path = typer.Option(Path("preflight-out"), "--out", help="Output directory"),
    emit_fixture: Path = typer.Option(
        None, "--emit-fixture", help="Write this run as the UI's demo fixture"
    ),
    offline: bool = typer.Option(False, "--offline", help="Never touch the network"),
    strategy: str = typer.Option(
        "",
        "--strategy",
        help="conservative|balanced|aggressive — the remediation preview in the "
        "report trades viewer impact for risk reduction. Default trusts each "
        "finding's own suggested fix directly.",
    ),
) -> None:
    """Analyse a video, print findings, and score it against the full triad.

    Ingest, every perception agent, retrieval, the adversarial triad and
    scoring all run in this one pass — the exit code is the readiness
    verdict: 0 when the video is ready to publish, 1 otherwise.
    """
    if not ffmpeg.available():
        console.print("[red]ffmpeg and ffprobe are required.[/red]")
        raise typer.Exit(EXIT_UPSTREAM)

    if strategy and strategy not in {"conservative", "balanced", "aggressive"}:
        console.print(
            f"[red]--strategy must be conservative, balanced or aggressive, "
            f"not {strategy!r}[/red]"
        )
        raise typer.Exit(EXIT_INPUT)

    store = cas.Store(cache_dir)
    settings = Settings.load(offline=True) if offline else Settings.load()
    console.print(f"[dim]{settings.describe_mode()}[/dim]")
    try:
        result = run_perception(
            video,
            store,
            asr_model=asr_model,
            skip_speech=no_speech,
            settings=settings,
        )
    except (FileNotFoundError, UnsupportedInput, ffmpeg.FfmpegFailed) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_INPUT) from exc

    meta = result.ingested.meta
    console.print()
    console.print(
        f"[bold]PREFLIGHT[/bold] {__version__}   {meta.filename}   "
        f"{_timecode(meta.durationMs)}   {cas.prefixed(result.ingested.video_hash)[:14]}…"
    )
    console.print()

    # Agent roster — the same telemetry the report's terminal panel renders.
    roster = Table(show_header=True, box=None, pad_edge=False, header_style="dim")
    roster.add_column("AGENT", width=22)
    roster.add_column("STATUS", width=9)
    roster.add_column("COV", width=5, justify="right")
    roster.add_column("MS", width=7, justify="right")
    roster.add_column("FOUND", width=5, justify="right")
    roster.add_column("DETAIL", overflow="fold")

    tone = {
        "OK": "green",
        "DEGRADED": "yellow",
        "FAILED": "red",
        "SKIPPED": "dim",
    }
    for agent in result.agents:
        roster.add_row(
            agent.name,
            f"[{tone.get(agent.status, 'white')}]{agent.status}[/]",
            f"{agent.coverage * 100:.0f}%",
            str(agent.elapsed_ms),
            str(len(agent.findings)) if agent.findings else "-",
            agent.detail,
        )
    console.print(roster)
    console.print()

    findings = sorted(
        result.findings,
        key=lambda f: (SEVERITY_RANK[f.severity], -f.confidence, f.startMs),
    )

    if not findings:
        console.print("  [green]no findings[/green]")
    else:
        console.print(f"  [bold]FINDINGS ({len(findings)})[/bold]")
        console.print()
        severity_tone = {
            "CRITICAL": "bright_red",
            "HIGH": "red",
            "MEDIUM": "yellow",
            "LOW": "cyan",
        }
        for finding in findings:
            span = (
                "file-scoped"
                if finding.endMs - finding.startMs >= meta.durationMs
                else f"{_timecode(finding.startMs)} → {_timecode(finding.endMs)}"
            )
            console.print(
                f"  [{severity_tone[finding.severity]}]â—[/] "
                f"[{severity_tone[finding.severity]}]{finding.severity:<8}[/] "
                f"[dim]{finding.clauseId:<8}[/] {finding.title}"
            )
            console.print(f"      [dim]span[/]        {span}")
            console.print(f"      [dim]clause[/]      {finding.policy.section}")
            console.print(
                f"      [dim]adjudicator[/] {finding.adversarial.rationale} "
                f"[dim]conf {finding.confidence:.2f}[/]"
            )
            if finding.suggestedFix != "NONE":
                console.print(f"      [dim]fix[/]         {finding.suggestedFix}")
            console.print()

    # Release readiness.
    sub = result.sub_scores
    readiness = result.readiness
    band = {
        "READY_TO_PUBLISH": "green",
        "PUBLISH_WITH_FIXES": "yellow",
        "NOT_READY": "dark_orange",
        "DO_NOT_PUBLISH": "red",
    }[readiness.verdict]

    console.print("  [bold]RELEASE READINESS[/bold]")
    console.print()
    for key in SUB_SCORE_ORDER:
        value = sub[key]
        filled = int(round(value / 5))
        bar = "█" * filled + "·" * (20 - filled)
        marker = " [dim]â† weakest[/dim]" if key == readiness.weakest else ""
        console.print(f"    {key:<14} {bar} {value:5.1f}{marker}")
    console.print()
    console.print(
        f"    [bold {band}]{readiness.overall} / 100[/bold {band}]   "
        f"[{band}]{readiness.verdict.replace('_', ' ')}[/{band}]"
    )
    if readiness.capped:
        console.print(
            f"    [dim]weighted mean {readiness.weighted:.1f}, capped at "
            f"weakest + 15 — one fatal flaw is never averaged away[/dim]"
        )
    console.print()

    coverage = result.coverage
    console.print(
        f"  coverage [bold]{coverage * 100:.0f}%[/bold]   "
        f"LLM calls [bold]{result.total_calls}[/bold]   "
        f"elapsed [bold]{sum(a.elapsed_ms for a in result.agents)} ms[/bold]"
    )
    if coverage < 0.95:
        # Distinguish an agent that ran badly from one that never ran at all.
        # Every agent SURFACE_WEIGHT carries is unconditionally invoked by
        # run_perception (asserted by
        # TestEveryWeightedAgentActuallyRuns), so "missing" here is a wiring
        # regression, not a feature that is legitimately still pending — the
        # message says so rather than the misleading "not yet implemented"
        # this used to print for A12 and the report writer, both of which
        # were fully built and simply run from different commands than
        # `check`.
        impaired = [
            a.name for a in result.agents if a.status in {"SKIPPED", "FAILED", "DEGRADED"}
        ]
        ran = {a.agent_id for a in result.agents}
        missing = [
            agent_id
            for agent_id, weight in SURFACE_WEIGHT.items()
            if weight > 0 and agent_id not in ran
        ]
        console.print(f"  [yellow]PARTIAL ANALYSIS[/yellow] — {coverage * 100:.0f}% coverage")
        if impaired:
            console.print(f"    impaired: {', '.join(impaired)}")
        if missing:
            console.print(
                f"    [red]did not run (wiring bug):[/red] {', '.join(sorted(missing))}"
            )
    # Emission.
    formats = {f.strip().lower() for f in fmt.split(",") if f.strip()}
    if html:
        formats.add("html")
    if "all" in formats:
        formats = {"json", "sarif", "certificate", "html"}
    if emit_fixture is not None:
        formats.add("json")

    if formats or emit_fixture is not None:
        _emit(result, formats, out, emit_fixture, settings, strategy=strategy or None)

    console.print()
    raise typer.Exit(EXIT_OK if readiness.verdict in PASSING else EXIT_FINDINGS)


def _emit(
    result,
    formats: set[str],
    out: Path,
    fixture: Path | None,
    settings: Settings,
    *,
    strategy: str | None = None,
) -> None:
    """Write the requested artifacts. Validates before writing anything.

    Settings are threaded in rather than re-loaded: reloading drops the
    --offline flag, and a provenance block that misreports how the run was
    produced is worse than no provenance block.
    """
    bundle = build_report(
        result,
        policy_version=result.corpus.version if result.corpus else "unknown",
        embed_media="html" in formats or fixture is not None,
        strategy=strategy,
    )

    schema_path = Path("schema/analysis-report.schema.json")
    if schema_path.is_file():
        try:
            validate(bundle.report, schema_path)
        except Exception as exc:  # noqa: BLE001
            # A contract violation means the page would render something wrong.
            # Fail here rather than ship a report the UI silently misreads.
            console.print(f"[red]report failed schema validation: {exc}[/red]")
            raise typer.Exit(EXIT_INPUT) from exc

    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if "json" in formats:
        path = out / "report.json"
        path.write_text(
            json.dumps(bundle.report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        written.append(path)

    if "sarif" in formats:
        path = out / "report.sarif"
        path.write_text(
            json.dumps(build_sarif(bundle.report), indent=2), encoding="utf-8"
        )
        written.append(path)

    if "certificate" in formats:
        certificate = build_certificate(
            bundle.report,
            models=settings.models.to_json(),
            policy_digest=result.corpus.digest if result.corpus else "none",
            video_hash=cas.prefixed(result.ingested.video_hash),
            retrieval_backend=result.retrieval_backend,
            provenance=Registry(offline=settings.offline).provenance(),
        )
        path = out / "certificate.json"
        path.write_text(json.dumps(certificate, indent=2), encoding="utf-8")
        written.append(path)

    if "html" in formats:
        try:
            path = emit_html(bundle.report, Path("dist"), out / "report.html")
            written.append(path)
        except BundleMissing as exc:
            console.print(f"[yellow]{exc}[/yellow]")

    if fixture is not None:
        written.append(emit_fixture_file(bundle.report, fixture))

    # Record into the archive so the Drift Watcher has something to monitor.
    # Clauses that were retrieved but did not fire are recorded too: when a
    # clause tightens, those are exactly the videos it will newly catch.
    considered = {
        chunk.clause_id
        for window in result.windows
        for chunk in getattr(window, "retrieved", []) or []
    }
    Archive(Path(".preflight/archive.db")).record(
        bundle.report,
        video_hash=result.ingested.video_hash,
        policy_digest=result.corpus.digest if result.corpus else "none",
        considered=considered,
    )

    if written:
        console.print()
        for path in written:
            size = path.stat().st_size
            console.print(f"  wrote [bold]{path}[/bold]  ({size:,} bytes)")


@app.command()
def fix(
    video: Path = typer.Argument(..., help="Video file to remediate"),
    apply: bool = typer.Option(
        False, "--apply", help="Actually render. Without this, dry-run only."
    ),
    strategy: str = typer.Option(
        "",
        "--strategy",
        help="conservative|balanced|aggressive — trade viewer impact for risk "
        "reduction. Default trusts each finding's own suggested fix directly.",
    ),
    out: Path = typer.Option(None, "--out", help="Output path (default <name>.safe.mp4)"),
    cache_dir: Path = typer.Option(Path(".preflight/cache"), "--cache-dir"),
    offline: bool = typer.Option(False, "--offline", help="Never touch the network"),
) -> None:
    """Compile findings into an ffmpeg program and optionally run it.

    Dry-runs by default. A tool that rewrites someone's master file because
    they typed the wrong command has lost more than it gained.
    """
    if not ffmpeg.available():
        console.print("[red]ffmpeg and ffprobe are required.[/red]")
        raise typer.Exit(EXIT_UPSTREAM)

    if strategy and strategy not in {"conservative", "balanced", "aggressive"}:
        console.print(
            f"[red]--strategy must be conservative, balanced or aggressive, "
            f"not {strategy!r}[/red]"
        )
        raise typer.Exit(EXIT_INPUT)

    store = cas.Store(cache_dir)
    settings = Settings.load(offline=True) if offline else Settings.load()
    try:
        result = run_perception(video, store, settings=settings)
    except (FileNotFoundError, UnsupportedInput, ffmpeg.FfmpegFailed) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(EXIT_INPUT) from exc

    destination = out or video.with_suffix("").with_name(f"{video.stem}.safe.mp4")

    try:
        edl = compile_edl(
            result.findings,
            str(video),
            result.ingested.meta.durationMs,
            result.transcript,
            strategy=strategy or None,  # type: ignore[arg-type]
        )
    except InvalidEDL as exc:
        console.print(f"[red]invalid EDL: {exc}[/red]")
        raise typer.Exit(EXIT_INPUT) from exc

    program = build_program(edl, video, destination)

    console.print()
    console.print(f"[bold]REMEDIATION PLAN[/bold]   {len(edl.ops)} operation(s)")
    console.print()
    if not edl.ops:
        console.print("  [green]nothing to remediate[/green]")
        console.print()
        raise typer.Exit(EXIT_OK)

    table = Table(show_header=True, box=None, pad_edge=False, header_style="dim")
    table.add_column("#", width=3, justify="right")
    table.add_column("START", width=9)
    table.add_column("END", width=9)
    table.add_column("ACTION", width=15)
    table.add_column("DETAILS", overflow="fold")
    for op in edl.ops:
        table.add_row(
            str(op.index),
            _timecode(op.start_ms),
            _timecode(op.end_ms),
            op.op.replace("_", " ").title(),
            op.details,
        )
    console.print(table)
    console.print()

    for line in edl.log:
        console.print(f"  [dim]·[/dim] {line}")
    for warning in edl.warnings:
        console.print(f"  [yellow]![/yellow] {warning}")
    console.print()

    console.print(
        "  video stream "
        + (
            "[green]copied[/green] (-c:v copy, audio-only EDL)"
            if program.video_stream_copied
            else "[yellow]re-encoded[/yellow] (EDL contains a video op)"
        )
    )
    console.print()
    # markup=False: stream labels like [aout] are rich markup tags, and letting
    # rich eat them prints a command that looks broken but is not.
    console.print(program.pretty(), style="dim", markup=False, highlight=False)
    console.print()

    out_dir = destination.parent
    script = write_fix_script(program, out_dir / "fix.sh", edl)
    edl_path = out_dir / "edl.json"
    edl_path.write_text(json.dumps(edl.to_json(), indent=2), encoding="utf-8")
    console.print(f"  wrote [bold]{edl_path}[/bold] and [bold]{script}[/bold]")

    if not apply:
        console.print()
        console.print("  [dim]dry run — pass --apply to render[/dim]")
        console.print()
        raise typer.Exit(EXIT_OK)

    # Atomic output. ffmpeg writes to a temp path in the same directory as the
    # real destination — same filesystem, so the final step is a rename, not
    # a copy — and only that succeeding temp file is ever moved onto the path
    # the user is expecting a good file at. A process killed mid-render
    # (Ctrl-C, OOM, disk full, a crash) leaves the temp file orphaned and the
    # real destination exactly as it was, rather than a half-written file
    # sitting where a repaired video should be.
    # The real extension has to stay LAST — ffmpeg's muxer is chosen from the
    # output filename, and a suffix like ".mp4.tmp1234" gives it nothing to
    # infer a container from. ".tmp1234.mp4" keeps the container recognisable
    # while still being distinct from the real destination.
    tmp_destination = destination.with_name(
        f"{destination.stem}.tmp{os.getpid()}{destination.suffix}"
    )
    atomic_command = program.command[:-1] + [tmp_destination.as_posix()]

    started = time.perf_counter()
    try:
        ffmpeg.run(atomic_command[1:])
    except ffmpeg.FfmpegFailed as exc:
        tmp_destination.unlink(missing_ok=True)
        console.print(f"[red]render failed: {exc}[/red]")
        raise typer.Exit(EXIT_INPUT) from exc
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    if not tmp_destination.is_file() or tmp_destination.stat().st_size == 0:
        tmp_destination.unlink(missing_ok=True)
        console.print("[red]render produced no output[/red]")
        raise typer.Exit(EXIT_INPUT)

    # Probe the actual output rather than trusting the exit code. ffmpeg can
    # return 0 having written a file whose duration does not match what the
    # EDL specified — a truncated render from a killed process that still
    # exited clean, or a filter graph mistake nothing else would catch.
    try:
        out_meta = probe_video(tmp_destination)
        cut_ms = sum(op.duration_ms for op in edl.ops if op.op == "CUT")
        expected_ms = result.ingested.meta.durationMs - cut_ms
        drift_ms = abs(out_meta.durationMs - expected_ms)
        if drift_ms > 1500:
            tmp_destination.unlink(missing_ok=True)
            console.print(
                f"[red]render verification failed:[/red] output is "
                f"{out_meta.durationMs}ms, expected {expected_ms}ms "
                f"(drift {drift_ms}ms) — nothing was written to {destination}"
            )
            raise typer.Exit(EXIT_INPUT)
    except (ffmpeg.FfmpegFailed, UnsupportedInput) as exc:
        tmp_destination.unlink(missing_ok=True)
        console.print(f"[red]could not verify the render: {exc}[/red]")
        raise typer.Exit(EXIT_INPUT) from exc

    tmp_destination.replace(destination)

    console.print()
    console.print(
        f"  [green]rendered[/green] {destination}  "
        f"({elapsed_ms} ms, video {'copied' if program.video_stream_copied else 're-encoded'})"
    )

    # Not every finding is fixed by a filter graph. Captions are repaired by
    # writing a file, and the word-level timings are already in hand.
    if result.transcript is not None:
        captions = write_captions(
            result.transcript, destination.with_suffix(".vtt")
        )
        if captions is not None:
            console.print(f"  [green]wrote[/green] {captions}  (from word-level timings)")

    console.print()
    raise typer.Exit(EXIT_OK)


@app.command()
def doctor(
    offline: bool = typer.Option(False, "--offline", help="Show the no-key plan"),
    deep: bool = typer.Option(False, "--deep", help="Make one real call per hosted provider"),
    strict: bool = typer.Option(False, "--strict", help="Exit 1 on any warning (for CI)"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Diagnose the environment, credentials and capability plan.

    Every failure line carries the command that fixes it.
    """
    report = run_doctor(offline=offline, deep=deep)

    if as_json:
        console.print_json(json.dumps(report.to_json()))
        raise typer.Exit(EXIT_OK if not report.failures else EXIT_INPUT)

    tone = {"ok": "green", "warn": "yellow", "fail": "red", "note": "dim"}
    console.print()
    for section, checks in report.sections.items():
        console.print(f"  [bold]{section}[/bold]")
        for check in checks:
            colour = tone[check.status]
            console.print(
                f"    [{colour}]{check.glyph}[/{colour}] {check.name:<22} {check.detail}"
            )
            if check.fix and check.status in {"fail", "warn", "note"}:
                console.print(f"        [dim]-> {check.fix}[/dim]")
        console.print()

    if report.failures:
        console.print(f"  [red]{len(report.failures)} problem(s) to fix.[/red]")
        console.print()
        raise typer.Exit(EXIT_INPUT)

    if report.warnings and strict:
        console.print(f"  [yellow]{len(report.warnings)} warning(s), --strict.[/yellow]")
        console.print()
        raise typer.Exit(EXIT_INPUT)

    console.print("  [green]Ready.[/green]")
    console.print()
    raise typer.Exit(EXIT_OK)


@app.command()
def agents() -> None:
    """Print the agent roster declared in prompts/, and its conformance."""
    roster = load_roster()

    table = Table(show_header=True, box=None, pad_edge=False, header_style="dim")
    table.add_column("ID", width=5, no_wrap=True)
    table.add_column("CODENAME", width=13, no_wrap=True)
    table.add_column("KIND", width=14, no_wrap=True)
    table.add_column("CAPABILITY", width=16, no_wrap=True)
    table.add_column("DEPENDS ON", width=20, no_wrap=True)
    table.add_column("STATE", width=10, no_wrap=True)

    for spec in roster.ordered:
        if not spec.implemented:
            state = "[yellow]not built[/yellow]"
        elif spec.is_model_driven:
            state = "[green]built[/green]"
        else:
            state = "[green]built[/green]"
        table.add_row(
            spec.agent_id,
            spec.codename,
            spec.kind,
            spec.model_capability if spec.model_capability != "none" else "[dim]—[/dim]",
            ", ".join(spec.parents) or "[dim]—[/dim]",
            state,
        )

    problems = roster.validate()
    built = sum(1 for s in roster.ordered if s.implemented)

    console.print()
    console.print("  [bold]AGENT ROSTER[/bold]   [dim]declared in prompts/[/dim]")
    console.print()
    console.print(table)
    console.print()
    console.print(
        f"  {len(roster.agents)} agents · {built} built · "
        f"{len(roster.model_driven)} model-driven · "
        f"roster digest {roster.digest[:16]}"
    )
    if problems:
        console.print()
        for problem in problems:
            console.print(f"  [red]![/red] {problem}")
        console.print()
        raise typer.Exit(EXIT_INPUT)
    console.print("  [dim]roster is a valid DAG[/dim]")
    console.print()
    raise typer.Exit(EXIT_OK)


@app.command()
def bench(
    ablation: bool = typer.Option(
        False, "--ablation", help="Report every layer, not just the shipped one"
    ),
    labels_path: Path = typer.Option(
        Path("data/corpus/labels.jsonl"), "--labels", help="Ground truth"
    ),
    clips_dir: Path = typer.Option(Path("data/corpus/clips"), "--clips"),
    cache_dir: Path = typer.Option(Path(".preflight/cache"), "--cache-dir"),
    out: Path = typer.Option(None, "--out", help="Write the full result as JSON"),
    offline: bool = typer.Option(False, "--offline", help="Never touch the network"),
    limit: int = typer.Option(0, "--limit", help="First N clips only, for a smoke run"),
) -> None:
    """Score the pipeline against the golden corpus.

    Pair accuracy is the number to read. Clip accuracy is near-worthless on a
    corpus built as twins — answer VIOLATION to everything and you score 50%.
    A pair counts only when BOTH twins are right, which a keyword filter
    cannot do by construction, because at the level of the words present the
    twins are identical.
    """
    if not ffmpeg.available():
        console.print("[red]ffmpeg and ffprobe are required.[/red]")
        raise typer.Exit(EXIT_UPSTREAM)

    labels = bench_mod.load_labels(labels_path)
    if not labels:
        console.print(f"[red]no ground truth at {labels_path}[/red]")
        raise typer.Exit(EXIT_INPUT)

    present = [
        label for label in labels if (Path(clips_dir) / label.clip).is_file()
    ]
    if not present:
        console.print(
            f"[red]no clips in {clips_dir}[/red] — "
            "the corpus is gitignored; run [bold]make corpus[/bold] to generate it"
        )
        raise typer.Exit(EXIT_INPUT)
    if limit:
        present = present[:limit]

    settings = Settings.load(offline=True) if offline else Settings.load()
    stages = bench_mod.STAGES if ablation else ("triad",)

    console.print()
    console.print(
        f"  [bold]BENCH[/bold]   {len(present)}/{len(labels)} clips · "
        f"{len(bench_mod.pairs(present))} pairs · [dim]{settings.describe_mode()}[/dim]"
    )
    console.print()

    with console.status("[dim]running…[/dim]") as status:
        def progress(row):
            mark = "[green]ok[/green]" if row[stages[-1]]["correct"] else "[red]x[/red]"
            status.update(f"[dim]{row['clip']} {mark}[/dim]")

        scored, per_clip = bench_mod.run_bench(
            present,
            clips_dir=clips_dir,
            settings=settings,
            store=cas.Store(cache_dir),
            stages=stages,
            progress=progress,
        )

    table = Table(show_header=True, box=None, pad_edge=False, header_style="dim")
    table.add_column("LAYER", width=10, no_wrap=True)
    table.add_column("PREC", width=7, justify="right")
    table.add_column("RECALL", width=7, justify="right")
    table.add_column("F1", width=7, justify="right")
    table.add_column("PAIRS", width=9, justify="right")
    table.add_column("SPAN", width=7, justify="right")
    table.add_column("CALLS", width=7, justify="right")

    describe = {
        "lexicon": "what a keyword filter sees",
        "auditor": "+ retrieval, charges before any defence",
        "triad": "+ advocate and adjudicator — shipped",
    }
    for stage in stages:
        metrics = scored[stage]
        table.add_row(
            stage,
            f"{metrics.precision:.2f}",
            f"{metrics.recall:.2f}",
            f"{metrics.f1:.2f}",
            f"{metrics.pairs_correct}/{metrics.pairs_total}",
            f"{metrics.span_accuracy:.2f}",
            str(metrics.calls),
        )

    console.print(table)
    console.print()
    for stage in stages:
        console.print(f"  [dim]{stage:<9} {describe.get(stage, '')}[/dim]")
    console.print()

    shipped = scored[stages[-1]]
    console.print(
        f"  [bold]{shipped.pairs_correct}/{shipped.pairs_total} pairs[/bold] "
        f"({shipped.pair_accuracy:.0%}) — both twins correct"
    )
    if len(stages) > 1 and "auditor" in scored:
        delta = shipped.pairs_correct - scored["auditor"].pairs_correct
        console.print(
            f"  [dim]the advocate and adjudicator are worth "
            f"{delta:+d} pair(s) over the charge sheet alone[/dim]"
        )
    console.print()

    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(bench_mod.to_json(scored, per_clip), indent=2) + "\n",
            encoding="utf-8",
        )
        console.print(f"  [dim]wrote {out}[/dim]")
        console.print()

    raise typer.Exit(EXIT_OK)


@app.command()
def capabilities(
    offline: bool = typer.Option(False, "--offline"),
) -> None:
    """Print the capability plan: which provider serves what, and why."""
    registry = Registry(offline=offline)

    table = Table(show_header=True, box=None, pad_edge=False, header_style="dim")
    table.add_column("CAPABILITY", width=18)
    table.add_column("PROVIDER", width=9)
    table.add_column("SERVED BY", width=42)
    table.add_column("TIER", width=12)

    for capability, resolution in registry.plan.items():
        if resolution.tier_label == "null":
            tier, colour = "unavailable", "yellow"
        elif resolution.degraded:
            tier, colour = "fallback", "cyan"
        else:
            tier, colour = "preferred", "green"
        table.add_row(
            capability,
            resolution.provider.id,
            resolution.label if resolution.tier_label != "null" else resolution.reason[:42],
            f"[{colour}]{tier}[/{colour}]",
        )

    preferred, fallback, unavailable = registry.summary()
    console.print()
    console.print("  [bold]CAPABILITY PLAN[/bold]")
    console.print()
    console.print(table)
    console.print()
    console.print(
        f"  {len(registry.plan)} capabilities · {preferred} preferred · "
        f"{fallback} fallback · {unavailable} unavailable"
    )
    console.print()
    raise typer.Exit(EXIT_OK)


@app.command()
def snapshot(
    out: Path = typer.Option(
        Path("data/policy-snapshots/latest.json"), "--out", help="Snapshot path"
    ),
    policy_dir: Path = typer.Option(Path("data/policy"), "--policy-dir"),
) -> None:
    """Capture the current policy corpus for later drift comparison."""
    corpus = load_corpus(policy_dir)
    path = write_snapshot(corpus, out)
    console.print()
    console.print(
        f"  captured [bold]{len(corpus.clauses)}[/bold] clauses "
        f"(version {corpus.version}, digest {corpus.digest[:16]}…)"
    )
    console.print(f"  wrote [bold]{path}[/bold]")
    console.print()
    raise typer.Exit(EXIT_OK)


@app.command()
def drift(
    against: Path = typer.Option(
        Path("data/policy-snapshots/latest.json"), "--against", help="Prior snapshot"
    ),
    policy_dir: Path = typer.Option(Path("data/policy"), "--policy-dir"),
    archive_path: Path = typer.Option(Path(".preflight/archive.db"), "--archive"),
    out: Path = typer.Option(None, "--out", help="Write the drift report as JSON"),
) -> None:
    """Detect policy changes and find which archived videos they put at risk.

    Your back catalogue was compliant when you uploaded it. The rules changed.
    """
    if not against.is_file():
        console.print(f"[red]no snapshot at {against}[/red]")
        console.print("  capture one first:  preflight snapshot")
        raise typer.Exit(EXIT_INPUT)

    settings = Settings.load()
    store = cas.Store(settings.cache_dir)
    archive = Archive(archive_path)

    # Semantic delta needs an embedder. Without one the diff still works from
    # text similarity, and the report says which method produced the number.
    embed = None
    backend = "text-similarity"
    if settings.online:
        client = NimClient(settings, store)

        def embed(texts):  # noqa: F811
            return client.embed(texts, model=settings.models.embed, input_type="passage")

        backend = f"embeddings:{settings.models.embed}"

    report = detect(against, policy_dir, archive, embed=embed)

    console.print()
    if not report.changes:
        console.print(f"  [green]no policy drift[/green] since {report.from_version}")
        console.print()
        raise typer.Exit(EXIT_OK)

    console.print(
        f"[bold]POLICY DRIFT DETECTED[/bold]   {report.detected_at[:10]}   "
        f"[dim]{report.from_version} → {report.to_version}[/dim]"
    )
    console.print()

    table = Table(show_header=True, box=None, pad_edge=False, header_style="dim")
    table.add_column("CLAUSE", width=8)
    table.add_column("TITLE", width=32)
    table.add_column("CHANGE", width=10)
    table.add_column("Δ", width=7, justify="right")
    table.add_column("", width=10)
    for change in report.changes:
        tone = {"ADDED": "green", "REMOVED": "red", "MODIFIED": "yellow"}[change.kind]
        table.add_row(
            change.clause_id,
            change.title[:32],
            f"[{tone}]{change.kind}[/{tone}]",
            f"{change.semantic_delta:.3f}",
            "" if change.material else "[dim]cosmetic[/dim]",
        )
    console.print(table)
    console.print()
    console.print(f"  [dim]semantic delta via {backend}[/dim]")
    console.print()

    if report.archive_size == 0:
        console.print(
            "  [dim]archive is empty — run `preflight check <video> --format json` "
            "to record reports for drift monitoring[/dim]"
        )
        console.print()
        raise typer.Exit(EXIT_OK)

    console.print(
        f"  Re-lint [bold]{len(report.affected)}[/bold] of "
        f"[bold]{report.archive_size}[/bold] archived videos "
        f"[dim](selective invalidation)[/dim]"
    )
    console.print()

    if report.affected:
        console.print("  [bold]NEWLY AT RISK[/bold]")
        console.print()
        for video in report.affected:
            touched = sorted(
                report.changed_clause_ids
                & (set(video.clauses) | set(video.near_miss_clauses))
            )
            reason = ", ".join(touched)
            near = set(touched) & set(video.near_miss_clauses)
            marker = " [dim](considered, did not fire)[/dim]" if near else ""
            console.print(
                f"    {video.filename:<28} readiness {video.overall:>3}   "
                f"[dim]{reason}[/dim]{marker}"
            )
        console.print()

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")
        console.print(f"  wrote [bold]{out}[/bold]")
        console.print()

    raise typer.Exit(EXIT_FINDINGS if report.affected else EXIT_OK)


@app.command()
def cache(
    cache_dir: Path = typer.Option(Path(".preflight/cache"), "--cache-dir"),
    clear: bool = typer.Option(False, "--clear", help="Delete the store"),
) -> None:
    """Inspect or clear the content-addressed store."""
    store = cas.Store(cache_dir)
    if clear:
        store.clear()
        console.print("[dim]cache cleared[/dim]")
        raise typer.Exit(EXIT_OK)

    labels = {
        "v": "video artifacts",
        "t": "transcripts",
        "f": "fingerprints",
        "p": "policy indexes",
        "r": "reports",
    }
    for namespace, count in store.stats().items():
        console.print(f"  {namespace}  {labels[namespace]:<18} {count}")
    raise typer.Exit(EXIT_OK)


if __name__ == "__main__":  # pragma: no cover
    app()

