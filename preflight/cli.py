"""PREFLIGHT command line.

Exit codes:
    0  pass
    1  findings exceed the configured threshold
    2  input or configuration error
    3  upstream unavailable and no fallback permitted
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from preflight import __version__, cas, ffmpeg
from preflight.ingest.pipeline import ingest
from preflight.ingest.probe import UnsupportedInput
from preflight.models import SEVERITY_RANK
from preflight.pipeline import SURFACE_WEIGHT, run_perception
from preflight.config import Settings
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
) -> None:
    """Analyse a video and print findings.

    Phase 2 runs ingest plus the offline perception agents. Retrieval, the
    adversarial triad and scoring arrive in later phases; the exit code is not
    yet a monetization verdict.
    """
    if not ffmpeg.available():
        console.print("[red]ffmpeg and ffprobe are required.[/red]")
        raise typer.Exit(EXIT_UPSTREAM)

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
                f"  [{severity_tone[finding.severity]}]●[/] "
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
        marker = " [dim]← weakest[/dim]" if key == readiness.weakest else ""
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
        # Distinguish an agent that ran badly from one that does not exist yet.
        # Reporting both as "degraded" would misstate why coverage is short.
        impaired = [
            a.name for a in result.agents if a.status in {"SKIPPED", "FAILED", "DEGRADED"}
        ]
        ran = {a.agent_id for a in result.agents}
        unbuilt = [
            agent_id
            for agent_id, weight in SURFACE_WEIGHT.items()
            if weight > 0 and agent_id not in ran
        ]
        console.print(f"  [yellow]PARTIAL ANALYSIS[/yellow] — {coverage * 100:.0f}% coverage")
        if impaired:
            console.print(f"    impaired: {', '.join(impaired)}")
        if unbuilt:
            console.print(f"    not yet implemented: {', '.join(sorted(unbuilt))}")
    # Emission.
    formats = {f.strip().lower() for f in fmt.split(",") if f.strip()}
    if html:
        formats.add("html")
    if "all" in formats:
        formats = {"json", "sarif", "certificate", "html"}
    if emit_fixture is not None:
        formats.add("json")

    if formats or emit_fixture is not None:
        _emit(result, formats, out, emit_fixture)

    console.print()
    raise typer.Exit(EXIT_OK if readiness.verdict in PASSING else EXIT_FINDINGS)


def _emit(result, formats: set[str], out: Path, fixture: Path | None) -> None:
    """Write the requested artifacts. Validates before writing anything."""
    settings = Settings.load()
    bundle = build_report(
        result,
        policy_version=result.corpus.version if result.corpus else "unknown",
        embed_media="html" in formats or fixture is not None,
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
    out: Path = typer.Option(None, "--out", help="Output path (default <name>.safe.mp4)"),
    cache_dir: Path = typer.Option(Path(".preflight/cache"), "--cache-dir"),
) -> None:
    """Compile findings into an ffmpeg program and optionally run it.

    Dry-runs by default. A tool that rewrites someone's master file because
    they typed the wrong command has lost more than it gained.
    """
    if not ffmpeg.available():
        console.print("[red]ffmpeg and ffprobe are required.[/red]")
        raise typer.Exit(EXIT_UPSTREAM)

    store = cas.Store(cache_dir)
    try:
        result = run_perception(video, store)
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

    started = time.perf_counter()
    try:
        ffmpeg.run(program.command[1:])
    except ffmpeg.FfmpegFailed as exc:
        console.print(f"[red]render failed: {exc}[/red]")
        raise typer.Exit(EXIT_INPUT) from exc
    elapsed_ms = int((time.perf_counter() - started) * 1000)

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
