"""PREFLIGHT command line.

Exit codes:
    0  pass
    1  findings exceed the configured threshold
    2  input or configuration error
    3  upstream unavailable and no fallback permitted
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from preflight import __version__, cas, ffmpeg
from preflight.ingest.pipeline import ingest
from preflight.ingest.probe import UnsupportedInput

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
