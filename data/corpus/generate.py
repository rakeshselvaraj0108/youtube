"""Build the labelled test corpus deterministically. No third-party media.

Everything here is synthesised: narration from the platform's built-in
text-to-speech, footage from ffmpeg's lavfi sources, and the marker music bed
from oscillators. Nothing is downloaded and nothing belongs to anyone.

The payoff is exact ground truth. A sourced clip requires a human to eyeball
where a violation starts, and that estimate becomes the label — so span IoU
measured against it is really measuring the annotator. Here the span is known
because the generator put it there.

    python data/corpus/generate.py
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    print("pyyaml required:  pip install pyyaml", file=sys.stderr)
    raise SystemExit(3)

CORPUS = Path("data/corpus")
CLIPS = CORPUS / "clips"
LEXICONS = Path("data/lexicons")
MARKER_BED = Path("data/assets/cc_music/marker_bed_01.wav")

FFMPEG = "ffmpeg"


def strong_profanity() -> str:
    """Read the tier-3 term from the lexicon rather than hardcoding it here."""
    for line in (LEXICONS / "profanity.tiered.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("tier") == 3:
            return record["term"]
    return "damn"


def run(args: list[str]) -> None:
    result = subprocess.run(
        [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-800:]}")


def tts(text: str, out: Path) -> bool:
    """Platform text-to-speech. Windows SAPI, macOS say, espeak-ng elsewhere."""
    system = platform.system()

    if system == "Windows":
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$s.Rate = 0; "
            f"$s.SetOutputToWaveFile('{out}'); "
            f"$s.Speak(@'\n{text}\n'@); $s.Dispose()"
        )
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True,
        )
        return proc.returncode == 0 and out.is_file()

    if system == "Darwin" and shutil.which("say"):
        aiff = out.with_suffix(".aiff")
        subprocess.run(["say", "-o", str(aiff), text], check=True, capture_output=True)
        run(["-i", str(aiff), str(out)])
        aiff.unlink(missing_ok=True)
        return out.is_file()

    binary = shutil.which("espeak-ng") or shutil.which("espeak")
    if binary:
        subprocess.run([binary, "-s", "150", "-w", str(out), text],
                       check=True, capture_output=True)
        return out.is_file()
    return False


def drawtext_font() -> str:
    """An explicit `fontfile=` clause for drawtext, or empty to use fontconfig.

    Windows ffmpeg builds ship without a fontconfig default, so drawtext fails
    with "Cannot load default config file" unless a font is named outright. The
    drive-letter colon also has to be escaped or drawtext reads it as an option
    separator.
    """
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    ]
    for font in candidates:
        if font.is_file():
            escaped = font.as_posix().replace(":", r"\:")
            return f"fontfile='{escaped}':"
    return ""


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out or 0.0)


def ensure_marker_bed() -> None:
    """Generated, so the corpus stays fully self-contained."""
    if MARKER_BED.is_file():
        return
    MARKER_BED.parent.mkdir(parents=True, exist_ok=True)
    run([
        "-f", "lavfi", "-i", "sine=frequency=220:duration=30",
        "-f", "lavfi", "-i", "sine=frequency=330:duration=30",
        "-filter_complex", "[0][1]amix=inputs=2,tremolo=f=0.5:d=0.3",
        str(MARKER_BED),
    ])


def build_video(spec: dict, out: Path) -> None:
    """Neutral generated footage, with visual defects applied inline."""
    duration = spec["duration_s"]
    res = spec["resolution"]
    fps = spec["fps"]
    visual = spec.get("visual")

    if visual == "strobe":
        s = spec["strobe"]
        hz, start, end = s["hz"], s["start_s"], s["end_s"]
        half = 1.0 / (2 * hz)
        # A white frame overlaid on alternating half-cycles inside the window.
        #
        # `geq` would express this more directly but evaluates per pixel — 600
        # frames of 720p is 550 million evaluations per clip, minutes each. An
        # overlay gated on the timeline produces an identical luminance series
        # and renders in about a second.
        gate = f"between(t,{start},{end})*lt(mod(t-{start},{2 * half}),{half})"
        run([
            "-f", "lavfi", "-i", f"gradients=s={res}:r={fps}:d={duration}",
            "-f", "lavfi", "-i", f"color=c=white:s={res}:r={fps}:d={duration}",
            "-filter_complex",
            f"[0:v]format=yuv420p[bg];[bg][1:v]overlay=enable='{gate}'[v]",
            "-map", "[v]", "-t", str(duration),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(out),
        ])
        return

    if visual == "fade":
        # The control for the strobe pair: the same total luminance travel,
        # delivered as a ramp. A detector that fires here is measuring
        # brightness change rather than counting flashes.
        half_d = duration / 2
        run([
            "-f", "lavfi", "-i", f"gradients=s={res}:r={fps}:d={duration}",
            "-vf",
            f"fade=t=in:st=0:d={half_d},fade=t=out:st={half_d}:d={half_d},"
            "format=yuv420p",
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "ultrafast", str(out),
        ])
        return

    run([
        "-f", "lavfi", "-i", f"gradients=s={res}:r={fps}:d={duration}",
        "-vf", "format=yuv420p",
        "-t", str(duration), "-c:v", "libx264", "-preset", "ultrafast",
        str(out),
    ])

    if "burn_in_text" in spec:
        b = spec["burn_in_text"]
        tmp = out.with_name(f"{out.stem}_txt.mp4")
        font = drawtext_font()
        run([
            "-i", str(out), "-vf",
            f"drawtext={font}text='{b['text']}':fontcolor=white:fontsize=44:"
            f"box=1:boxcolor=black@0.6:x=(w-tw)/2:y=h-th-40:"
            f"enable='between(t,{b['start_s']},{b['end_s']})'",
            "-c:v", "libx264", "-preset", "ultrafast", str(tmp),
        ])
        tmp.replace(out)


def build_audio(spec: dict, voice: Path, out: Path) -> None:
    duration = spec["duration_s"]
    inputs = ["-i", str(voice)]
    music = spec.get("music")

    if music:
        ensure_marker_bed()
        inputs += ["-i", str(MARKER_BED)]
        duck = music.get("duck_db", 0)
        level = music.get("level_db", -14)
        if duck > 0:
            # Sidechain the bed under the speech: the signature of a placed,
            # deliberate music bed rather than incidental background.
            fc = (
                f"[1:a]aloop=loop=-1:size=2e9,atrim=0:{duration},volume={level}dB[m];"
                f"[m][0:a]sidechaincompress=threshold=0.05:ratio=8:attack=20:"
                f"release=300[md];[0:a][md]amix=inputs=2:duration=first:"
                f"normalize=0[a]"
            )
        else:
            fc = (
                f"[1:a]aloop=loop=-1:size=2e9,atrim=0:{duration},volume={level}dB[m];"
                f"[0:a][m]amix=inputs=2:duration=first:normalize=0[a]"
            )
    else:
        fc = "[0:a]anull[a]"

    defect = spec.get("audio_defect") or {}
    kind = defect.get("kind")
    if kind == "dead_channel":
        pan = ("stereo|c0=c0|c1=0*c0" if defect.get("channel") == "right"
               else "stereo|c0=0*c1|c1=c1")
        fc += f";[a]aformat=channel_layouts=stereo,pan={pan}[d]"
    elif kind == "hot":
        fc += f";[a]volume={defect.get('gain_db', 12)}dB,alimiter=limit=0.999[d]"
    elif kind == "normalize":
        fc += ";[a]loudnorm=I=-14:TP=-1.0:LRA=11[d]"
    else:
        fc += ";[a]anull[d]"

    # Pad to the declared duration. The narration is almost always shorter than
    # the clip, and `-t` caps without extending — so `-shortest` at mux time
    # truncated every clip to its speech and left visual defects specified at
    # 6-9s sitting past the end of the file. g010's strobe was simply not in
    # the rendered video.
    fc += f";[d]apad,atrim=0:{duration},asetpts=N/SR/TB[out]"

    run([*inputs, "-filter_complex", fc, "-map", "[out]",
         "-t", str(duration), "-ar", "44100", "-ac", "2", str(out)])


def build() -> int:
    if not shutil.which(FFMPEG):
        print("ffmpeg not found on PATH", file=sys.stderr)
        return 3

    manifest = yaml.safe_load((CORPUS / "manifest.yaml").read_text(encoding="utf-8"))
    defaults = manifest["defaults"]
    profanity = strong_profanity()

    CLIPS.mkdir(parents=True, exist_ok=True)
    labels: list[dict] = []
    built = 0

    for raw in manifest["clips"]:
        spec = {**defaults, **raw}
        clip_id = spec["id"]
        script = " ".join(spec["script"].split()).replace("[PROFANITY_STRONG]", profanity)

        with tempfile.TemporaryDirectory() as tmp:
            voice = Path(tmp) / "voice.wav"
            if not tts(script, voice):
                print("no text-to-speech backend available:", file=sys.stderr)
                print("  Windows: built in   macOS: built in   "
                      "Linux: sudo apt install espeak-ng", file=sys.stderr)
                return 3

            silent = Path(tmp) / "video.mp4"
            audio = Path(tmp) / "audio.wav"
            build_video(spec, silent)
            build_audio(spec, voice, audio)

            out = CLIPS / f"{clip_id}.mp4"
            run(["-i", str(silent), "-i", str(audio),
                 "-map", "0:v", "-map", "1:a",
                 "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                 "-shortest", str(out)])

        if "meta" in spec:
            meta = dict(spec["meta"])
            (CLIPS / f"{clip_id}.meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )

        span = spec.get("expect_span_s")
        labels.append({
            "clip": f"{clip_id}.mp4",
            "label": spec["label"],
            "clause": spec.get("clause"),
            "span_ms": [int(x * 1000) for x in span] if span else None,
            "expect_findings": spec.get("expect_findings"),
            "twin_of": spec.get("twin_of"),
            "note": " ".join(str(spec.get("note", "")).split()),
        })
        built += 1
        print(f"  {clip_id}  {spec['label']:<10} {spec.get('clause') or '-'}")

    (CORPUS / "labels.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in labels) + "\n", encoding="utf-8"
    )

    violations = sum(1 for entry in labels if entry["label"] == "VIOLATION")
    print(f"\nbuilt {built} clips: {violations} violation / {built - violations} clean")
    print(f"wrote {CORPUS / 'labels.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
