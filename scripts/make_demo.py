"""Build a demo clip with real narration, so the whole pipeline can run.

`samples/synthetic.mp4` is a test pattern with a sine tone. It exercises ingest
and the audio agent, but it contains no speech — so ASR returns nothing, the
policy triad has nothing to adjudicate, and `check -> fix -> check` cannot move
the score. This builds a clip that can.

The narration is synthesised with the platform's built-in text-to-speech (SAPI
on Windows, `say` on macOS, espeak elsewhere), so no audio is downloaded and
nothing here belongs to anyone. It deliberately contains material that maps to
specific clauses:

    profanity          -> AF-01 -> BLEEP
    injury and blood   -> AF-02 -> BLUR_REGION / MUTE
    casualty figures   -> AF-10 -> MUTE
    alcohol reference  -> AF-12 -> advisory

Real footage is still better for the submission video. This exists so the
engine can be developed and demonstrated without waiting for it.

    python scripts/make_demo.py
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

OUT = Path("samples/demo.mp4")
META = Path("samples/demo.meta.json")

NARRATION = (
    "Morning of day four, and the weather window on the north col finally opened up. "
    "We moved fast on the fixed lines. "
    "Then the anchor pulled clean out of the ice and the whole shelf just went. "
    "This is fucked. We need to be off this face in ten minutes. "
    "He slipped and went straight down onto the rocks. "
    "Look at the blood. That is a bad one. Get the kit open, get pressure on it now. "
    "Everyone here still talks about the two thousand nineteen slide on the north col. "
    "Eleven people went out that morning, and four of them did not come back down. "
    "Back at camp we cracked the whisky we had been hauling since base camp. "
    "One finger each, purely medicinal. "
    "Tomorrow we descend."
)

META_JSON = """{
  "title": "NORTH COL - THE FULL UNCUT ASCENT DOCUMENTARY 2026 EDITION",
  "description": "Our climb. Gear at amzn.to/3xKp2Qw",
  "tags": [
    "climbing", "mountaineering", "expedition", "documentary", "north col",
    "alpine", "himalaya", "vlog", "adventure", "outdoors", "survival",
    "extreme sports", "travel", "nature", "4k", "summit push", "high altitude"
  ],
  "category": "Travel & Events"
}
"""


def synthesise(text: str, destination: Path) -> bool:
    """Platform text-to-speech to a WAV. False when unavailable."""
    system = platform.system()

    if system == "Windows":
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$s.Rate = -1; "
            f"$s.SetOutputToWaveFile('{destination}'); "
            f"$s.Speak(@'\n{text}\n'@); "
            "$s.Dispose()"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 and destination.is_file()

    if system == "Darwin" and shutil.which("say"):
        aiff = destination.with_suffix(".aiff")
        subprocess.run(["say", "-o", str(aiff), text], check=True)
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(aiff), str(destination)],
            check=True,
        )
        aiff.unlink(missing_ok=True)
        return destination.is_file()

    if shutil.which("espeak-ng") or shutil.which("espeak"):
        binary = shutil.which("espeak-ng") or shutil.which("espeak")
        subprocess.run([binary, "-s", "150", "-w", str(destination), text], check=True)
        return destination.is_file()

    return False


def duration_of(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out or 0.0)


def main() -> int:
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found on PATH.", file=sys.stderr)
        return 3

    OUT.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        speech = Path(tmp) / "speech.wav"
        if not synthesise(NARRATION, speech):
            print(
                "No text-to-speech backend available.\n"
                "  Windows: built in (SAPI)\n"
                "  macOS:   built in (say)\n"
                "  Linux:   sudo apt install espeak-ng",
                file=sys.stderr,
            )
            return 3

        seconds = duration_of(speech)
        if seconds < 1:
            print("synthesised speech was empty", file=sys.stderr)
            return 3
        print(f"narration: {seconds:.1f}s")

        # Three visually distinct segments so scene detection has real cuts.
        segment = seconds / 3
        bed = Path("assets/cc_music/glacier_calm.mp3")
        has_bed = bed.is_file()

        inputs = [
            "-f", "lavfi", "-i",
            f"gradients=size=1280x720:rate=30:duration={segment:.2f}:n=3",
            "-f", "lavfi", "-i",
            f"color=c=0x1b2838:size=1280x720:rate=30:duration={segment:.2f}",
            "-f", "lavfi", "-i",
            f"testsrc2=size=1280x720:rate=30:duration={segment:.2f}",
            "-i", str(speech),
        ]
        if has_bed:
            inputs += ["-i", str(bed)]

        # Speech over a quiet bed, so the copyright detector has something real
        # to find under the narration.
        if has_bed:
            audio_filter = (
                "[3:a]volume=1.0,aresample=44100[sp];"
                "[4:a]atrim=0:{d},asetpts=PTS-STARTPTS,volume=0.22,"
                "aresample=44100[bed];"
                "[sp][bed]amix=inputs=2:duration=first:normalize=0[aout]"
            ).format(d=f"{seconds:.2f}")
        else:
            audio_filter = "[3:a]aresample=44100[aout]"

        filter_complex = (
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0[vout];" + audio_filter
        )

        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs,
             "-filter_complex", filter_complex,
             "-map", "[vout]", "-map", "[aout]",
             "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "160k", "-shortest", str(OUT)],
            check=True,
        )

    META.write_text(META_JSON, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")
    print(f"wrote {META}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
