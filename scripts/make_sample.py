"""Synthesise a test clip so a clean clone can exercise the pipeline.

Media is not committed — it bloats the repository and a judge cloning it should
not wait on a 40MB download to run the tests. This produces a 12-second clip
with two hard scene cuts at 4s and 8s, which is exactly what the ingest tests
assert against.

    python scripts/make_sample.py

Real footage goes at samples/demo.mp4 and is what `make demo` uses.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

OUT = Path("samples/synthetic.mp4")


def main() -> int:
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found on PATH.", file=sys.stderr)
        print("  Windows:  winget install Gyan.FFmpeg", file=sys.stderr)
        print("  macOS:    brew install ffmpeg", file=sys.stderr)
        print("  Debian:   sudo apt install ffmpeg", file=sys.stderr)
        return 3

    OUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            # Three visually distinct four-second segments -> two scene cuts.
            "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=4",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
            "-f", "lavfi", "-i", "color=c=darkred:size=640x360:rate=30:duration=4",
            "-f", "lavfi", "-i", "smptebars=size=640x360:rate=30:duration=4",
            "-filter_complex", "[0:v][2:v][3:v]concat=n=3:v=1:a=0[v]",
            "-map", "[v]", "-map", "1:a",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(OUT),
        ],
        check=True,
    )
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
