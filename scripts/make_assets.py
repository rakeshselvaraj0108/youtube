"""Synthesise the CC0 replacement audio bed.

REPLACE_AUDIO swaps a copyright-matched music bed for a licence-free one. That
replacement cannot itself be someone else's recording, so it is generated here
rather than downloaded: a soft filtered-noise pad with a slow amplitude
envelope. Nobody will release it as a single, but it sits under narration
without drawing attention, which is the entire job.

Generated audio has no rights holder, which is the point — the file that fixes
a copyright finding must not create one.

    python scripts/make_assets.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

OUT = Path("assets/cc_music/glacier_calm.mp3")
DURATION_S = 120


def main() -> int:
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found on PATH.", file=sys.stderr)
        return 3

    OUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            # Two detuned sines plus filtered noise: a pad, not a melody.
            "-f", "lavfi", "-i", f"sine=frequency=110:duration={DURATION_S}",
            "-f", "lavfi", "-i", f"sine=frequency=164.81:duration={DURATION_S}",
            "-f", "lavfi", "-i", f"anoisesrc=d={DURATION_S}:c=pink:a=0.03",
            "-filter_complex",
            "[0:a]volume=0.16[a];"
            "[1:a]volume=0.10[b];"
            "[2:a]lowpass=f=900,volume=0.5[c];"
            "[a][b][c]amix=inputs=3:duration=first:normalize=0,"
            # Slow swell so it does not read as a test tone. 0.1Hz is the
            # lowest rate the tremolo filter accepts.
            "tremolo=f=0.1:d=0.4,"
            "afade=t=in:st=0:d=3,"
            f"afade=t=out:st={DURATION_S - 3}:d=3,"
            "loudnorm=I=-24:TP=-3.0:LRA=7[out]",
            "-map", "[out]",
            "-c:a", "libmp3lame", "-b:a", "128k",
            str(OUT),
        ],
        check=True,
    )
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes, {DURATION_S}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
