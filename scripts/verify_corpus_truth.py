"""Check that the detectors measure what the generator constructed.

The corpus claims exact ground truth by construction. That claim is only worth
anything if the constructed defect is actually present in the rendered file —
an ffmpeg filter that silently did nothing would leave a clip labelled
VIOLATION with nothing in it to find, and every metric computed against it
would be quietly wrong.

This probes the deterministic detectors against the pairs they are supposed to
separate. It does not touch an LLM.

    python scripts/verify_corpus_truth.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from preflight.perception import signal as sig
from preflight.perception.accessibility import flash_risk
from preflight.ingest.audio import loudness

CLIPS = Path("data/corpus/clips")


def check_photosensitive() -> list[str]:
    print("\nPHOTOSENSITIVE — constructed flash rate vs measured")
    failures = []
    cases = [
        ("g010", "10Hz strobe", "HIGH"),
        ("g012", "5Hz strobe", "HIGH"),
        ("g011", "1Hz strobe", "LOW"),
        ("g013", "luminance ramp", "LOW"),
    ]
    for clip, described, expected in cases:
        path = CLIPS / f"{clip}.mp4"
        if not path.is_file():
            failures.append(f"{clip} missing")
            continue
        from preflight.perception.accessibility import SAMPLE_FPS

        series = sig.luminance_series(path, fps=SAMPLE_FPS)
        result = flash_risk(series, SAMPLE_FPS)
        peak = result["max_flashes_per_second"]
        risk = result["risk"]
        ok = risk == expected
        print(
            f"  {clip}  {described:<16} peak {peak}/s  risk {risk:<8} "
            f"expected {expected:<8} {'ok' if ok else 'MISMATCH'}"
        )
        if not ok:
            failures.append(f"{clip}: risk {risk}, expected {expected}")
    return failures


def check_channels() -> list[str]:
    print("\nCHANNEL BALANCE — dead mic vs balanced")
    failures = []
    for clip, described, expect_dead in [
        ("g050", "right channel killed", True),
        ("g051", "balanced stereo", False),
    ]:
        path = CLIPS / f"{clip}.mp4"
        if not path.is_file():
            failures.append(f"{clip} missing")
            continue
        wav = path.with_suffix(".probe.wav")
        import subprocess

        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(path), "-vn", "-ac", "2", "-ar", "44100", str(wav)],
            check=True, capture_output=True,
        )
        audio = sig.read_wav(wav)
        levels = np.sqrt((audio.samples.astype(np.float64) ** 2).mean(axis=1))
        levels = np.maximum(levels, 1e-9)
        db = 20 * np.log10(levels)
        delta = float(db.max() - db.min())
        wav.unlink(missing_ok=True)

        detected = delta >= 12.0
        ok = detected == expect_dead
        print(
            f"  {clip}  {described:<22} spread {delta:6.1f} dB  "
            f"{'dead' if detected else 'balanced':<9} {'ok' if ok else 'MISMATCH'}"
        )
        if not ok:
            failures.append(f"{clip}: spread {delta:.1f}dB, expected dead={expect_dead}")
    return failures


def check_loudness() -> list[str]:
    """Relative AND absolute.

    A relative-only check ("hot louder than clean") passed while the actual
    production detector fired on neither clip: g052 measured -12.4 LUFS
    against a -14+-2 target, 0.4 LUFS inside tolerance, and the VIOLATION
    clip did not violate. The relative comparison was blind to that, because
    it never asks the question the detector actually asks. Both are checked
    now, and the absolute one uses the real target and tolerance from the
    module that ships, not a duplicate copied here to drift out of sync.
    """
    from preflight.perception.audio import LUFS_TOLERANCE, TARGET_LUFS

    print("\nLOUDNESS — hot master vs normalised, against the real detector")
    failures = []
    measured: dict[str, float] = {}
    for clip, described in [("g052", "hot master"), ("g053", "normalised")]:
        path = CLIPS / f"{clip}.mp4"
        if not path.is_file():
            failures.append(f"{clip} missing")
            continue
        result = loudness(path)
        if not result:
            failures.append(f"{clip}: loudness unavailable")
            continue
        lufs = result["integrated_lufs"]
        measured[clip] = lufs
        fires = abs(lufs - TARGET_LUFS) > LUFS_TOLERANCE
        print(f"  {clip}  {described:<14} {lufs:7.1f} LUFS  fires={fires}")

    if len(measured) == 2 and measured["g052"] <= measured["g053"]:
        failures.append(
            f"hot master ({measured['g052']:.1f}) is not louder than "
            f"normalised ({measured['g053']:.1f})"
        )

    if "g052" in measured and abs(measured["g052"] - TARGET_LUFS) <= LUFS_TOLERANCE:
        failures.append(
            f"g052 (VIOLATION) measures {measured['g052']:.1f} LUFS, inside "
            f"the real {TARGET_LUFS}+-{LUFS_TOLERANCE} tolerance — the "
            "detector would not fire on the clip built to trigger it"
        )
    if "g053" in measured and abs(measured["g053"] - TARGET_LUFS) > LUFS_TOLERANCE:
        failures.append(
            f"g053 (CLEAN) measures {measured['g053']:.1f} LUFS, outside "
            f"the real {TARGET_LUFS}+-{LUFS_TOLERANCE} tolerance — the "
            "detector would fire on the clip built to stay clean"
        )
    return failures


def check_pairs() -> list[str]:
    """Every VIOLATION must have a CLEAN twin. The pairs are the discipline."""
    print("\nPAIRING — every violation has a clean twin")
    import json

    labels = [
        json.loads(line)
        for line in (Path("data/corpus/labels.jsonl")).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_clip = {entry["clip"].replace(".mp4", ""): entry for entry in labels}
    violations = [e for e in labels if e["label"] == "VIOLATION"]
    cleans = [e for e in labels if e["label"] == "CLEAN"]

    failures = []
    twinned = {e["twin_of"] for e in cleans if e.get("twin_of")}
    for entry in violations:
        clip = entry["clip"].replace(".mp4", "")
        if clip not in twinned:
            failures.append(f"{clip} has no CLEAN twin")

    print(f"  {len(violations)} violation / {len(cleans)} clean")
    print(f"  {len(twinned)} twins declared, {len(by_clip)} clips total")
    return failures


def main() -> int:
    if not CLIPS.is_dir() or not any(CLIPS.glob("*.mp4")):
        print("no corpus — run: python data/corpus/generate.py", file=sys.stderr)
        return 3

    failures: list[str] = []
    failures += check_pairs()
    failures += check_photosensitive()
    failures += check_channels()
    failures += check_loudness()

    print()
    if failures:
        print(f"{len(failures)} PROBLEM(S):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("ground truth verified: every constructed defect is measurable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
