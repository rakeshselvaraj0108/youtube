"""Author the reference constants.

Small files, but they exist so the thresholds are data a reader can inspect and
a profile can override, rather than magic numbers buried at their call sites.
Each records the basis for its value — a threshold with no stated basis is
indistinguishable from one somebody guessed.

    python scripts/build_reference.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("data/reference")

LOUDNESS = {
    "platform_target_lufs": -14,
    "true_peak_ceiling_dbtp": -1.0,
    "acceptable_range_lufs": [-16, -12],
    "lra_warn_above": 12,
    "basis": (
        "Playback normalises loud uploads downward toward roughly -14 LUFS. "
        "Delivering above target does not increase perceived volume; it only "
        "reduces headroom and dynamic range after normalisation, so mastering "
        "louder is a loss with no corresponding gain."
    ),
    "measurement": "EBU R128 via ffmpeg loudnorm print_format=json, one pass",
}

PSE = {
    "flashes_per_second_high": 3,
    "flashes_per_second_moderate": 2,
    "luminance_delta_threshold": 0.10,
    "red_saturation_delta_threshold": 0.20,
    "sample_rate_fps_min": 30,
    "window_s": 1.0,
    "sample_rate_basis": (
        "Nyquist. Sampling a strobe at its own rate lands on the same phase "
        "every time and measures a flat series — a 10Hz strobe sampled at 10fps "
        "reports ZERO flashes. 30fps resolves flashes to about 15Hz, past the "
        "range that matters. Verified against corpus clip g010, which "
        "constructs exactly that aliasing case."
    ),
    "basis": (
        "WCAG general flash and red flash thresholds, and the widely used "
        "three-flashes-per-second harm threshold. Transitions to and from "
        "saturated red are treated more strictly than luminance alone."
    ),
    "sampling_note": (
        "Scene-cut keyframes cannot detect this. A strobe lives entirely between "
        "two cuts, so the luminance series is sampled at a fixed rate instead — "
        "10fps minimum, independent of the keyframe extractor."
    ),
}

ACCESSIBILITY = {
    "wpm_comfortable": [120, 160],
    "wpm_warn_above": 180,
    "max_unbroken_speech_ms": 45000,
    "min_pause_ms": 300,
    "text_contrast_min_ratio": 4.5,
    "chapters_required_above_ms": 480000,
    "caption_cue_max_ms": 6000,
    "caption_max_line_chars": 42,
    "basis": (
        "WCAG contrast minimum for normal text is 4.5:1. Caption cue length and "
        "line width follow broadcast subtitling convention: a cue longer than "
        "about six seconds is read twice and then ignored."
    ),
}

MODALITY_WEIGHTS = {
    "speech": 1.00,
    "music": 0.95,
    "access": 0.95,
    "meta": 0.90,
    "vision": 0.85,
    "ocr": 0.80,
    "metadata": 0.70,
    "audio": 0.60,
    "default": 0.50,
    "basis": (
        "Reliability discount applied before noisy-OR fusion. Speech is exact — "
        "the word was said or it was not. Vision is inferential and VLMs "
        "hallucinate objects. Audio DSP is a proxy for a proxy."
    ),
    "coverage_scaling": (
        "Each weight is further scaled by that agent's actual coverage. A vision "
        "confidence of 0.9 from an agent that reached 42% of keyframes is not "
        "worth 0.9, and reporting it as such claims certainty the run never "
        "earned."
    ),
}

CLAUSE_MULTIPLIERS = {
    "_basis": (
        "Per-clause risk tuning. Clauses whose violations cause the largest "
        "revenue loss, or the most serious harm, weigh more than their raw "
        "severity implies."
    ),
    "default": 1.0,
    "clauses": {
        "AF-01": 1.00,
        "AF-02": 1.15,
        "AF-03": 1.15,
        "AF-04": 1.10,
        "AF-05": 1.10,
        "AF-06": 1.30,
        "AF-07": 1.05,
        "AF-08": 1.05,
        "AF-09": 1.10,
        "AF-10": 1.20,
        "AF-11": 1.05,
        "AF-12": 0.95,
        "AF-13": 1.25,
        "AF-14": 1.10,
        "META-01": 1.20,
        "ACC-01": 1.00,
        "COPY-01": 1.25,
    },
    "profiles": {
        "general": {},
        "kids": {
            "AF-01": 1.60,
            "AF-02": 1.50,
            "AF-03": 2.00,
            "AF-07": 1.80,
            "AF-12": 1.80,
            "AF-13": 2.00,
            "_note": "Everything that AF-13 modifies is raised, not just AF-13.",
        },
        "mature-commentary": {
            "AF-01": 0.70,
            "AF-09": 0.80,
            "AF-14": 0.90,
            "_note": (
                "Language and controversy are the register. Hate, kids content "
                "and copyright are NOT relaxed — a profile tunes tolerance, it "
                "does not disable a clause."
            ),
        },
    },
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    files = {
        "loudness_targets.json": LOUDNESS,
        "pse_thresholds.json": PSE,
        "accessibility_norms.json": ACCESSIBILITY,
        "modality_weights.json": MODALITY_WEIGHTS,
        "clause_multipliers.json": CLAUSE_MULTIPLIERS,
    }
    for name, payload in files.items():
        path = OUT / name
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"  {path}")
    print(f"wrote {len(files)} reference files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
