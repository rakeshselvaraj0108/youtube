"""Author the A04 audio taxonomy.

Reference mappings plus, unusually, an explicit statement of what each
detection tier can and cannot resolve. That second part is the point of this
file.

A04 has two tiers:

  **DSP (always available, numpy only).** Loudness, clipping, silence, mains
  hum, noise floor, speech/music segmentation, tempo, energy, impulsive
  transients, applause. All of it is real signal processing on real samples.

  **Classifier (optional, `audio.classify`).** The five-hundred-odd AudioSet
  classes — rain, birdsong, traffic, a dog barking, a gunshot specifically.
  These need learned features; no amount of spectral arithmetic separates a
  gunshot from a slammed door.

The distinction is load-bearing rather than academic. DSP can establish that an
impulsive broadband transient with a six-millisecond attack occurred at 5:32.
It cannot establish that the transient was a gunshot. Most implementations
label it anyway. A04 reports the acoustic evidence and says the label is
unresolved, because a confident wrong label is worse than an honest gap — the
fusion layer will weigh whatever it is given.

    python scripts/build_audio_taxonomy.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("data/audio")

NOT_A_VERDICT = (
    "Acoustic EVIDENCE, never a verdict. A04 reports what the waveform shows. "
    "Whether a transient at 5:32 breaches a clause is A11's decision, made with "
    "the clause text and an advocate."
)

DSP = "dsp"
CLASSIFIER = "classifier"


def taxonomy(category: str, note: str, entries: list[dict]) -> dict:
    return {
        "_category": category,
        "_note": NOT_A_VERDICT,
        "_purpose": note,
        "resolvable_by": sorted({e["tier"] for e in entries}),
        "labels": entries,
    }


def label(name: str, tier: str, *, evidence: str = "", aliases: list[str] | None = None) -> dict:
    return {
        "label": name,
        "tier": tier,
        "evidence": evidence,
        "aliases": aliases or [],
    }


FILES = {
    # ── What DSP genuinely resolves ──────────────────────────────────
    "technical_metrics.json": taxonomy(
        "technical",
        "Measurements. Reproducible to the sample, no model involved.",
        [
            label("loudness_lufs", DSP, evidence="EBU R128 integrated, one ffmpeg pass"),
            label("true_peak_dbtp", DSP, evidence="inter-sample peak"),
            label("rms", DSP, evidence="windowed root mean square"),
            label("dynamic_range", DSP, evidence="loudness range, LRA"),
            label("crest_factor", DSP, evidence="peak over RMS"),
            label("clipping", DSP, evidence="samples at or beyond full scale"),
            label("noise_floor", DSP, evidence="tenth percentile of the RMS envelope"),
            label("mains_hum", DSP, evidence="narrowband peak at 50/60Hz and harmonics"),
            label("channel_imbalance", DSP, evidence="per-channel RMS difference"),
        ],
    ),
    "segmentation.json": taxonomy(
        "segment",
        (
            "The speech / music / ambient / silence timeline. Feeds the "
            "remediation compiler directly: REPLACE_AUDIO needs to know where "
            "the bed actually starts and stops."
        ),
        [
            label("silence", DSP, evidence="RMS below the floor"),
            label("speech", DSP, evidence="high spectral flux, moderate flatness, 4Hz envelope modulation"),
            label("music", DSP, evidence="low spectral flatness with a stable beat"),
            label("ambient", DSP, evidence="steady broadband energy, no beat, low flux"),
            label("noise", DSP, evidence="high flatness, high energy, no structure"),
        ],
    ),
    "music_labels.json": taxonomy(
        "music",
        "Musical attributes. Tempo and energy are measured; genre is not.",
        [
            label("music_present", DSP, evidence="sustained low spectral flatness"),
            label("tempo_bpm", DSP, evidence="autocorrelation of the onset envelope"),
            label("energy_calm", DSP, evidence="low RMS, low spectral centroid"),
            label("energy_medium", DSP),
            label("energy_high", DSP, evidence="high RMS, high centroid, dense onsets"),
            label("music_onset", DSP, evidence="transition into a music segment"),
            label("music_offset", DSP, evidence="transition out of a music segment"),
            label("genre", CLASSIFIER, evidence="not resolvable by DSP"),
            label("commercial_recording", CLASSIFIER, evidence="requires fingerprint lookup"),
        ],
    ),
    "transients.json": taxonomy(
        "transient",
        (
            "Impulsive events. THE honest boundary in this agent: DSP resolves "
            "that a transient occurred and characterises it — attack time, "
            "crest factor, bandwidth, decay. It does NOT resolve what made it. "
            "A gunshot, a door slam, a dropped microphone and a balloon pop are "
            "acoustically similar at this level of analysis."
        ),
        [
            label(
                "impulsive_transient", DSP,
                evidence="sub-20ms attack, crest factor above 8, broadband",
            ),
            label("sustained_transient", DSP, evidence="fast attack with a long decay"),
            label("gunshot", CLASSIFIER, evidence="not separable from a door slam by DSP"),
            label("explosion", CLASSIFIER, evidence="not separable from a mic bump by DSP"),
            label("glass_break", CLASSIFIER, evidence="needs learned spectral structure"),
            label("scream", CLASSIFIER, evidence="needs learned features"),
        ],
    ),
    "human_reactions.json": taxonomy(
        "reaction",
        (
            "Applause is genuinely DSP-detectable — dense broadband impulses "
            "with no periodicity is a distinctive signature. Laughter is not; "
            "it overlaps speech too closely."
        ),
        [
            label("applause", DSP, evidence="dense aperiodic broadband impulse train"),
            label("crowd_noise", DSP, evidence="sustained broadband with speech-like modulation"),
            label("laughter", CLASSIFIER, evidence="overlaps speech in every DSP feature"),
            label("crying", CLASSIFIER),
            label("cheering", CLASSIFIER),
            label("coughing", CLASSIFIER),
        ],
    ),
    # ── What needs a classifier, stated plainly ──────────────────────
    "environment.json": taxonomy(
        "environment",
        (
            "Entirely classifier-tier. Rain, wind and applause are all "
            "broadband noise; sirens and birdsong are both pitched sweeps. "
            "Spectral arithmetic does not separate them, and pretending "
            "otherwise would produce confident nonsense."
        ),
        [
            label(n, CLASSIFIER)
            for n in [
                "rain", "thunder", "ocean", "wind", "birdsong", "traffic",
                "construction", "crowd", "police_siren", "ambulance_siren",
                "fire_alarm", "engine", "footsteps", "door",
            ]
        ],
    ),
    "animals.json": taxonomy(
        "animal",
        "Classifier-tier. Included so the vocabulary is complete when one is available.",
        [
            label(n, CLASSIFIER)
            for n in ["dog_bark", "cat", "bird", "horse", "lion", "elephant", "insect"]
        ],
    ),
    "audio_quality.json": taxonomy(
        "quality",
        "Production defects. Mostly measurable; a few need a model.",
        [
            label("hum", DSP, evidence="narrowband peak at mains frequency"),
            label("static", DSP, evidence="broadband high-flatness energy under speech"),
            label("dead_channel", DSP, evidence="one channel far below the other"),
            label("dead_air", DSP, evidence="sustained RMS below the floor"),
            label("distortion", DSP, evidence="clipping plus harmonic spread"),
            label("reverb", DSP, evidence="envelope autocorrelation decay estimate"),
            label("echo", DSP, evidence="discrete envelope autocorrelation peak"),
            label("mic_quality", CLASSIFIER, evidence="subjective, needs a learned model"),
        ],
    ),
    "speakers.json": taxonomy(
        "speakers",
        (
            "Deliberately thin. Speaker COUNT needs diarisation — clustering "
            "speaker embeddings — and estimating it from energy statistics "
            "produces a number that looks authoritative and is guesswork. A04 "
            "reports speech PRESENCE and leaves counting to a tier that can "
            "actually do it. The specification says never guess unknown "
            "sounds; this is the same rule applied to unknown speakers."
        ),
        [
            label("speech_present", DSP, evidence="4Hz envelope modulation with high flux"),
            label("speaker_count", CLASSIFIER, evidence="requires diarisation"),
            label("speaker_change", CLASSIFIER, evidence="requires speaker embeddings"),
        ],
    ),
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    dsp_total = classifier_total = 0
    for name, payload in FILES.items():
        path = OUT / name
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        dsp = sum(1 for entry in payload["labels"] if entry["tier"] == DSP)
        clf = sum(1 for entry in payload["labels"] if entry["tier"] == CLASSIFIER)
        dsp_total += dsp
        classifier_total += clf
        print(f"  {path.name:<26} {payload['_category']:<12} {dsp:>2} dsp  {clf:>2} classifier")

    print(f"\nwrote {len(FILES)} taxonomy files to {OUT}")
    print(f"{dsp_total} labels resolvable with no dependencies, "
          f"{classifier_total}需 a classifier tier".replace("需", "need"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
