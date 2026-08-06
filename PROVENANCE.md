# Data provenance

Where every file under `data/` and `assets/` came from, and what it is
allowed to be used for. Checked mechanically by
[`scripts/verify_data.py`](scripts/verify_data.py) (`make verify-data`, and
wired into the test suite as `tests/test_data_provenance.py`) — this document
explains the *why* behind each rule; the script enforces it on every build so
the explanation cannot quietly drift out of sync with the data.

The build prompts for this project stated a short list of hard constraints,
verbatim, before any of this data existed:

> Ship `.env.example`, never `.env`.
> No bulk verbatim copies of policy pages. Restatements with source URLs and
> fetch dates.
> No hate-speech corpora, extremist text, or slur lists sourced from
> third-party datasets.
> No graphic or disturbing media for the demo or the corpus.
> Never download video or audio from YouTube. Not for the demo, not for the
> test corpus, not "just to check."

Everything below exists to make each of those true and keep it true.

## `data/policy/` — 28 clauses, two kinds

`data/policy/manifest.json` (rebuilt by [`scripts/build_corpus.py`](scripts/build_corpus.py))
labels every clause `kind: policy_restatement` or `kind: house_rule`. The
distinction is load-bearing: this project's whole claim is that a finding
names the specific clause it breaches, and a citation is worthless if a
reader cannot tell whose rule it is.

**17 `policy_restatement` clauses** (AF-01…AF-14, ACC-01, COPY-01, META-01) are
structured restatements, in this project's own words, of publicly published
platform guidance. Each carries the page it was read from and the date:

| Source | Used for |
|---|---|
| `support.google.com/youtube/answer/6162278` | Advertiser-friendly content guidelines (AF-01…AF-14) |
| `support.google.com/youtube/answer/154235` | Paid promotion disclosure (META-01) |
| `support.google.com/youtube/answer/2797370` | Content ID / third-party content (COPY-01) |
| `w3.org/WAI/WCAG22/.../three-flashes-or-below-threshold.html` | Photosensitive flash thresholds (ACC-01) |

Fetched 2026-08-05, corpus version `2026-08`. None of these files are a copy
of the source page — each is chunked at heading level (Scope / Fully
monetized when / Limited ads when / No ads when / Documented exemptions /
Signals / Remediation guidance) and written for retrieval, not reproduced
verbatim. `verify_data.py` asserts every such clause's `derivation` field says
so explicitly and that its `source_url` resolves to one of the four domains
above — a restatement that drifted onto an unreviewed domain, or lost its
disclaimer, fails the build rather than shipping quietly.

**14 `house_rule` clauses** (ACC-02…ACC-04, AUD-01…AUD-05, VID-01…VID-02,
META-02…META-05) are PREFLIGHT's own engineering thresholds — a
caption-availability check, a -14 LUFS loudness target, tag-stuffing
detection — and are explicitly **not** platform policy. Their `source` is the
literal string `PREFLIGHT engineering ruleset` and their `derivation` says,
verbatim, "not a restatement of any platform policy." `verify_data.py`
asserts a house rule never carries a `support.google.com` or `w3.org` URL,
which is the specific failure mode this distinction exists to prevent: a
loudness target that looks like it was handed down by YouTube.

The first eleven were added after the fact, and the reason is worth
recording. Every deterministic agent (`accessibility.py`, `audio.py`,
`metadata.py`) was emitting a `clauseId` for its findings — ACC-02 through
ACC-04, AUD-01 through AUD-04, META-02 through META-05 — that referred to
nothing in the manifest. AUD-05 (mono/phase compatibility) and VID-01/VID-02
(black and frozen frames) were added later as genuinely new detectors, not
retrofits for an existing gap, and follow the same house-rule discipline from
the start.
Eleven of fourteen cited clause ids resolved to no text at all, and the one
accessibility id that *did* exist, ACC-01, was on the wrong finding: the
manifest defined it as photosensitive content, but the caption-availability
finding cited it, while the photosensitivity finding cited the undefined
ACC-02. `preflight bench --ablation` surfacing implausibly low precision on
its first real run is what caught it — see `data/corpus/labels.jsonl`'s
`ACC-01` entries and [`preflight/perception/accessibility.py`](preflight/perception/accessibility.py).

## `data/corpus/` — synthetic, gitignored, regenerated on demand

Thirty clips in fifteen VIOLATION/CLEAN pairs, built by
[`data/corpus/generate.py`](data/corpus/generate.py) from
[`data/corpus/manifest.yaml`](data/corpus/manifest.yaml). Nothing here is
downloaded or sourced from a third party:

- **Narration** — the platform's own built-in text-to-speech (`say` on
  macOS, SAPI on Windows, `espeak`/`festival` on Linux), never a recorded
  voice.
- **Footage** — ffmpeg `lavfi` synthetic sources (`testsrc2`, `color`,
  `smptebars`), never a downloaded or captured clip.
- **The marker music bed** (`data/assets/cc_music/marker_bed_01.wav`) —
  detuned oscillators plus filtered noise, generated by
  [`scripts/make_assets.py`](scripts/make_assets.py). Nobody holds the
  rights to it because nobody performed it.

Ground truth is exact *by construction*: the profanity was inserted at
4,200ms because the generator put it there at 4,200ms, not because a human
scrubbed the timeline and guessed. `scripts/verify_corpus_truth.py` checks the
deterministic detectors actually measure what the generator constructed — the
one link in this chain that could silently break (an ffmpeg filter that did
nothing) and invalidate every metric computed against the corpus without
anyone noticing.

The pairs are the point, and the profanity lexicon is deliberately thin about
why: **no clip contains a slur, a hate-speech corpus, or extremist text.**
`data/lexicons/profanity.tiered.jsonl` grades ordinary profanity into three
tiers and declares a fourth tier for slurs that is *never populated* —
`verify_data.py` asserts that tier stays empty on every build. AF-06
(hateful and derogatory content) is judged from its clause text by a model,
not from a list a keyword filter matches against, which is the Scunthorpe
problem in exactly the shape this project exists to avoid enumerating.

`data/corpus/clips/*.mp4` are gitignored (`.gitignore` line 61) and
regenerate with `make corpus`. A clone of this repository never receives
rendered media of any kind from this project — it receives the generator.

## `data/lexicons/`, `data/vision/`, `data/audio/` — hand-authored vocabularies

Every term list, synonym map, role-classification pattern and taxonomy label
under these three directories was written for this project by hand, not
scraped or sourced from an external dataset. None of them are the kind of
third-party corpus the build constraints prohibit — they are closed,
reviewed vocabularies with an explicit purpose stated in each file's `_note`
and `_purpose` fields, built by
[`scripts/build_speech_lexicons.py`](scripts/build_speech_lexicons.py),
[`scripts/build_vision_vocab.py`](scripts/build_vision_vocab.py) and
[`scripts/build_audio_taxonomy.py`](scripts/build_audio_taxonomy.py). A hit
in any of them is never a finding on its own — it raises retrieval priority
for a clause, and the adversarial triad decides using the clause text.

## `data/reference/` — measured or documented constants

`loudness_targets.json`, `pse_thresholds.json`, `accessibility_norms.json`,
`clause_multipliers.json`, `modality_weights.json` — the numeric constants
the scoring pipeline reads so no threshold is a magic number buried in code.
`loudness_targets` (-14 LUFS, -1.0 dBTP ceiling) and `pse_thresholds` (3
flashes/second) are widely documented broadcast and accessibility norms
(EBU R128; WCAG 2.2's three-flashes threshold, already cited above for
ACC-01). `clause_multipliers` and `modality_weights` are this project's own
tuning, in the same spirit as the house rules above.

## `assets/cc_music/` — synthesized replacement audio

`glacier_calm.mp3`, built by
[`scripts/make_assets.py`](scripts/make_assets.py), is what the remediation
compiler's `REPLACE_AUDIO` operation swaps in for a copyright-matched bed.
Two detuned sine oscillators plus filtered pink noise — a pad, not a melody,
generated rather than downloaded, because the file that *fixes* a copyright
finding cannot itself be someone else's recording. Nobody holds rights to it.

## What is deliberately absent

- **No credentials.** `.env` is gitignored; `.env.example` documents the
  shape without a value. `preflight/providers/secrets.py` is the only module
  permitted to read one, and installs a logging filter so a key cannot reach
  a log line even by accident.
- **No downloaded YouTube media**, ever, for any purpose — not the corpus,
  not the demo, not a one-off check. `verify_data.py` greps the entire data
  layer for a YouTube media URL pattern (`youtube.com/watch`, `youtu.be/`,
  `googlevideo.com`) and fails the build if one appears; a citation to a
  YouTube *policy page* is fine and expected, a URL that would fetch video
  or audio is not.
- **No rendered media tracked by git**, checked directly against
  `git ls-files` rather than the working tree, so a file that was generated
  locally and accidentally `git add`-ed is caught the same way as one that
  never should have existed.
