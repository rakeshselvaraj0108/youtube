# Compliance posture

PREFLIGHT analyses video files. That puts it near two sets of rules — YouTube's
terms of service and the provenance of the policy text it lints against — so
this document states exactly what it does and does not do.

**PREFLIGHT is not affiliated with, endorsed by, or connected to YouTube or
Google.**

---

## What it touches

| Operation | Network | YouTube API | Notes |
|---|---|---|---|
| `preflight check` | Optional | **None** | Local file only. Hosted models are used for policy adjudication when a key is present; without one the run is local and says so. |
| `preflight fix` | None | **None** | ffmpeg on a local file. |
| `preflight probe` | None | **None** | ffprobe on a local file. |
| `preflight drift` | Optional | **None** | Diffs two local corpus snapshots. Embeddings are used for semantic delta when available. |
| `preflight snapshot` | None | **None** | Writes a local file. |

The core loop — analyse, remediate, re-analyse — makes **zero YouTube API
calls**. It operates on a file on your disk, before that file has ever been
uploaded anywhere.

### What it explicitly does not do

- Download videos from YouTube, or any other platform
- Scrape YouTube pages, or parse the site's HTML
- Probe, reverse-engineer, or attempt to replicate YouTube's monetization
  classifier
- Automate uploads, comments, likes, subscriptions, or any other account action
- Circumvent, evade, or defeat any platform enforcement mechanism

That last point deserves emphasis: PREFLIGHT helps a creator **comply** with
published guidelines before publishing. It is a linter, in the same sense that
a security scanner helps you find your own vulnerabilities. It is not an
evasion tool, and the remediation operations it emits — mute, bleep, blur,
replace audio, cut — are the ordinary edits a human editor would make.

---

## Policy corpus provenance

`data/policy/` is a **structured restatement** of publicly published platform
guidance, authored in `scripts/build_corpus.py`. It is not a copy of YouTube's
guidelines and does not claim to be authoritative.

Each clause file carries frontmatter recording where the underlying guidance
came from and when:

```yaml
clause_id: AF-02
title: Violence
severity_default: DEMONETIZING
version: 2026-08
source_url: https://support.google.com/youtube/answer/6162278
fetched_at: 2026-08-05
```

`manifest.json` records a SHA-256 per clause. That is what makes a report
reproducible against a known snapshot, and it is what the Drift Watcher diffs.

Every finding cites the clause it was judged under, verbatim, so a human can
read the rule and overrule the machine. A tool that says "this seems risky" is
not auditable; one that quotes the clause and the evidence is.

---

## Optional YouTube Data API use

Back-catalogue triage is the one feature that would touch the YouTube API. It
is **optional, off by default, and operates only on the authenticated user's
own channel** under OAuth.

The free quota is 10,000 units/day. The naive implementation blows it:

```
channels.list                1
playlistItems.list × 4       4      (200 videos, 50 per page)
videos.list × 4              4      (batched, 50 ids per call)
captions.list × 200      10,000     ← the entire daily quota, on one operation
```

`captions.list` costs 50 units. Calling it per video is the mistake. Triage
therefore runs in two tiers:

- **Tier 1 — whole archive, 9 units.** Metadata-only heuristics over titles,
  descriptions and tags, ranking videos by textual risk signal.
- **Tier 2 — top 20 only, 5,000 units.** `captions.list` + `captions.download`
  on the highest-ranked videos.

Total: **5,009 of 10,000 units**, leaving headroom for a second pass in the same
day. A default budget cap of 8,000 units is enforced in configuration so a run
cannot exhaust the quota an account needs for anything else.

---

## Limits of the tool, stated plainly

**It is not YouTube's classifier, and cannot be.** It predicts risk against
published policy using retrieval-grounded classification. The published
guidelines are the specification; the classifier that enforces them is not
public. A finding is a prediction with a citation, not a verdict.

**A copyright non-match does not prove safety.** Content ID's reference
database is private and substantially larger than any public fingerprint
service. PREFLIGHT reports `CLAIM_LIKELY` when a match is found and
`MUSIC_BED_PRESENT` when tonal content is detected without identification. It
never reports `SAFE`.

**Coverage is reported, never hidden.** Every report states what fraction of
the analysis surface each agent actually inspected. A run where the vision
agent reached 42% of keyframes says so, in the header chip, in the SARIF
invocation, and in the certificate. A compliance tool that quietly returns a
clean bill of health after inspecting half the input is worse than no tool.

**The score is recomputable.** The certificate ships the dimension weights and
the clamp rule alongside the number, so a client can check the arithmetic
rather than trust it.

---

## Data handling

- Video files never leave the machine. Only derived text — transcript windows
  and retrieved clause text — is sent to a hosted model, and only when a key is
  configured.
- `--offline` guarantees no network access regardless of configuration.
- The content-addressed cache lives under `.preflight/` in the working
  directory and contains derived artifacts from your own files.
- No telemetry, analytics, or usage reporting of any kind.
- `.env` is gitignored. `.env.example` documents the variables.

---

## Licensing of generated assets

`REPLACE_AUDIO` substitutes a copyright-matched music bed for a licence-free
one. That replacement is **generated**, not downloaded — `scripts/make_assets.py`
synthesises it with ffmpeg from oscillators and filtered noise. Generated audio
has no rights holder, which is the point: the file that fixes a copyright
finding must not create one.
