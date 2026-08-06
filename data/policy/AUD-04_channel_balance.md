---
clause_id: AUD-04
title: Channel balance
severity_default: LIMITING
version: 2026-08
source_url: PREFLIGHT engineering ruleset
fetched_at: 2026-08-05
derivation: PREFLIGHT's own production rule, not a restatement of any platform policy
---

## Scope

Per-channel RMS difference. A recording where one channel sits far below the other is a dead microphone — real, common, and expensive, because nobody notices until the video is live and half the audience is hearing silence.

## Fully monetized when

- Channels within a few dB of each other, or genuinely mono

## Limited ads when

- Noticeable imbalance between channels

## No ads when

- One channel effectively silent — a dead mic

## Documented exemptions

- Deliberate artistic choice where the content itself makes the intent evident rather than the metadata asserting it
- A measurement taken over a span too short to characterise the file

## Signals that distinguish this clause from neighbours

- Intentional hard-panning exists but is rare outside music, and a channel that is silent for the whole runtime is not a pan.

## Remediation guidance

- Preferred fix: REPLACE_AUDIO
- Typical span: File-scoped — measured across the whole track.
