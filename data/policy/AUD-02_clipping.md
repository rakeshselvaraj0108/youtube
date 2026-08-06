---
clause_id: AUD-02
title: Clipping
severity_default: LIMITING
version: 2026-08
source_url: PREFLIGHT engineering ruleset
fetched_at: 2026-08-05
derivation: PREFLIGHT's own production rule, not a restatement of any platform policy
---

## Scope

Samples at or beyond full scale. Clipping is destroyed signal: the waveform above the ceiling is gone and no later processing restores it.

## Fully monetized when

- No samples at full scale

## Limited ads when

- Isolated clipped samples, likely inaudible

## No ads when

- Sustained clipping across a span, audible as distortion

## Documented exemptions

- Deliberate artistic choice where the content itself makes the intent evident rather than the metadata asserting it
- A measurement taken over a span too short to characterise the file

## Signals that distinguish this clause from neighbours

- vs AUD-01 (loudness): clipping is a defect in the recording. Turning the file down does not repair it.

## Remediation guidance

- Preferred fix: REPLACE_AUDIO
- Typical span: The clipped region, padded to the nearest zero crossing.
