---
clause_id: AUD-01
title: Loudness normalisation
severity_default: ADVISORY
version: 2026-08
source_url: PREFLIGHT engineering ruleset
fetched_at: 2026-08-05
derivation: PREFLIGHT's own production rule, not a restatement of any platform policy
---

## Scope

Integrated loudness against the platform's normalisation target, measured to EBU R128. Content far from target is turned down on playback, and a mix built loud loses its dynamics in the process.

## Fully monetized when

- Integrated loudness within tolerance of the target

## Limited ads when

- Loudness outside tolerance in either direction

## No ads when

- Loudness far enough from target that playback normalisation will materially change the mix

## Documented exemptions

- Deliberate artistic choice where the content itself makes the intent evident rather than the metadata asserting it
- A measurement taken over a span too short to characterise the file

## Signals that distinguish this clause from neighbours

- vs AUD-02 (clipping): loudness is where the whole file sits; clipping is samples destroyed at individual peaks. A quiet file can clip and a loud one need not.

## Remediation guidance

- Preferred fix: REPLACE_AUDIO
- Typical span: File-scoped — an integrated measurement has no span.
