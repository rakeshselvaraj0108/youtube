---
clause_id: AUD-05
title: Mono compatibility
severity_default: LIMITING
version: 2026-08
source_url: PREFLIGHT engineering ruleset
fetched_at: 2026-08-05
derivation: PREFLIGHT's own production rule, not a restatement of any platform policy
---

## Scope

Correlation between the left and right channels. Content recorded or mixed out of phase collapses toward silence when the two channels are summed — which happens on a phone speaker, a laptop, a single earbud, or any playback system that is not true stereo, which is most of a video platform's audience most of the time.

## Fully monetized when

- Channels positively correlated — safe when summed to mono

## Limited ads when

- Correlation near zero — no consistent phase relationship

## No ads when

- Channels negatively correlated — audibly hollow or silent in mono

## Documented exemptions

- Deliberate artistic choice where the content itself makes the intent evident rather than the metadata asserting it
- A measurement taken over a span too short to characterise the file

## Signals that distinguish this clause from neighbours

- vs AUD-04 (channel balance): balance is a LEVEL difference between channels; phase is a TIMING/POLARITY relationship. A file can fail either independently of the other — balanced channels can still be out of phase, and a dead channel has no phase relationship to measure.

## Remediation guidance

- Preferred fix: NONE
- Typical span: File-scoped — measured across the whole track.
