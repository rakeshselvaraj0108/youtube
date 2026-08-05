---
clause_id: ACC-01
title: Photosensitive content
severity_default: LIMITING
version: 2026-08
source_url: https://www.w3.org/WAI/WCAG22/Understanding/three-flashes-or-below-threshold.html
fetched_at: 2026-08-05
derivation: structured restatement in own words; not a verbatim copy
---

## Scope

Rapid luminance change that can provoke a seizure in photosensitive viewers. The widely used threshold is three flashes within any one second, with transitions to and from saturated red treated more strictly. This is a safety property, measurable directly, and it is not a monetisation rule — it is included because a creator has no other way to discover it.

## Fully monetized when

- No sequence exceeding two flashes per second
- Gradual transitions and cross-fades

## Limited ads when

- Two flashes per second sustained across a sequence
- Rapid cuts producing large luminance swings without a warning card

## No ads when

- Three or more flashes within any one second
- Rapid transitions to and from saturated red

## Documented exemptions

- An explicit photosensitivity warning shown before the sequence — this mitigates the harm but does not remove it, so it lowers severity rather than dismissing the finding
- The flashing area occupies a small enough proportion of the frame that it falls under the small-area exception in the underlying guidance
- Luminance change stays below the general flash threshold even where transitions are frequent — rapid but low-contrast cutting is not a flash
- The sequence is a single transition rather than a repeating pattern

## Signals that distinguish this clause from neighbours

- Measured, not inferred: a luminance series sampled at 10fps or better, differenced, and counted in a one-second sliding window.
- Scene-cut keyframes cannot detect this. A strobe lives entirely between two cuts, so this check requires its own sampling rate.

## Remediation guidance

- Preferred fix: NONE
- Typical span: The remedy is a warning card or a re-edit of the sequence; a filter cannot make a strobe safe.
