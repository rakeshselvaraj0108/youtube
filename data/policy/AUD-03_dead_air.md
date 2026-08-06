---
clause_id: AUD-03
title: Dead air
severity_default: ADVISORY
version: 2026-08
source_url: PREFLIGHT engineering ruleset
fetched_at: 2026-08-05
derivation: PREFLIGHT's own production rule, not a restatement of any platform policy
---

## Scope

A sustained span with RMS below the noise floor. Usually an editing error — a muted track, a dropped clip, a gap left in the timeline.

## Fully monetized when

- No silent span longer than a natural pause

## Limited ads when

- A silent span long enough to read as a mistake

## No ads when

- Extended silence where content was clearly intended

## Documented exemptions

- Deliberate artistic choice where the content itself makes the intent evident rather than the metadata asserting it
- A measurement taken over a span too short to characterise the file

## Signals that distinguish this clause from neighbours

- A deliberate pause for effect is short. This clause is scoped to spans long enough that a viewer checks whether their audio broke.

## Remediation guidance

- Preferred fix: CUT
- Typical span: The silent span itself.
