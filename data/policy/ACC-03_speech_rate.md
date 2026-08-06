---
clause_id: ACC-03
title: Speech rate
severity_default: ADVISORY
version: 2026-08
source_url: PREFLIGHT engineering ruleset
fetched_at: 2026-08-05
derivation: PREFLIGHT's own production rule, not a restatement of any platform policy
---

## Scope

Sustained words per minute over a rolling window. Fast delivery reduces comprehension for non-native speakers and for anyone relying on automatic captions, which degrade as rate rises.

## Fully monetized when

- Sustained rate within a comfortable listening range

## Limited ads when

- Sustained rate well above conversational pace

## No ads when

- Rate high enough that automatic captions are unlikely to track it

## Documented exemptions

- Deliberate artistic choice where the content itself makes the intent evident rather than the metadata asserting it
- A measurement taken over a span too short to characterise the file

## Signals that distinguish this clause from neighbours

- Advisory only. A fast talker is not a policy problem, and this clause exists to inform rather than to gate.

## Remediation guidance

- Preferred fix: NONE
- Typical span: The window that exceeded the threshold, not the whole file.
