---
clause_id: DISC-02
title: Personal data visible on screen
severity_default: ADVISORY
version: 2026-08
source_url: PREFLIGHT engineering ruleset
fetched_at: 2026-08-05
derivation: PREFLIGHT's own production rule, not a restatement of any platform policy
---

## Scope

A phone number, email address or payment card legible in the picture — a notification banner, a browser autofill, a document left open in shot.

## Fully monetized when

- No personal data legible on screen

## Limited ads when

- An email or phone number visible briefly

## No ads when

- A payment card number legible, checksum-valid

## Documented exemptions

- Deliberate artistic choice where the content itself makes the intent evident rather than the metadata asserting it
- A measurement taken over a span too short to characterise the file

## Signals that distinguish this clause from neighbours

- Card numbers are Luhn-checked. Sixteen digits that fail the checksum are an order number, not a card, and reporting them as one teaches a creator to ignore this finding.
- Phone numbers require a separator or a country code: a bare run of ten digits is far more often a timestamp or a score.

## Remediation guidance

- Preferred fix: BLUR_REGION
- Typical span: The span the text was legible for, merged across frames.
