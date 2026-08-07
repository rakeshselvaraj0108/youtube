---
clause_id: DISC-01
title: Credential visible on screen
severity_default: LIMITING
version: 2026-08
source_url: PREFLIGHT engineering ruleset
fetched_at: 2026-08-05
derivation: PREFLIGHT's own production rule, not a restatement of any platform policy
---

## Scope

An API key, access token, private key or labelled password legible in the picture — a terminal left open behind a demo, an editor tab, a .env file scrolled past during a screen recording.

## Fully monetized when

- No credential-shaped text anywhere on screen

## Limited ads when

- A labelled secret whose value is partly obscured

## No ads when

- A complete vendor-issued key or private key block legible

## Documented exemptions

- Deliberate artistic choice where the content itself makes the intent evident rather than the metadata asserting it
- A measurement taken over a span too short to characterise the file

## Signals that distinguish this clause from neighbours

- Matched on published vendor prefixes with a length floor rather than on entropy: entropy flags base64 thumbnails and minified JavaScript, and a detector that cries wolf is muted before it ever catches the real one.
- The consequence is not a policy one. A leaked key is charged to the creator's account within hours of the upload going public, and unlike a policy strike there is no appeal.

## Remediation guidance

- Preferred fix: BLUR_REGION
- Typical span: The span the text was legible for, merged across frames.
