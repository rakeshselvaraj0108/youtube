---
agent_id: A06
codename: METADATA
kind: deterministic
status: implemented
implementation: preflight/perception/metadata.py
model: none
tier: 2
parents: [A01]
produces: Finding[]
---

# A06 — METADATA

## Identity

You lint the title, description and tags. Zero model calls. Your most valuable check is the cheapest one: sponsorship language in the transcript with no disclosure in the description is real regulatory exposure, and both inputs are already in hand.

Deterministic. No model is called and no prompt is sent; this file is the
contract the implementation is tested against.

## Responsibilities

- Cross-check spoken sponsorship against description disclosure
- Detect affiliate hosts in the description
- Description depth, title length and case, tag count

## Inputs

- `<video>.meta.json` sidecar, and the transcript from A02

## Outputs

- `Finding[]` on META-01..META-05

## Prohibitions

- Never assert non-disclosure from the description alone — an on-screen disclosure card is invisible here, so the finding is always rebuttable

## Failure behaviour

No sidecar: SKIPPED. Metadata cannot be inferred from the video.
