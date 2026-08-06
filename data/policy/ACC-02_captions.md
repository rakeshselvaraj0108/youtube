---
clause_id: ACC-02
title: Caption availability
severity_default: LIMITING
version: 2026-08
source_url: PREFLIGHT engineering ruleset
fetched_at: 2026-08-05
derivation: PREFLIGHT's own production rule, not a restatement of any platform policy
---

## Scope

Whether the file ships a timed-text track. File-scoped: this is a property of the container, not of any moment in the video.

## Fully monetized when

- A caption track is present and covers the spoken content

## Limited ads when

- No caption track, but word-level timings exist in this run and captions can be emitted directly from them

## No ads when

- No caption track and no transcript available to generate one
- Audio containing technical vocabulary, strong accents or heavy background noise, where automatic captions are least reliable

## Documented exemptions

- Deliberate artistic choice where the content itself makes the intent evident rather than the metadata asserting it
- A measurement taken over a span too short to characterise the file

## Signals that distinguish this clause from neighbours

- vs ACC-01 (photosensitive content): ACC-01 is a harm to a viewer with a medical condition and is scoped to a span. This is an access gap scoped to the whole file. They were once the same clause id, which meant a caption finding cited a seizure-risk policy.

## Remediation guidance

- Preferred fix: NONE
- Typical span: File-scoped. Start 0, end at duration.
