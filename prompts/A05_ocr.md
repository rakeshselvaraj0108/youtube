---
agent_id: A05
codename: OCR
kind: deterministic
status: unimplemented
implementation: preflight/perception/ocr.py
model: none
tier: 3
parents: [A03]
produces: Finding[]
---

# A05 — OCR

## Identity

You read text burned into the picture. On-screen text is assessed the same way as speech, but it behaves differently: it persists across hundreds of frames, and counting frames instead of spans is the classic way to turn one caption into two hundred findings.

Deterministic. No model is called and no prompt is sent; this file is the
contract the implementation is tested against.

## Responsibilities

- Extract text from scene-cut keyframes
- Classify the role of the text using data/lexicons/ocr_role_patterns.json
- Deduplicate temporally — persistent text is ONE finding

## Inputs

- Scene-cut keyframes with millisecond timestamps

## Outputs

- `Finding[]` with a text role and a merged span

## Prohibitions

- Never emit one finding per frame
- Never treat a corner watermark as a language finding — it is a third-party footage cue and routes to COPY-01

## Failure behaviour

No tesseract: SKIPPED with coverage 0 and the install command.
