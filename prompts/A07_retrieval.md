---
agent_id: A07
codename: RETRIEVAL
kind: deterministic
status: implemented
implementation: preflight/policy/index.py
model: none
tier: 4
parents: [A02, A03, A04, A05]
produces: Chunk[]
---

# A07 — RETRIEVAL

## Identity

You select which policy clauses a window is judged against. You decide nothing about the content; you decide what the judges are allowed to read, which is a narrower and more consequential job than it sounds.

Deterministic. No model is called and no prompt is sent; this file is the
contract the implementation is tested against.

## Responsibilities

- Embed each window and each sentence within it
- BM25 in parallel, fused by reciprocal rank
- Return the top clauses, deduplicated so three hits are three different clauses rather than three sections of one

## Inputs

- Windows with transcript, OCR text and visual flags

## Outputs

- Up to five verbatim clause chunks per window

## Prohibitions

- Never search outside the requesting agent's scope — the triad sees advertiser-friendly clauses only
- Never index advisory sections; remediation guidance is instruction for the compiler, not a rule to match against

## Failure behaviour

No embedder: BM25 alone, reported as sparse-only rather than silently pretending dense retrieval happened.
