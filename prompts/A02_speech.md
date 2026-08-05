---
agent_id: A02
codename: SPEECH
kind: deterministic
status: implemented
implementation: preflight/perception/asr.py
model: none
tier: 2
parents: [A01]
produces: Transcript
---

# A02 — SPEECH

## Identity

You convert audio into a word-level transcript. You are the only load-bearing perception agent: every downstream text judgement rests on your timings, and a span you place wrongly becomes a bleep over the wrong word.

Deterministic. No model is called and no prompt is sent; this file is the
contract the implementation is tested against.

## Responsibilities

- Transcribe with word-level start and end timestamps
- Report per-word confidence
- Emit segments for readability and windows for analysis
- Cache by hash of audio plus model id

## Inputs

- 16 kHz mono WAV, loudness-normalised

## Outputs

- `Transcript` — words with start_ms, end_ms, conf; segments; language

## Prohibitions

- Never round a timestamp to a second boundary
- Never drop a word because confidence is low — report the confidence
- Never paraphrase; the transcript is evidence, not a summary

## Failure behaviour

Without a local backend, report SKIPPED with the install command. Speech is required: if it produces nothing, text-derived findings are absent and coverage falls accordingly.
