---
agent_id: A12
codename: REMEDIATION
kind: deterministic
status: implemented
implementation: preflight/remediate/
model: none
tier: 9
parents: [A08]
produces: EDL, ffmpeg program
---

# A12 — REMEDIATION

## Identity

You compile findings into an executable repair. This is a real compiler: lowering, seven optimiser passes in a fixed order, and codegen. Findings are not advice.

Deterministic. No model is called and no prompt is sent; this file is the
contract the implementation is tested against.

## Responsibilities

- Lower each remediable finding to a typed EDL operation
- Snap to word boundaries, pad, coalesce, resolve dominance and conflicts
- Enforce the cut budget
- Validate before codegen, and emit ffmpeg plus a readable fix.sh

## Inputs

- Fused findings, the transcript, the runtime

## Outputs

- `edl.json`, the ffmpeg command, `fix.sh`, captions

## Prohibitions

- Never re-encode the video for an audio-only EDL
- Never silently delete more than the cut budget allows — demote to MUTE and warn
- Never render over a master file without --apply

## Failure behaviour

An invalid EDL raises rather than rendering something corrupt.
