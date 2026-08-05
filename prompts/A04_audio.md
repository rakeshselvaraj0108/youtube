---
agent_id: A04
codename: AUDIO
kind: deterministic
status: implemented
implementation: preflight/perception/audio.py
model: none
tier: 2
parents: [A01]
produces: Finding[]
---

# A04 — AUDIO

## Identity

You measure the audio signal. Every finding you emit is a measurement rather than a judgement, which is why they carry confidence near 1.0 and an empty defence — there is no arguing with an LUFS reading.

Deterministic. No model is called and no prompt is sent; this file is the
contract the implementation is tested against.

## Responsibilities

- EBU R128 integrated loudness and true peak
- Clipped sample count
- Dead air spans
- Per-channel balance — a dead microphone is silent on one side
- Music-bed presence via spectral flatness

## Inputs

- Untouched stereo 44.1 kHz WAV — NOT the normalised ASR track, because loudnorm perturbs the spectral peaks a fingerprint hashes

## Outputs

- `Finding[]` on AUD-01..AUD-04 and COPY-01

## Prohibitions

- Never report music as SAFE — only MUSIC_BED_PRESENT or CLAIM_LIKELY
- Never claim to identify a recording from spectral flatness alone

## Failure behaviour

No audio stream: report SKIPPED. A decode error: FAILED with the reason.
