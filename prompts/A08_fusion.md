---
agent_id: A08
codename: FUSION
kind: deterministic
status: implemented
implementation: preflight/scoring/fusion.py
model: none
tier: 8
parents: [A11]
produces: Finding[]
---

# A08 — FUSION

## Identity

You combine evidence across modalities. Independent agents agreeing is evidence; one agent shouting is not.

Deterministic. No model is called and no prompt is sent; this file is the
contract the implementation is tested against.

## Responsibilities

- Noisy-or over per-modality confidences, weighted by reliability
- Scale each modality by that agent's ACTUAL coverage
- PROMOTE on multi-modal agreement, DEMOTE a lone weak visual claim, flag CONTRADICTION for review

## Inputs

- Adjudicated findings and per-agent coverage

## Outputs

- The same findings with fused confidence and severity

## Prohibitions

- Never raise confidence above what coverage supports
- Never let a single vision claim below the floor drive a demonetising verdict — VLMs hallucinate objects

## Failure behaviour

Cannot fail independently; with one modality it is identity.
