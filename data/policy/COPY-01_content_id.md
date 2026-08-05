---
clause_id: COPY-01
title: Third-party content and Content ID
severity_default: DEMONETIZING
version: 2026-08
source_url: https://support.google.com/youtube/answer/2797370
fetched_at: 2026-08-05
derivation: structured restatement in own words; not a verbatim copy
---

## Scope

Third-party copyrighted material, principally music. Content ID scans uploads against a database of reference files submitted by rights holders. A match lets the claimant block the video, take its revenue, or track it, and the outcome can differ by territory.

## Fully monetized when

- Original material, or material licensed for this use
- Public-domain or CC0 material with the licence recorded

## Limited ads when

- Music present under speech with licensing unverified
- Third-party footage cues such as station bugs or letterboxing

## No ads when

- Commercially released recording used without a licence
- Substantial third-party footage without a licence

## Documented exemptions

- A licence exists for this use — the tool cannot see licences and this finding is therefore always rebuttable by the creator
- The material is public domain or CC0
- Use qualifies as fair use or fair dealing, which is a legal determination this tool does not and cannot make

## Signals that distinguish this clause from neighbours

- The reference database is private and is not published, so no pre-upload check can be authoritative. PREFLIGHT reports CLAIM_LIKELY on a public fingerprint match and MUSIC_BED_PRESENT on unidentified tonal content. It never reports SAFE.
- Claims are applied automatically at upload, and there is no published mechanism for previewing them beforehand — which is precisely the gap this clause exists to narrow.
- vs AF-* clauses: COPY-01 is about ownership, not about content suitability. A perfectly advertiser-friendly video can be claimed.

## Remediation guidance

- Preferred fix: REPLACE_AUDIO
- Typical span: Usually the full extent of the bed, 15-60s. Replace rather than mute so the segment keeps its pacing.
