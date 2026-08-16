<div align="center">

# PREFLIGHT

### Autonomous Video Assurance Engine

**Don't just publish. Prove it's ready.**

PREFLIGHT is a multi-agent video assurance system. Twelve specialised agents independently observe a video, every claim they produce is bound to the evidence that created it, findings are grounded in a citable policy corpus, adversarially adjudicated, correlated into incidents, compiled into an executable ffmpeg program — and the rendered artifact is then **put back through the same pipeline** to prove the repair actually worked.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![React 18](https://img.shields.io/badge/React-18.3-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-required-007808?logo=ffmpeg&logoColor=white)](https://ffmpeg.org/)
[![Tests](https://img.shields.io/badge/tests-1%2C406_Python_%2B_109_TS-success)](#-testing)
[![SARIF 2.1.0](https://img.shields.io/badge/SARIF-2.1.0-blue)](https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Deployed on Render](https://img.shields.io/badge/deployed-Render-46E3B7?logo=render&logoColor=white)](https://preflight-deck.onrender.com)

[**Live App**](https://preflight-deck.onrender.com) · [**API**](https://preflight-api-vax3.onrender.com/api/health) · [**Demo Video**](https://youtu.be/4MqqM_7RoZE) · [**Source**](https://github.com/rakeshselvaraj0108/youtube)

</div>

---

## 🚀 Live Demo

| Surface | Link | Status |
|---|---|---|
| 🌐 **Web App** (Command Deck) | https://preflight-deck.onrender.com | Live — static Vite build on Render |
| ⚙️ **API** | https://preflight-api-vax3.onrender.com/api/health | Live — Docker web service on Render |
| 🎥 **Demo Video** | https://youtu.be/4MqqM_7RoZE | Published |
| 📦 **GitHub** | https://github.com/rakeshselvaraj0108/youtube | Public |
| 💻 **CLI** | `pip install -e '.[asr]'` → `preflight --help` | 13 commands |
| 📑 **Devpost** | Not linked in this repository | — |

> **Note on the free tier.** Both Render services run on the free plan, which spins down after ~15 minutes idle. The first request after a cold start takes a few seconds. This is a deliberate cost decision, recorded in [`render.yaml`](render.yaml).

**Verified live capability plan** (`GET /api/health`, at time of writing):

```json
{ "status": "ok", "engineVersion": "0.1.0",
  "ffmpeg": "ffmpeg version 7.1.5-0+deb13u1",
  "capabilitySummary": { "preferred": 7, "fallback": 1, "unavailable": 0 } }
```

---

## 📋 Table of Contents

<details>
<summary><b>Expand</b></summary>

- [The Problem](#-the-problem)
- [The Core Idea](#-the-core-idea-assurance-not-detection)
- [Why "PREFLIGHT"](#-why-preflight)
- [System Architecture](#-system-architecture)
- [The Agent Fleet](#-the-agent-fleet)
- [Agent Interaction Sequence](#-agent-interaction-sequence)
- [Coverage-Aware Reasoning](#-coverage-aware-reasoning)
- [Adaptive Sampling](#-adaptive-sampling)
- [Evidence Fusion](#-evidence-fusion)
- [Policy Retrieval](#-policy-retrieval)
- [Incident Intelligence](#-incident-intelligence)
- [No-Hallucination Architecture](#-no-hallucination-architecture)
- [Secret & PII Detection](#-secret--pii-detection)
- [Decision Simulation](#-decision-simulation)
- [Remediation](#-remediation)
- [Closed-Loop Verification](#-closed-loop-verification)
- [Before/After Comparison](#-beforeafter-comparison)
- [Risk Scoring](#-risk-scoring)
- [Mathematical Foundation](#-mathematical-foundation)
- [Data Model](#-data-model)
- [Lifecycle State Machine](#-lifecycle-state-machine)
- [Artifacts](#-artifacts)
- [API Reference](#-api-reference)
- [CLI Reference](#-cli-reference)
- [Web Application](#-web-application)
- [Deployment](#-deployment)
- [Environment Variables](#-environment-variables)
- [Local Development](#-local-development)
- [Docker](#-docker)
- [Testing](#-testing)
- [Benchmarks](#-benchmarks)
- [Failure Modes](#-failure-modes)
- [Threat Model](#-threat-model)
- [Cost & Efficiency](#-cost--efficiency)
- [Technical Decisions](#-technical-decisions)
- [What Makes PREFLIGHT Different](#-what-makes-preflight-different)
- [End-to-End Example](#-end-to-end-example)
- [Limitations](#-current-limitations)
- [Roadmap](#-roadmap)
- [License](#-license)

</details>

---

## 🎯 The Problem

A creator uploads to YouTube and only *afterwards* learns whether the video will be monetized. By then the first 48 hours — where most of a video's revenue lives — are gone. There is no dry-run on the most financially consequential action in the entire creator workflow.

Every other tool in the creator ecosystem is a **generator**: make me a thumbnail, write me a title, cut me a Short. PREFLIGHT is a **verifier** — the missing static-analysis step in a pipeline that currently ships straight to production with no tests.

But "run a classifier over it" is not the same as assurance. Most video analysis stops here:

```
VIDEO ──▶ DETECTION ──▶ REPORT
```

That pipeline cannot answer the questions that actually decide whether to publish:

- Did anything **look** at minute nine, or is "nothing found there" an artifact of never sampling it?
- Which **policy clause** is this judged under, and is that clause the platform's rule or the tool's own?
- Do **independent** modalities agree, or is one hallucinating model driving the verdict?
- If I apply the suggested fix, **what actually changes** — and did it work?

PREFLIGHT extends the pipeline to answer them:

```mermaid
flowchart LR
    A[VIDEO] --> B[INVESTIGATE]
    B --> C[EVIDENCE]
    C --> D[ADJUDICATE]
    D --> E[SIMULATE]
    E --> F[REMEDIATE]
    F --> G[RE-ANALYZE]
    G --> H[VERIFY]
    H --> I[FINAL VERDICT]
```

The difference that matters: **steps G and H exist.** A successful ffmpeg exit proves a file was written. It proves nothing about whether the problem is still in the picture. Only the same pipeline finding fewer problems in the output is evidence that the fix worked.

---

## 💡 The Core Idea: Assurance, not Detection

PREFLIGHT is built to answer a specific list of questions, and every architectural decision traces back to one of them:

| Question | Answered by |
|---|---|
| **What happened?** | Agent observations → `Finding` |
| **Where did it happen?** | `startMs` / `endMs` spans, mapped through edits |
| **Which modality observed it?** | `Finding.modalities` — per-modality confidence |
| **How confident is the observation?** | Coverage-scaled noisy-OR fusion |
| **Was the relevant span actually examined?** | [`coverage.py`](preflight/coverage.py) — per-band temporal coverage |
| **Which policy applies?** | Hybrid BM25 ⊕ dense retrieval over a 33-clause corpus |
| **Which independent observations support it?** | Incident correlation — cross-agent, non-self-corroborating |
| **Can it be remediated?** | EDL compilation → typed operations |
| **What will the remediation change?** | [`simulation.py`](preflight/scoring/simulation.py) — evidence removal + real rescore |
| **Did the rendered artifact improve?** | [`verify.py`](preflight/verify.py) — closed-loop re-analysis |
| **Did anything *new* appear?** | `NEW_RISK_DETECTED` verdict |

---

## ✈️ Why "PREFLIGHT"

An aviation preflight check does not ask *"did one sensor say OK?"* It independently verifies multiple systems, records what was inspected, and — critically — records what was **not** inspected. An unchecked system is never logged as airworthy.

| Aviation | PREFLIGHT |
|---|---|
| The aircraft | The video artifact |
| Specialised inspection systems | 12 independent agents |
| Instrumentation readings | Evidence, bound to claims by type |
| Operational constraints | The policy corpus, cited by clause id |
| Corrective maintenance | The remediation compiler → ffmpeg |
| Post-maintenance inspection | Closed-loop re-analysis of the rendered file |
| Airworthiness decision | Release Readiness + verification verdict |
| **"Not inspected" ≠ "serviceable"** | **`UNEXAMINED` ≠ `CLEAN`** |

That last row is the entire thesis.

---

## 🏗 System Architecture

```mermaid
flowchart TB
    subgraph Clients
        CLI["CLI · 13 commands"]
        DECK["Command Deck<br/>React + Vite"]
        GHA["GitHub Action<br/>action.yml"]
    end

    subgraph Server["HTTP API · Python stdlib http.server"]
        REST["REST routes"]
        SSE["SSE /api/events/:id"]
        JOBS["Background job registry"]
    end

    subgraph Engine["Analysis Engine"]
        ORCH["A01 ORCHESTRATOR<br/>DAG scheduler + retries"]
        ING["Ingest · ffprobe<br/>audio + keyframes"]

        subgraph Perception["Perception Agents"]
            SP["A02 SPEECH"]
            VI["A03 VISION"]
            AU["A04 AUDIO"]
            OC["A05 OCR"]
            MT["A06 METADATA"]
            AC["Accessibility"]
        end

        RET["A07 RETRIEVAL<br/>BM25 + dense, RRF"]

        subgraph Triad["Adversarial Triad"]
            AUD["A09 AUDITOR<br/>prosecutes"]
            ADV["A10 ADVOCATE<br/>defends"]
            ADJ["A11 ADJUDICATOR<br/>rules"]
        end

        FUS["A08 FUSION<br/>noisy-OR + rules"]
        INC["Incident correlation"]
        SCO["Release Readiness"]
        SIM["Decision Simulation"]
        REM["A12 REMEDIATION<br/>EDL to ffmpeg"]
    end

    subgraph Storage
        CAS["Content-addressed store<br/>BLAKE3"]
        LIN["Lineage · SQLite<br/>7 tables"]
        POL["Policy corpus<br/>33 clauses"]
    end

    subgraph Providers["Capability Registry"]
        NIM["NVIDIA NIM<br/>hosted"]
        LOC["Local<br/>whisper · tesseract · numpy"]
    end

    CLI --> Engine
    DECK --> Server
    GHA --> CLI
    Server --> Engine

    ORCH --> ING
    ING --> Perception
    VI --> OC
    Perception --> RET
    RET --> POL
    RET --> AUD
    AUD --> ADV
    ADV --> ADJ
    ADJ --> FUS
    FUS --> INC
    INC --> SCO
    SCO --> SIM
    SIM --> REM

    Engine <--> CAS
    Engine <--> LIN
    Perception <--> Providers
    Triad <--> NIM

    REM --> ART["Rendered artifact<br/>*.safe.mp4"]
    ART --> REANALYSIS["Re-analysis<br/>same pipeline"]
    REANALYSIS --> VER["Verification<br/>before/after comparison"]
    VER --> VERDICT["Final verdict<br/>+ certificate"]
```

**Verified component counts:** 12 agents · 74 Python modules (23,862 LOC) · 29 React components (5,943 LOC) · 33 policy clauses · 7 SQLite tables · 19 lifecycle states.

---

## 🤖 The Agent Fleet

The roster is **declared in [`prompts/`](prompts/)** as YAML frontmatter and validated as a DAG at runtime. `preflight agents` prints it and checks conformance:

```
  12 agents · 12 built · 4 model-driven · roster digest 6a0ae1ec23ef4c89
  roster is a valid DAG
```

| ID | Codename | Tier | Kind | Capability | Depends on | Produces |
|---|---|---|---|---|---|---|
| **A01** | ORCHESTRATOR | 0 | deterministic | — | — | `ExecutionReport` |
| **A02** | SPEECH | 2 | deterministic | local ASR | A01 | `Transcript` |
| **A03** | VISION | 2 | **model** | `vision.describe` | A01 | `Finding[]` |
| **A04** | AUDIO | 2 | deterministic | — | A01 | `Finding[]` |
| **A05** | OCR | 3 | deterministic | `ocr.image` | A03 | `Finding[]` |
| **A06** | METADATA | 2 | deterministic | — | A01 | `Finding[]` |
| **A07** | RETRIEVAL | 4 | deterministic | `embed.text`, `rerank.text` | A02, A03, A04, A05 | `Chunk[]` |
| **A09** | AUDITOR | 5 | **model** | `chat.extraction` | A07 | `Candidate[]` |
| **A10** | ADVOCATE | 6 | **model** | `chat.reasoning` | A09 | `Defense[]` |
| **A11** | ADJUDICATOR | 7 | **model** | `chat.reasoning` | A09, A10 | `Verdict[]` |
| **A08** | FUSION | 8 | deterministic | — | A11 | `Finding[]` |
| **A12** | REMEDIATION | 9 | deterministic | — | A08 | EDL, ffmpeg program |

**8 deterministic, 4 model-driven.** The ratio is the design: models are used where judgement is genuinely required (visual description, policy interpretation), and everything measurable — loudness, flash rate, Luhn checksums, timestamps, scoring — is computed, not asked.

### Analysis surface weights

Coverage is a weighted mean over each agent's share of the analysis surface ([`pipeline.py`](preflight/pipeline.py)):

| Agent | Share | Agent | Share |
|---|---|---|---|
| vision | 22% | policy | 10% |
| speech | 20% | access | 6% |
| audio | 15% | ingest | 5% |
| ocr | 13% | score | 5% |
| meta | 4% | orchestrator | 0% |

An agent that never ran contributes **zero coverage, not zero weight** — silently dropping it would let a run that skipped vision entirely still claim 100% coverage.

<details>
<summary><b>Per-agent detail — what each sees, what it does not, and how it moves the verdict</b></summary>

### A01 — ORCHESTRATOR *(deterministic)*
- **Sees:** nothing about the video. It schedules.
- **Does not see:** any frame, sample or word. It never classifies.
- **Produces:** a stage timeline, retry records, and per-stage status.
- **Effect on verdict:** none directly. It guarantees that an agent failure degrades coverage rather than failing the run.
- **Why this matters:** a coordinator that also judges is a coordinator whose judgement cannot be audited separately. A01 is explicitly forbidden from producing findings.

### A02 — SPEECH *(deterministic, local model weights)*
- **Sees:** the audio track, transcribed to word-level timestamps via faster-whisper.
- **Does not see:** anything visual; anything in a language the model does not handle well.
- **Produces:** `Transcript` with word timings, plus quotation spans and framing cues consumed by the triad as exemption evidence.
- **Effect:** load-bearing. Every downstream text judgement rests on its timings — a span placed wrongly becomes a bleep over the wrong word.

### A03 — VISION *(model — `vision.describe`)*
- **Sees:** sampled keyframes, described by a hosted vision-language model.
- **Does not see:** anything between sampled frames; anything it cannot name.
- **Produces:** `Observation` objects — labels, never verdicts. Whether a visible firearm breaches a clause is A11's decision.
- **Effect:** 22% of the analysis surface. Subject to the `DEMOTE` fusion rule — a lone visual claim below the corroboration floor is held at advisory, because VLMs hallucinate objects.

### A04 — AUDIO *(deterministic)*
- **Sees:** the waveform — EBU R128 integrated loudness, true-peak clipping, dead air, channel balance, phase, music-bed presence and ducking.
- **Does not see:** *what* the audio means. It measures.
- **Produces:** findings with confidence near 1.0 and empty defences — there is no arguing with an LUFS reading.
- **Effect:** drives the `audio` and part of the `copyright` sub-score.

### A05 — OCR *(deterministic, `ocr.image` via tesseract)*
- **Sees:** text burned into the picture, across evenly-spaced keyframes, read in parallel.
- **Does not see:** text too small, too fast, or stylised past recognition.
- **Produces:** deduplicated text items with spans — plus the disclosure scan (credentials, PII, cards).
- **Effect:** the highest-consequence findings in the engine (`DISC-01` credential on screen) originate here.
- **Design note:** on-screen text persists across hundreds of frames. Counting frames instead of spans is the classic way to turn one caption into two hundred findings, so items are span-deduplicated.

### A06 — METADATA *(deterministic)*
- **Sees:** the sidecar metadata — title, description, tags, declared category and audience.
- **Does not see:** the video itself.
- **Produces:** disclosure, description, title and tag-hygiene findings.
- **Effect:** the `metadata` sub-score.

### A07 — RETRIEVAL *(deterministic)*
- **Sees:** analysis windows built from the transcript, keyframes and OCR items.
- **Produces:** ranked policy chunks per window, fused by RRF.
- **Effect:** determines *which clause* a window is judged under. A retrieval miss is a finding that never happens.

### A09 — AUDITOR *(model — `chat.extraction`)*
- **Sees:** a window plus its retrieved clauses.
- **Produces:** candidate charges — the prosecution case. Batched (multiple windows per call) for quota efficiency.
- **Effect:** only windows producing a candidate advance, which is also the cost-control cascade.

### A10 — ADVOCATE *(model — `chat.reasoning`)*
- **Sees:** the charge and the clause text.
- **Produces:** a defence built **only** from exemptions the clause itself documents.
- **Effect:** stops the single-pass over-firing that makes a linter uninstallable.

### A11 — ADJUDICATOR *(model — `chat.reasoning`)*
- **Sees:** charge, defence, and clause.
- **Produces:** a ruling with calibrated confidence and a written rationale.
- **Effect:** produces the surviving findings and their confidences.

### A08 — FUSION *(deterministic)*
- **Sees:** every finding from every agent at once, plus per-agent coverage.
- **Produces:** fused confidence and severity adjustments under three named rules.
- **Effect:** can promote, demote, or flag for review — and logs which rule fired.

### A12 — REMEDIATION *(deterministic)*
- **Sees:** the fused findings and the transcript.
- **Produces:** a typed Edit Decision List and an ffmpeg program.
- **Effect:** turns a report into an executable repair.

</details>

---

## 🔄 Agent Interaction Sequence

```mermaid
sequenceDiagram
    autonumber
    actor C as Creator
    participant O as A01 ORCHESTRATOR
    participant I as Ingest
    participant P as Perception<br/>A02·A03·A04·A05·A06
    participant R as A07 RETRIEVAL
    participant T as Triad<br/>A09·A10·A11
    participant F as A08 FUSION
    participant N as Incidents
    participant S as Scoring
    participant M as Simulation
    participant X as A12 REMEDIATION
    participant V as Verification

    C->>O: check video.mp4
    O->>I: probe + extract
    I-->>O: streams, keyframes, audio (BLAKE3-keyed)

    par Independent observation
        O->>P: run perception agents
        P-->>O: Findings + Observations + coverage
    end

    O->>R: build windows, retrieve clauses
    R-->>O: ranked chunks (BM25 ⊕ dense, RRF)

    O->>T: adjudicate each window
    T->>T: AUDITOR prosecutes
    T->>T: ADVOCATE defends (clause exemptions only)
    T->>T: ADJUDICATOR rules + rationale
    T-->>O: Findings with confidence

    O->>F: fuse across modalities
    F-->>O: fusedConfidence, severity rules fired

    O->>N: correlate into incidents
    N-->>O: cross-agent grouped events

    O->>S: sub-scores + Release Readiness
    S-->>C: verdict + coverage + citations

    C->>M: simulate an edit
    M-->>C: predicted score (real scorer, evidence removed)

    C->>X: apply fix
    X->>X: compile EDL, run ffmpeg
    X-->>V: rendered artifact

    V->>V: structural check (duration vs EDL)
    V->>P: RE-ANALYSE the rendered file
    P-->>V: second report
    V->>V: match findings across runs via TimeMap
    V-->>C: RESOLVED / PERSISTING / NEW + certificate
```

---

## 🔍 Coverage-Aware Reasoning

> **"I did not find a problem" is not "I proved there is no problem."**

This is the property most video tooling gets wrong, and it is enforced here at two levels.

### Level 1 — Temporal bands

[`coverage.py`](preflight/coverage.py) projects each modality's evidence back onto the timeline in **one-minute bands** (`DEFAULT_BAND_MS = 60_000`) and classifies each band:

| Band state | Meaning |
|---|---|
| `EXAMINED` | Samples ≥ the thin floor. This band was genuinely inspected. |
| `THIN` | Touched, but below `THIN_BAND_RATIO = 0.5` of expected samples. |
| `UNEXAMINED` | Zero samples. **Nothing looked here.** |

The floor is derived from what the modality *actually produced*, not a fixed target, so a modality that legitimately samples sparsely is not punished for it:

```python
floor = max(MIN_SAMPLES_FOR_EXAMINED, expected[modality] * THIN_BAND_RATIO)
```

### Level 2 — Absence classification

An absence claim over a span resolves to one of four states — and **only one of them supports "this is clean"**:

```mermaid
stateDiagram-v2
    [*] --> Query: "found nothing in span"
    Query --> NOT_RUN: modality never ran
    Query --> NO_COVERAGE: any touched band UNEXAMINED
    Query --> INSUFFICIENT_COVERAGE: any touched band THIN
    Query --> NEGATIVE_EVIDENCE: all touched bands EXAMINED

    NEGATIVE_EVIDENCE --> [*]: absence is supported
    INSUFFICIENT_COVERAGE --> [*]: not clean, unproven
    NO_COVERAGE --> [*]: a hole in the audit
    NOT_RUN --> [*]: contributes no evidence
```

The classification is **pessimistic across a span** — one unexamined minute inside an otherwise covered stretch downgrades the whole claim:

```python
states = {b.state_of(modality) for b in touched}
if "UNEXAMINED" in states: return NO_COVERAGE
if "THIN"       in states: return INSUFFICIENT_COVERAGE
return NEGATIVE_EVIDENCE
```

**Why a simple detector cannot do this.** A detector returns a list of hits. The absence of a hit is indistinguishable from the absence of a look. PREFLIGHT keeps the sample ledger so the two can be told apart — and surfaces it in the UI as a clickable per-minute grid ([`CoverageMap.tsx`](src/components/CoverageMap.tsx)), not as a decorative number.

The same principle runs on the confidence axis in verification: `MIN_COVERAGE_FOR_ABSENCE = 0.5` — an agent that reached under half coverage in the re-analysis cannot have its silence counted as proof a finding was resolved.

---

## 📐 Adaptive Sampling

A flat frame ceiling is fine for a 20-second clip and quietly useless for a long one. At 90 frames over fourteen minutes, samples land **9.3 seconds apart** — an API key visible on screen for two seconds has roughly a one-in-five chance of being seen at all.

**Keyframe budget** ([`ingest/frames.py`](preflight/ingest/frames.py)):

$$\text{budget}(d) = \max\left(90,\ \min\left(\left\lfloor \frac{d}{2.5} \right\rfloor,\ 400\right)\right)$$

where $d$ is duration in seconds. Constants: `MAX_FRAMES = 90` (floor), `TARGET_SAMPLE_INTERVAL_S = 2.5`, `MAX_FRAMES_CEILING = 400`.

**Vision baseline budget** ([`perception/vision.py`](preflight/perception/vision.py)) — vision is billed per call, so it scales on a longer interval with a tighter ceiling:

$$\text{baseline}(d) = \max\left(\text{BASELINE\_FRAMES},\ \min\left(\left\lfloor \frac{d}{30} \right\rfloor,\ 48\right)\right)$$

Constants: `BASELINE_INTERVAL_S = 30.0`, `BASELINE_FRAMES_CEILING = 48`.

Beyond the uniform baseline, vision frame selection is **motion-aware**: a motion signal computed during the single quality-analysis decode pass is handed to the vision agent so the budget follows activity rather than the clock. Static content is preserved by the uniform floor; busy content attracts extra samples up to the ceiling.

> OCR reads its frames on an **evenly spaced** schedule rather than head-slicing the budget (`keyframes[:budget]` would read only the opening), and does so in parallel via a thread pool.

---

## 🔗 Evidence Fusion

Independent agents agreeing is evidence. One agent shouting is not.

[`scoring/fusion.py`](preflight/scoring/fusion.py) combines per-modality confidences with a **coverage-scaled weighted noisy-OR**:

$$C_{\text{fused}} = 1 - \prod_{m \in M}\left(1 - w_m \cdot \text{cov}_m \cdot c_m\right)$$

| Symbol | Meaning |
|---|---|
| $M$ | modalities with confidence > 0 for this finding |
| $c_m$ | that modality's confidence, clamped to $[0,1]$ |
| $w_m$ | modality weight (below) |
| $\text{cov}_m$ | that agent's **actual coverage**, clamped to $[0,1]$ |

**Modality weights** — speech is exact (the word was said or it was not); audio DSP is a proxy for a proxy:

| speech | music | access | meta | vision | ocr | metadata | audio | *default* |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 0.95 | 0.95 | 0.90 | 0.85 | 0.80 | 0.70 | 0.60 | 0.50 |

The coverage term is the honest part: when vision only reached 42% of keyframes, a vision confidence of 0.9 is **not worth 0.9**.

### Three corroboration rules

```mermaid
flowchart TD
    F[Finding with per-modality confidences] --> N[Noisy-OR fusion]
    N --> C{≥2 modalities AND<br/>spread ≥ 0.5?}
    C -->|yes| CON["CONTRADICTION<br/>×0.75, flag for human review"]
    C -->|no| D{vision only AND<br/>fused &lt; 0.70?}
    D -->|yes| DEM["DEMOTE<br/>severity → LOW (advisory)"]
    D -->|no| P{≥2 modalities AND<br/>fused ≥ 0.90 AND<br/>severity MEDIUM/HIGH?}
    P -->|yes| PRO["PROMOTE<br/>severity +1 step"]
    P -->|no| K[unchanged]
```

- **`CONTRADICTION`** (`CONTRADICTION_GAP = 0.5`, `CONTRADICTION_SCALE = 0.75`) — modalities that should agree don't. Something is wrong with the evidence: confidence drops and a human is asked to look.
- **`DEMOTE`** (`DEMOTE_VISION_ONLY = 0.70`) — the rule that earns its place. Vision-language models report a weapon in a frame containing a tripod. A lone visual claim below the floor becomes advisory rather than driving a demonetisation verdict.
- **`PROMOTE`** (`PROMOTE_FUSED = 0.90`, `PROMOTE_MODALITIES = 2`) — several independent modalities agreeing at high confidence raises severity one step.

Every rule that fires is **logged with the before/after severity** and appended to the finding's rationale — the adjustment is auditable, not silent.

**Independence assumption, stated plainly:** noisy-OR assumes the modalities are conditionally independent given the event. They are not perfectly so (OCR reads frames vision also saw). This is why the corroboration bonus in incident correlation is capped and why one agent can never corroborate itself.

---

## 📚 Policy Retrieval

**Corpus:** 33 clauses ([`data/policy/`](data/policy/)), version `2026-08`, `corpus_hash 4dbc352d65185451cc6c57449ab3466c`, each file SHA-256'd in [`manifest.json`](data/policy/manifest.json).

The corpus is explicitly split by **who owns the rule** — this is a correctness property, not documentation:

| `kind` | Count | Meaning |
|---|---|---|
| `policy_restatement` | 17 | Structured restatements **in our own words** of publicly published guidance. Not verbatim, not authoritative, not affiliated. |
| `house_rule` | 16 | PREFLIGHT's **own** production thresholds — loudness targets, caption availability, tag hygiene. **Not platform policy.** |

> A finding cites the clause it was judged under, and the citation is worthless if a reader cannot tell whose rule it is. This is why `DISC-*` (credential on screen) scores under `metadata`, not `policy` — no platform clause says anything about it, and scoring it as policy would attribute the tool's own engineering rule to the platform.

### Hybrid retrieval

BM25 is **implemented in-repo** (~30 lines, [`policy/retrieval.py`](preflight/policy/retrieval.py)) rather than imported. Dense and sparse rankings are fused by weighted **Reciprocal Rank Fusion**:

$$\text{RRF}(d) = \frac{w_{\text{dense}}}{k + \text{rank}_{\text{dense}}(d)} + \frac{w_{\text{sparse}}}{k + \text{rank}_{\text{sparse}}(d)}$$

Constants: `RRF_K = 60`, `DENSE_WEIGHT = 1.0`, `SPARSE_WEIGHT = 0.5`, `SPARSE_FLOOR = 0.35`.

```mermaid
flowchart LR
    Q["Window text"] --> B["BM25<br/>lexical"]
    Q --> D["Dense<br/>nv-embedqa-e5-v5"]
    B --> FL{"BM25 score ≥ 35%<br/>of top score?"}
    FL -->|no| DROP["dropped: incidental<br/>common-word match"]
    FL -->|yes| RRF
    D --> RRF["Weighted RRF<br/>k = 60"]
    RRF --> RR{Reranker<br/>available?}
    RR -->|yes| CE["Cross-encoder over the<br/>FULL fused pool"]
    RR -->|no| OUT[RRF order]
    CE --> OUT
```

**Why hybrid.** Transcript language and policy language rarely share tokens — *"this is fucked"* vs *"strong profanity"*. Dense carries the semantic load; BM25 exists to catch exact tokens dense misses — a specific slur, a named event. Equal weighting on a small corpus lets BM25's incidental word matches outrank real semantic hits, hence the 1.0 / 0.5 split and the sparse floor.

**Reranking covers the full fused pool, not the truncated head** — reordering a top-k the fusion already chose cannot rescue a clause RRF ranked 11th, which is exactly the case a cross-encoder exists to catch. Every reranker failure path returns `None` and leaves RRF order untouched.

---

## 🧩 Incident Intelligence

Four agents that each noticed something at 02:14 have not found four problems. They have found **one problem four times**.

[`scoring/incidents.py`](preflight/scoring/incidents.py) decides which findings describe the same event. Three traps govern the design — and the real corpus contains all three:

```mermaid
flowchart TD
    A[Findings] --> B{Spans ≥ 90% of video?}
    B -->|yes| FS["FILE-SCOPED<br/>never correlated"]
    B -->|no| C{Within 2500ms?}
    C -->|no| SEP[separate incidents]
    C -->|yes| D{Categories compatible?}
    D -->|no| SEP
    D -->|yes| E{Different agent?}
    E -->|no| MEM["member added,<br/>NO confidence bonus"]
    E -->|yes| COR["corroborated:<br/>+0.05, capped 0.97"]
```

| Trap | Why it breaks naive grouping | Rule |
|---|---|---|
| **File-scoped findings correlate with everything** | "No caption track" spans the whole video, so temporal overlap merges it into every incident and produces one meaningless super-incident | `FILE_SCOPED_SHARE = 0.9` — excluded from correlation entirely |
| **Proximity is not relatedness** | Profanity at 02:14 and a frozen frame at 02:14 are two unrelated problems sharing a timestamp | `COMPATIBLE` category map — narrow by design; an unlisted pair is two incidents |
| **One agent cannot corroborate itself** | Two speech findings a second apart are two events, not agreement | Corroboration requires an *independent* agent; a repeat from the same agent adds a member, not confidence |

Corroboration is bounded (`CORROBORATION_STEP = 0.05`, `CONFIDENCE_CEILING = 0.97`) because **certainty is not additive** — a scheme that walks to 0.99 on volume alone has stopped measuring anything.

> **A false split reports two real problems. A false merge hides one behind a label that does not describe it.** The compatibility map is deliberately narrow because those errors are not symmetric.

Note that incident correlation deliberately does **not** re-run fusion's noisy-OR. Fusion combines *modalities within one finding*; correlation combines *separate findings*. Running noisy-OR again here would double-count the same agreement.

---

## 🚫 No-Hallucination Architecture

**The rule is enforced by the type system, not by discipline.**

[`scoring/reasoning.py`](preflight/scoring/reasoning.py) builds a citable chain for every conclusion. `Claim` cannot be constructed without a `source`, and every source references something that already exists in the run:

```mermaid
flowchart LR
    CL[Claim] -->|requires| SRC[Source]
    SRC --> F["finding id"]
    SRC --> C["clause id"]
    SRC --> A["agent id"]
    SRC --> M["measurement"]
```

```
SourceKind = Literal["finding", "clause", "agent", "measurement"]
```

There is **no code path that produces a sentence without a citation attached**. Prose assembled from a template still has to name the finding it was assembled from.

**Nothing in the reasoning layer calls a model.** Every step reads material the run already produced:

| Chain step | Sourced from |
|---|---|
| `observation` | the agent that saw it |
| `evidence` | finding spans and modalities |
| `policy` | the retrieved clause text |
| `risk_argument` | the AUDITOR's charge |
| `counter_argument` | the ADVOCATE's defence |
| `decision` | the ADJUDICATOR's ruling |
| `uncertainty` | dismissed charges, silent agents, coverage lost |

Generating new reasoning at report time would be a fifth opinion nobody adjudicated.

**Role separation is strict.** The orchestrator coordinates and never classifies. Perception agents observe and never rule — A03 VISION reports "a firearm is visible", never "this violates AF-08". Only A11 ADJUDICATOR rules, and only on a charge A09 raised and A10 answered.

The most valuable section of a reasoning chain is the one most reports omit — `uncertainty`: what was **not** concluded, and why. Agents whose coverage fell below `SILENCE_IS_MEANINGFUL_ABOVE = 0.5` are recorded as having barely looked, rather than as having found nothing.

---

## 🔐 Secret & PII Detection

On-screen credentials are the highest-consequence finding the engine produces — a leaked key outlives the video.

**Detected patterns** ([`perception/disclosure.py`](preflight/perception/disclosure.py)):

| Class | Detection |
|---|---|
| AWS access key id | `AKIA…` / `ASIA…` prefixed |
| GitHub token | `ghp_` `gho_` `ghu_` `ghs_` `ghr_` |
| Google API key | `AIza…` |
| Slack token | `xoxa-` `xoxb-` `xoxp-` `xoxr-` `xoxs-` |
| Stripe key | `sk_live_` / `pk_test_` family |
| Private key block | `-----BEGIN … PRIVATE KEY-----` |
| JSON Web Token | three base64url segments |
| Labelled credential | `password` `passwd` `pwd` `secret` `api_key` `access_key` `token` `auth` followed by `:` or `=` |
| Bearer / Basic header | `Authorization: Bearer <…>` |
| Connection string | `scheme://user:pass@host` — credentials in userinfo |
| Payment card | 13–19 digits **validated by Luhn** |
| Email / phone / URL | conservative patterns |

Two details that make this work on real footage rather than in a demo:

**1. SCREAMING_SNAKE_CASE.** A plain `\b` before the keyword cannot match inside `DB_PASSWORD` — `_` is a word character, so there is no boundary. That made the detector miss *every* environment variable, which is the single most common way a real secret appears on screen (`.env` files, `export` lines, docker-compose blocks name things `DB_PASSWORD`, `AWS_SECRET_ACCESS_KEY`, `GITHUB_TOKEN` — never bare `password`). Fixed with a lookbehind alternation:

```python
r"(?:\b|(?<=[_.\-]))(?:password|passwd|pwd|secret|api[_ .\-]?key|…)\b\s*[:=]\s*\S{4,}"
```

**2. Luhn, not "sixteen digits".** Roughly nine in ten random digit runs of card length fail the checksum — it removes almost all of the false positives that would otherwise train a creator to ignore the finding.

### Redaction

```python
def redact(value: str, keep: int = 4) -> str:
    """Enough to recognise, never enough to use."""
```

Reporting *"your key is on screen"* by printing the key would put the secret into `report.json`, the HTML, the terminal scrollback and any CI log that captured the run — **more copies than the video ever made**. A finding shows `AKIA********` and its timestamp.

A separate redaction sweep runs in [`providers/secrets.py`](preflight/providers/secrets.py) over anything crossing a provider boundary, and `preflight check` refuses to persist a report containing the configured API key at all:

```python
if len(key) >= 12 and key in text:
    raise ApiError(500, "refusing to persist a report containing the API key")
```

There is a regression test asserting the report contains **zero** credential material after a run over footage that contains some.

---

## 🎲 Decision Simulation

The report says a video is risky. The question a creator actually has is *what to do about it, and which option is worth the edit.*

[`scoring/simulation.py`](preflight/scoring/simulation.py) answers that **without touching the video** — every scenario is computed from findings the run already produced, so a what-if costs microseconds and no ffmpeg, no Whisper, no hosted call.

```mermaid
flowchart TD
    A[Original findings + evidence] --> B[Choose candidate edit]
    B --> C["Remove the OBSERVATIONS<br/>that edit destroys"]
    C --> D[Re-fuse remaining modalities]
    D --> E["Re-score with the REAL scorer<br/>sub_scores + compute_readiness"]
    E --> F[Predicted readiness]
    F -.separate path.-> G[Apply real remediation]
    G --> H[Actual re-analysis]
    H --> I[Compare prediction vs actual]
```

**Two decisions carry the whole module.**

**1. Edits remove evidence, not findings.** Muting 10–12s does not delete a finding — it deletes the *speech observation* inside that span. A finding corroborated by vision survives the mute with lower confidence; one that existed only in audio disappears. Simulating at the finding level ("remove the finding, subtract its risk") gets both wrong, and gets them wrong **in the flattering direction**, promising reductions the edit cannot deliver.

| Edit | Observations destroyed |
|---|---|
| `BLEEP` | speech |
| `MUTE` | speech, audio, music |
| `BLUR_REGION` | vision, ocr |
| `REPLACE_AUDIO` | music, audio |
| `CUT` | speech, audio, music, vision, ocr, access, meta |
| `INSERT_DISCLOSURE` | meta |
| `REPLACE_THUMBNAIL` | meta |
| `ADD_CAPTIONS` | access |

**2. The predicted score is computed by the real scorer.** `sub_scores` and `compute_readiness` are the same functions that produced the current score, called on modified findings. There is no second risk model to drift out of step — which matters because the real one is deliberately non-linear: the combiner saturates, so removing one of two `CRITICAL` findings barely moves the number, while the anti-masking clamp means removing the *worst* finding can move it a great deal. Any independent estimate would have to reproduce both behaviours exactly, and would eventually fail to.

**Prediction and verification stay strictly separate.** The simulation is a forecast; the verification is a measurement. The comparison records `predictionIsForThisEdit` explicitly, because the compiler routinely picks a different (balanced) operation set than the highest-scoring scenario — comparing a prediction about one edit against the result of a different one would report `UNDERESTIMATED` for a simulation that was never wrong.

---

## 🛠 Remediation

Findings are **compiled, not suggested**.

```mermaid
flowchart LR
    F[Findings] --> EDL["Typed Edit Decision List"]
    EDL --> OPT["Optimiser passes<br/>fixed order"]
    OPT --> CG[ffmpeg codegen]
    CG --> TMP["Staged temp file<br/>*.tmpPID.mp4"]
    TMP --> SV{"Duration within<br/>1500ms of expected?"}
    SV -->|no| FAIL["unlink + FAILED<br/>nothing promoted"]
    SV -->|yes| PROM["Atomic promote<br/>→ *.safe.mp4"]
    PROM --> RE[Re-analysis]
```

**Operations** (`OpKind` in [`remediate/edl.py`](preflight/remediate/edl.py)):

| Op | Impact | Touches video? | Risk reduction |
|---|---|---|---|
| `REPLACE_AUDIO` | 0.10 | no | 0.95 |
| `BLEEP` | 0.15 | no | 0.90 |
| `BLUR_REGION` | 0.20 | yes | 0.85 |
| `MUTE` | 0.25 | no | 0.90 |
| `CUT` | 0.80 | yes | 1.00 |

Three properties worth reading:

- **Ordering is load-bearing.** Snap-to-word runs *before* coalesce, because coalescing first would merge spans that word boundaries had not yet widened.
- **The cut budget demotes rather than deletes.** `MAX_CUT_RATIO = 0.08` — beyond that, a `CUT` becomes a `MUTE` instead of silently removing a third of someone's footage.
- **If the EDL contains no video ops, the video stream is never re-encoded.** A long repair becomes a stream copy plus an audio pass. `videoStreamCopied` is reported in `report.json`.

**Atomicity.** The render goes to a staged temp file whose extension stays last (so ffmpeg's muxer can still infer the container), is verified on duration against the EDL, and only then is promoted with `Path.replace`. A half-written `.safe.mp4` sitting next to the original is worse than no output, because it *looks* like a finished render. The lifecycle row moves to `RENDERING` **before** ffmpeg starts and `RENDERED` **after** the file lands, so a process killed mid-render leaves a row that says `RENDERING` — which is true — rather than no row at all.

```bash
preflight fix samples/demo.mp4              # dry run — prints the program
preflight fix samples/demo.mp4 --apply      # render to demo.safe.mp4
preflight fix samples/demo.mp4 --apply --strategy conservative
```

---

## ✅ Closed-Loop Verification

> **Detection is not verification. A successful ffmpeg process does not mean the video is safe.**

This is the section that separates PREFLIGHT from a linter. The rendered artifact goes back through the **same** `run_perception` — not a cheaper checker, not a diff of the original report — and the two reports are compared.

```mermaid
stateDiagram-v2
    direction LR
    [*] --> RENDERING
    RENDERING --> RENDERED: ffmpeg output written
    RENDERED --> STRUCTURAL_VERIFYING
    STRUCTURAL_VERIFYING --> STRUCTURALLY_VALID: duration matches EDL
    STRUCTURALLY_VALID --> REANALYSIS_QUEUED
    REANALYSIS_QUEUED --> REANALYSING
    REANALYSING --> REANALYSIS_COMPLETE
    REANALYSING --> INCONCLUSIVE: re-analysis failed
    REANALYSIS_COMPLETE --> COMPARING
    COMPARING --> VERIFIED
    COMPARING --> PARTIALLY_REMEDIATED
    COMPARING --> NEW_RISK_DETECTED
    COMPARING --> NO_CHANGE
    COMPARING --> INCONCLUSIVE
```

`RENDERING` leads **only** to `RENDERED`. No amount of optimism gets a render to a verdict without passing through the three stages that actually check it.

### The real status vocabulary

**Finding-level** — `RESOLVED` · `PERSISTING` · `NEW` · `CHANGED` · `INCONCLUSIVE`

**Incident-level** — the same five plus `PARTIALLY_REMEDIATED`. An incident is a group, so it can be genuinely half-fixed in a way a single finding cannot; collapsing that into `PERSISTING` would hide real progress, and into `RESOLVED` would hide a real remaining problem.

**Run verdict** — `VERIFIED_SAFE` · `PARTIALLY_REMEDIATED` · `REMEDIATION_FAILED` · `NEW_RISK_DETECTED` · `INCONCLUSIVE` · `NO_CHANGE`

**Prediction outcome** — `MATCHED` · `PARTIALLY_MATCHED` · `OVERESTIMATED` · `UNDERESTIMATED` · `FAILED` · `INCONCLUSIVE`

### New-risk detection

A remediation can *create* problems — a `CUT` can produce a jarring transition, `REPLACE_AUDIO` can introduce a different copyright exposure. Findings present in the second report but absent from the first are classified `NEW`, and a run with new incidents returns **`NEW_RISK_DETECTED`** regardless of how many original findings were resolved.

### The certificate

Each verification issues a signed-shape certificate ([`certificate.py`](preflight/certificate.py)) carrying both artifact hashes, both run ids, predicted vs actual score, and the coverage the second run reached. It is issued from the **stored verification's timestamp**, not the clock, so re-issuing it for the same verification reproduces the same document and the same hash — a hash that changed on every read would certify nothing. Certificate integrity is re-checkable (`VALID` / `MISMATCH`).

---

## ⚖️ Before/After Comparison

Three things make this harder than diffing two lists ([`verify.py`](preflight/verify.py)):

### 1. Ids do not survive

The second run assigns its own finding ids. Matching on them would report every finding as resolved **and** every one as new, simultaneously. Identity here is **what the finding is, not what it was called**: clause + category + mapped span.

### 2. Timestamps move

A `CUT` removes ten seconds, so everything after it shifts earlier by ten seconds. Comparing raw timestamps across a cut compares unrelated moments. The EDL is the authority, and `TimeMap` applies it:

```mermaid
flowchart LR
    O["Original span<br/>45.0s – 47.0s"] --> TM["TimeMap<br/>cuts: (10.0s, 20.0s)"]
    TM --> M["Mapped span<br/>35.0s – 37.0s"]
    TM --> N["Inside a removed span<br/>→ None"]
```

Only `CUT` changes the timeline — `MUTE`, `BLEEP`, `BLUR_REGION` and `REPLACE_AUDIO` all preserve duration, so a report full of those maps one-to-one and `TimeMap` is the identity function.

A finding **inside** a removed span maps to `None`. That is not a failure — it is the correct answer, and it is why a span that was cut cannot be "still detected".

### 3. Absence is not proof

Matching uses `MATCH_IOU = 0.3` with a `MATCH_TOLERANCE_MS = 1_500` fallback — loose on purpose, because a re-encode moves a detected span by a frame or two and demanding exact equality would report one persisting finding as one resolved plus one new.

But the critical guard is coverage: where the mapping cannot place a finding, **or the agent that would have seen it fell below `MIN_COVERAGE_FOR_ABSENCE = 0.5`**, the comparison returns `INCONCLUSIVE` rather than claiming a resolution it cannot support.

> This is what stops "make re-analysis cheaper" from becoming "make success more likely" — the worst incentive it is possible to build into a verification loop.

---

## 📊 Risk Scoring

Five dimensions, **all running the same direction — 100 is always good**, so a 90%-full bar means the same thing whether it measures copyright exposure or caption coverage.

| Dimension | Weight | Fed by clause families |
|---|---|---|
| `policy` | 0.40 | `AF-*` |
| `copyright` | 0.30 | `COPY-*`, `CID-*` |
| `metadata` | 0.12 | `META-*`, `DISC-*` |
| `accessibility` | 0.10 | `ACC-*`, `VID-*` |
| `audio` | 0.08 | `AUD-*` |

### Per-finding risk

$$r_f = \min\left(0.99,\ \; s_f \cdot c_f \cdot d_f \cdot p_f\right)$$

| Term | Definition |
|---|---|
| $s_f$ | severity risk — `CRITICAL` 1.00, `HIGH` 0.55, `MEDIUM` 0.28, `LOW` 0.10 |
| $c_f$ | `fusedConfidence` (falls back to raw confidence), clamped $[0,1]$ |
| $d_f$ | duration weight — $0.35 + 0.65\cdot\min(1, \text{span}/10000\text{ms})$, **fixed at 1.0 for `CRITICAL`** |
| $p_f$ | position multiplier — **1.35** if `startMs < 30000`, else 1.0 |

Two deliberate asymmetries: a violation in the opening weighs more because it is what a classifier *and* a human reviewer both see first; and `CRITICAL` findings are **not duration-discounted**, because a three-second graphic injury demonetises the whole upload and discounting it to 48% for brevity would score that video as merely mediocre.

### Saturating combiner

$$\text{risk} = 100 \cdot \left(1 - \prod_{f}(1 - r_f)\right), \qquad \text{sub} = 100 - \text{risk}$$

Many small findings can never exceed one severe one — the property that stops a video full of advisories outranking a video with a Content ID match.

### The anti-masking clamp

$$\text{overall} = \text{round}_{\text{half-up}}\Big(\min\big(\textstyle\sum_d w_d \cdot \text{sub}_d,\ \ \text{worst} + 15\big)\Big)$$

**That clamp is the whole design.** From a real verified run on `samples/synthetic.mp4`:

```
    policy         ████████████████████ 100.0
    copyright      ██████████████████··  87.5
    metadata       ██████··············  29.5 ← weakest
    accessibility  ██████··············  30.2
    audio          ████████████████····  78.5

    45 / 100   DO NOT PUBLISH
    weighted mean 79.1, capped at weakest + 15 — one fatal flaw is never averaged away
```

A plain weighted average scores **79.1** — comfortably passing. The clamp brings it to **45 / DO NOT PUBLISH**.

### Verdicts

| Verdict | Condition |
|---|---|
| `READY_TO_PUBLISH` | overall ≥ 85 **and** worst ≥ 70 |
| `PUBLISH_WITH_FIXES` | overall ≥ 70 **and** worst ≥ 50 |
| `NOT_READY` | overall ≥ 50 |
| `DO_NOT_PUBLISH` | otherwise |

### Cross-language contract

The page renders the TypeScript scorer ([`src/lib/scoring.ts`](src/lib/scoring.ts)) and `report.json` carries the Python one — a one-point disagreement makes the certificate a lie. Shared vectors pin them together in **both** test suites, including the half-integer cases where Python's banker's-rounding `round()` and JavaScript's `Math.round` disagree:

```python
def js_round(value: float) -> int:
    """Round half UP, the way JavaScript's Math.round does."""
    return int(math.floor(value + 0.5))
```

CI regenerates the vectors and fails on any diff.

---

## 🧮 Mathematical Foundation

Every formula below is extracted from source. Nothing here is decorative.

<details>
<summary><b>Formal system model</b></summary>

Let $V$ be a video artifact and $A = \{a_1, \dots, a_{12}\}$ the agent fleet.

| Symbol | Definition |
|---|---|
| $E = A(V)$ | evidence — per-agent observations with timestamps |
| $\Gamma$ | temporal coverage: $\text{bands} \times \text{modalities} \to \{\text{EXAMINED}, \text{THIN}, \text{UNEXAMINED}\}$ |
| $P$ | policy corpus, $\lvert P \rvert = 33$ clauses |
| $F$ | findings, each carrying (clause, span, severity, modality confidences) |
| $I$ | incidents — a partition-like grouping over non-file-scoped $F$ |
| $R$ | remediation: an EDL over 5 typed operations |
| $V'$ | the rendered artifact |
| $\Delta$ | the comparison $F \leftrightarrow F'$ through `TimeMap` |

The pipeline is:

$$V \xrightarrow{A} E \xrightarrow{\text{retrieve}(P)} \xrightarrow{\text{triad}} F \xrightarrow{\text{fuse}} F^{*} \xrightarrow{\text{correlate}} I \xrightarrow{\text{compile}} R \xrightarrow{\text{ffmpeg}} V' \xrightarrow{A} F' \xrightarrow{\Delta} \text{verdict}$$

with the coverage constraint applied at every absence claim:

$$\text{claim}(\text{"no } x \text{ in } [t_0,t_1]\text{"}) \text{ is valid} \iff \Gamma(b, m) = \text{EXAMINED} \ \ \forall b \cap [t_0, t_1] \neq \emptyset$$

</details>

| # | Model | Equation | Where |
|---|---|---|---|
| 1 | Coverage-scaled noisy-OR fusion | $1 - \prod_m (1 - w_m \text{cov}_m c_m)$ | [`fusion.py`](preflight/scoring/fusion.py) |
| 2 | Weighted RRF | $\sum_r w_r / (60 + \text{rank}_r(d))$ | [`retrieval.py`](preflight/policy/retrieval.py) |
| 3 | Per-finding risk | $\min(0.99, s\cdot c\cdot d\cdot p)$ | [`readiness.py`](preflight/scoring/readiness.py) |
| 4 | Saturating risk combiner | $100(1 - \prod_f (1 - r_f))$ | [`readiness.py`](preflight/scoring/readiness.py) |
| 5 | Anti-masking clamp | $\min(\bar{w}, \text{worst} + 15)$ | [`readiness.py`](preflight/scoring/readiness.py) |
| 6 | Weighted coverage | $\sum_a \omega_a \text{cov}_a / \sum_a \omega_a$ | [`pipeline.py`](preflight/pipeline.py) |
| 7 | Band thin-floor | $\max(1, \bar{n}_m \cdot 0.5)$ | [`coverage.py`](preflight/coverage.py) |
| 8 | Keyframe budget | $\max(90, \min(\lfloor d/2.5 \rfloor, 400))$ | [`frames.py`](preflight/ingest/frames.py) |
| 9 | Vision baseline budget | $\max(b, \min(\lfloor d/30 \rfloor, 48))$ | [`vision.py`](preflight/perception/vision.py) |
| 10 | Span match | $\text{IoU} \ge 0.3 \ \lor\ \lvert\Delta t\rvert \le 1500\text{ms}$ | [`verify.py`](preflight/verify.py) |
| 11 | Corroboration bonus | $\min(0.97, c + 0.05 k)$ | [`incidents.py`](preflight/scoring/incidents.py) |
| 12 | Luhn checksum | $\sum \text{doubled-odd digits} \equiv 0 \pmod{10}$ | [`disclosure.py`](preflight/perception/disclosure.py) |

**Stated limitations of the math:**
- Noisy-OR assumes conditional independence between modalities. OCR and vision read the same frames, so this is approximate — mitigated by the self-corroboration ban and the bounded corroboration bonus, not eliminated.
- The severity → risk mapping is a calibrated design choice, not a fitted parameter. It has not been optimised against labelled outcome data.
- Confidence values from the model-driven agents are model-reported and calibrated by prompt, not by post-hoc calibration curves.

---

## 🗃 Data Model

Lineage is persisted in **SQLite** ([`lineage.py`](preflight/lineage.py)) — 7 tables. This is what makes a restart informative rather than lossy.

```mermaid
erDiagram
    ARTIFACTS ||--o{ RUNS : "measured in"
    RUNS ||--o| SIMULATIONS : "produces"
    RUNS ||--o{ REMEDIATIONS : "source of"
    REMEDIATIONS ||--o{ TRANSITIONS : "records"
    REMEDIATIONS ||--o| VERIFICATIONS : "verified by"
    VERIFICATIONS ||--o| CERTIFICATES : "attests"
    ARTIFACTS ||--o{ REMEDIATIONS : "output of"

    ARTIFACTS {
        string artifact_id PK
        string path
        string hash "BLAKE3"
        int duration_ms
    }
    RUNS {
        string run_id PK
        string role "ORIGINAL | VERIFICATION"
        string video_hash
        string report_path
        string artifact_id FK
        string parent_run_id FK
    }
    SIMULATIONS {
        string simulation_id PK
        string run_id FK
        json scenarios
    }
    REMEDIATIONS {
        string remediation_id PK
        string source_run_id FK
        string state
        json edl_json
        json finding_ids
        json incident_ids
        string artifact_id FK
    }
    TRANSITIONS {
        int id PK
        string remediation_id FK
        string state
        string detail
        string created_at
    }
    VERIFICATIONS {
        string verification_id PK
        string remediation_id FK
        string verdict
        json comparison
        json telemetry
    }
    CERTIFICATES {
        string certificate_id PK
        string verification_id FK
        json body
        string digest
    }
```

An artifact is only reused if **the row names it, the file exists, and its bytes still hash to the recorded digest** — trusting a path because a row mentions it would make persistence a correctness *regression*.

---

## 🔁 Lifecycle State Machine

19 states, 6 terminal ([`lifecycle.py`](preflight/lifecycle.py)). Transitions not listed are **impossible**, and the omissions carry the meaning.

```mermaid
stateDiagram-v2
    [*] --> ANALYSIS_COMPLETE
    ANALYSIS_COMPLETE --> SIMULATION_READY
    ANALYSIS_COMPLETE --> REMEDIATION_REQUESTED
    SIMULATION_READY --> SIMULATING
    SIMULATION_READY --> REMEDIATION_REQUESTED
    SIMULATING --> SIMULATED
    SIMULATED --> SIMULATING: simulate again
    SIMULATED --> REMEDIATION_REQUESTED
    REMEDIATION_REQUESTED --> RENDERING
    RENDERING --> RENDERED
    RENDERED --> STRUCTURAL_VERIFYING
    STRUCTURAL_VERIFYING --> STRUCTURALLY_VALID
    STRUCTURALLY_VALID --> REANALYSIS_QUEUED
    REANALYSIS_QUEUED --> REANALYSING
    REANALYSING --> REANALYSIS_COMPLETE
    REANALYSING --> INCONCLUSIVE
    REANALYSIS_COMPLETE --> COMPARING
    COMPARING --> VERIFIED
    COMPARING --> PARTIALLY_REMEDIATED
    COMPARING --> NEW_RISK_DETECTED
    COMPARING --> NO_CHANGE
    COMPARING --> INCONCLUSIVE
    VERIFIED --> [*]
    PARTIALLY_REMEDIATED --> [*]
    NEW_RISK_DETECTED --> [*]
    NO_CHANGE --> [*]
    INCONCLUSIVE --> [*]
    FAILED --> [*]
```

Because every transition is persisted **before** the work it describes, a crashed process leaves a truthful row. `GET /api/remediations` reports interrupted work explicitly, and the engine can resume from it — a lifecycle that only becomes durable on success can only ever record successes, and the failures are the interesting part.

---

## 📦 Artifacts

| Artifact | Purpose | Produced by | Consumed by |
|---|---|---|---|
| `report.json` | The full analysis — findings, incidents, reasoning, simulation, coverage, agents, cost | [`report/build.py`](preflight/report/build.py) | Command Deck, CI, `verify` |
| `report.sarif` | SARIF 2.1.0 static-analysis results | [`report/sarif.py`](preflight/report/sarif.py) | GitHub Security tab |
| `certificate.json` | Attestation — score, weights, clamp rule, corpus hash | [`certificate.py`](preflight/certificate.py) | Auditors, clients recomputing the score |
| `report.html` | **One self-contained file** — JS, CSS, poster, evidence frames all inlined | [`report/html.py`](preflight/report/html.py) | Anyone, offline |
| `<name>.safe.mp4` | The rendered remediation | [`remediate/codegen.py`](preflight/remediate/codegen.py) → ffmpeg | Re-analysis, the creator |
| `fix.sh` | The ffmpeg program as a runnable script | `preflight fix` | Manual inspection |
| `edl.json` | The typed edit decision list | [`remediate/edl.py`](preflight/remediate/edl.py) | `TimeMap`, verification |
| `drift.json` | Policy delta between two corpus snapshots | [`drift.py`](preflight/drift.py) | Archive re-lint selection |
| `.preflight/lineage.db` | Runs, remediations, verifications, certificates | [`lineage.py`](preflight/lineage.py) | `/api/lineage`, resume |
| `.preflight/cache/` | Content-addressed intermediate artifacts | [`cas.py`](preflight/cas.py) | Every subsequent run |

`report.html` has **no external references at all** — there is a test asserting it. Double-click it and it works offline, on a plane, forever.

---

## 🌐 API Reference

Served by [`preflight/server.py`](preflight/server.py) — Python's stdlib `ThreadingHTTPServer`. **No web framework.** The project ships numpy and typer and nothing else heavy; adding FastAPI and an ASGI server to serve a handful of JSON routes would be the largest dependency in the tree.

> **Two rules govern every route.** (1) The engine is never reimplemented — every route calls the same `run_perception` / `build_report` the CLI calls. (2) Nothing leaves the process that a credential could ride out on.

**Base URL (deployed):** `https://preflight-api-vax3.onrender.com` · **Local:** `http://127.0.0.1:8000`

### `GET /api/health`

Engine version, ffmpeg availability, and the resolved capability plan. Reports capability **names and tiers, never secrets**.

```bash
curl -s https://preflight-api-vax3.onrender.com/api/health
```

```json
{
  "status": "ok",
  "engineVersion": "0.1.0",
  "ffmpeg": "ffmpeg version 7.1.5-0+deb13u1",
  "ffmpegAvailable": true,
  "online": true,
  "capabilities": {
    "chat.reasoning":  { "tier": "hosted", "degraded": false },
    "chat.extraction": { "tier": "hosted", "degraded": false },
    "asr.transcribe":  { "tier": "local",  "degraded": false },
    "embed.text":      { "tier": "hosted", "degraded": false },
    "rerank.text":     { "tier": "hosted", "degraded": false },
    "vector.search":   { "tier": "local",  "degraded": true  },
    "vision.describe": { "tier": "hosted", "degraded": false },
    "ocr.image":       { "tier": "local",  "degraded": false }
  },
  "capabilitySummary": { "preferred": 7, "fallback": 1, "unavailable": 0 },
  "time": "2026-08-16T13:22:18Z"
}
```

### `GET /api/agents`

The roster as `preflight agents` prints it — ids, codenames, kinds, capabilities, dependencies, and DAG conformance problems.

### `GET /api/runs` · `GET /api/runs/{id}`

Every report on disk (newest first), and one full report.

```bash
curl -s http://127.0.0.1:8000/api/runs | head -c 400
```

### `GET /api/runs/{id}/media`

The measured artifact, **with HTTP range support** so the player can seek to an exact evidence timestamp without downloading the whole file. The path is resolved from the lineage record, **never** from a URL component.

Returns `206 Partial Content` for a `Range` request, `416` for an unsatisfiable one.

### `POST /api/upload`

Multipart (browser-owned encoding) or raw octet-stream with `X-Filename`. Storage ids are **server-generated** (`vid_<uuid>.mp4`); the user filename is preserved as metadata rather than becoming a filesystem path. Unicode filenames travel in the multipart body, never a header.

```bash
curl -s -X POST http://127.0.0.1:8000/api/upload \
  -F "file=@samples/demo.mp4"
```

```json
{ "id": "vid_9ddfdf1595334839a90990df55a1fca9",
  "path": ".preflight/uploads/vid_9ddfdf1595334839a90990df55a1fca9.mp4",
  "name": "demo.mp4", "bytes": 2379241 }
```

### `POST /api/jobs` → `GET /api/events/{id}` (SSE)

Start an analysis and **watch it happen**. The synchronous route answers only when everything is finished — minutes of silence on a hosted run, indistinguishable from a hang.

```bash
JOB=$(curl -s -X POST http://127.0.0.1:8000/api/jobs \
  -H 'Content-Type: application/json' \
  -d '{"video":"samples/demo.mp4","offline":true}' | python -c "import json,sys;print(json.load(sys.stdin)['id'])")

curl -N http://127.0.0.1:8000/api/events/$JOB
```

Returns `202` with `{"id": "...", "events": "/api/events/..."}`. A missing file is rejected **before** a job exists (`404`), rather than returning `202` and failing seconds later over the stream.

Event stream (`text/event-stream`), sequence-numbered for idempotent delivery:

| `type` | Carries |
|---|---|
| `run.start` | source name + full topology (tier, parents) |
| `stage.start` | stage, agentId, name |
| `stage.end` | status, coverage, elapsedMs, findings, calls |
| `fix.progress` | stage, detail, persisted lifecycle `state` |
| `run.complete` | verification, certificate, evidence, telemetry, afterReport |
| `run.error` | error |

Keepalive comment frames (`: keepalive`) hold the connection open without reaching the `EventSource` message handler.

### `GET /api/jobs/{id}`

The authoritative job record — used to reconcile if a proxy drops the SSE connection. This is recovery, not a second execution path.

### `POST /api/analyze`

Synchronous analysis. Returns `{ "id": ..., "report": {...} }`. Guarded by a lock — a second concurrent call gets `409`.

### `POST /api/fix`

Compile the remediation, render it, then **prove it worked**. Returns `202` + job id; progress and the terminal payload arrive over SSE.

```bash
curl -s -X POST http://127.0.0.1:8000/api/fix \
  -H 'Content-Type: application/json' \
  -d '{"video":"samples/synthetic.mp4","offline":true}'
```

Terminal payload includes `rendered`, `output`, `ops`, `renderMs`, `videoStreamCopied`, `command`, `verification`, `certificate`, `evidence`, `telemetry`, `sourceRunId`, `remediationId`, `verificationId`, `verificationRunId`, `certificateId`, `lifecycle`, and `afterReport`.

> `verificationId` (`VER-0007`) and `verificationRunId` (`verify-…`) are **different ids** — the first is the comparison record, the second is the runs-table key whose media is fetchable at `/api/runs/{id}/media`.

### `POST /api/plan`

Estimate the work before spending any of it — the analysis plan and projected call count.

### `GET /api/remediations` · `GET /api/remediations/{id}`

Every remediation on disk **including ones a crash left open**, with the state each was interrupted in. Resolving one id returns it through to its certificate and integrity check.

### `GET /api/lineage/{run_id}`

One original run and everything derived from it.

### Errors

Uniform JSON: `{"error": "..."}`. A stack trace **never** crosses the wire — there is a test asserting the body contains neither `Traceback` nor the string `preflight`. Run ids are regex-validated (`^[A-Za-z0-9_.-]{1,120}$`) and traversal attempts return `400`/`404`.

CORS is open (`*`) with `OPTIONS` preflight support, since the deck is a different origin.

---

## 💻 CLI Reference

13 commands. `preflight --help` for the full list.

```
version       Print versions of PREFLIGHT and its media toolchain.
probe         Probe a video and populate the artifact store.
check         Analyse a video, print findings, and score it against the full triad.
fix           Compile findings into an ffmpeg program and optionally run it.
doctor        Diagnose the environment, credentials and capability plan.
serve         Serve the HTTP API the Command Deck reads.
agents        Print the agent roster declared in prompts/, and its conformance.
bench         Score the pipeline against the golden corpus.
capabilities  Print the capability plan: which provider serves what, and why.
snapshot      Capture the current policy corpus for later drift comparison.
drift         Detect policy changes and find which archived videos they put at risk.
cache         Inspect or clear the content-addressed store.
models        Cache the local models offline analysis depends on.
```

### `preflight check`

```bash
preflight check samples/demo.mp4                    # analyse and score
preflight check samples/demo.mp4 --html             # + self-contained report.html
preflight check samples/demo.mp4 --format all       # json,sarif,certificate,html
preflight check samples/demo.mp4 --offline          # never touch the network
preflight check samples/demo.mp4 --budget 20        # cap hosted model calls
preflight check samples/demo.mp4 --strategy conservative
```

| Flag | Effect |
|---|---|
| `--cache-dir <path>` | default `.preflight/cache` |
| `--asr-model <str>` | faster-whisper model, default `base.en` |
| `--no-speech` | skip transcription |
| `--html` | emit a self-contained `report.html` |
| `--format <str>` | comma-separated: `json,sarif,certificate,html,all` |
| `--out <path>` | output directory, default `preflight-out` |
| `--emit-fixture <path>` | write this run as the UI's demo fixture |
| `--offline` | never touch the network |
| `--strategy <str>` | `conservative` \| `balanced` \| `aggressive` |
| `--budget <int>` | ceiling on hosted model calls; the run **sheds work to stay inside it and reports what it gave up**, and coverage falls to match. `0` = no ceiling |

**Exit codes:** `0` pass · `1` findings exceed threshold · `2` input/config error · `3` upstream unavailable and no fallback permitted. A video file can fail CI.

### `preflight fix`

```bash
preflight fix samples/demo.mp4                      # dry run — print the program
preflight fix samples/demo.mp4 --apply              # render <name>.safe.mp4
preflight fix samples/demo.mp4 --apply --strategy aggressive --out fixed.mp4
```

| Flag | Effect |
|---|---|
| `--apply` | actually render; without it, dry-run only |
| `--strategy <str>` | `conservative` \| `balanced` \| `aggressive` |
| `--out <path>` | output path, default `<name>.safe.mp4` |
| `--cache-dir <path>` | default `.preflight/cache` |
| `--offline` | never touch the network |

### `preflight doctor` / `capabilities`

Prints the full capability plan — which provider serves what, why, and **the exact command that fixes each gap**.

### `preflight bench`

```bash
preflight bench --offline --limit 6      # smoke run
preflight bench --ablation --out bench.json
```

| Flag | Effect |
|---|---|
| `--ablation` | report every layer, not just the shipped one |
| `--labels <path>` | ground truth, default `data/corpus/labels.jsonl` |
| `--clips <path>` | default `data/corpus/clips` |
| `--out <path>` | write the full result as JSON |
| `--limit <int>` | first N clips only |

### `preflight drift`

```bash
preflight snapshot            # capture the current corpus
preflight drift               # compare, and find which archived videos are at risk
```

### `preflight models` / `cache`

```bash
preflight models pull asr     # cache faster-whisper weights
preflight models pull embed
preflight models pull         # everything
preflight cache               # inspect the content-addressed store
```

---

## 🖥 Web Application

The **Command Deck** — React 18 + TypeScript + Vite + Zustand + Tailwind. 29 components (23 workspace + 6 landing), 5,943 lines.

> **Screenshots are not committed to this repository.** The deck is live at https://preflight-deck.onrender.com and demonstrated in the [demo video](https://youtu.be/4MqqM_7RoZE).

### Landing → workspace

A 3D activation intro ([`landing/`](src/components/landing/)) built with `@react-three/fiber` + `drei` — an agent constellation, animated pipeline steps, and an activation wipe into the workspace. The 3D scene is **lazy-loaded** so it never costs the analysis workspace anything.

### Workspace panels

| Panel | What it does | Interactive |
|---|---|---|
| [`RunBar`](src/components/RunBar.tsx) | Upload a file, start a run, watch stages settle | ✅ upload + live SSE |
| [`AgentFlow`](src/components/AgentFlow.tsx) | The live DAG — every agent, tier, parent edge, status | ✅ streams from `run.start` topology |
| [`TerminalColumn`](src/components/TerminalColumn.tsx) | Live engine log | ✅ streamed |
| [`FileCard`](src/components/FileCard.tsx) | Media metadata, attestation hash, poster | ✅ real frame grab via `requestVideoFrameCallback` |
| [`ScoreGauge`](src/components/ScoreGauge.tsx) / [`SubScorePanel`](src/components/SubScorePanel.tsx) | Release Readiness + five dimensions | ✅ |
| [`CoverageMap`](src/components/CoverageMap.tsx) | **Per-minute × per-modality coverage grid** | ✅ click a band to seek |
| [`RiskTimeline`](src/components/RiskTimeline.tsx) | Risk bands across the timeline | ✅ click to seek |
| [`IncidentsPanel`](src/components/IncidentsPanel.tsx) | Correlated incidents, members, thumbnails | ✅ expand, select, seek |
| [`FindingsList`](src/components/FindingsList.tsx) / [`DetailPanel`](src/components/DetailPanel.tsx) | Findings + full reasoning chain | ✅ sort, filter, select |
| [`PolicyBreakdown`](src/components/PolicyBreakdown.tsx) | Clause citations with retrieved text | ✅ |
| [`EvidencePanel`](src/components/EvidencePanel.tsx) | Before/after evidence pairs | ✅ |
| [`DecisionSimulator`](src/components/DecisionSimulator.tsx) | What-if scenarios and predicted score | ✅ |
| [`RemediationPlan`](src/components/RemediationPlan.tsx) | The compiled op list + **Apply fix** | ✅ triggers real render |
| [`FfmpegBlock`](src/components/FfmpegBlock.tsx) | The generated ffmpeg command | ✅ copy |
| [`BeforeAfterPlayers`](src/components/BeforeAfterPlayers.tsx) | **Seek-linked** before/after players | ✅ one playhead drives both |
| [`VerificationPanel`](src/components/VerificationPanel.tsx) | Per-finding RESOLVED/PERSISTING/NEW + verdict | ✅ click a change to seek |
| [`LifecycleStrip`](src/components/LifecycleStrip.tsx) | The persisted state machine, live | ✅ backend-sent states |

**The after-player only shows a verified render.** It is gated on `hasVerifiedAfter`, not on a successful ffmpeg exit — a toggle must never manufacture an "after" result from a plan.

The deck degrades in a defined order: an injected report from the CLI (in `report.html`) outranks the API, which outranks the bundled fixture. Every call fails soft — a deck that renders the fixture is useful; one that renders a stack trace because the API is not running is not.

---

## 🚢 Deployment

Deployed to **Render** via [`render.yaml`](render.yaml) as two services.

```mermaid
flowchart LR
    U[Browser] -->|HTTPS| D["preflight-deck<br/>Static Site<br/>Vite build → /dist"]
    D -->|"VITE_PREFLIGHT_API"| A["preflight-api<br/>Docker Web Service<br/>Python 3.11-slim"]
    A --> FF["ffmpeg 7.1.5<br/>+ tesseract-ocr<br/>+ espeak-ng"]
    A --> W["faster-whisper base.en<br/>baked into the image"]
    A --> SQ[("SQLite<br/>lineage.db")]
    A --> FS[("Filesystem<br/>CAS + runs")]
    A -.->|"optional, on key"| NIM["NVIDIA NIM<br/>integrate.api.nvidia.com"]
```

| Layer | Actual technology |
|---|---|
| **Frontend** | Render Static Site — Vite build, `staticPublishPath: ./dist`, SPA rewrite |
| **Backend** | Render Docker Web Service — `python:3.11-slim`, health-checked at `/api/health` |
| **Database** | SQLite (`.preflight/lineage.db`) — no external DB |
| **Vector store** | numpy in-memory by default. Qdrant is supported (`vector.search`) but **optional and not provisioned** |
| **Object storage** | Local filesystem, content-addressed by BLAKE3. No S3/GCS |
| **LLM** | NVIDIA NIM (`integrate.api.nvidia.com`) — optional; the engine runs degraded-but-correct without a key |
| **ASR** | faster-whisper `base.en`, **baked into the image** — no download at runtime |
| **OCR** | tesseract, apt-installed in the image |
| **ffmpeg** | Hard runtime requirement — apt-installed (7.1.5 in the deployed image) |

### Models actually used

| Capability | Preference chain |
|---|---|
| `chat.reasoning` | `nvidia/llama-3.3-nemotron-super-49b-v1` → `meta/llama-3.3-70b-instruct` → `qwen/qwen2.5-72b-instruct` |
| `chat.extraction` | `meta/llama-3.3-70b-instruct` → `nvidia/llama-3.3-nemotron-super-49b-v1` |
| `vision.describe` | `meta/llama-3.2-11b-vision-instruct` → `nvidia/nemotron-nano-12b-v2-vl` → `meta/llama-3.2-90b-vision-instruct` |
| `embed.text` | `nvidia/nv-embedqa-e5-v5` |
| `rerank.text` | `nvidia/nv-rerankqa-mistral-4b-v3` |
| `asr.transcribe` | faster-whisper `base.en` (local, int8) |
| `ocr.image` | tesseract (local) |

Chains are ordered by **measured availability, not parameter count**. The 90B vision model was once the sole entry and stopped answering — every request hit the full read timeout while smaller models answered in under 15s. It stays in the chain (it is the better model when up), but it can no longer take the whole modality down with it.

### A real deployment bug, and its fix

The image cached faster-whisper's weights as `root` (`HOME=/root`) but served as the unprivileged `preflight` user (`HOME=/home/preflight`). `_hf_cached()` resolves through `Path.home()` **per process**, so at runtime it found nothing and reported `not cached` — no error, just a silent downgrade of ASR to the hosted tier. Fixed by copying the cache into the runtime user's own `HOME`, with a regression test pinning `Path.home()` behaviour.

### CI/CD

Two GitHub Actions workflows ([`.github/workflows/`](.github/workflows/)):

- **`ci.yml`** — UI job (typecheck, **schema-drift check**, vitest, build) + engine job (build corpus, pytest, **scoring-vector drift check**). Both drift checks `git diff --exit-code`, so a contract change that isn't committed fails the build.
- **`preflight.yml`** — PREFLIGHT running on itself.
- **[`action.yml`](action.yml)** — a reusable GitHub Action that emits SARIF into the Security tab.

---

## 🔑 Environment Variables

**Every value below is a placeholder. Never commit real keys.** See [`.env.example`](.env.example).

| Variable | Required | Purpose | Example |
|---|---|---|---|
| `NVIDIA_API_KEY` | No | Upgrades hosted capabilities (triad, vision, embed, rerank). **Without it the engine runs fully — degraded and reported.** | `nvapi-YOUR_KEY_HERE` |
| `NVIDIA_BASE_URL` | No | NIM endpoint | `https://integrate.api.nvidia.com/v1` |
| `ACOUSTID_API_KEY` | No | Acoustic fingerprint lookup | `YOUR_ACOUSTID_KEY` |
| `PREFLIGHT_RPM` | No | Token-bucket rate limit | `30` |
| `PREFLIGHT_OFFLINE` | No | `1` = never touch the network, even with a key | `0` |
| `PREFLIGHT_CACHE_DIR` | No | Content-addressed store location | `.preflight/cache` |
| `PREFLIGHT_POLICY_DIR` | No | Policy corpus location | `data/policy` |
| `PREFLIGHT_HTTP_TIMEOUT` | No | Provider HTTP timeout (seconds) | `300` |
| `PREFLIGHT_MODEL_*` | No | Override a specific model id | `PREFLIGHT_MODEL_ASR=base.en` |
| `PORT` | Deploy | Server bind port | `10000` |
| `VITE_PREFLIGHT_API` | Frontend | API base URL baked into the build | `https://preflight-api-vax3.onrender.com` |

In `render.yaml`, secrets are declared `sync: false` — the blueprint **cannot see or set their values**, by design.

---

## 🧑‍💻 Local Development

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11+ |
| Node | 20+ |
| ffmpeg + ffprobe | on `PATH` (hard requirement) |
| tesseract | optional — enables `ocr.image` |

### Quick start

```bash
git clone https://github.com/rakeshselvaraj0108/youtube.git
cd youtube

make setup     # venv, Python + Node dependencies, policy corpus
make demo      # generate a clip, then check → fix → check
```

### By hand

```bash
# 1. Engine
pip install -e '.[asr,ocr,dev]'

# 2. Policy corpus (33 clauses, hash-verified)
python scripts/build_corpus.py

# 3. Media assets + a narrated demo clip
python scripts/make_assets.py
python scripts/make_demo.py

# 4. Verify the environment
preflight doctor

# 5. Analyse
preflight check samples/demo.mp4 --html
```

> **No footage is committed.** Every media artifact is generated by a script, so a clean clone reproduces the demo without a large download.

### Running the full stack

```bash
# Terminal 1 — the engine
preflight serve --host 127.0.0.1 --port 8000

# Terminal 2 — the deck
npm install
npm run dev        # http://localhost:5173
```

The deck defaults to `http://127.0.0.1:8000`; override with `VITE_PREFLIGHT_API`.

### Optional extras

```bash
pip install -e '.[retrieval]'   # sentence-transformers, faiss-cpu, rank-bm25
pip install -e '.[audio]'       # librosa, soundfile
preflight models pull           # cache local model weights offline
```

---

## 🐳 Docker

```bash
make docker         # build the image with models baked in
make docker-demo    # run the full demo inside the container
docker compose up   # see docker-compose.yml
```

The [`Dockerfile`](Dockerfile) is built for **cold-clone reproducibility**:

- ASR weights are **baked into the image**, not downloaded on first run — a judge on a flaky connection still gets a working tool.
- ffmpeg, espeak-ng and tesseract-ocr are apt-installed.
- The policy corpus is generated from its authoring script during build, so the image always carries a corpus whose manifest hashes match its clause files.
- Runs as a **non-root user** (`preflight`, uid 1000). A container that writes root-owned files into a mounted working directory is a small cruelty to whoever runs it.
- `CMD` serves on `0.0.0.0:10000` by default; `docker run <image> check foo.mp4` still overrides only the CMD half, preserving CLI passthrough.

---

## 🧪 Testing

```bash
make test        # everything
make test-py     # Python
make test-ui     # TypeScript
make verify      # regenerate scoring vectors, prove both languages agree
```

**Verified counts** — `pytest` reports `1405 passed, 1 skipped in 121.09s`; `vitest run` reports `7 passed (7) · 109 passed (109)`:

| Suite | Collected | Result | Files |
|---|---|---|---|
| **Python** | **1,406** | 1,405 passed · 1 skipped | 43 |
| **TypeScript** | **109** | 109 passed | 7 |
| **Total** | **1,515** | 1,514 passed · 1 skipped | 50 |

The single skip is a deliberately-documented edge case in [`test_disclosure.py`](tests/test_disclosure.py) — an all-zero digit run technically satisfies Luhn, and length/context rules handle it instead. It is not a failing test.

<details>
<summary><b>Per-file Python breakdown (top 20)</b></summary>

| File | Tests | File | Tests |
|---|---|---|---|
| `test_audio_intel.py` | 53 | `test_lineage.py` | 32 |
| `test_disclosure.py` | 51 | `test_certificate.py` | 28 |
| `test_agents.py` | 48 | `test_coverage.py` | 27 |
| `test_ingest.py` | 26 | `test_lifecycle.py` | 27 |
| `test_incident_compare.py` | 25 | `test_cli_fix.py` | 24 |
| `test_audio.py` | 23 | `test_incidents.py` | 23 |
| `test_budget.py` | 20 | `test_drift.py` | 19 |
| `test_cas.py` | 16 | `test_evidence.py` | 14 |
| `test_decode_reuse.py` | 8 | `test_nim_transport.py` | 7 |
| `test_data_provenance.py` | 5 | `test_corpus_truth.py` | 4 |

</details>

### Regression tests worth reading

These exist because the bug happened, not because the category sounded good:

| Test | The bug it pins |
|---|---|
| [`test_decode_reuse.py`](tests/test_decode_reuse.py) | The pipeline decoded the same video twice — one pass, verified by counting ffmpeg invocations |
| [`test_cas.py`](tests/test_cas.py) | A run that crashed mid-extraction served a half-built frame set as complete. Fixed with a `.ready` marker |
| [`test_disclosure.py`](tests/test_disclosure.py) (51) | `\b` cannot match inside `DB_PASSWORD` — the detector missed every SCREAMING_SNAKE_CASE env var |
| [`test_coverage.py`](tests/test_coverage.py) (27) | Includes `TestNegativeEvidenceIsNotClean` — `UNEXAMINED` must never read as clean |
| [`test_sampling.py`](tests/test_sampling.py) | A flat frame budget starved long videos; `TestLongVideoDensity` pins duration-aware scaling |
| [`test_incidents.py`](tests/test_incidents.py) | File-scoped findings merging into one meaningless super-incident |
| [`test_incident_compare.py`](tests/test_incident_compare.py) (25) | Matching findings across runs when ids don't survive and timestamps move |
| [`test_verify.py`](tests/test_verify.py) | Absence in an unexamined region reported as `RESOLVED` |
| [`test_evidence.py`](tests/test_evidence.py) | A claim without a source must not be constructible |
| [`test_nim_transport.py`](tests/test_nim_transport.py) | A trickling HTTP response stalled a run for 25 minutes — `read()` blocks until EOF, so a deadline check never runs. Tests use **real sockets** |
| [`test_providers.py`](tests/test_providers.py) | The circuit breaker could not open until a call's own retries were already exhausted — it protected nothing |
| [`test_server.py`](tests/test_server.py) | `TestApplyFixExposesPlayableMedia` — the render succeeded but the browser was handed an unresolvable relative URL |
| [`test_report.py`](tests/test_report.py) | The emitted HTML must have **zero** external references — the offline-open guarantee |
| [`contract.test.ts`](src/lib/contract.test.ts) | Python and TypeScript scorers must agree to the decimal, including half-integer rounding |
| [`test_lifecycle.py`](tests/test_lifecycle.py) (27) | Illegal state transitions; a killed render must leave a truthful row |
| [`test_data_provenance.py`](tests/test_data_provenance.py) | Every clause file's SHA-256 must match the manifest |

CI additionally enforces two **drift checks** that fail the build on an uncommitted contract change: the JSON schema regenerated from `src/types/analysis.ts`, and the shared scoring vectors.

---

## 📈 Benchmarks

A benchmark harness and a labelled golden corpus **both exist and run**:

- **Corpus:** 31 generated clips ([`data/corpus/clips/`](data/corpus/)), 30 labelled examples in [`labels.jsonl`](data/corpus/labels.jsonl).
- **Design:** the corpus is built around **twin pairs** — two clips containing the *same word or scene*, one a violation and one exempt under a clause's own documented exemptions. Example:

```json
{"clip": "g001.mp4", "label": "VIOLATION", "clause": "AF-01",
 "note": "strong profanity as an intensifier, no exemption available"}
{"clip": "g002.mp4", "label": "CLEAN", "clause": null, "twin_of": "g001",
 "note": "SAME WORD. Educational framing plus attributed quotation, both
          documented exemptions in AF-01. Must be DISMISSED."}
```

Pair accuracy is the metric that matters: a system that fires on both twins scores high recall and is **useless**, because it cannot tell a violation from its exemption.

- **Metrics computed:** precision, recall, F1, pair accuracy, span accuracy, call count — per ablation layer.
- **Run it:** `preflight bench --ablation --out bench.json`

> **Full benchmark results are not published in this repository.** The harness and the corpus are committed and reproducible; the scored table is not, and inventing numbers would be worse than not having them. Reproduce with the command above.

---

## 🧯 Failure Modes

An honest account of what happens when things go wrong. This section exists because a compliance tool that fails silently is worse than no tool.

| Failure | Actual behaviour |
|---|---|
| **No API key** | The engine runs. Hosted capabilities report `SKIPPED` with a reason; the triad is skipped and deterministic agents still score. Coverage falls to match and is reported in the header, SARIF invocation and certificate. |
| **Model timeout** | Bounded by `REQUEST_DEADLINE_S = 90` using `read1()` in a deadline-checked loop (`read()` blocks until EOF, which is what caused a 25-minute stall). Transport failures get a **smaller** retry budget than HTTP 429/5xx — a vendor that answered "try again" deserves the full budget; one that never answered does not. |
| **Vendor down** | Circuit breaker (`FAILURE_THRESHOLD = 3`) opens **before** a single call's transport budget is exhausted, so it can actually cut a bad run short rather than only protecting calls after the first. |
| **Model chain exhausted** | `_resolve()` walks the preference chain in order. If all entries fail, the capability reports unavailable, not silently degraded. |
| **OCR unavailable** | `ocr.image` reports unavailable with the exact install command. 13% of the analysis surface reports as uncovered. |
| **Vision unavailable** | 22% of the surface uncovered. `DEMOTE` never fires because there are no vision-only findings. |
| **Insufficient coverage** | Absence claims resolve to `INSUFFICIENT_COVERAGE` or `NO_COVERAGE`, never `NEGATIVE_EVIDENCE`. The UI shows hatched bands. |
| **Corrupted / unsupported video** | `UnsupportedInput` is caught by name at the CLI boundary → exit code `2`, not a traceback. |
| **ffmpeg missing** | `preflight check` warns; `/api/analyze` returns `503`. |
| **ffmpeg render failure** | Staged temp file is unlinked, lifecycle row moves to `FAILED`, nothing is promoted. |
| **Render duration mismatch** | If output drifts >1500ms from the EDL prediction, the staged file is deleted and the remediation fails — a truncated render from a killed process still exits zero, so exit code is not trusted. |
| **Re-analysis fails** | Verdict is `INCONCLUSIVE` and the lifecycle records it. The file exists and is structurally sound; nothing is known about whether it is safe. That is precisely inconclusive. |
| **Process killed mid-render** | The lifecycle row says `RENDERING` — which is true. `GET /api/remediations` lists it as interrupted, and the engine can resume. |
| **Rate limit (429)** | Token bucket at 30 RPM plus exponential backoff with jitter. |
| **Budget exhausted** | The run **sheds work and reports what it gave up**; coverage falls to match what was actually examined. |
| **Malformed model JSON** | A JSON repair cascade handles prose preambles and trailing commas — models emit them regardless of instruction. |
| **SSE connection dropped** | The deck reconciles against `GET /api/jobs/{id}`, delivering by sequence number. Recovery, not re-execution. |
| **Cache entry half-written** | Only a committed `.ready` marker makes an entry a hit. |

---

## 🛡 Threat Model

| Threat | Mitigation actually implemented |
|---|---|
| **Secret leaking into artifacts** | Redaction at detection (`redact()` keeps 4 chars), a second sweep at provider boundaries, and `preflight check` **refuses to persist** a report containing the configured API key. Tested. |
| **Secret in logs** | `providers/secrets.py::redact` runs over provider error strings before they reach any log. |
| **Path traversal via run id** | Run ids regex-validated; media paths resolved from **lineage records**, never from URL components. Tested with encoded traversal. |
| **Stack trace disclosure** | `_fail` emits `{"error": "<ExceptionType>"}`. Tested that no body contains `Traceback` or internal module names. |
| **Malicious upload filename** | Storage ids are server-generated; the user filename is metadata only and never becomes a path. |
| **Model hallucination** | `DEMOTE` rule holds lone vision claims at advisory. `Claim` requires a `source`. The reasoning layer calls no model. |
| **OCR false positives** | Luhn validation for cards; a keyword-plus-separator requirement for credentials; a sparse floor for retrieval. |
| **Artifact tampering** | Every artifact is BLAKE3-hashed; reuse requires the bytes to still match the recorded digest. Certificates carry both artifact hashes and are integrity-checkable. |
| **Cache staleness** | Content-addressed — the key *is* the hash of the inputs. A changed input is a different key. |
| **Policy drift** | The Drift Watcher measures per-section semantic deltas between corpus snapshots and selectively re-lints only archived videos where the changed clause was already in contention. |
| **Remediation side effects** | New-risk detection: findings present after but not before are `NEW`, and any new incident forces `NEW_RISK_DETECTED`. |
| **Report claiming false safety** | Coverage gating at every absence claim; `MIN_COVERAGE_FOR_ABSENCE` in verification; `UNEXAMINED` never renders as clean. |

**Not implemented:** authentication, authorization, rate limiting per client, or multi-tenancy. The API is designed to be run locally or behind your own perimeter. CORS is open. Do not expose it to untrusted networks as-is.

---

## 💰 Cost & Efficiency

The community NIM tier sits around 40 requests/minute. The architecture assumes that.

| Mechanism | Effect |
|---|---|
| **Content-addressed cache** | Keyed by BLAKE3 of inputs. A second run over identical input costs **zero** API calls and produces an identical report. |
| **Read-through model cache** | Keyed on model + prompt; a re-run costs nothing. |
| **Cascade** | Only windows that produce a candidate reach ADVOCATE and ADJUDICATOR — three stages cost far less than 3× one. |
| **Batching** | Multiple windows per AUDITOR call, not one call per window. |
| **Token bucket at 30 RPM** | Headroom so a burst never trips a 429 mid-demo. |
| **Exponential backoff with jitter** | Recovers when one lands anyway. |
| **Bounded + adaptive sampling** | Frame budget scales with duration up to a hard ceiling (400), so a three-hour upload cannot fill the disk. |
| **Motion-aware vision budget** | Calls follow activity rather than the clock, capped at 48. |
| **Single decode pass** | Quality, motion and thumbnail candidates come from **one** decode. There is a test counting ffmpeg invocations. |
| **Parallel OCR** | Frames read concurrently via a thread pool. |
| **Stream copy when possible** | No video ops in the EDL → the video stream is never re-encoded. |
| **Artifact reuse** | A verified render that still hashes correctly is not re-rendered — but re-analysis is **never** skipped, because that is the part that proves anything. |
| **`--budget` ceiling** | Caps hosted calls; the run sheds work and reports what it gave up. |
| **`--offline`** | Zero network, by construction. |

Cost accounting is reported in `report.json` under `cost`: `estimatedCalls`, `actualCalls`, `ceiling`, `shed`.

> Specific speedup percentages are **not published** here — the mechanisms are real and inspectable, but a clean before/after measurement across hardware is not committed to this repository.

---

## 🎛 Technical Decisions

| Decision | Why | Alternative | Why not |
|---|---|---|---|
| **stdlib `http.server`** | The project ships numpy + typer and nothing heavy. A handful of JSON routes doesn't justify a framework | FastAPI + uvicorn | Would be the **largest dependency in the tree**, for a surface that fits in one file |
| **BM25 implemented in-repo** | ~30 lines, no dependency, fully inspectable | `rank-bm25` | An optional extra; the core must work with zero extras installed |
| **SQLite for lineage** | Zero-config, single file, ships everywhere | Postgres | A remediation history is not a multi-tenant workload; an external DB would break "runs on a stranger's machine" |
| **numpy vector search by default** | The corpus is ~120 chunks. Exact cosine over a 120×1024 matrix is one dot product — microseconds, no index, no recall failure mode | Mandatory Qdrant | A vector DB with nothing to do is a dependency with nothing to do. Qdrant **is** supported for the archive case |
| **SSE, not WebSocket** | One-directional progress. `EventSource` reconnects for free and replays by sequence number | WebSocket | Bidirectional machinery for a unidirectional problem |
| **BLAKE3 content addressing** | The cache key *is* the input hash — staleness is structurally impossible | mtime/path keys | Silently serve stale artifacts after an edit |
| **Deterministic core, models at the edges** | 8 of 12 agents are deterministic. Loudness, flash rate, Luhn, timestamps and scoring are computed | LLM-for-everything | Non-reproducible, expensive, and unauditable for things with exact answers |
| **Adversarial triad** | A single classification pass over-fires, and an over-firing linter is an uninstalled linter | Single classifier | No structured place for a clause's own exemptions to be argued |
| **Re-analyse the render** | The only evidence a fix worked | Trust ffmpeg's exit code | Proves a file was written, nothing more |
| **Zustand** | The deck is one workspace with shared playhead/selection state | Redux | Ceremony for a single-view app |
| **Typer + Rich** | Exit codes and readable output matter for CI | argparse | No exit-code discipline, worse help |
| **SARIF 2.1.0** | What CodeQL and Semgrep speak — findings render natively in GitHub's Security tab | Custom JSON only | A video file could not fail CI in a way GitHub understands |
| **Self-contained `report.html`** | Everything inlined — works offline, forever, emailable | Static-site bundle | Breaks the moment it leaves its directory |
| **Corpus split by rule ownership** | A citation is worthless if a reader cannot tell whose rule it is | One flat clause list | Attributes the tool's own thresholds to the platform |

---

## 🆚 What Makes PREFLIGHT Different

Compared against the general class of single-pass video moderation/detection tooling. **No named competitor is being characterised here.**

| Capability | Typical detector | PREFLIGHT |
|---|---|---|
| Multimodal analysis | Often 1–2 modalities | 6 observation modalities across 12 agents |
| Multi-agent investigation | ❌ single pass | ✅ 12-agent DAG, tiered, retried, roster-validated |
| Evidence provenance | ❌ score only | ✅ `Claim` **cannot exist** without a `Source` |
| Coverage awareness | ❌ absence = clean | ✅ 3 band states + 4 absence states, pessimistic across spans |
| Policy grounding | ❌ opaque label | ✅ 33 hash-verified clauses, cited verbatim, split by rule ownership |
| Adversarial adjudication | ❌ | ✅ prosecute → defend → rule, with written rationale |
| Cross-modal fusion | ❌ or naive max | ✅ coverage-scaled noisy-OR + 3 named, logged rules |
| Incident correlation | ❌ flat list | ✅ temporal + semantic + cross-agent, non-self-corroborating |
| Simulation | ❌ | ✅ evidence-level removal, scored by the **real** scorer |
| Remediation | ⚠️ suggestions | ✅ compiled to an executable ffmpeg program |
| Closed-loop re-analysis | ❌ | ✅ rendered artifact goes back through the same pipeline |
| Timestamp mapping across edits | ❌ | ✅ `TimeMap` through cuts; in-cut findings map to `None` |
| New-risk detection | ❌ | ✅ `NEW_RISK_DETECTED` verdict |
| Auditable verdict | ❌ | ✅ certificate with both hashes, weights, clamp rule, coverage |
| Reproducibility | ⚠️ | ✅ content-addressed; identical inputs → identical report, 0 calls |
| CI integration | ⚠️ | ✅ SARIF 2.1.0 + exit codes + GitHub Action |
| Runs with no API key | ⚠️ | ✅ fully — degraded and **reported** |

---

## 🎬 End-to-End Example

A real walkthrough, matching actual engine behaviour.

```bash
$ preflight check samples/synthetic.mp4 --offline
```

**1. Ingest** — ffprobe reads the streams; audio is extracted twice (analysis + ASR sample rates); scene-cut keyframes are extracted on a duration-aware budget. Everything is keyed by BLAKE3.

**2. Perception runs** — SPEECH transcribes with word timings. AUDIO measures EBU R128 loudness, clipping, dead air, and detects a sustained music bed with ducking. METADATA reads title/description/tags. ACCESSIBILITY checks flash rate and caption presence. VISION and OCR resolve their capabilities — offline here, so both report `SKIPPED` with a reason.

**3. Retrieval** — windows are built from the transcript, keyframes and OCR items; each is matched against the 33-clause corpus by BM25 ⊕ dense, fused by RRF.

**4. Triad** — offline, so skipped. Deterministic findings still stand.

**5. Fusion** — noisy-OR across modalities, scaled by each agent's real coverage.

**6. Correlation** — file-scoped findings (loudness, captions) are excluded from grouping; the rest are correlated by proximity + compatibility + agent independence.

**7. Scoring** — five dimensions, then the anti-masking clamp. Real output:

```
  ● MEDIUM   AUD-01   Integrated loudness 7.8 LU below target
      span        file-scoped
      clause      PREFLIGHT audio ruleset § 3.1
      adjudicator Measured value, not a classification. Correctable with a
                  loudnorm pass. conf 0.95

  ● MEDIUM   COPY-01  Sustained music bed, 12s
      span        file-scoped
      clause      Copyright policy — Content ID
      adjudicator Reported as MUSIC_BED_PRESENT, never as a match and never as
                  safe. Identifying the recording requires a fingerprint lookup.
                  conf 0.55
      fix         REPLACE_AUDIO

  ● LOW      META-02  Description is 34 characters
  ● LOW      META-05  20 tags
  ● LOW      META-04  Title is mostly uppercase

  RELEASE READINESS

    policy         ████████████████████ 100.0
    copyright      ██████████████████··  87.5
    metadata       ██████··············  29.5 ← weakest
    accessibility  ██████··············  30.2
    audio          ████████████████····  78.5

    45 / 100   DO NOT PUBLISH
    weighted mean 79.1, capped at weakest + 15 — one fatal flaw is never averaged away

  coverage 65%   LLM calls 0 (plan: at most 29)   elapsed 7056 ms
  PARTIAL ANALYSIS — 65% coverage
    impaired: Vision Agent, OCR Agent, Policy Agent
```

Note what the tool does **not** do: it does not claim the video is clean. Coverage is 65% and the three impaired agents are named.

**8. Simulate** — in the deck, `DecisionSimulator` shows what each candidate edit would score, computed by removing the *observations* the edit destroys and re-running the real scorer.

**9. Remediate**

```bash
$ preflight fix samples/synthetic.mp4 --apply
```

Findings lower to a typed EDL; optimiser passes run in fixed order; codegen emits ffmpeg; the render goes to a staged temp file, is verified on duration against the EDL, then atomically promoted to `synthetic.safe.mp4`.

**10. Verify** — the rendered file goes back through **the same pipeline**. Findings are matched across runs by clause + category + `TimeMap`-mapped span. Each becomes `RESOLVED`, `PERSISTING`, `CHANGED`, `NEW` or `INCONCLUSIVE`; the run gets a verdict and a certificate binding both artifact hashes.

---

## ⚠️ Current Limitations

Stated plainly. A tool that hides these is not trustworthy.

**Prediction, not adjudication**
- **This is not YouTube's classifier and cannot be.** The published guidelines are the specification; the classifier enforcing them is not public. A finding is a prediction with a citation, not a verdict.
- **Not affiliated with YouTube or Google.** See [COMPLIANCE.md](COMPLIANCE.md).
- 16 of 33 clauses are **PREFLIGHT's own house rules**, not platform policy. The report says which.

**Copyright**
- **A non-match does not prove safety.** Content ID's reference database is private and larger than any public fingerprint service. PREFLIGHT reports `CLAIM_LIKELY` or `MUSIC_BED_PRESENT` — **never** `SAFE`.

**Coverage and sampling**
- Vision and OCR are **frame-sampled**. Content shorter-lived than the sampling interval can be missed entirely; the coverage map shows where.
- The frame ceiling (400) means very long videos are sampled more sparsely per minute than short ones.

**Models**
- Three capabilities (`chat.*`, `vision.describe`, `rerank.text`) have **no honest local fallback** and report `SKIPPED` rather than degrading silently.
- Model-reported confidences are prompt-calibrated, **not** post-hoc calibrated against outcome data.
- `available()` checks key shape and package presence, not endpoint reachability. Chain ordering provides the resilience; `doctor` probes for real.

**Verification**
- Incomplete re-analysis yields `INCONCLUSIVE`, which is correct but not useful — a low-coverage second run cannot prove a fix worked.
- Prediction-vs-actual comparison of *resolved counts* only happens when the compiled operation set exactly matches a simulated scenario.

**Benchmarks**
- The harness and labelled corpus exist and run. **Scored results are not published in this repository.**

**Not implemented**
- No authentication, authorization, per-client rate limiting or multi-tenancy on the API.
- Qdrant is supported but **not provisioned** in the deployment; `vector.search` runs on the local numpy tier and reports as degraded.
- No object storage — artifacts are local filesystem only, which on Render's free tier means **they do not survive a restart**.
- No screenshots committed to the repository.

---

## 🗺 Roadmap

**✅ DONE**
- 12-agent DAG with roster validation, retries and per-stage coverage
- Coverage-aware reasoning: 3 band states, 4 absence states, UI coverage map
- Coverage-scaled noisy-OR fusion with 3 logged corroboration rules
- Hybrid BM25 ⊕ dense retrieval with weighted RRF and optional reranking
- Adversarial triad (prosecute → defend → rule) with written rationale
- Incident correlation: temporal + semantic + cross-agent
- Evidence-bound claims — `Claim` cannot exist without a `Source`
- Credential/PII detection with redaction and Luhn validation
- Decision simulation at the evidence level, scored by the real scorer
- Remediation compiler → EDL → ffmpeg, atomic promotion
- **Closed-loop verification** with `TimeMap`, 6 verdicts, new-risk detection
- Persisted lifecycle (19 states) with crash resume
- Certificates with integrity checking
- Content-addressed caching; SARIF 2.1.0; GitHub Action; Policy Drift Watcher
- Deployed to Render (API + deck); 1,515 tests; CI with contract-drift gates

**⏭ NEXT**
- Publish scored benchmark results from the existing harness and corpus
- Persistent object storage so artifacts survive a free-tier restart
- Endpoint reachability probing in `available()`, not just key shape
- Post-hoc confidence calibration against labelled outcomes

**🔮 FUTURE**
- API authentication and per-client rate limiting
- Provisioned Qdrant for the archive/back-catalogue case
- Expanded golden corpus with more twin pairs per clause
- Additional locales for ASR and on-screen text

---

## 🤝 Contributing

```bash
make setup && make test      # everything must pass before a PR
make lint                    # typecheck the UI
make verify                  # prove both scorers still agree
```

- Branch from `main`; name branches `fix/…`, `feat/…`, `docs/…`.
- **A regression test is expected with a bug fix.** Every test listed in [Testing](#-testing) exists because the bug happened.
- CI enforces two contract-drift gates (JSON schema, scoring vectors). Run `make verify` and commit the regenerated files.
- Match the surrounding comment density — this codebase explains *why*, not *what*.

---

## 📄 License

**MIT** — see [LICENSE](LICENSE). Copyright © 2026 Rakesh Selvaraj.

---

## 🙏 Acknowledgements

Built with [ffmpeg](https://ffmpeg.org/), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [Tesseract](https://github.com/tesseract-ocr/tesseract), [NumPy](https://numpy.org/), [Typer](https://typer.tiangolo.com/), [Rich](https://rich.readthedocs.io/), [Pydantic](https://docs.pydantic.dev/), [BLAKE3](https://github.com/BLAKE3-team/BLAKE3), [React](https://react.dev/), [Vite](https://vitejs.dev/), [Zustand](https://zustand-demo.pmnd.rs/), [Tailwind CSS](https://tailwindcss.com/), [react-three-fiber](https://docs.pmnd.rs/react-three-fiber), and [Three.js](https://threejs.org/). Hosted model capabilities are served by [NVIDIA NIM](https://build.nvidia.com/). Accessibility thresholds reference [WCAG 2.2](https://www.w3.org/WAI/WCAG22/Understanding/three-flashes-or-below-threshold.html).

No sponsorship or affiliation is claimed with any of the above, or with YouTube or Google.

---

<div align="center">

## Why PREFLIGHT?

**Traditional systems:**
`Detect → Report`

**PREFLIGHT:**
`Detect → Investigate → Prove → Simulate → Remediate → Re-analyze → Verify`

<br/>

*Before a video reaches millions of viewers, PREFLIGHT asks whether the artifact itself can withstand inspection.*

<br/>

### PREFLIGHT — Don't just publish. Prove it's ready.

<br/>

[**Live App**](https://preflight-deck.onrender.com) · [**API**](https://preflight-api-vax3.onrender.com/api/health) · [**Demo Video**](https://youtu.be/4MqqM_7RoZE) · [**Source**](https://github.com/rakeshselvaraj0108/youtube)

<sub>Built for the YouTube Automation Hackathon · August 2026 · by Rakesh Selvaraj</sub>

</div>
