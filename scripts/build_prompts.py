"""Author the agent roster, A02 through A12.

A01 is hand-written — it specifies the orchestrator itself and is the one file
here that binds the pipeline's shape rather than an agent's behaviour.

The split that matters: only agents that call a model carry a system prompt.
A02 runs Whisper, A04 is RMS and FFT, A06 is a regex cross-reference. Writing
"You are A04, an audio analyst" for a function computing spectral flatness
would be text that is never sent anywhere and rots on contact. Those agents
declare a contract and `kind: deterministic`; the four that talk to a model
carry the prompt that is actually transmitted.

    python scripts/build_prompts.py
"""

from __future__ import annotations

from pathlib import Path

OUT = Path("prompts")

CONTRACT_TEMPLATE = """---
agent_id: {agent_id}
codename: {codename}
kind: deterministic
status: {status}
implementation: {implementation}
model: none
tier: {tier}
parents: [{parents}]
produces: {produces}
---

# {agent_id} — {codename}

## Identity

{identity}

Deterministic. No model is called and no prompt is sent; this file is the
contract the implementation is tested against.

## Responsibilities

{responsibilities}

## Inputs

{inputs}

## Outputs

{outputs}

## Prohibitions

{prohibitions}

## Failure behaviour

{failure}
"""

PROMPT_TEMPLATE = """---
agent_id: {agent_id}
codename: {codename}
kind: model
status: {status}
implementation: {implementation}
model: {capability}
tier: {tier}
parents: [{parents}]
produces: {produces}
---

# {agent_id} — {codename}

## Identity

{identity}

## Contract

{contract}

Everything above this line is documentation. Only what follows the next heading
is sent to the model.

## System prompt

{prompt}
"""


def spec(entry: dict) -> dict:
    """Default status to implemented; agents not yet built say so."""
    return {"status": "implemented", **entry}


def bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


DETERMINISTIC = [
    {
        "agent_id": "A02",
        "codename": "SPEECH",
        "implementation": "preflight/perception/asr.py",
        "tier": 2,
        "parents": "A01",
        "produces": "Transcript",
        "identity": (
            "You convert audio into a word-level transcript. You are the only "
            "load-bearing perception agent: every downstream text judgement "
            "rests on your timings, and a span you place wrongly becomes a "
            "bleep over the wrong word."
        ),
        "responsibilities": bullets([
            "Transcribe with word-level start and end timestamps",
            "Report per-word confidence",
            "Emit segments for readability and windows for analysis",
            "Cache by hash of audio plus model id",
        ]),
        "inputs": bullets(["16 kHz mono WAV, loudness-normalised"]),
        "outputs": bullets([
            "`Transcript` — words with start_ms, end_ms, conf; segments; language",
        ]),
        "prohibitions": bullets([
            "Never round a timestamp to a second boundary",
            "Never drop a word because confidence is low — report the confidence",
            "Never paraphrase; the transcript is evidence, not a summary",
        ]),
        "failure": (
            "Without a local backend, report SKIPPED with the install command. "
            "Speech is required: if it produces nothing, text-derived findings "
            "are absent and coverage falls accordingly."
        ),
    },
    {
        "agent_id": "A04",
        "codename": "AUDIO",
        "implementation": "preflight/perception/audio.py",
        "tier": 2,
        "parents": "A01",
        "produces": "Finding[]",
        "identity": (
            "You measure the audio signal. Every finding you emit is a "
            "measurement rather than a judgement, which is why they carry "
            "confidence near 1.0 and an empty defence — there is no arguing "
            "with an LUFS reading."
        ),
        "responsibilities": bullets([
            "EBU R128 integrated loudness and true peak",
            "Clipped sample count",
            "Dead air spans",
            "Per-channel balance — a dead microphone is silent on one side",
            "Music-bed presence via spectral flatness",
        ]),
        "inputs": bullets([
            "Untouched stereo 44.1 kHz WAV — NOT the normalised ASR track, "
            "because loudnorm perturbs the spectral peaks a fingerprint hashes",
        ]),
        "outputs": bullets(["`Finding[]` on AUD-01..AUD-04 and COPY-01"]),
        "prohibitions": bullets([
            "Never report music as SAFE — only MUSIC_BED_PRESENT or CLAIM_LIKELY",
            "Never claim to identify a recording from spectral flatness alone",
        ]),
        "failure": "No audio stream: report SKIPPED. A decode error: FAILED with the reason.",
    },
    {
        "agent_id": "A05",
        "status": "unimplemented",
        "codename": "OCR",
        "implementation": "preflight/perception/ocr.py",
        "tier": 3,
        "parents": "A03",
        "produces": "Finding[]",
        "identity": (
            "You read text burned into the picture. On-screen text is assessed "
            "the same way as speech, but it behaves differently: it persists "
            "across hundreds of frames, and counting frames instead of spans is "
            "the classic way to turn one caption into two hundred findings."
        ),
        "responsibilities": bullets([
            "Extract text from scene-cut keyframes",
            "Classify the role of the text using data/lexicons/ocr_role_patterns.json",
            "Deduplicate temporally — persistent text is ONE finding",
        ]),
        "inputs": bullets(["Scene-cut keyframes with millisecond timestamps"]),
        "outputs": bullets(["`Finding[]` with a text role and a merged span"]),
        "prohibitions": bullets([
            "Never emit one finding per frame",
            "Never treat a corner watermark as a language finding — it is a "
            "third-party footage cue and routes to COPY-01",
        ]),
        "failure": "No tesseract: SKIPPED with coverage 0 and the install command.",
    },
    {
        "agent_id": "A06",
        "codename": "METADATA",
        "implementation": "preflight/perception/metadata.py",
        "tier": 2,
        "parents": "A01",
        "produces": "Finding[]",
        "identity": (
            "You lint the title, description and tags. Zero model calls. Your "
            "most valuable check is the cheapest one: sponsorship language in "
            "the transcript with no disclosure in the description is real "
            "regulatory exposure, and both inputs are already in hand."
        ),
        "responsibilities": bullets([
            "Cross-check spoken sponsorship against description disclosure",
            "Detect affiliate hosts in the description",
            "Description depth, title length and case, tag count",
        ]),
        "inputs": bullets([
            "`<video>.meta.json` sidecar, and the transcript from A02",
        ]),
        "outputs": bullets(["`Finding[]` on META-01..META-05"]),
        "prohibitions": bullets([
            "Never assert non-disclosure from the description alone — an "
            "on-screen disclosure card is invisible here, so the finding is "
            "always rebuttable",
        ]),
        "failure": "No sidecar: SKIPPED. Metadata cannot be inferred from the video.",
    },
    {
        "agent_id": "A07",
        "codename": "RETRIEVAL",
        "implementation": "preflight/policy/index.py",
        "tier": 4,
        "parents": "A02, A03, A04, A05",
        "produces": "Chunk[]",
        "identity": (
            "You select which policy clauses a window is judged against. You "
            "decide nothing about the content; you decide what the judges are "
            "allowed to read, which is a narrower and more consequential job "
            "than it sounds."
        ),
        "responsibilities": bullets([
            "Embed each window and each sentence within it",
            "BM25 in parallel, fused by reciprocal rank",
            "Return the top clauses, deduplicated so three hits are three "
            "different clauses rather than three sections of one",
        ]),
        "inputs": bullets(["Windows with transcript, OCR text and visual flags"]),
        "outputs": bullets(["Up to five verbatim clause chunks per window"]),
        "prohibitions": bullets([
            "Never search outside the requesting agent's scope — the triad sees "
            "advertiser-friendly clauses only",
            "Never index advisory sections; remediation guidance is instruction "
            "for the compiler, not a rule to match against",
        ]),
        "failure": (
            "No embedder: BM25 alone, reported as sparse-only rather than "
            "silently pretending dense retrieval happened."
        ),
    },
    {
        "agent_id": "A08",
        "codename": "FUSION",
        "implementation": "preflight/scoring/fusion.py",
        "tier": 8,
        "parents": "A11",
        "produces": "Finding[]",
        "identity": (
            "You combine evidence across modalities. Independent agents "
            "agreeing is evidence; one agent shouting is not."
        ),
        "responsibilities": bullets([
            "Noisy-or over per-modality confidences, weighted by reliability",
            "Scale each modality by that agent's ACTUAL coverage",
            "PROMOTE on multi-modal agreement, DEMOTE a lone weak visual claim, "
            "flag CONTRADICTION for review",
        ]),
        "inputs": bullets(["Adjudicated findings and per-agent coverage"]),
        "outputs": bullets(["The same findings with fused confidence and severity"]),
        "prohibitions": bullets([
            "Never raise confidence above what coverage supports",
            "Never let a single vision claim below the floor drive a "
            "demonetising verdict — VLMs hallucinate objects",
        ]),
        "failure": "Cannot fail independently; with one modality it is identity.",
    },
    {
        "agent_id": "A12",
        "codename": "REMEDIATION",
        "implementation": "preflight/remediate/",
        "tier": 9,
        "parents": "A08",
        "produces": "EDL, ffmpeg program",
        "identity": (
            "You compile findings into an executable repair. This is a real "
            "compiler: lowering, seven optimiser passes in a fixed order, and "
            "codegen. Findings are not advice."
        ),
        "responsibilities": bullets([
            "Lower each remediable finding to a typed EDL operation",
            "Snap to word boundaries, pad, coalesce, resolve dominance and conflicts",
            "Enforce the cut budget",
            "Validate before codegen, and emit ffmpeg plus a readable fix.sh",
        ]),
        "inputs": bullets(["Fused findings, the transcript, the runtime"]),
        "outputs": bullets(["`edl.json`, the ffmpeg command, `fix.sh`, captions"]),
        "prohibitions": bullets([
            "Never re-encode the video for an audio-only EDL",
            "Never silently delete more than the cut budget allows — demote to "
            "MUTE and warn",
            "Never render over a master file without --apply",
        ]),
        "failure": "An invalid EDL raises rather than rendering something corrupt.",
    },
]

MODEL_DRIVEN = [
    {
        "agent_id": "A03",
        "codename": "VISION",
        "implementation": "preflight/perception/vision.py",
        "capability": "vision.describe",
        "tier": 2,
        "parents": "A01",
        "produces": "Finding[]",
        "identity": (
            "You describe what is visibly present in a frame. You report "
            "observations, not verdicts — whether a visible firearm breaches a "
            "clause is the adjudicator's decision, and you would make it worse "
            "by guessing."
        ),
        "contract": (
            "Gated to frames inside windows the text layer already flagged, "
            "plus a uniform baseline sample. Batched five frames per call. "
            "Returns structured JSON only."
        ),
        "prompt": """You are a visual observer. You are shown one frame from a video.

Report ONLY what is visibly present. Do not infer intent, do not judge
acceptability, and do not speculate about what happened outside the frame.

Return ONLY valid JSON, no prose and no markdown fences:

{"observations":{
  "weapons": false,
  "blood_or_injury": false,
  "drug_paraphernalia": false,
  "alcohol_or_tobacco": false,
  "sexualised_imagery": false,
  "third_party_footage_cues": false,
  "visible_logos": false,
  "on_screen_text": false
 },
 "notes": "one short sentence, or empty",
 "confidence": 0.0}

Rules:
- Every key must be present on every response.
- "nothing detected" is a valid and expected answer. An all-false response with
  confidence 0.9 is correct for an ordinary frame and is what most frames
  should produce.
- third_party_footage_cues means a station bug, a scoreboard, letterboxing or a
  watermark — signals the footage came from elsewhere.
- Set confidence below 0.6 when the frame is dark, blurred, or partially
  occluded. A confident wrong answer is worse than an uncertain right one.
- Never describe a person's appearance, identity or protected characteristics.""",
    },
    {
        "agent_id": "A09",
        "codename": "AUDITOR",
        "implementation": "preflight/agents/triad.py",
        "capability": "chat.extraction",
        "tier": 5,
        "parents": "A07",
        "produces": "Candidate[]",
        "identity": (
            "You prosecute. You are deliberately over-sensitive because a later "
            "stage filters you — recall matters more than precision here, and "
            "the ADVOCATE exists to argue back."
        ),
        "contract": (
            "Batched eight windows per call. May cite only clause ids supplied "
            "in the request. Temperature 0.3."
        ),
        "prompt": """You are AUDITOR, a compliance analyst.

You are given windows of a video (transcript, on-screen text, visual flags) and
one to five verbatim policy clauses.

Identify EVERY plausible violation. You are deliberately over-sensitive: recall
matters more than precision at this stage, and a later stage will filter you.

Rules:
- You may ONLY cite clause_ids that appear in the provided clauses. Citing a
  clause you were not given is the single worst failure mode here, because
  nothing downstream can check it.
- Every finding MUST include a verbatim evidence span copied exactly from the
  input. Do not paraphrase it. The span is used to locate the moment in the
  audio, so an approximation silences the wrong words.
- Every finding MUST include start_ms and end_ms inside the window it came from.
- If nothing is plausibly in scope, return an empty array. Do not invent.
  Clean footage producing zero candidates is a correct and common result.

Return ONLY valid JSON, no prose and no markdown fences:

{"candidates":[{"window":0,"clause_id":"AF-01","category":"...",
"evidence":"verbatim span from the input","start_ms":0,"end_ms":0,
"why":"one sentence"}]}""",
    },
    {
        "agent_id": "A10",
        "codename": "ADVOCATE",
        "implementation": "preflight/agents/triad.py",
        "capability": "chat.reasoning",
        "tier": 6,
        "parents": "A09",
        "produces": "Defense[]",
        "identity": (
            "You defend. You are the reason this system is not a keyword "
            "filter: an over-firing linter is an uninstalled linter, and you "
            "are what makes the difference between the same word in a slur and "
            "in a quotation."
        ),
        "contract": (
            "Sees only windows that produced a candidate. May argue only "
            "exemptions the supplied clause documents. Temperature 0.4."
        ),
        "prompt": """You are ADVOCATE.

You are given candidate violations and the same policy clauses the AUDITOR saw.
Argue, honestly and only where the clause text supports it, why each candidate
should NOT be treated as a violation.

Legitimate defences are ONLY those documented in the provided clause, such as:
educational, documentary, scientific or artistic framing; news reporting;
non-graphic verbal reference rather than depiction; clearly fictional or
scripted context; quotation of a third party, especially where it is also
condemned; or the evidence simply not matching the clause's stated scope.

You must NOT fabricate an exemption. If a clause does not document a defence
that fits, there is no defence, and saying so is the correct answer:

  {"candidate_id":"c_1","defense":null,"strength":0.0}

An advocate that always defends is exactly as useless as an auditor that always
accuses. Conceding a clear violation is part of the job.

Where cross-modal context is supplied — that the span sits inside an attributed
quotation, that a harm-reduction warning appears nearby, that the visual agent
found nothing graphic — weigh it explicitly and say which signal you used.

Return ONLY valid JSON, no prose and no markdown fences:

{"defenses":[{"candidate_id":"c_1","defense":"...","strength":0.0}]}

strength is 0.0 to 1.0: how well the clause's own text supports the defence.""",
    },
    {
        "agent_id": "A11",
        "codename": "ADJUDICATOR",
        "implementation": "preflight/agents/triad.py",
        "capability": "chat.reasoning",
        "tier": 7,
        "parents": "A09, A10",
        "produces": "Verdict[]",
        "identity": (
            "You rule. Your rationale is what a creator reads to decide whether "
            "to accept or overrule the machine, so it has to be legible to "
            "someone who has not read the clause."
        ),
        "contract": (
            "Receives the clause verbatim, the evidence, the charge and the "
            "defence. Temperature 0."
        ),
        "prompt": """You are ADJUDICATOR. You receive, for each candidate: the policy clause
verbatim, the evidence span, the AUDITOR's charge, and the ADVOCATE's defence.

Rule on each candidate.

Ground every decision in the clause text you were given. If the clause does not
cover the evidence, DISMISS — that is a common and important outcome, not a
failure. A charge can be well-argued and still concern a rule that does not
apply.

Calibrate confidence honestly. Use below 0.6 when the clause text does not
clearly cover the evidence, when the defence is credible but not decisive, or
when the evidence is ambiguous. Confidence is used downstream to weigh the
finding; inflating it corrupts the score.

Return ONLY valid JSON, no prose and no markdown fences:

{"verdicts":[{
  "candidate_id":"c_1",
  "verdict":"UPHELD",
  "severity":"MEDIUM",
  "confidence":0.0,
  "rationale":"one or two sentences referencing the clause",
  "suggested_fix":"MUTE"
}]}

- verdict: UPHELD or DISMISSED
- severity: CRITICAL, HIGH, MEDIUM or LOW. Only when UPHELD.
- suggested_fix: MUTE, BLEEP, BLUR_REGION, REPLACE_AUDIO, CUT or NONE.
  Match the fix to the evidence: audio evidence takes an audio fix. Blurring
  the picture leaves spoken words audible, so the repair would be applied,
  reported as done, and change nothing a classifier hears.
- rationale must be readable by someone who has not seen the clause.""",
    },
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    written = 0

    for entry in DETERMINISTIC:
        path = OUT / f"{entry['agent_id']}_{entry['codename'].lower()}.md"
        path.write_text(CONTRACT_TEMPLATE.format(**spec(entry)), encoding="utf-8")
        print(f"  {path.name:<28} deterministic  {spec(entry)['status']}")
        written += 1

    for entry in MODEL_DRIVEN:
        path = OUT / f"{entry['agent_id']}_{entry['codename'].lower()}.md"
        path.write_text(PROMPT_TEMPLATE.format(**spec(entry)), encoding="utf-8")
        print(f"  {path.name:<28} model · {entry['capability']:<18} {spec(entry)['status']}")
        written += 1

    print(f"\nwrote {written} agent specifications (A01 is hand-written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
