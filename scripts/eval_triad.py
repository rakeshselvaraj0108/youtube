"""Exercise the adversarial triad against the live API.

A test harness, not product data. The synthetic sample clip contains no speech,
so this feeds the triad a hand-written transcript with known content — some of
it in scope, some of it deliberately clean — and prints the full three-agent
record for each verdict.

What it demonstrates:
  * findings cite a clause that was actually retrieved
  * evidence spans are copied verbatim from the input
  * spans are recovered from word-level timings, not trusted from the model
  * ADVOCATE returns null when the clause offers no exemption
  * clean windows come back clean

    python scripts/eval_triad.py
"""

from __future__ import annotations

import textwrap

from preflight import cas
from preflight.agents.nim import NimClient
from preflight.agents.triad import run_triad
from preflight.chunking import build_windows
from preflight.config import Settings
from preflight.perception.asr import Segment, Transcript, Word
from preflight.policy.corpus import load_corpus
from preflight.policy.index import build_index

# (start_ms, text). Windows are 30s, so these land in distinct chunks.
SCRIPT: list[tuple[int, str]] = [
    (2_000, "Morning of day four and the weather window finally opened up on the col."),
    (
        32_000,
        "The anchor pulled clean out of the ice and the whole shelf just went. "
        "This is fucked, we need to be off this face in ten minutes.",
    ),
    (
        62_000,
        "He slipped and went straight down onto the rocks. Look at the blood, "
        "that's a bad one. Get the kit open, get pressure on it now.",
    ),
    (
        92_000,
        "Everyone here still talks about the 2019 slide on the north col. Eleven "
        "people went out that morning and four of them didn't come back down.",
    ),
    (
        122_000,
        "We skipped the fixed line entirely here. No rope, no anchor, just "
        "downclimbing the gully on wet rock because we were losing light.",
    ),
    # Deliberately clean: nothing here should survive adjudication.
    (
        152_000,
        "Back at base camp we sorted the gear, dried the ropes out, and I spent "
        "the evening colour grading the footage from the ridge.",
    ),
]

DURATION_MS = 182_000


def span_contains_evidence(transcript: Transcript, finding) -> bool:
    """Does the recovered span actually cover the quoted words?

    The real check on span recovery. Comparing against a segment's start time
    would be wrong — a phrase sits partway into its segment, so the correct
    span is not the segment boundary.
    """
    spoken = transcript.text_between(finding.startMs, finding.endMs).lower()
    quoted = [
        t.strip(".,!?;:\"'").lower()
        for t in finding.evidence.transcript.split()
        if t.strip(".,!?;:\"'")
    ]
    if not quoted:
        return False
    hits = sum(1 for token in quoted if token in spoken)
    return hits / len(quoted) >= 0.8


def build_transcript() -> Transcript:
    words: list[Word] = []
    segments: list[Segment] = []
    for start, text in SCRIPT:
        tokens = text.split()
        step = max(1, 25_000 // max(len(tokens), 1))
        for i, token in enumerate(tokens):
            words.append(
                Word(
                    w=token,
                    start_ms=start + i * step,
                    end_ms=start + i * step + step,
                    conf=0.95,
                )
            )
        segments.append(
            Segment(start_ms=start, end_ms=start + len(tokens) * step, text=text)
        )
    return Transcript(
        language="en", duration_ms=DURATION_MS, words=words, segments=segments
    )


def main() -> int:
    settings = Settings.load()
    store = cas.Store(settings.cache_dir)
    client = NimClient(settings, store)

    print(f"mode         {settings.describe_mode()}")
    if not settings.online:
        print()
        print("No API key - the triad reports SKIPPED and the report still ships")
        print("with deterministic findings only. Set NVIDIA_API_KEY to run it.")
        return 0

    corpus = load_corpus(settings.policy_dir)
    index = build_index(corpus, settings, store, client)
    transcript = build_transcript()
    windows = build_windows(
        transcript, DURATION_MS, [], chunk_ms=30_000, overlap_ms=5_000
    )

    print(f"retrieval    {index.backend}")
    print(
        f"windows      {len(windows)} "
        f"({sum(1 for w in windows if w.has_content)} with content)"
    )
    print()

    result = run_triad(windows, corpus, index, client, settings, transcript)

    for line in result.log:
        print(f"  - {line}")
    print()

    rule = "-" * 78
    if not result.findings:
        print("  no findings upheld")

    aligned = 0
    for finding in result.findings:
        adv = finding.adversarial
        ok = span_contains_evidence(transcript, finding)
        aligned += ok

        print(rule)
        print(
            f"  {finding.severity:<8} {finding.clauseId:<7} {finding.title}"
            f"   conf {finding.confidence:.2f}"
        )
        print(
            f"  span        {finding.startMs}ms -> {finding.endMs}ms   "
            f"[{'evidence inside span' if ok else 'SPAN MISMATCH'}]"
        )
        print(f"  clause      {finding.policy.section}")
        print(f'  evidence    "{finding.evidence.transcript[:92]}"')

        defense = (
            adv.defense
            if adv.defense
            else f"none available in the clause text (strength {adv.defense_strength:.2f})"
        )
        for label, body in (
            ("AUDITOR", adv.charge),
            ("ADVOCATE", defense),
            ("ADJUDICATOR", adv.rationale),
        ):
            wrapped = textwrap.fill(body, 62, subsequent_indent=" " * 14)
            print(f"  {label:<11} {wrapped}")
        if finding.suggestedFix != "NONE":
            print(f"  fix         {finding.suggestedFix}")

    print(rule)
    print()
    if result.findings:
        print(f"  spans       {aligned}/{len(result.findings)} contain their evidence")
    print(f"  LLM calls   {client.usage.calls} live, {client.usage.cached} cached")
    print(f"  retries     {client.usage.retries}")
    print(
        f"  tokens      {client.usage.prompt_tokens} in / "
        f"{client.usage.completion_tokens} out"
    )
    for line in client.usage.log:
        print(f"  ! {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
