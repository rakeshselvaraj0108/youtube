"""The Adversarial Adjudication Triad.

A single classification pass over-fires, and an over-firing linter is an
uninstalled linter. Every candidate is therefore contested:

    AUDITOR    temp 0.3   prosecutes — deliberately over-sensitive
    ADVOCATE   temp 0.4   defends — only with exemptions the clause documents
    ADJUDICATOR temp 0.0  rules — verdict, severity, calibrated confidence

The cascade is also the quota strategy. AUDITOR sees every window with content,
batched eight at a time. ADVOCATE and ADJUDICATOR only ever see windows that
produced a candidate — typically a small minority — so three stages cost far
less than three times one stage.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from preflight.agents.nim import NimClient, NimUnavailable
from preflight.agents import prompts
from preflight.chunking import Window, iou
from preflight.config import Settings
from preflight.models import Adversarial, AgentResult, Evidence, Finding, PolicyRef
from preflight.policy.corpus import Chunk, Corpus
from preflight.policy.index import IndexBuild

AUDITOR_BATCH = 8
ADVOCATE_BATCH = 6
ADJUDICATOR_BATCH = 6

VALID_SEVERITY = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
VALID_FIX = {"MUTE", "BLEEP", "BLUR_REGION", "REPLACE_AUDIO", "CUT", "NONE"}

# Two findings on the same clause overlapping by more than this are the same
# violation seen through two overlapping windows.
DEDUPE_IOU = 0.5


@dataclass
class Candidate:
    id: str
    window: int
    clause_id: str
    category: str
    evidence: str
    start_ms: int
    end_ms: int
    why: str
    chunk: Chunk

    defense: str | None = None
    defense_strength: float = 0.0

    # Cross-modal context, attached before the ADVOCATE runs — one line per
    # signal a different agent's independent read of the same moment
    # produced, none of which the AUDITOR was shown. A charge can come from
    # clause text alone with no lexicon hit of its own anywhere near it, so
    # every one of these is computed from the candidate's span directly
    # rather than depending on A02's own findings having fired.
    quotation_context: str | None = None
    edsa_context: str | None = None
    harm_reduction_context: str | None = None
    vision_context: str | None = None
    video_context: str | None = None

    @property
    def cross_modal_lines(self) -> list[str]:
        return [
            line
            for line in (
                self.quotation_context,
                self.edsa_context,
                self.harm_reduction_context,
                self.vision_context,
                self.video_context,
            )
            if line
        ]

    verdict: str = "UPHELD"
    severity: str = "MEDIUM"
    confidence: float = 0.5
    rationale: str = ""
    suggested_fix: str = "NONE"


@dataclass(frozen=True)
class CrossModalContext:
    """Everything the ADVOCATE gets that the AUDITOR did not — bundled so
    `run_triad` takes one parameter instead of five that grow independently.

    Every field is optional and defaults to empty/absent: a run with no
    vision agent, no metadata sidecar, or an ASR-only transcript still tags
    whatever it has. The brief degrades gracefully field by field rather than
    all at once.
    """

    quotation_spans: list = field(default_factory=list)
    framing_cues: list = field(default_factory=list)
    visual_tracks: list = field(default_factory=list)
    vision_coverage: float = 0.0
    declared_category: str = ""
    declared_audience: str = ""
    music_deliberate_bed: bool | None = None


@dataclass
class TriadResult:
    findings: list[Finding] = field(default_factory=list)
    calls: int = 0
    log: list[str] = field(default_factory=list)
    status: str = "OK"
    coverage: float = 1.0
    error: str | None = None
    windows_seen: int = 0
    windows_with_candidates: int = 0

    # Every candidate the AUDITOR raised, carrying whatever the ADVOCATE and
    # ADJUDICATOR did to it. Retained rather than discarded for two reasons.
    #
    # A dismissal is evidence the triad works. "AF-01 charged at 02:14,
    # dismissed — attributed quotation, exemption applied" is the single most
    # convincing line this system produces, and it only exists if the losing
    # candidates survive the function that ruled on them.
    #
    # And it makes the ablation a real ablation. The charge set, the defended
    # set and the upheld set are three depths of the SAME run on the SAME
    # inputs, so the difference between them is attributable to the stage
    # rather than to two runs disagreeing.
    candidates: list[Candidate] = field(default_factory=list)


def _clauses_block(chunks: list[Chunk]) -> str:
    return "\n\n".join(chunk.for_prompt() for chunk in chunks)


def _batched(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _clamp(value, low: float, high: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _items(payload, key: str) -> list[dict]:
    """Pull a list of records out of a model response.

    The prompt asks for `{"candidates": [...]}`. Models routinely return the
    bare array instead, or wrap it under a differently-named key. All three are
    the same answer, and refusing two of them loses real findings for no reason.
    """
    if payload is None:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        # Sole list-valued key: the model renamed the wrapper.
        lists = [v for v in payload.values() if isinstance(v, list)]
        if len(lists) == 1:
            return [item for item in lists[0] if isinstance(item, dict)]
        # A single record returned unwrapped.
        if key.rstrip("s") in {"candidate", "defense", "verdict"} and "clause_id" in payload:
            return [payload]
    return []


MAX_CLAUSES_PER_WINDOW = 5
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
MIN_SENTENCE_CHARS = 12


def _retrieve_for_window(window: Window, index: IndexBuild) -> list[Chunk]:
    """Retrieve clauses for a window, per sentence as well as per window.

    A 30-second window holds one profanity and twenty-five seconds of neutral
    narration. Embedded whole, it averages toward neutral and the language
    clause never reaches the AUDITOR — measured on the demo clip, the profanity
    was transcribed correctly and still went unflagged for exactly this reason.

    Retrieving per sentence and unioning fixes it. Embeddings are batched and
    cached, so the extra recall costs effectively nothing after the first run.
    """
    query = window.query()
    if not query.strip():
        return []

    ordered: list[Chunk] = []
    seen: set[str] = set()

    def take(hits, limit: int) -> None:
        for hit in hits[:limit]:
            if hit.chunk.clause_id not in seen:
                seen.add(hit.chunk.clause_id)
                ordered.append(hit.chunk)

    # Window-level first: it carries the overall topic.
    take(index.retriever.clauses_for(query, top_k=3, query_vector=index.embed_query(query)), 3)

    # Then each sentence, which is where a single sharp phrase lives.
    for sentence in SENTENCE_SPLIT.split(query):
        sentence = sentence.strip()
        if len(sentence) < MIN_SENTENCE_CHARS:
            continue
        if len(ordered) >= MAX_CLAUSES_PER_WINDOW:
            break
        take(
            index.retriever.clauses_for(
                sentence, top_k=2, query_vector=index.embed_query(sentence)
            ),
            2,
        )

    return ordered[:MAX_CLAUSES_PER_WINDOW]


def run_triad(
    windows: list[Window],
    corpus: Corpus,
    index: IndexBuild,
    client: NimClient,
    settings: Settings,
    transcript: "Transcript | None" = None,
    *,
    cross_modal: CrossModalContext | None = None,
) -> TriadResult:
    started = time.perf_counter()
    result = TriadResult()

    active = [w for w in windows if w.has_content]
    result.windows_seen = len(active)

    if not active:
        result.status = "SKIPPED"
        result.error = "no windows contained speech or on-screen text"
        result.log.append(result.error)
        return result

    if not client.online:
        result.status = "SKIPPED"
        result.coverage = 0.0
        result.error = (
            "no API key — policy adjudication skipped, deterministic agents only"
        )
        result.log.append(result.error)
        return result

    retrieved: dict[int, list[Chunk]] = {
        window.index: _retrieve_for_window(window, index) for window in active
    }

    calls_before = client.usage.calls

    try:
        candidates = _audit(active, retrieved, client, settings, transcript)
        result.candidates = candidates
        result.windows_with_candidates = len({c.window for c in candidates})
        result.log.append(
            f"AUDITOR: {len(candidates)} candidate(s) across "
            f"{result.windows_with_candidates}/{len(active)} windows"
        )

        if candidates:
            tagged = _tag_cross_modal_context(candidates, cross_modal or CrossModalContext())
            _defend(candidates, client, settings)
            defended = sum(1 for c in candidates if c.defense)
            signal_summary = ", ".join(
                f"{name} {count}" for name, count in tagged.items() if count
            )
            result.log.append(
                f"ADVOCATE: {defended}/{len(candidates)} candidate(s) defended"
                + (f" · cross-modal: {signal_summary}" if signal_summary else "")
            )

            _adjudicate(candidates, client, settings)
            upheld = [c for c in candidates if c.verdict == "UPHELD"]
            result.log.append(
                f"ADJUDICATOR: {len(upheld)} upheld, "
                f"{len(candidates) - len(upheld)} dismissed"
            )
            result.findings = _to_findings(_dedupe(upheld))

    except NimUnavailable as exc:
        result.status = "DEGRADED"
        result.coverage = 0.4
        result.error = str(exc)
        result.log.append(f"policy adjudication degraded: {exc}")

    result.calls = client.usage.calls - calls_before
    elapsed = int((time.perf_counter() - started) * 1000)
    result.log.append(
        f"{result.calls} LLM call(s) for {len(active)} window(s) in {elapsed}ms"
    )
    return result


def _audit(
    windows: list[Window],
    retrieved: dict[int, list[Chunk]],
    client: NimClient,
    settings: Settings,
    transcript: "Transcript | None" = None,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    counter = 0

    for batch in _batched(windows, AUDITOR_BATCH):
        chunks: dict[str, Chunk] = {}
        for window in batch:
            for chunk in retrieved.get(window.index, []):
                chunks[chunk.id] = chunk
        if not chunks:
            continue

        payload = client.chat_json(
            model=settings.models.auditor,
            system=prompts.AUDITOR_SYSTEM,
            user=prompts.auditor_user(
                "\n\n".join(w.for_prompt() for w in batch),
                _clauses_block(list(chunks.values())),
            ),
            temperature=0.3,
            max_tokens=2048,
        )

        allowed = {chunk.clause_id: chunk for chunk in chunks.values()}
        by_index = {w.index: w for w in batch}

        for raw in _items(payload, "candidates"):
            clause_id = str(raw.get("clause_id", "")).strip()
            chunk = allowed.get(clause_id)
            # Silently drop hallucinated clause ids. A finding citing a rule
            # that was never provided is worse than no finding at all.
            if chunk is None:
                continue

            window = by_index.get(int(raw.get("window", -1)))
            if window is None:
                window = batch[0]

            evidence = str(raw.get("evidence", "")).strip()
            if not evidence:
                continue

            # Prefer the span recovered from word-level ASR timings by locating
            # the quoted evidence. Model-reported timestamps are routinely
            # seconds out, and this span becomes the remediation op's span — a
            # bleep placed early silences the wrong word.
            located = (
                transcript.locate(evidence, near_ms=window.start_ms)
                if transcript is not None
                else None
            )
            if located is not None:
                start, end = located
            else:
                start = int(
                    _clamp(raw.get("start_ms", window.start_ms), 0, 1e12, window.start_ms)
                )
                end = int(_clamp(raw.get("end_ms", window.end_ms), 0, 1e12, window.end_ms))
                start = max(window.start_ms, min(start, window.end_ms))
                end = max(start + 200, min(end, window.end_ms))

            counter += 1
            candidates.append(
                Candidate(
                    id=f"c{counter}",
                    window=window.index,
                    clause_id=clause_id,
                    category=str(raw.get("category") or chunk.clause_title),
                    evidence=evidence,
                    start_ms=start,
                    end_ms=end,
                    why=str(raw.get("why", "")).strip(),
                    chunk=chunk,
                )
            )
    return candidates


def _tag_quotation_context(candidates: list[Candidate], spans: list) -> int:
    """Attach quotation cross-modal context to every candidate it overlaps.

    This is what makes the ADVOCATE more than a second opinion running the
    same charge back through a different prompt: it gets to see something the
    AUDITOR was deliberately not shown, drawn from a different agent's read of
    the same moment. A candidate charged from clause text alone — no lexicon
    hit of A02's own — still gets tagged if its evidence sits inside a span
    A02 found by a different route.
    """
    tagged = 0
    for candidate in candidates:
        hit = next(
            (s for s in spans if s.overlaps(candidate.start_ms, candidate.end_ms)),
            None,
        )
        if hit is None:
            continue
        candidate.quotation_context = (
            f'inside a quotation attributed by the cue "{hit.cue}"'
            + (
                ", which the speaker then condemned"
                if hit.kind == "attributed_and_condemned"
                else ""
            )
        )
        tagged += 1
    return tagged


# Vision categories that stand for depicted, not merely referenced, harm.
# "weapon" and "injury" are the closest this vocabulary comes to "graphic" —
# there is no separate blood/sexual category, and inventing one here rather
# than in the vocabulary itself would create a second, unreviewed taxonomy.
GRAPHIC_VISION_CATEGORIES = {"weapon", "injury"}

# Framing-cue window and default harm-reduction cap, mirrored from
# speech_intel so triad.py does not need to import the module just to read
# two numbers — the values are equally defensible constants either place.
EDSA_WINDOW_MS = 8_000


def _tag_edsa_and_harm_reduction(candidates: list[Candidate], cues: list) -> int:
    """EDSA framing nearby, and distance to the nearest harm-reduction phrase.

    Both use the SAME framing-cue occurrences speech_intel already computed;
    this is the routing step, not a second detector. A candidate near "in
    this lesson" reads differently from the identical evidence with nothing
    around it, and "never do this — you could get seriously hurt" four
    seconds before a dangerous act is a materially different video from the
    same act with no warning anywhere.
    """
    from preflight.perception.speech_intel import (
        edsa_categories_near,
        harm_reduction_distance_ms,
    )

    tagged = 0
    for candidate in candidates:
        categories = edsa_categories_near(
            cues, candidate.start_ms, candidate.end_ms, EDSA_WINDOW_MS
        )
        if categories:
            candidate.edsa_context = (
                f"within {EDSA_WINDOW_MS // 1000}s of {'/'.join(categories)} framing language"
            )
            tagged += 1

        distance = harm_reduction_distance_ms(cues, candidate.start_ms, candidate.end_ms)
        if distance is not None:
            candidate.harm_reduction_context = (
                f"a harm-reduction warning appears {distance / 1000:.1f}s away"
            )
            tagged += 1
    return tagged


def _tag_vision_context(
    candidates: list[Candidate], tracks: list, coverage: float
) -> int:
    """Whether A03 saw graphic imagery in the same window, and how much of
    the video A03 actually looked at.

    Coverage travels WITH the finding rather than sitting in a separate
    field, because "vision found nothing graphic" is a materially different
    claim at coverage 1.0 than at 0.42 — the second is silence from an agent
    that only saw four windows in ten, and the ADVOCATE should not lean on a
    negative that thin.
    """
    if not tracks:
        return 0
    tagged = 0
    for candidate in candidates:
        overlapping = [
            t for t in tracks if t.start_ms < candidate.end_ms and t.end_ms > candidate.start_ms
        ]
        graphic = any(t.category in GRAPHIC_VISION_CATEGORIES for t in overlapping)
        candidate.vision_context = (
            f"vision {'found' if graphic else 'found no'} graphic imagery in this window "
            f"(coverage {coverage:.0%})"
        )
        tagged += 1
    return tagged


def _tag_video_context(
    candidates: list[Candidate], category: str, audience: str
) -> int:
    """The creator's own declared category and audience, unconditionally —
    this is metadata about the whole video, not a per-window signal, so every
    candidate gets it if it exists at all."""
    if not category and not audience:
        return 0
    parts = []
    if category:
        parts.append(f'declared category "{category}"')
    if audience:
        parts.append(f'declared audience "{audience}"')
    line = "video metadata: " + ", ".join(parts)
    for candidate in candidates:
        candidate.video_context = line
    return len(candidates)


def _tag_cross_modal_context(candidates: list[Candidate], context: CrossModalContext) -> dict[str, int]:
    """Every cross-modal signal, tagged in one pass before the ADVOCATE runs."""
    return {
        "quotation": _tag_quotation_context(candidates, context.quotation_spans),
        "framing": _tag_edsa_and_harm_reduction(candidates, context.framing_cues),
        "vision": _tag_vision_context(candidates, context.visual_tracks, context.vision_coverage),
        "video": _tag_video_context(candidates, context.declared_category, context.declared_audience),
    }


def _defend(candidates: list[Candidate], client: NimClient, settings: Settings) -> None:
    for batch in _batched(candidates, ADVOCATE_BATCH):
        chunks = {c.chunk.id: c.chunk for c in batch}
        block = "\n\n".join(
            f'candidate_id: {c.id}\nclause_id: {c.clause_id}\n'
            f'evidence: "{c.evidence}"\ncharge: {c.why}'
            + (
                "\ncross_modal_context: " + " · ".join(c.cross_modal_lines)
                if c.cross_modal_lines
                else ""
            )
            for c in batch
        )
        payload = client.chat_json(
            model=settings.models.advocate,
            system=prompts.ADVOCATE_SYSTEM,
            user=prompts.advocate_user(block, _clauses_block(list(chunks.values()))),
            temperature=0.4,
            max_tokens=1536,
        )

        by_id = {c.id: c for c in batch}
        for raw in _items(payload, "defenses"):
            candidate = by_id.get(str(raw.get("candidate_id", "")))
            if candidate is None:
                continue
            # Models emit the *string* "null" as often as a JSON null. Treating
            # that as a defence puts the word "null" in front of a creator as
            # the reason their video was flagged.
            defense = raw.get("defense")
            text = str(defense).strip() if defense is not None else ""
            candidate.defense = (
                None if text.lower() in {"", "null", "none", "n/a", "no defense", "no defence"}
                else text
            )
            candidate.defense_strength = (
                _clamp(raw.get("strength", 0.0), 0.0, 1.0, 0.0)
                if candidate.defense
                else 0.0
            )


def _adjudicate(candidates: list[Candidate], client: NimClient, settings: Settings) -> None:
    for batch in _batched(candidates, ADJUDICATOR_BATCH):
        block = "\n\n".join(
            f"candidate_id: {c.id}\n"
            f"CLAUSE [{c.clause_id}] {c.chunk.clause_title} — {c.chunk.section}\n"
            f"{c.chunk.text}\n"
            f'EVIDENCE: "{c.evidence}"\n'
            f"AUDITOR: {c.why}\n"
            f"ADVOCATE: {c.defense or 'No defence available.'}"
            for c in batch
        )
        payload = client.chat_json(
            model=settings.models.adjudicator,
            system=prompts.ADJUDICATOR_SYSTEM,
            user=prompts.adjudicator_user(block),
            temperature=0.0,
            max_tokens=1536,
        )

        by_id = {c.id: c for c in batch}
        for raw in _items(payload, "verdicts"):
            candidate = by_id.get(str(raw.get("candidate_id", "")))
            if candidate is None:
                continue
            verdict = str(raw.get("verdict", "UPHELD")).upper()
            candidate.verdict = "DISMISSED" if verdict.startswith("DIS") else "UPHELD"

            severity = str(raw.get("severity", "")).upper()
            candidate.severity = (
                severity if severity in VALID_SEVERITY else _default_severity(candidate)
            )
            candidate.confidence = _clamp(raw.get("confidence", 0.5), 0.0, 1.0, 0.5)
            candidate.rationale = str(raw.get("rationale", "")).strip() or (
                "Ruled against the cited clause."
            )
            fix = str(raw.get("suggested_fix", "NONE")).upper()
            candidate.suggested_fix = _sane_fix(
                fix if fix in VALID_FIX else "NONE", candidate
            )


def _sane_fix(fix: str, candidate: Candidate) -> str:
    """Reconcile the suggested fix with what the evidence actually is.

    Adjudicators pick video fixes for spoken evidence — the demo run suggested
    BLUR_REGION for a casualty figure that exists only in the audio. Blurring
    the picture leaves the words audible, so the fix would be applied, reported
    as done, and change nothing a classifier hears.

    Evidence quoted from the transcript is audio evidence, and audio evidence
    gets an audio fix.
    """
    spoken = bool(candidate.evidence.strip()) and not candidate.evidence.startswith("[")
    if not spoken:
        return fix

    if fix == "BLUR_REGION":
        # A short quote is one phrase to bleep; a long one is a passage to mute.
        return "BLEEP" if len(candidate.evidence.split()) <= 3 else "MUTE"
    return fix


def _default_severity(candidate: Candidate) -> str:
    return "HIGH" if candidate.chunk.severity_default == "DEMONETIZING" else "MEDIUM"


def _dedupe(candidates: list[Candidate]) -> list[Candidate]:
    """Collapse the same violation seen through two overlapping windows."""
    kept: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda c: (-c.confidence, c.start_ms)):
        span = (candidate.start_ms, candidate.end_ms)
        duplicate = next(
            (
                k
                for k in kept
                if k.clause_id == candidate.clause_id
                and iou(span, (k.start_ms, k.end_ms)) > DEDUPE_IOU
            ),
            None,
        )
        if duplicate is None:
            kept.append(candidate)
        else:
            # Keep the union span; the higher confidence already won by sort.
            duplicate.start_ms = min(duplicate.start_ms, candidate.start_ms)
            duplicate.end_ms = max(duplicate.end_ms, candidate.end_ms)
    return sorted(kept, key=lambda c: c.start_ms)


def _to_findings(candidates: list[Candidate]) -> list[Finding]:
    findings: list[Finding] = []
    for index, candidate in enumerate(candidates):
        findings.append(
            Finding(
                id=f"p_{index:02d}",
                clauseId=candidate.clause_id,
                category=candidate.category,
                title=candidate.chunk.clause_title,
                description=candidate.why or candidate.rationale,
                startMs=candidate.start_ms,
                endMs=candidate.end_ms,
                severity=candidate.severity,  # type: ignore[arg-type]
                confidence=candidate.confidence,
                modalities={"speech": candidate.confidence},
                evidence=Evidence.marking(candidate.evidence, candidate.evidence),
                policy=PolicyRef(
                    clauseId=candidate.clause_id,
                    title=candidate.chunk.clause_title,
                    section=candidate.chunk.citation,
                    text=candidate.chunk.text,
                ),
                adversarial=Adversarial(
                    charge=candidate.why or "Candidate raised by AUDITOR.",
                    rationale=candidate.rationale,
                    confidence=candidate.confidence,
                    defense=candidate.defense,
                    defense_strength=candidate.defense_strength,
                    verdict="UPHELD",
                ),
                suggestedFix=candidate.suggested_fix,  # type: ignore[arg-type]
            )
        )
    return findings


def to_agent_result(result: TriadResult) -> AgentResult:
    return AgentResult(
        agent_id="policy",
        name="Policy Agent",
        status=result.status,  # type: ignore[arg-type]
        findings=result.findings,
        coverage=result.coverage,
        error=result.error,
        calls=result.calls,
        log=result.log,
        artifacts={
            "windows_seen": result.windows_seen,
            "windows_with_candidates": result.windows_with_candidates,
            "candidates": [
                {
                    "id": c.id,
                    "clause_id": c.clause_id,
                    "evidence": c.evidence,
                    "start_ms": c.start_ms,
                    "end_ms": c.end_ms,
                    "why": c.why,
                    "defense": c.defense,
                    "defense_strength": round(c.defense_strength, 3),
                    "verdict": c.verdict,
                    "rationale": c.rationale,
                }
                for c in result.candidates
            ],
            "dismissed": [
                {
                    "clause_id": c.clause_id,
                    "evidence": c.evidence,
                    "start_ms": c.start_ms,
                    "rationale": c.rationale,
                }
                for c in result.candidates
                if c.verdict != "UPHELD"
            ],
        },
    )


