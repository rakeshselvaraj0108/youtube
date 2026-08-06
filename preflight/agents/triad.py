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

    # Cross-modal context, attached before the ADVOCATE runs. Set by
    # `_tag_quotation_context` when this candidate's evidence span overlaps a
    # quotation A02 found in the transcript — independent of whether A02's own
    # lexicons happened to also fire on the same words, since a charge can
    # come from clause text alone.
    quotation_context: str | None = None

    verdict: str = "UPHELD"
    severity: str = "MEDIUM"
    confidence: float = 0.5
    rationale: str = ""
    suggested_fix: str = "NONE"


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
    quotation_spans: list | None = None,
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
            in_quotation = _tag_quotation_context(candidates, quotation_spans or [])
            _defend(candidates, client, settings)
            defended = sum(1 for c in candidates if c.defense)
            result.log.append(
                f"ADVOCATE: {defended}/{len(candidates)} candidate(s) defended"
                + (f" · {in_quotation} inside a quotation span" if in_quotation else "")
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


def _defend(candidates: list[Candidate], client: NimClient, settings: Settings) -> None:
    for batch in _batched(candidates, ADVOCATE_BATCH):
        chunks = {c.chunk.id: c.chunk for c in batch}
        block = "\n\n".join(
            f'candidate_id: {c.id}\nclause_id: {c.clause_id}\n'
            f'evidence: "{c.evidence}"\ncharge: {c.why}'
            + (
                f'\ncross_modal_context: {c.quotation_context}'
                if c.quotation_context
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


