"""A02 — SPEECH INTELLIGENCE.

Turns a transcript into structured speech events. Deterministic: no model, no
network, no retrieval. Every event carries a span taken from real word timings
rather than an estimate.

It emits evidence and never a verdict. `PROFANITY` with a span is an
observation; `VIOLATION` is A11's decision, and A02 would make it worse by
guessing.

Two problems separate this from a word filter, and both are the reason naive
matching produces roughly nine false positives in ten:

**The Scunthorpe problem.** Substring matching flags `bass` for containing
`ass`, `assessment` for the same reason, and `Cocaine` in `Cocaine Bear` as
drug promotion. Matching is therefore done on *whole normalised tokens*, using
the word-level timings the transcript already carries — which solves the
timing problem and the substring problem with one decision.

**Attribution.** The same sentence containing the same word is a violation when
asserted and an exemption when quoted and condemned. A02 does not rule on that,
but it records whether the span sits inside an attributed quotation, near a
negation, or near a harm-reduction warning. That is evidence, and it is what
gives the ADVOCATE something to argue from instead of inventing an exemption.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from preflight.models import AgentResult
from preflight.perception.asr import Transcript, Word

AGENT_ID = "speech_intel"
AGENT_NAME = "Speech Intelligence"
LEXICON_DIR = Path("data/lexicons/speech")
CUE_DIR = Path("data/lexicons")

# Confidence by evidence source. A lexicon hit on an unambiguous token is
# strong; the same hit inside a quotation is not, which is why context
# demotes rather than the source alone deciding.
CONFIDENCE = {
    "lexicon": 0.95,
    "regex": 0.92,
    "phrase": 0.93,
    "ner": 0.90,
    "uncertain": 0.65,
}

# How far either side of an event to look for framing signals.
CONTEXT_WORDS = 14
QUOTE_SPAN_MS = 20_000

# Characters used to obfuscate inside a token. Stripped before comparison, but
# only within a token — stripping globally would join "a smart" into "asmart".
SEPARATORS = str.maketrans("", "", "*._-+^~`'\"()[]{}!?,;:")

# Only substitutions that are unambiguous in this direction. Deliberately no
# 1->i or 1->l: that reading turns "1nformation" into a hit and is the kind of
# cleverness that manufactures false positives.
LEET = str.maketrans({"4": "a", "@": "a", "3": "e", "0": "o", "$": "s", "5": "s"})

_SPACED_LETTERS = re.compile(r"\b(?:[a-z]\s+){2,}[a-z]\b", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9']+")


def normalize(token: str) -> str:
    """Canonical form of one token: lowercase, unpunctuated, de-leeted.

    Repeated characters collapse to two — `fuuuuck` and `fuuck` both reduce to
    `fuuck`, which then needs an explicit variant rather than an infinite family
    of them. Collapsing to one instead would map `hell` onto `hel` and `boo`
    onto `bo`, breaking ordinary words.
    """
    lowered = token.lower().strip()
    stripped = lowered.translate(SEPARATORS).translate(LEET)
    return re.sub(r"(.)\1{2,}", r"\1\1", stripped)


def collapse_spaced_letters(text: str) -> str:
    """`f u c k` -> `fuck`, without touching `a smart idea`.

    Only runs of three or more single letters are collapsed. Two-letter runs
    are ordinary English ("a b test", "I a m" is not real speech) and
    collapsing them costs more than it catches.
    """
    return _SPACED_LETTERS.sub(lambda m: m.group(0).replace(" ", ""), text)


@dataclass(frozen=True)
class LexEntry:
    event: str
    term: str
    severity: str
    forms: frozenset[str]
    words: int


@dataclass
class SpeechEvent:
    """One observation. Never a verdict."""

    type: str
    severity: str
    start_ms: int
    end_ms: int
    text: str
    matched: str
    confidence: float
    source: str
    context: dict[str, Any] = field(default_factory=dict)
    word_index: int = 0

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.type, self.matched, self.start_ms)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.type,
            "severity": self.severity,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
            "matched": self.matched,
            "confidence": round(self.confidence, 3),
            "source": self.source,
        }
        if self.context:
            payload["context"] = self.context
        return payload


# ------------------------------------------------------------------ #
# Lexicon loading                                                     #
# ------------------------------------------------------------------ #


class Lexicons:
    """Every speech lexicon, in RAM. Never indexed, never embedded."""

    def __init__(self, directory: Path = LEXICON_DIR) -> None:
        self.directory = Path(directory)
        self.by_form: dict[str, list[LexEntry]] = {}
        self.phrases: list[tuple[str, LexEntry]] = []
        self.events: set[str] = set()
        self.loaded: list[str] = []
        self._load()

    def _load(self) -> None:
        if not self.directory.is_dir():
            return
        for path in sorted(self.directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            event = payload.get("_event", path.stem.upper())
            default = payload.get("severity_default", "MEDIUM")
            self.events.add(event)
            self.loaded.append(path.name)

            for record in payload.get("terms", []):
                term = record["term"]
                severity = record.get("severity", default)
                forms = {normalize(term)}
                forms.update(normalize(v) for v in record.get("variants", []))
                forms.discard("")
                entry = LexEntry(event, term, severity, frozenset(forms), len(term.split()))
                for form in forms:
                    self.by_form.setdefault(form, []).append(entry)

            for phrase in payload.get("phrases", []):
                normalised = " ".join(normalize(p) for p in phrase.split())
                self.phrases.append(
                    (normalised, LexEntry(event, phrase, default, frozenset(), len(phrase.split())))
                )

        # Longest phrase first, so "this video is sponsored by" wins over
        # "sponsored by" and the event is reported once at full extent.
        self.phrases.sort(key=lambda item: -len(item[0]))

    @property
    def term_count(self) -> int:
        return sum(len(v) for v in self.by_form.values())

    def lookup(self, form: str) -> list[LexEntry]:
        return self.by_form.get(form, [])


# ------------------------------------------------------------------ #
# PII patterns, with validators                                       #
# ------------------------------------------------------------------ #


def _luhn(digits: str) -> bool:
    """Card checksum.

    Without it, any sixteen-digit run is a 'credit card' — a timestamp, an
    order number, a phone number with the spaces removed. The regex alone is
    a false-positive generator.
    """
    total, alternate = 0, False
    for char in reversed(digits):
        value = int(char)
        if alternate:
            value *= 2
            if value > 9:
                value -= 9
        total += value
        alternate = not alternate
    return total % 10 == 0


PII_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"), "HIGH"),
    ("URL", re.compile(r"\bhttps?://[^\s]+|\b(?:www\.)[\w.-]+\.\w{2,}\b"), "LOW"),
    (
        "PHONE",
        re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b"),
        "HIGH",
    ),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b"), "HIGH"),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), "HIGH"),
]

CASUALTY = re.compile(
    r"\b\d+\s+(?:people\s+)?(?:were\s+)?(?:killed|dead|died|injured|wounded)\b"
    r"|\bdeath toll\b|\bcasualt(?:y|ies)\b",
    re.IGNORECASE,
)


# ------------------------------------------------------------------ #
# Context                                                             #
# ------------------------------------------------------------------ #

NEGATIONS = {"not", "never", "no", "dont", "doesnt", "didnt", "wont", "cannot", "cant"}
HARM_REDUCTION = {
    "dangerous", "warning", "never", "dont", "do not", "unsafe", "seriously hurt",
}

ATTRIBUTION_CUES_PATH = Path("data/lexicons/attribution_cues.json")


class AttributionCues:
    """Reporting verbs, quote markers and condemnation phrases.

    `data/lexicons/attribution_cues.json` existed from the day the data layer
    was authored and was never loaded — this module carried its own smaller,
    hardcoded set of the same vocabulary instead, which meant the file on
    disk was decorative and editing it changed nothing. Loading it here makes
    the lexicon the single source rather than two lists that can drift.

    Phrases (multi-word cues like "in his words", "which is disgusting") are
    matched against the joined, normalised context window rather than
    token-by-token, since a reporting-verb-shaped set membership test cannot
    see a two-word phrase.
    """

    def __init__(self, path: Path = ATTRIBUTION_CUES_PATH) -> None:
        payload: dict[str, Any] = {}
        if Path(path).is_file():
            payload = json.loads(Path(path).read_text(encoding="utf-8"))

        self.reporting_verbs = {
            normalize(v) for v in payload.get("reporting_verbs", [])
        }
        self.quote_markers = [
            normalize(m) for m in payload.get("quote_markers", [])
        ]
        self.condemnation_markers = [
            normalize(m) for m in payload.get("condemnation_markers", [])
        ]
        self.max_quote_span_ms = int(payload.get("max_quote_span_ms", QUOTE_SPAN_MS))
        self.loaded = bool(self.reporting_verbs)


_DEFAULT_CUES = AttributionCues()


# A phrase's words matching in order but not touching — "which is frankly
# disgusting" for the lexicon's "which is disgusting" — is still the phrase.
# Real speech inserts adverbs and hedges between the words that carry the
# meaning ("which is, frankly, disgusting"), and a literal substring match
# would miss the phrase specifically where it sounds most natural. The gap is
# capped so "which is a totally different thing that happens to also be
# disgusting" does not count as the same phrase stretched thin.
PHRASE_GAP_TOLERANCE = 2


def _phrase_matches_from(tokens: list[str], words: list[str], start: int) -> bool:
    """Do `words` appear in order starting at or after `start`, gaps allowed?"""
    cursor = start
    for word in words:
        found_at = None
        for offset in range(cursor, min(cursor + PHRASE_GAP_TOLERANCE + 1, len(tokens))):
            if tokens[offset] == word:
                found_at = offset
                break
        if found_at is None:
            return False
        cursor = found_at + 1
    return True


def _phrase_hit(tokens: list[str], phrases: list[str]) -> str | None:
    """Does any phrase's words appear in order in `tokens`, gaps allowed?"""
    for phrase in phrases:
        words = phrase.split()
        if not words:
            continue
        if any(
            _phrase_matches_from(tokens, words, start)
            for start in range(len(tokens))
        ):
            return phrase
    return None


def _context_for(
    words: list[Word], index: int, cues: AttributionCues | None = None
) -> dict[str, Any]:
    """Framing signals around an event. Evidence, not judgement."""
    cues = cues or _DEFAULT_CUES
    low = max(0, index - CONTEXT_WORDS)
    high = min(len(words), index + CONTEXT_WORDS + 1)
    before = [normalize(w.w) for w in words[low:index]]
    after = [normalize(w.w) for w in words[index + 1 : high]]
    window = before + after

    context: dict[str, Any] = {}

    # Attribution: a reporting verb *before* the span, inside the quote window.
    for offset, token in enumerate(reversed(before)):
        if token in cues.reporting_verbs:
            distance_ms = words[index].start_ms - words[index - offset - 1].start_ms
            if distance_ms <= cues.max_quote_span_ms:
                context["in_quotation"] = True
                context["attribution_cue"] = token
                context["attribution_distance_ms"] = int(distance_ms)
            break

    # Markers cut both ways: "unquote" and "end quote" close a span from
    # after it, while "in his words" and "according to" open one from before
    # it. The lexicon mixes both kinds in one list, so both directions are
    # checked rather than assuming every marker is a closer.
    marker = _phrase_hit(before[-8:], cues.quote_markers) or _phrase_hit(
        after[:8], cues.quote_markers
    )
    if marker:
        context["in_quotation"] = True
        context["attribution_cue"] = context.get("attribution_cue", marker)

    if any(token in NEGATIONS for token in before[-4:]):
        context["negated"] = True

    condemned = _phrase_hit(window, cues.condemnation_markers)
    if condemned:
        context["condemned"] = True
        context["condemnation_cue"] = condemned

    warning = [t for t in window if t in HARM_REDUCTION]
    if warning:
        context["harm_reduction_nearby"] = True
        context["harm_reduction_cue"] = warning[0]

    return context


def _confidence(source: str, context: dict[str, Any]) -> float:
    """Context demotes. A quoted term is genuinely less certain evidence of
    the speaker asserting it, and reporting 0.95 either way would be a
    calibration the downstream fusion step then trusts."""
    base = CONFIDENCE.get(source, CONFIDENCE["uncertain"])
    if context.get("in_quotation") or context.get("negated"):
        return CONFIDENCE["uncertain"]
    return base


# ------------------------------------------------------------------ #
# Extraction                                                          #
# ------------------------------------------------------------------ #


def _sentence_around(words: list[Word], index: int) -> str:
    """The utterance containing the span, for the evidence field."""
    low = index
    while low > 0 and not words[low - 1].w.endswith((".", "?", "!")):
        low -= 1
        if index - low > 40:
            break
    high = index
    while high < len(words) - 1 and not words[high].w.endswith((".", "?", "!")):
        high += 1
        if high - index > 40:
            break
    return " ".join(w.w for w in words[low : high + 1]).strip()


def _dedupe(events: list[SpeechEvent]) -> list[SpeechEvent]:
    """One utterance is one event.

    Overlapping spans of the same type and term are the same observation seen
    twice — a phrase matched inside a longer phrase, or a term matched by two
    variants. Keeping both would inflate every count downstream.
    """
    ordered = sorted(events, key=lambda e: (e.start_ms, -(e.end_ms - e.start_ms)))
    kept: list[SpeechEvent] = []
    for event in ordered:
        clash = next(
            (
                k
                for k in kept
                if k.type == event.type
                and event.start_ms < k.end_ms
                and event.end_ms > k.start_ms
            ),
            None,
        )
        if clash is None:
            kept.append(event)
        elif event.confidence > clash.confidence:
            kept[kept.index(clash)] = event
    return kept


def extract_events(
    transcript: Transcript, lexicons: Lexicons | None = None
) -> list[SpeechEvent]:
    lexicons = lexicons or Lexicons()
    words = transcript.words
    if not words:
        return []

    normalised = [normalize(w.w) for w in words]
    events: list[SpeechEvent] = []

    # ---- single tokens, whole-word only -----------------------------
    for index, form in enumerate(normalised):
        if not form:
            continue
        for entry in lexicons.lookup(form):
            context = _context_for(words, index)
            events.append(
                SpeechEvent(
                    type=entry.event,
                    severity=entry.severity,
                    start_ms=words[index].start_ms,
                    end_ms=words[index].end_ms,
                    text=_sentence_around(words, index),
                    matched=entry.term,
                    confidence=_confidence("lexicon", context),
                    source="lexicon",
                    context=context,
                    word_index=index,
                )
            )

    # ---- phrases, as n-grams over the same normalised tokens --------
    for phrase, entry in lexicons.phrases:
        parts = phrase.split()
        span = len(parts)
        if span == 0 or span > len(normalised):
            continue
        for index in range(len(normalised) - span + 1):
            if normalised[index : index + span] != parts:
                continue
            last = index + span - 1
            context = _context_for(words, index)
            events.append(
                SpeechEvent(
                    type=entry.event,
                    severity=entry.severity,
                    start_ms=words[index].start_ms,
                    end_ms=words[last].end_ms,
                    text=_sentence_around(words, index),
                    matched=entry.term,
                    confidence=_confidence("phrase", context),
                    source="phrase",
                    context=context,
                    word_index=index,
                )
            )

    events.extend(_extract_patterns(words))
    return _dedupe(events)


def _extract_patterns(words: list[Word]) -> list[SpeechEvent]:
    """Regex over the joined utterance, mapped back to word spans.

    Character offsets are converted to word indices so every event still
    carries a real timing rather than an interpolation.
    """
    if not words:
        return []

    offsets: list[tuple[int, int, int]] = []  # start, end, word index
    cursor = 0
    pieces: list[str] = []
    for index, word in enumerate(words):
        pieces.append(word.w)
        offsets.append((cursor, cursor + len(word.w), index))
        cursor += len(word.w) + 1
    joined = " ".join(pieces)

    def words_for(start: int, end: int) -> tuple[int, int] | None:
        touched = [i for s, e, i in offsets if e > start and s < end]
        return (touched[0], touched[-1]) if touched else None

    found: list[SpeechEvent] = []

    for label, pattern, severity in PII_PATTERNS:
        for match in pattern.finditer(joined):
            raw = match.group(0)
            if label == "CARD":
                digits = re.sub(r"\D", "", raw)
                if not (13 <= len(digits) <= 19) or not _luhn(digits):
                    continue
            bounds = words_for(match.start(), match.end())
            if bounds is None:
                continue
            first, last = bounds
            context = _context_for(words, first)
            found.append(
                SpeechEvent(
                    type="PERSONAL_INFO",
                    severity=severity,
                    start_ms=words[first].start_ms,
                    end_ms=words[last].end_ms,
                    # The value itself is not reproduced. Copying a card number
                    # into a report to warn about a card number in a report is
                    # not a defensible trade.
                    text=f"[{label} redacted]",
                    matched=label,
                    confidence=CONFIDENCE["regex"],
                    source="regex",
                    context=context,
                    word_index=first,
                )
            )

    for match in CASUALTY.finditer(joined):
        bounds = words_for(match.start(), match.end())
        if bounds is None:
            continue
        first, last = bounds
        context = _context_for(words, first)
        found.append(
            SpeechEvent(
                type="SENSITIVE_EVENT",
                severity="MEDIUM",
                start_ms=words[first].start_ms,
                end_ms=words[last].end_ms,
                text=_sentence_around(words, first),
                matched=match.group(0),
                confidence=_confidence("regex", context),
                source="regex",
                context=context,
                word_index=first,
            )
        )

    return found


# ------------------------------------------------------------------ #
# Quotation spans — cross-modal context for the ADVOCATE               #
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class QuotationSpan:
    """A stretch of the transcript introduced by a reporting cue.

    Policy treats quoting objectionable speech differently from asserting
    it, and this is what gives the ADVOCATE a real, evidenced defence to
    argue instead of an invented one. Deliberately independent of any
    lexicon hit: a span exists wherever the transcript reads like a
    quotation, whether or not A02's own lexicons found anything inside it,
    because the ADVOCATE needs this for candidates the AUDITOR raised from
    the clause text alone.
    """

    start_ms: int
    end_ms: int
    cue: str
    kind: str  # "attributed" | "attributed_and_condemned"
    strength: float

    def overlaps(self, start_ms: int, end_ms: int) -> bool:
        return self.start_ms < end_ms and self.end_ms > start_ms

    def to_json(self) -> dict[str, Any]:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "cue": self.cue,
            "kind": self.kind,
            "strength": round(self.strength, 3),
        }


def _next_terminator(words: list[Word], index: int, limit: int = 60) -> int:
    """Index of the word ending the sentence that starts at `index`."""
    cursor = index
    while cursor < len(words) - 1 and not words[cursor].w.endswith((".", "?", "!")):
        cursor += 1
        if cursor - index > limit:
            break
    return cursor


def quotation_spans(
    transcript: Transcript | None, cues: AttributionCues | None = None
) -> list[QuotationSpan]:
    """Every attributed-quotation span in the transcript, merged and scored.

    Two passes. The first finds cues — a reporting verb or an opening marker
    — and builds a span from the cue to the end of that sentence, capped by
    the lexicon's `max_quote_span_ms` so one missing terminator cannot swallow
    the rest of the file. The second checks the text immediately after each
    span for a condemnation marker: quoting AND condemning is a materially
    stronger exemption than quoting alone, since it forecloses the reading
    that the speaker endorses what was quoted.

    No prosody signal. The published design for this also measures pitch
    deviation against the speaker's baseline — quoted speech commonly shifts
    register — which needs a pitch tracker this project does not depend on.
    Lexical detection alone is the honest scope: it catches an attributed
    quotation that says it is one, and does not claim to catch a silent
    tonal shift no cue announces.
    """
    cues = cues or _DEFAULT_CUES
    if transcript is None or not transcript.words:
        return []

    words = transcript.words
    normalised = [normalize(w.w) for w in words]
    spans: list[QuotationSpan] = []

    for index, token in enumerate(normalised):
        cue_text: str | None = None
        if token in cues.reporting_verbs:
            cue_text = token
        else:
            marker = _phrase_hit(normalised[index : index + 4], cues.quote_markers)
            if marker and normalised[index : index + 1] == [marker.split()[0]]:
                cue_text = marker

        if cue_text is None:
            continue

        end_index = _next_terminator(words, index)
        start_ms = words[index].end_ms
        end_ms = min(words[end_index].end_ms, start_ms + cues.max_quote_span_ms)
        if end_ms <= start_ms:
            continue

        after = normalised[end_index + 1 : end_index + 9]
        condemned = _phrase_hit(after, cues.condemnation_markers)
        kind = "attributed_and_condemned" if condemned else "attributed"
        # A condemned quotation is a stronger, more legible exemption than a
        # bare one, and the strength communicates that ordering to the
        # ADVOCATE without asserting a confidence this lexical signal alone
        # cannot support.
        strength = 0.75 if condemned else 0.55

        spans.append(
            QuotationSpan(
                start_ms=start_ms, end_ms=end_ms, cue=cue_text, kind=kind,
                strength=strength,
            )
        )

    return _merge_quotation_spans(spans, cues.max_quote_span_ms)


def _merge_quotation_spans(
    spans: list[QuotationSpan], max_span_ms: int = QUOTE_SPAN_MS
) -> list[QuotationSpan]:
    """Two cues inside one sentence ('she said, and I quote,') should not
    produce two overlapping spans that double-count as corroboration.

    Each individual span already respects `max_quote_span_ms` on its own, but
    "he said quote ..." triggers on BOTH "said" and "quote" one word apart,
    producing two independently-capped spans whose union can still exceed the
    cap — measured at 20,400ms against a 20,000ms limit. The merge re-clamps
    to the earliest span's own budget rather than trusting that a union of
    two valid spans is itself valid.
    """
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: s.start_ms)
    merged = [ordered[0]]
    for span in ordered[1:]:
        last = merged[-1]
        if span.start_ms <= last.end_ms:
            stronger = span if span.strength > last.strength else last
            end_ms = min(max(last.end_ms, span.end_ms), last.start_ms + max_span_ms)
            merged[-1] = QuotationSpan(
                start_ms=last.start_ms,
                end_ms=end_ms,
                cue=stronger.cue,
                kind=stronger.kind,
                strength=stronger.strength,
            )
        else:
            merged.append(span)
    return merged


# ------------------------------------------------------------------ #
# Agent entry point                                                   #
# ------------------------------------------------------------------ #


def analyse(
    transcript: Transcript | None,
    lexicons: Lexicons | None = None,
    cues: AttributionCues | None = None,
) -> tuple[AgentResult, list[SpeechEvent], list[QuotationSpan]]:
    started = time.perf_counter()

    if transcript is None:
        return (
            AgentResult.skipped(AGENT_ID, AGENT_NAME, "Transcript unavailable"),
            [],
            [],
        )
    if not transcript.words:
        return (
            AgentResult(
                agent_id=AGENT_ID,
                name=AGENT_NAME,
                status="OK",
                coverage=1.0,
                log=["transcript contains no words — no speech events"],
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            ),
            [],
            [],
        )

    lexicons = lexicons or Lexicons()
    cues = cues or _DEFAULT_CUES
    events = extract_events(transcript, lexicons)
    spans = quotation_spans(transcript, cues)

    by_type: dict[str, int] = {}
    for event in events:
        by_type[event.type] = by_type.get(event.type, 0) + 1
    summary = ", ".join(f"{k.lower()} {v}" for k, v in sorted(by_type.items())) or "none"

    log = [
        f"{len(lexicons.loaded)} lexicons, {lexicons.term_count} terms, "
        f"{len(lexicons.phrases)} phrases in RAM",
        f"{len(events)} speech event(s): {summary}",
    ]
    if spans:
        condemned = sum(1 for s in spans if s.kind == "attributed_and_condemned")
        log.append(
            f"{len(spans)} quotation span(s), {condemned} with condemnation — "
            "cross-modal context for the ADVOCATE"
        )

    return (
        AgentResult(
            agent_id=AGENT_ID,
            name=AGENT_NAME,
            status="OK",
            coverage=1.0,
            artifacts={
                "events": [e.to_json() for e in events],
                "quotation_spans": [s.to_json() for s in spans],
            },
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            log=log,
        ),
        events,
        spans,
    )


def to_json(
    events: Iterable[SpeechEvent], spans: Iterable[QuotationSpan] = ()
) -> dict[str, Any]:
    """The A02 output contract. JSON only — never a paragraph."""
    return {
        "speech_events": [event.to_json() for event in events],
        "quotation_spans": [span.to_json() for span in spans],
    }
