"""NIM client resilience and triad post-processing.

No network. These test the parts that decide whether a live run survives
contact with a rate-limited endpoint and an instruction-tuned model that
ignores your JSON instructions.
"""

from __future__ import annotations

import time

import pytest

from preflight.agents.nim import TokenBucket, extract_json
from preflight.agents.triad import Candidate, _dedupe, _to_findings
from preflight.perception.asr import Segment, Transcript, Word
from preflight.policy.corpus import Chunk


def chunk(clause_id: str = "AF-01") -> Chunk:
    return Chunk(
        clause_id=clause_id,
        clause_title="Inappropriate language",
        section="Yellow",
        text="Strong profanity used more than occasionally through the video.",
        severity_default="LIMITING",
        source_url="https://example.invalid",
    )


class TestExtractJson:
    """Every case here is output a model actually produces."""

    def test_plain_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_unlabelled_fence(self):
        assert extract_json('```\n{"a": 1}\n```') == {"a": 1}

    def test_prose_preamble(self):
        assert extract_json('Here is the JSON you asked for:\n{"a": 1}') == {"a": 1}

    def test_prose_on_both_sides(self):
        assert extract_json('Sure!\n{"a": 1}\nLet me know if you need more.') == {"a": 1}

    def test_trailing_comma_in_object(self):
        assert extract_json('{"a": 1,}') == {"a": 1}

    def test_trailing_comma_in_array(self):
        assert extract_json('{"candidates": [1, 2,]}') == {"candidates": [1, 2]}

    def test_top_level_array(self):
        assert extract_json("[1, 2, 3]") == [1, 2, 3]

    def test_braces_inside_strings_do_not_confuse_the_scanner(self):
        assert extract_json('{"evidence": "he said {this}"}') == {
            "evidence": "he said {this}"
        }

    def test_escaped_quotes_inside_strings(self):
        assert extract_json('{"q": "she said \\"go\\""}') == {"q": 'she said "go"'}

    def test_nested_structures_survive(self):
        payload = '{"candidates":[{"clause_id":"AF-01","span":{"a":1}}]}'
        assert extract_json(payload)["candidates"][0]["span"] == {"a": 1}

    def test_empty_response_raises(self):
        with pytest.raises(ValueError):
            extract_json("")

    def test_unrecoverable_response_raises(self):
        with pytest.raises(ValueError):
            extract_json("I cannot help with that request.")


class TestTokenBucket:
    def test_allows_a_burst_up_to_capacity(self):
        bucket = TokenBucket(rpm=60)
        started = time.monotonic()
        for _ in range(5):
            bucket.take()
        assert time.monotonic() - started < 0.2

    def test_blocks_once_the_bucket_is_drained(self):
        bucket = TokenBucket(rpm=60)  # one token per second
        for _ in range(60):
            bucket.take()
        started = time.monotonic()
        waited = bucket.take()
        assert waited > 0
        assert time.monotonic() - started >= 0.5

    def test_capacity_is_never_below_one(self):
        assert TokenBucket(rpm=0).capacity == 1


class TestDedupe:
    """Overlapping windows surface the same violation twice."""

    def _candidate(self, cid: str, start: int, end: int, conf: float, clause="AF-01"):
        return Candidate(
            id=cid,
            window=0,
            clause_id=clause,
            category="Language",
            evidence="this is fucked",
            start_ms=start,
            end_ms=end,
            why="profanity",
            chunk=chunk(clause),
            confidence=conf,
        )

    def test_merges_overlapping_findings_on_the_same_clause(self):
        kept = _dedupe(
            [
                self._candidate("a", 30_000, 32_000, 0.8),
                self._candidate("b", 30_500, 32_500, 0.9),
            ]
        )
        assert len(kept) == 1
        assert kept[0].confidence == 0.9  # higher confidence wins

    def test_merged_span_is_the_union(self):
        """Real duplicates resolve to near-identical spans, because both are
        recovered from the same word timings. The union then widens slightly."""
        kept = _dedupe(
            [
                self._candidate("a", 30_000, 32_000, 0.9),
                self._candidate("b", 30_200, 32_400, 0.8),
            ]
        )
        assert len(kept) == 1
        assert kept[0].start_ms == 30_000
        assert kept[0].end_ms == 32_400

    def test_partial_overlap_below_the_threshold_stays_separate(self):
        """Two findings sharing a third of their span are two violations, not
        one seen twice. Merging them would smear a bleep across clean audio."""
        kept = _dedupe(
            [
                self._candidate("a", 30_000, 32_000, 0.9),
                self._candidate("b", 31_000, 33_000, 0.8),
            ]
        )
        assert len(kept) == 2

    def test_keeps_distant_findings_on_the_same_clause(self):
        kept = _dedupe(
            [
                self._candidate("a", 10_000, 12_000, 0.9),
                self._candidate("b", 90_000, 92_000, 0.8),
            ]
        )
        assert len(kept) == 2

    def test_keeps_overlapping_findings_on_different_clauses(self):
        kept = _dedupe(
            [
                self._candidate("a", 30_000, 32_000, 0.9, clause="AF-01"),
                self._candidate("b", 30_000, 32_000, 0.8, clause="AF-02"),
            ]
        )
        assert len(kept) == 2

    def test_output_is_ordered_by_time(self):
        kept = _dedupe(
            [
                self._candidate("a", 90_000, 92_000, 0.9),
                self._candidate("b", 10_000, 12_000, 0.8),
            ]
        )
        assert [c.start_ms for c in kept] == [10_000, 90_000]


class TestToFindings:
    def test_carries_the_full_adversarial_record(self):
        candidate = Candidate(
            id="c1",
            window=0,
            clause_id="AF-01",
            category="Language",
            evidence="this is fucked",
            start_ms=30_000,
            end_ms=31_000,
            why="strong profanity",
            chunk=chunk(),
            defense=None,
            confidence=0.9,
            rationale="Matches the Yellow condition.",
            suggested_fix="BLEEP",
        )
        finding = _to_findings([candidate])[0]

        assert finding.clauseId == "AF-01"
        assert finding.policy.text  # the clause travels with the finding
        assert finding.adversarial.charge == "strong profanity"
        assert finding.adversarial.defense is None
        assert finding.suggestedFix == "BLEEP"

    def test_evidence_span_is_resolvable(self):
        candidate = Candidate(
            id="c1", window=0, clause_id="AF-01", category="Language",
            evidence="this is fucked", start_ms=0, end_ms=1000, why="w",
            chunk=chunk(),
        )
        finding = _to_findings([candidate])[0]
        start, end = finding.evidence.highlightSpan
        assert finding.evidence.transcript[start:end] == "this is fucked"


class TestTranscriptLocate:
    """Span recovery. Models return plausible-looking timestamps that are
    frequently seconds out; the quote plus word timings gives the truth."""

    def _transcript(self) -> Transcript:
        text = "the anchor pulled clean out and this is fucked we need to move"
        words = [
            Word(w=token, start_ms=30_000 + i * 500, end_ms=30_000 + i * 500 + 450, conf=0.9)
            for i, token in enumerate(text.split())
        ]
        return Transcript(
            language="en",
            duration_ms=60_000,
            words=words,
            segments=[Segment(start_ms=30_000, end_ms=36_000, text=text)],
        )

    def test_recovers_the_true_span_of_a_quote(self):
        located = self._transcript().locate("this is fucked")
        assert located is not None
        start, end = located
        assert start == 33_000  # index 6 -> 30000 + 6*500
        assert end > start

    def test_tolerates_punctuation_and_case(self):
        assert self._transcript().locate("This is FUCKED,") is not None

    def test_returns_none_for_absent_text(self):
        assert self._transcript().locate("entirely unrelated wording here") is None

    def test_returns_none_without_words(self):
        empty = Transcript(language="en", duration_ms=0, words=[], segments=[])
        assert empty.locate("anything") is None
