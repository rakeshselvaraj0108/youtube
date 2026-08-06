"""NIM client resilience and triad post-processing.

No network. These test the parts that decide whether a live run survives
contact with a rate-limited endpoint and an instruction-tuned model that
ignores your JSON instructions.
"""

from __future__ import annotations

import time

import pytest

from preflight.agents.nim import TokenBucket, extract_json
from preflight.agents.triad import Candidate, _dedupe, _tag_quotation_context, _to_findings
from preflight.perception.asr import Segment, Transcript, Word
from preflight.perception.speech_intel import QuotationSpan
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


class TestQuotationCrossModalContext:
    """What makes the ADVOCATE more than a second opinion: it sees a signal
    from a different agent's independent read of the same moment, which the
    AUDITOR was deliberately never shown."""

    def _candidate(self, start: int, end: int, clause="AF-01"):
        return Candidate(
            id="c1", window=0, clause_id=clause, category="Language",
            evidence="this is fucked", start_ms=start, end_ms=end,
            why="profanity", chunk=chunk(clause),
        )

    def test_a_candidate_inside_a_quotation_span_is_tagged(self):
        candidate = self._candidate(31_000, 32_000)
        span = QuotationSpan(30_000, 33_000, "said", "attributed", 0.55)
        tagged = _tag_quotation_context([candidate], [span])
        assert tagged == 1
        assert candidate.quotation_context is not None
        assert "said" in candidate.quotation_context

    def test_a_condemned_span_says_so_in_the_context_line(self):
        candidate = self._candidate(31_000, 32_000)
        span = QuotationSpan(30_000, 33_000, "said", "attributed_and_condemned", 0.75)
        _tag_quotation_context([candidate], [span])
        assert "condemned" in candidate.quotation_context

    def test_a_bare_attribution_does_not_claim_condemnation(self):
        candidate = self._candidate(31_000, 32_000)
        span = QuotationSpan(30_000, 33_000, "said", "attributed", 0.55)
        _tag_quotation_context([candidate], [span])
        assert "condemned" not in candidate.quotation_context

    def test_a_candidate_outside_every_span_is_not_tagged(self):
        candidate = self._candidate(50_000, 51_000)
        span = QuotationSpan(30_000, 33_000, "said", "attributed", 0.55)
        tagged = _tag_quotation_context([candidate], [span])
        assert tagged == 0
        assert candidate.quotation_context is None

    def test_no_spans_at_all_tags_nothing_and_does_not_crash(self):
        candidate = self._candidate(31_000, 32_000)
        assert _tag_quotation_context([candidate], []) == 0
        assert candidate.quotation_context is None

    def test_a_candidate_from_a_charge_with_no_matching_lexicon_hit_can_still_be_tagged(self):
        """The AUDITOR can charge from clause text alone, with no A02 lexicon
        hit anywhere near it. The tag must not depend on one."""
        candidate = self._candidate(31_000, 32_000, clause="AF-06")
        span = QuotationSpan(30_000, 33_000, "according to", "attributed", 0.55)
        _tag_quotation_context([candidate], [span])
        assert candidate.quotation_context is not None


class TestFramingAndHarmReductionContext:
    def _candidate(self, start: int, end: int, clause="AF-05"):
        return Candidate(
            id="c1", window=0, clause_id=clause, category="Dangerous acts",
            evidence="bypass the safety cutout", start_ms=start, end_ms=end,
            why="imitable act", chunk=chunk(clause),
        )

    def test_a_candidate_near_an_edsa_cue_is_tagged(self):
        from preflight.agents.triad import _tag_edsa_and_harm_reduction
        from preflight.perception.speech_intel import FramingCue

        candidate = self._candidate(5000, 6000)
        cue = FramingCue(ts_ms=3000, category="educational", cue="in this lesson")
        tagged = _tag_edsa_and_harm_reduction([candidate], [cue])
        assert tagged == 1
        assert "educational" in candidate.edsa_context

    def test_a_candidate_far_from_any_cue_is_not_tagged(self):
        from preflight.agents.triad import _tag_edsa_and_harm_reduction
        from preflight.perception.speech_intel import FramingCue

        candidate = self._candidate(500_000, 501_000)
        cue = FramingCue(ts_ms=3000, category="educational", cue="in this lesson")
        assert _tag_edsa_and_harm_reduction([candidate], [cue]) == 0
        assert candidate.edsa_context is None

    def test_a_nearby_harm_reduction_cue_is_tagged_with_its_distance(self):
        from preflight.agents.triad import _tag_edsa_and_harm_reduction
        from preflight.perception.speech_intel import FramingCue

        candidate = self._candidate(10_000, 11_000)
        cue = FramingCue(ts_ms=8000, category="harm_reduction", cue="never do this")
        _tag_edsa_and_harm_reduction([candidate], [cue])
        assert candidate.harm_reduction_context is not None
        assert "2.0s" in candidate.harm_reduction_context

    def test_no_cues_tags_nothing_and_does_not_crash(self):
        from preflight.agents.triad import _tag_edsa_and_harm_reduction

        candidate = self._candidate(5000, 6000)
        assert _tag_edsa_and_harm_reduction([candidate], []) == 0


class TestVisionCrossModalContext:
    def _candidate(self, start=5000, end=6000):
        return Candidate(
            id="c1", window=0, clause_id="AF-02", category="Violence",
            evidence="there is a knife", start_ms=start, end_ms=end,
            why="weapon reference", chunk=chunk("AF-02"),
        )

    @staticmethod
    def _track(category: str, start=4000, end=7000):
        from preflight.perception.vision import Track

        return Track(
            label=category, category=category, start_ms=start, end_ms=end,
            frames=3, peak_confidence=0.9, confidence=0.85,
        )

    def test_overlapping_graphic_track_is_reported(self):
        from preflight.agents.triad import _tag_vision_context

        candidate = self._candidate()
        tagged = _tag_vision_context([candidate], [self._track("weapon")], coverage=1.0)
        assert tagged == 1
        assert "found graphic imagery" in candidate.vision_context
        assert "100%" in candidate.vision_context

    def test_overlapping_non_graphic_track_says_so(self):
        from preflight.agents.triad import _tag_vision_context

        candidate = self._candidate()
        _tag_vision_context([candidate], [self._track("scene")], coverage=0.8)
        assert "found no graphic imagery" in candidate.vision_context

    def test_low_coverage_is_reported_alongside_the_negative(self):
        """A 'found no graphic imagery' claim at 42% coverage is a much
        weaker defence than the same claim at 100%, and the ADVOCATE needs
        the number to weigh it correctly."""
        from preflight.agents.triad import _tag_vision_context

        candidate = self._candidate()
        _tag_vision_context([candidate], [self._track("scene")], coverage=0.42)
        assert "42%" in candidate.vision_context

    def test_no_tracks_at_all_tags_nothing(self):
        from preflight.agents.triad import _tag_vision_context

        candidate = self._candidate()
        assert _tag_vision_context([candidate], [], coverage=0.0) == 0
        assert candidate.vision_context is None

    def test_a_non_overlapping_track_still_tags_as_no_graphic_imagery(self):
        """Absence in THIS window is itself the signal — a weapon seen
        elsewhere in the video does not make this window graphic."""
        from preflight.agents.triad import _tag_vision_context

        candidate = self._candidate(5000, 6000)
        distant = self._track("weapon", start=90_000, end=91_000)
        _tag_vision_context([candidate], [distant], coverage=1.0)
        assert "found no graphic imagery" in candidate.vision_context


class TestVideoCrossModalContext:
    def _candidate(self):
        return Candidate(
            id="c1", window=0, clause_id="AF-01", category="Language",
            evidence="this is fucked", start_ms=1000, end_ms=2000,
            why="profanity", chunk=chunk("AF-01"),
        )

    def test_category_and_audience_are_both_reported(self):
        from preflight.agents.triad import _tag_video_context

        candidate = self._candidate()
        tagged = _tag_video_context([candidate], "Education", "general")
        assert tagged == 1
        assert "Education" in candidate.video_context
        assert "general" in candidate.video_context

    def test_neither_present_tags_nothing(self):
        from preflight.agents.triad import _tag_video_context

        candidate = self._candidate()
        assert _tag_video_context([candidate], "", "") == 0
        assert candidate.video_context is None

    def test_every_candidate_gets_the_same_video_level_line(self):
        """Category/audience describe the whole video, not one window — every
        candidate carries it, not just ones near some cue."""
        from preflight.agents.triad import _tag_video_context

        candidates = [self._candidate(), self._candidate()]
        _tag_video_context(candidates, "News", "")
        assert candidates[0].video_context == candidates[1].video_context


class TestUnifiedCrossModalTagging:
    def test_tags_every_signal_in_one_pass_and_reports_counts(self):
        from preflight.agents.triad import CrossModalContext, _tag_cross_modal_context
        from preflight.perception.speech_intel import FramingCue, QuotationSpan
        from preflight.perception.vision import Track

        candidate = Candidate(
            id="c1", window=0, clause_id="AF-01", category="Language",
            evidence="this is fucked", start_ms=5000, end_ms=6000,
            why="profanity", chunk=chunk("AF-01"),
        )
        context = CrossModalContext(
            quotation_spans=[QuotationSpan(4000, 7000, "said", "attributed", 0.55)],
            framing_cues=[FramingCue(4500, "educational", "in this lesson")],
            visual_tracks=[Track("weapon", "weapon", 4000, 7000, 2, 0.9, 0.85)],
            vision_coverage=1.0,
            declared_category="Education",
            declared_audience="general",
        )
        counts = _tag_cross_modal_context([candidate], context)
        assert counts == {"quotation": 1, "framing": 1, "vision": 1, "video": 1}
        assert len(candidate.cross_modal_lines) == 4

    def test_an_empty_context_tags_nothing_and_does_not_crash(self):
        from preflight.agents.triad import CrossModalContext, _tag_cross_modal_context

        candidate = Candidate(
            id="c1", window=0, clause_id="AF-01", category="Language",
            evidence="this is fucked", start_ms=5000, end_ms=6000,
            why="profanity", chunk=chunk("AF-01"),
        )
        counts = _tag_cross_modal_context([candidate], CrossModalContext())
        assert sum(counts.values()) == 0
        assert candidate.cross_modal_lines == []


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
