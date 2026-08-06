"""A02 — speech intelligence.

The tests that matter are the false-positive ones. Any word list finds "fuck";
the question is whether it also finds "bass", "assessment", "Cocaine Bear" and
a slur being quoted in order to condemn it. Those are the four ways a naive
matcher reaches a 90% false-positive rate, and each has a test here.
"""

from __future__ import annotations

import pytest

from preflight.perception.asr import Segment, Transcript, Word
from preflight.perception.speech_intel import (
    Lexicons,
    analyse,
    collapse_spaced_letters,
    extract_events,
    normalize,
    to_json,
)

LEXICONS = Lexicons()


def transcript(text: str, *, start_ms: int = 1000, gap_ms: int = 400) -> Transcript:
    """Build a transcript with real word-level timings from a sentence."""
    words = []
    cursor = start_ms
    for token in text.split():
        words.append(Word(w=token, start_ms=cursor, end_ms=cursor + 300, conf=0.95))
        cursor += gap_ms
    return Transcript(
        language="en",
        duration_ms=cursor + 1000,
        words=words,
        segments=[Segment(start_ms=start_ms, end_ms=cursor, text=text)],
    )


def events_of(text: str, event_type: str | None = None):
    found = extract_events(transcript(text), LEXICONS)
    return [e for e in found if event_type is None or e.type == event_type]


class TestNormalisation:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Fuck", "fuck"),
            ("f*ck", "fck"),
            ("f**k", "fk"),
            ("SH1T", "sh1t"),
            ("$hit", "shit"),
            ("d@mn", "damn"),
            ("h3ll", "hell"),
            ("fuuuuck", "fuuck"),
            ("word,", "word"),
        ],
    )
    def test_canonicalises(self, raw, expected):
        assert normalize(raw) == expected

    def test_does_not_map_one_to_i_or_l(self):
        """That reading turns 1nformation into a hit. Cleverness that
        manufactures false positives is worse than no cleverness."""
        assert normalize("1nformation") == "1nformation"

    def test_repeat_collapse_stops_at_two(self):
        """Collapsing to one would map hell to hel and boo to bo."""
        assert normalize("hell") == "hell"
        assert normalize("boo") == "boo"

    def test_collapses_spaced_letters(self):
        assert collapse_spaced_letters("that is f u c k right there") == (
            "that is fuck right there"
        )

    def test_leaves_ordinary_short_words_alone(self):
        """A two-letter run is ordinary English; collapsing it costs more than
        it catches."""
        assert collapse_spaced_letters("a smart idea") == "a smart idea"


class TestScunthorpeProblem:
    """The single most important test class in this file."""

    @pytest.mark.parametrize(
        "sentence",
        [
            "the bass guitar sounds great",
            "we ran a full assessment yesterday",
            "the class was cancelled",
            "pass me the grass clippings",
            "he is an assassin in the game",
            "the shitake mushrooms were good",
            "cocktail hour starts at six",
        ],
    )
    def test_substrings_are_not_matches(self, sentence):
        """Substring matching flags `bass` for containing `ass`. Whole-token
        matching cannot, by construction."""
        assert events_of(sentence, "PROFANITY") == []

    def test_the_real_word_still_matches(self):
        assert len(events_of("that is complete bullshit honestly", "PROFANITY")) == 1

    def test_hell_matches_but_hello_does_not(self):
        assert len(events_of("what the hell was that", "PROFANITY")) == 1
        assert events_of("hello everyone welcome back", "PROFANITY") == []

    def test_shoot_in_basketball_is_recorded_not_judged(self):
        """A02 records the term. Whether a basketball video breaches AF-02 is
        A11's call, and A02 must not pre-empt it in either direction."""
        found = events_of("he can really shoot from range", "VIOLENCE")
        assert len(found) == 1
        assert found[0].severity in {"LOW", "MEDIUM", "HIGH"}
        assert "VIOLATION" not in found[0].type


class TestObfuscation:
    @pytest.mark.parametrize(
        "sentence",
        [
            "this is f*ck right here",
            "this is fuk right here",
            "this is phuck right here",
        ],
    )
    def test_declared_variants_match(self, sentence):
        assert len(events_of(sentence, "PROFANITY")) == 1

    def test_undeclared_obfuscation_is_missed_and_that_is_correct(self):
        """Generating variants algorithmically catches more and invents more.
        Missing an exotic spelling is a recall cost; inventing a hit on
        ordinary text is a credibility cost, and only one of those is
        recoverable downstream."""
        assert events_of("this is fvck right here", "PROFANITY") == []


class TestAttributionContext:
    """A02 does not rule on exemptions. It records the signals that let the
    ADVOCATE argue one from the clause instead of inventing it."""

    def test_marks_a_quoted_span(self):
        found = events_of("he said quote this is bullshit unquote to the room")
        profanity = [e for e in found if e.type == "PROFANITY"]
        assert profanity
        assert profanity[0].context.get("in_quotation") is True

    def test_quotation_demotes_confidence(self):
        asserted = events_of("this is complete bullshit", "PROFANITY")[0]
        quoted = events_of("she said quote this is complete bullshit unquote", "PROFANITY")[0]
        assert quoted.confidence < asserted.confidence

    def test_marks_condemnation_separately_from_quotation(self):
        found = events_of(
            "he said that is bullshit which is frankly disgusting behaviour",
            "PROFANITY",
        )
        assert found[0].context.get("condemned") is True

    def test_marks_negation(self):
        found = events_of("I would never say bullshit on this channel", "PROFANITY")
        assert found[0].context.get("negated") is True

    def test_marks_a_nearby_harm_reduction_warning(self):
        found = events_of(
            "never do this it is dangerous you could get a serious wound",
            "VIOLENCE",
        )
        assert any(e.context.get("harm_reduction_nearby") for e in found)

    def test_plain_assertion_carries_no_exempting_context(self):
        found = events_of("that is complete bullshit", "PROFANITY")[0]
        assert not found.context.get("in_quotation")
        assert not found.context.get("negated")
        assert found.confidence == pytest.approx(0.95)

    def test_a_condemnation_phrase_stretched_too_thin_does_not_match(self):
        """The gap tolerance is bounded — two unrelated clauses sharing one
        word each side of 'disgusting' is not the lexicon's phrase."""
        found = events_of(
            "that is bullshit and which is a totally different thing that "
            "happens to also be disgusting somehow",
            "PROFANITY",
        )
        assert not found[0].context.get("condemned")

    def test_past_tense_quoted_is_a_reporting_verb(self):
        """The old hardcoded set had 'quote' and 'quoted'. Switching to the
        lexicon file first lost 'quoted' — the file only had 'quote' as a
        marker, not as a verb form — which silently stopped catching 'she
        quoted him saying bullshit'. Fixed in the lexicon, not patched around
        in code, since the file is the source now."""
        found = events_of("he quoted the article saying that is bullshit", "PROFANITY")
        assert found[0].context.get("in_quotation") is True

    def test_a_quote_marker_phrase_from_the_lexicon_is_recognised(self):
        """'in his words' is only in the JSON lexicon, never in the old
        hardcoded set — proof the file drives behaviour now."""
        found = events_of(
            "in his words this is complete bullshit and nothing more", "PROFANITY"
        )
        assert found[0].context.get("in_quotation") is True


class TestQuotationSpans:
    """Standalone spans over the transcript, independent of any lexicon hit —
    the ADVOCATE needs this for candidates the AUDITOR raised from clause
    text alone, not only for A02's own findings."""

    def test_a_span_exists_even_with_no_lexicon_hit_inside_it(self):
        """The AUDITOR can charge a candidate under a clause A02's own
        lexicons never touch. The span still has to be there."""
        from preflight.perception.speech_intel import quotation_spans

        spans = quotation_spans(
            transcript("she said quote the plan is completely reckless unquote")
        )
        assert spans
        assert spans[0].kind == "attributed"

    def test_condemnation_after_the_quote_upgrades_the_kind(self):
        """A sentence terminator on 'unquote.' is what tells the span where
        the quoted material ends — the same signal a punctuated ASR
        transcript would carry — so the condemnation that follows reads as
        the speaker's own words, not part of what was quoted."""
        from preflight.perception.speech_intel import quotation_spans

        spans = quotation_spans(
            transcript(
                "he said quote this is fine unquote. which is disgusting and wrong"
            )
        )
        assert spans
        assert spans[0].kind == "attributed_and_condemned"
        assert spans[0].strength > 0.55

    def test_a_condemned_span_scores_higher_than_a_bare_one(self):
        from preflight.perception.speech_intel import quotation_spans

        bare = quotation_spans(transcript("he said quote this is fine unquote."))[0]
        condemned = quotation_spans(
            transcript("he said quote this is fine unquote. which is disgusting")
        )[0]
        assert condemned.strength > bare.strength

    def test_span_covers_the_quoted_material_not_the_cue_itself(self):
        from preflight.perception.speech_intel import quotation_spans

        text = "he said quote this is completely reckless unquote to everyone"
        spans = quotation_spans(transcript(text))
        assert spans
        span = spans[0]
        words = text.split()
        cue_end_ms = 1000 + words.index("said") * 400 + 300
        assert span.start_ms >= cue_end_ms

    def test_no_cue_yields_no_span(self):
        from preflight.perception.speech_intel import quotation_spans

        assert quotation_spans(transcript("the weather today is quite pleasant")) == []

    def test_none_transcript_yields_no_span(self):
        from preflight.perception.speech_intel import quotation_spans

        assert quotation_spans(None) == []

    def test_span_length_is_capped_by_max_quote_span_ms(self):
        """One missing sentence terminator must not swallow the rest of the
        file into a single 'quotation'."""
        from preflight.perception.speech_intel import AttributionCues, quotation_spans

        long_text = "he said quote " + " ".join(f"word{i}" for i in range(200))
        cues = AttributionCues()
        spans = quotation_spans(transcript(long_text), cues)
        assert spans
        assert spans[0].end_ms - spans[0].start_ms <= cues.max_quote_span_ms

    def test_two_cues_in_one_sentence_merge_into_one_span(self):
        """'She said, and I quote,' must not double-count as two pieces of
        corroborating evidence for the same quotation."""
        from preflight.perception.speech_intel import quotation_spans

        spans = quotation_spans(
            transcript("she said and i quote this plan is reckless unquote")
        )
        assert len(spans) == 1

    def test_overlaps_is_the_span_membership_test_the_triad_uses(self):
        from preflight.perception.speech_intel import QuotationSpan

        span = QuotationSpan(1000, 5000, "said", "attributed", 0.55)
        assert span.overlaps(2000, 3000) is True
        assert span.overlaps(6000, 7000) is False
        assert span.overlaps(4500, 6000) is True

    def test_to_json_round_trips_every_field(self):
        from preflight.perception.speech_intel import QuotationSpan

        span = QuotationSpan(1000, 5000, "said", "attributed_and_condemned", 0.75)
        payload = span.to_json()
        assert payload == {
            "start_ms": 1000, "end_ms": 5000, "cue": "said",
            "kind": "attributed_and_condemned", "strength": 0.75,
        }


class TestFramingCues:
    """EDSA framing and harm-reduction language, timestamped — the other half
    of the cross-modal brief. `data/lexicons/edsa_framing_cues.json` existed
    since the data layer was authored and nothing loaded it, same as
    attribution_cues.json before it was wired in."""

    def test_an_educational_cue_is_found(self):
        from preflight.perception.speech_intel import find_framing_cues

        cues = find_framing_cues(transcript("in this lesson we cover register shifts"))
        assert any(c.category == "educational" for c in cues)

    def test_a_harm_reduction_cue_is_found(self):
        from preflight.perception.speech_intel import find_framing_cues

        cues = find_framing_cues(transcript("never do this it is extremely dangerous"))
        assert any(c.category == "harm_reduction" for c in cues)

    def test_no_cues_in_ordinary_text(self):
        from preflight.perception.speech_intel import find_framing_cues

        assert find_framing_cues(transcript("the weather today is quite pleasant")) == []

    def test_none_transcript_yields_no_cues(self):
        from preflight.perception.speech_intel import find_framing_cues

        assert find_framing_cues(None) == []

    def test_one_occurrence_is_not_reported_three_times(self):
        """The tolerant matcher searches forward from wherever it starts, so
        naively scanning every word position found 'this is dangerous' three
        times in 'never do this it is extremely dangerous' — once starting
        from 'do', again from 'this', again from 'it', all within the same
        tolerance window as the real occurrence. Advancing past a match once
        found is what fixes it."""
        from preflight.perception.speech_intel import find_framing_cues

        cues = find_framing_cues(
            transcript("never do this it is extremely dangerous and risky")
        )
        assert len(cues) == 1

    def test_edsa_categories_near_finds_cues_inside_the_window(self):
        from preflight.perception.speech_intel import (
            edsa_categories_near,
            find_framing_cues,
        )

        cues = find_framing_cues(
            transcript("in this lesson we will bypass the safety cutout")
        )
        near = edsa_categories_near(cues, 1000, 2000, window_ms=8000)
        assert "educational" in near

    def test_edsa_categories_near_excludes_cues_outside_the_window(self):
        from preflight.perception.speech_intel import (
            edsa_categories_near,
            find_framing_cues,
        )

        cues = find_framing_cues(
            transcript("in this lesson we will bypass the safety cutout")
        )
        near = edsa_categories_near(cues, 500_000, 501_000, window_ms=8000)
        assert near == []

    def test_edsa_categories_near_never_returns_harm_reduction(self):
        """harm_reduction is its own signal, scored by distance rather than
        presence-in-a-window — it must not leak into the EDSA category list."""
        from preflight.perception.speech_intel import (
            edsa_categories_near,
            find_framing_cues,
        )

        cues = find_framing_cues(transcript("never do this it is dangerous"))
        assert "harm_reduction" not in edsa_categories_near(cues, 0, 100_000, window_ms=999_999)

    def test_harm_reduction_distance_measures_proximity(self):
        from preflight.perception.speech_intel import (
            find_framing_cues,
            harm_reduction_distance_ms,
        )

        cues = find_framing_cues(
            transcript("never do this it is dangerous you could get seriously hurt")
        )
        near = harm_reduction_distance_ms(cues, 0, 1000)
        far = harm_reduction_distance_ms(cues, 500_000, 501_000)
        assert near is not None
        assert far is None

    def test_a_missing_lexicon_degrades_without_crashing(self, tmp_path):
        from preflight.perception.speech_intel import FramingCues, find_framing_cues

        cues = FramingCues(tmp_path / "does_not_exist.json")
        assert not cues.loaded
        assert find_framing_cues(transcript("in this lesson"), cues) == []


class TestAttributionCuesLexicon:
    """The lexicon file, not a hardcoded duplicate, is what `_context_for`
    reads. `data/lexicons/attribution_cues.json` existed since the data layer
    was authored and was never loaded by anything."""

    def test_the_lexicon_file_loads(self):
        from preflight.perception.speech_intel import AttributionCues

        cues = AttributionCues()
        assert cues.loaded
        assert "said" in cues.reporting_verbs
        assert any("unquote" in m for m in cues.quote_markers)
        assert cues.condemnation_markers

    def test_a_missing_lexicon_degrades_without_crashing(self, tmp_path):
        from preflight.perception.speech_intel import AttributionCues

        cues = AttributionCues(tmp_path / "does_not_exist.json")
        assert not cues.loaded
        assert cues.reporting_verbs == set()


class TestPhrases:
    def test_detects_sponsorship(self):
        found = events_of("this video is sponsored by a company I like", "SPONSOR")
        assert found
        assert "sponsored by" in found[0].matched

    def test_longest_phrase_wins(self):
        """Reporting both `sponsored by` and `this video is sponsored by` would
        double-count one utterance."""
        found = events_of("this video is sponsored by them", "SPONSOR")
        assert len(found) == 1
        assert found[0].matched == "this video is sponsored by"

    def test_financial_claim_needs_the_guarantee_not_the_noun(self):
        """`profit` and `return` are ordinary business vocabulary."""
        assert events_of("our profit margin improved this quarter", "FINANCIAL_CLAIM") == []
        assert events_of("I can double your money in a week", "FINANCIAL_CLAIM")

    def test_medical_claim_needs_the_assertion_not_the_condition(self):
        assert events_of("my uncle is recovering from cancer", "MEDICAL_CLAIM") == []
        assert events_of("this supplement cures cancer completely", "MEDICAL_CLAIM")

    def test_ai_disclosure_is_recorded_as_a_positive_signal(self):
        found = events_of("the narration was generated by ai for this video")
        assert any(e.type == "AI_DISCLOSURE" for e in found)


class TestPersonalInfo:
    def test_detects_an_email(self):
        found = events_of("write to me at hello@example.com any time", "PERSONAL_INFO")
        assert found
        assert found[0].matched == "EMAIL"

    def test_never_reproduces_the_value(self):
        """Copying a card number into a report warning about a card number in
        a report is not a defensible trade."""
        found = events_of("email me at secret@example.com", "PERSONAL_INFO")[0]
        assert "secret@example.com" not in found.text
        assert "redacted" in found.text

    def test_card_requires_a_valid_checksum(self):
        """Without Luhn, any sixteen-digit run is a card — a timestamp, an
        order number, a phone number with the spaces removed."""
        valid = events_of("the number is 4539578763621486 exactly", "PERSONAL_INFO")
        invalid = events_of("the number is 1234567812345678 exactly", "PERSONAL_INFO")
        assert any(e.matched == "CARD" for e in valid)
        assert not any(e.matched == "CARD" for e in invalid)

    def test_detects_a_phone_number(self):
        found = events_of("call me on 415 555 2671 tomorrow", "PERSONAL_INFO")
        assert any(e.matched == "PHONE" for e in found)


class TestSensitiveEvents:
    def test_detects_a_casualty_figure(self):
        found = events_of("that morning 4 people died on the mountain", "SENSITIVE_EVENT")
        assert found

    def test_detects_death_toll(self):
        assert events_of("the death toll rose overnight", "SENSITIVE_EVENT")

    def test_ordinary_numbers_are_not_casualties(self):
        assert events_of("4 people joined the call this morning", "SENSITIVE_EVENT") == []


class TestSpansAndDedupe:
    def test_span_comes_from_real_word_timings(self):
        source = transcript("this is complete bullshit honestly")
        found = extract_events(source, LEXICONS)[0]
        target = next(w for w in source.words if "bullshit" in w.w)
        assert found.start_ms == target.start_ms
        assert found.end_ms == target.end_ms

    def test_no_duplicate_events_for_one_utterance(self):
        found = events_of("this is bullshit", "PROFANITY")
        assert len(found) == 1

    def test_two_separate_utterances_are_two_events(self):
        found = events_of("bullshit and later also damn right", "PROFANITY")
        assert len(found) == 2

    def test_events_are_ordered_by_time(self):
        found = extract_events(
            transcript("damn this is bullshit and the casino was open"), LEXICONS
        )
        assert [e.start_ms for e in found] == sorted(e.start_ms for e in found)


class TestAgentContract:
    def test_missing_transcript_reports_the_specified_failure(self):
        result, events, spans, framing = analyse(None)
        assert result.status == "SKIPPED"
        assert result.error == "Transcript unavailable"
        assert events == []
        assert spans == []
        assert framing == []

    def test_empty_transcript_is_success_with_no_events(self):
        result, events, spans, framing = analyse(Transcript(language="en", duration_ms=0))
        assert result.status == "OK"
        assert events == []
        assert spans == []
        assert framing == []

    def test_reports_lexicon_load_in_the_log(self):
        result, _, _, _ = analyse(transcript("hello everyone"))
        assert any("lexicons" in line for line in result.log)

    def test_output_is_json_only(self):
        _, events, spans, framing = analyse(transcript("this is bullshit"))
        payload = to_json(events, spans, framing)
        assert set(payload) == {"speech_events", "quotation_spans", "framing_cues"}
        assert isinstance(payload["speech_events"], list)
        assert isinstance(payload["quotation_spans"], list)
        assert isinstance(payload["framing_cues"], list)

    def test_never_emits_a_verdict(self):
        """'Never write Limited Ads. Never write Violation. Never write
        Unsafe.' Those belong to later agents."""
        _, events, spans, framing = analyse(
            transcript(
                "this is bullshit and someone got shot and the casino paid out"
            )
        )
        blob = str(to_json(events, spans, framing)).upper()
        for forbidden in ("LIMITED ADS", "VIOLATION", "UNSAFE", "DEMONETIZ"):
            assert forbidden not in blob

    def test_every_event_is_timestamped_and_scored(self):
        _, events, _, _ = analyse(transcript("this is bullshit and there was a knife"))
        assert events
        for event in events:
            assert event.end_ms > event.start_ms
            assert 0.0 < event.confidence <= 1.0
            assert event.severity in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

    def test_quotation_spans_are_reported_in_the_artifacts_and_the_log(self):
        result, _, spans, _ = analyse(
            transcript("he said quote this is bullshit unquote to the room")
        )
        assert spans
        assert result.artifacts["quotation_spans"]
        assert any("quotation span" in line for line in result.log)

    def test_framing_cues_are_reported_in_the_artifacts_and_the_log(self):
        result, _, _, framing = analyse(
            transcript("in this lesson we will bypass the safety cutout")
        )
        assert framing
        assert result.artifacts["framing_cues"]
        assert any("framing cues" in line for line in result.log)


class TestLexicons:
    def test_loads_every_file(self):
        assert len(LEXICONS.loaded) >= 12
        assert LEXICONS.term_count > 80

    def test_hate_ships_no_slur_list(self):
        """A slur list in a public repository is a liability with no upside,
        and a list cannot tell use from mention."""
        import json
        from pathlib import Path

        payload = json.loads(
            (Path("data/lexicons/speech/hate.json")).read_text(encoding="utf-8")
        )
        assert payload["terms"] == []
        assert payload["phrases"]

    def test_every_lexicon_states_it_is_not_a_verdict(self):
        import json
        from pathlib import Path

        for path in Path("data/lexicons/speech").glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert "never a verdict" in payload["_note"].lower(), path.name
