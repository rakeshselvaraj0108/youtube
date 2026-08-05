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
        result, events = analyse(None)
        assert result.status == "SKIPPED"
        assert result.error == "Transcript unavailable"
        assert events == []

    def test_empty_transcript_is_success_with_no_events(self):
        result, events = analyse(Transcript(language="en", duration_ms=0))
        assert result.status == "OK"
        assert events == []

    def test_reports_lexicon_load_in_the_log(self):
        result, _ = analyse(transcript("hello everyone"))
        assert any("lexicons" in line for line in result.log)

    def test_output_is_json_only(self):
        _, events = analyse(transcript("this is bullshit"))
        payload = to_json(events)
        assert set(payload) == {"speech_events"}
        assert isinstance(payload["speech_events"], list)

    def test_never_emits_a_verdict(self):
        """'Never write Limited Ads. Never write Violation. Never write
        Unsafe.' Those belong to later agents."""
        _, events = analyse(
            transcript(
                "this is bullshit and someone got shot and the casino paid out"
            )
        )
        blob = str(to_json(events)).upper()
        for forbidden in ("LIMITED ADS", "VIOLATION", "UNSAFE", "DEMONETIZ"):
            assert forbidden not in blob

    def test_every_event_is_timestamped_and_scored(self):
        _, events = analyse(transcript("this is bullshit and there was a knife"))
        assert events
        for event in events:
            assert event.end_ms > event.start_ms
            assert 0.0 < event.confidence <= 1.0
            assert event.severity in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


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
