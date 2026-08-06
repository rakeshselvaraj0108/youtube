"""A05 — OCR.

The extraction is tesseract's job and is not tested here; what is tested is
everything that happens to the extraction afterwards, because that is where a
naive implementation goes wrong.

The load-bearing assertion is the counting one. A caption across forty frames
must produce ONE item. If it produces forty, every count downstream inflates
by the sampler's frame rate rather than by anything about the video, and the
risk score becomes a function of how densely the video was sampled.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from preflight.ingest.frames import Keyframe
from preflight.perception.ocr import (
    DEGRADED_CONFIDENCE,
    MIN_FINDING_CONFIDENCE,
    OcrReport,
    RoleRules,
    TextItem,
    TextSighting,
    analyse,
    box_iou,
    classify_role,
    cluster,
    edit_distance,
    group_lines,
    merge_boxes,
    normalise,
    similar_enough,
    similarity,
    tracks_speech,
)

RULES = RoleRules()
DURATION = 600_000


def sighting(text: str, ts_ms: int, box=(0.1, 0.8, 0.4, 0.05), conf: float = 0.9):
    return TextSighting(text=text, box=box, conf=conf, ts_ms=ts_ms)


def item(text: str, box, start_ms=0, end_ms=3000, conf=0.9, frames=3, tracks=False):
    return TextItem(
        id="ocr_000",
        text=text,
        box=box,
        start_ms=start_ms,
        end_ms=end_ms,
        frames=frames,
        conf=conf,
        tracks_speech=tracks,
    )


class TestGeometry:
    def test_identical_boxes_fully_overlap(self):
        assert box_iou((0.1, 0.1, 0.2, 0.2), (0.1, 0.1, 0.2, 0.2)) == pytest.approx(1.0)

    def test_disjoint_boxes_do_not_overlap(self):
        assert box_iou((0.0, 0.0, 0.1, 0.1), (0.8, 0.8, 0.1, 0.1)) == 0.0

    def test_touching_edges_are_not_an_overlap(self):
        assert box_iou((0.0, 0.0, 0.2, 0.2), (0.2, 0.0, 0.2, 0.2)) == 0.0

    def test_partial_overlap_is_between(self):
        iou = box_iou((0.0, 0.0, 0.2, 0.2), (0.1, 0.0, 0.2, 0.2))
        assert 0.0 < iou < 1.0

    def test_merge_spans_every_box(self):
        merged = merge_boxes([(0.1, 0.1, 0.1, 0.1), (0.5, 0.5, 0.1, 0.1)])
        assert merged == pytest.approx((0.1, 0.1, 0.5, 0.5))


class TestSimilarity:
    def test_identical_strings(self):
        assert similarity("hello world", "hello world") == 1.0

    def test_the_proportional_threshold_alone_would_split_a_noisy_caption(self):
        """Two edits on a thirteen-character line scores 0.846 — under the
        0.85 ratio, which is why the absolute allowance exists."""
        assert similarity("subscribe now", "subscnbe now") == pytest.approx(0.846, abs=0.01)

    def test_two_characters_of_ocr_noise_is_still_a_match(self):
        assert similar_enough("subscribe now", "subscnbe now") is True

    def test_different_strings_are_not(self):
        assert similarity("subscribe now", "buy my course") < 0.5
        assert similar_enough("subscribe now", "buy my course") is False

    def test_the_absolute_allowance_does_not_reach_short_strings(self):
        """At four characters, two edits turns one word into another."""
        assert similar_enough("cat.", "dog.") is False

    def test_edit_distance_counts_operations(self):
        assert edit_distance("kitten", "sitting") == 3
        assert edit_distance("same", "same") == 0
        assert edit_distance("", "abc") == 3

    def test_empty_against_text(self):
        assert similarity("", "text") == 0.0

    def test_normalise_collapses_whitespace_and_case(self):
        assert normalise("  Hello   WORLD ") == "hello world"

    def test_normalise_does_not_strip_punctuation(self):
        """The evidence has to show what the model actually read."""
        assert normalise("What?!") == "what?!"


class TestTemporalDedup:
    """The counting problem."""

    def test_forty_frames_of_one_caption_is_one_item(self):
        sightings = [sighting("Follow for more", ts) for ts in range(0, 40_000, 1000)]
        items = cluster(sightings)
        assert len(items) == 1
        assert items[0].frames == 40

    def test_the_item_spans_first_to_last_sighting(self):
        sightings = [sighting("Follow for more", ts) for ts in range(0, 4000, 1000)]
        merged = cluster(sightings)[0]
        assert merged.start_ms == 0
        assert merged.end_ms == 3000
        assert merged.persist_ms == 3000

    def test_ocr_noise_across_frames_does_not_split_the_item(self):
        sightings = [
            sighting("Subscribe now", 0),
            sighting("Subscnbe now", 1000),
            sighting("Subscribe n0w", 2000),
        ]
        assert len(cluster(sightings)) == 1

    def test_different_text_in_the_same_box_is_two_items(self):
        """A caption template holds still while its words change."""
        sightings = [
            sighting("First caption line", 0),
            sighting("Completely other words", 1000),
        ]
        assert len(cluster(sightings)) == 2

    def test_same_text_in_a_different_corner_is_two_items(self):
        sightings = [
            sighting("BREAKING", 0, box=(0.05, 0.05, 0.2, 0.06)),
            sighting("BREAKING", 1000, box=(0.75, 0.85, 0.2, 0.06)),
        ]
        assert len(cluster(sightings)) == 2

    def test_the_same_title_card_much_later_is_a_second_item(self):
        """Without a temporal gap guard, a card at 0:03 and the same card at
        14:00 become one fourteen-minute element."""
        sightings = [sighting("EPISODE ONE", 3000), sighting("EPISODE ONE", 840_000)]
        assert len(cluster(sightings)) == 2

    def test_the_best_sighting_supplies_the_text(self):
        sightings = [
            sighting("bl_rry text", 0, conf=0.41),
            sighting("blurry text", 1000, conf=0.93),
        ]
        merged = cluster(sightings)[0]
        assert merged.text == "blurry text"
        assert merged.conf == pytest.approx(0.93)

    def test_no_sightings_is_no_items(self):
        assert cluster([]) == []


class TestLineGrouping:
    def test_words_on_one_line_become_one_sighting(self):
        words = [
            {"text": "Follow", "box": (0.10, 0.80, 0.10, 0.05), "conf": 0.9, "line": (1, 1)},
            {"text": "for", "box": (0.21, 0.80, 0.05, 0.05), "conf": 0.8, "line": (1, 1)},
            {"text": "more", "box": (0.27, 0.80, 0.08, 0.05), "conf": 0.95, "line": (1, 1)},
        ]
        sightings = group_lines(words, 5000)
        assert len(sightings) == 1
        assert sightings[0].text == "Follow for more"

    def test_the_weakest_word_caps_the_line(self):
        """A caption is only as trustworthy as the word most likely to be
        misread; averaging lets one confident article carry five guesses."""
        words = [
            {"text": "clear", "box": (0.1, 0.8, 0.1, 0.05), "conf": 0.99, "line": (1, 1)},
            {"text": "gu3ss", "box": (0.2, 0.8, 0.1, 0.05), "conf": 0.31, "line": (1, 1)},
        ]
        assert group_lines(words, 0)[0].conf == pytest.approx(0.31)

    def test_separate_lines_stay_separate(self):
        words = [
            {"text": "top", "box": (0.1, 0.10, 0.1, 0.05), "conf": 0.9, "line": (1, 1)},
            {"text": "bottom", "box": (0.1, 0.85, 0.1, 0.05), "conf": 0.9, "line": (2, 1)},
        ]
        assert len(group_lines(words, 0)) == 2

    def test_empty_words_are_dropped(self):
        words = [{"text": "   ", "box": (0.1, 0.1, 0.1, 0.1), "conf": 0.9, "line": (1, 1)}]
        assert group_lines(words, 0) == []


class TestRoleClassification:
    def test_a_persistent_corner_mark_is_a_watermark(self):
        mark = item("CHANNEL", (0.82, 0.04, 0.14, 0.05), 0, 500_000)
        assert classify_role(mark, DURATION, None, RULES) == "watermark"

    def test_a_brief_corner_mark_is_not_a_watermark(self):
        """Persistence is what separates a watermark from a corner caption."""
        mark = item("CHANNEL", (0.82, 0.04, 0.14, 0.05), 0, 3000)
        assert classify_role(mark, DURATION, None, RULES) != "watermark"

    def test_low_text_that_tracks_speech_is_a_burned_in_caption(self):
        caption = item("we are going to talk about", (0.15, 0.85, 0.6, 0.05), tracks=True)
        assert classify_role(caption, DURATION, None, RULES) == "burned_in_caption"

    def test_low_text_that_does_not_track_speech_is_not_a_caption(self):
        other = item("buy my course", (0.15, 0.85, 0.3, 0.05), tracks=False)
        assert classify_role(other, DURATION, None, RULES) != "burned_in_caption"

    def test_full_width_text_at_the_bottom_is_a_chyron(self):
        chyron = item("LIVE FROM THE SCENE", (0.02, 0.88, 0.96, 0.06))
        assert classify_role(chyron, DURATION, None, RULES) == "chyron"

    def test_a_short_persistent_lower_third_is_a_lower_third(self):
        name = item("Dr Jane Smith", (0.10, 0.75, 0.3, 0.05), 0, 6000)
        assert classify_role(name, DURATION, None, RULES) == "lower_third"

    def test_large_brief_text_at_the_top_is_meme_text(self):
        meme = item("WHEN YOU REALISE", (0.1, 0.05, 0.8, 0.10), 0, 2000)
        assert classify_role(meme, DURATION, None, RULES) == "meme_text"

    def test_unrecognised_placement_is_unclassified_not_guessed(self):
        stray = item("x", (0.45, 0.45, 0.02, 0.02), 0, 1000)
        assert classify_role(stray, DURATION, None, RULES) == "unclassified"

    def test_every_role_is_a_declared_one(self):
        from preflight.perception.ocr import ROLES

        cases = [
            item("CHANNEL", (0.82, 0.04, 0.14, 0.05), 0, 500_000),
            item("LIVE FROM THE SCENE", (0.02, 0.88, 0.96, 0.06)),
            item("WHEN YOU REALISE", (0.1, 0.05, 0.8, 0.10), 0, 2000),
            item("x", (0.45, 0.45, 0.02, 0.02), 0, 1000),
        ]
        for case in cases:
            assert classify_role(case, DURATION, None, RULES) in ROLES

    def test_missing_role_patterns_do_not_crash(self):
        empty = RoleRules(Path("data/lexicons/does_not_exist.json"))
        assert not empty.loaded
        assert classify_role(item("text", (0.4, 0.4, 0.1, 0.05)), DURATION, None, empty)


class TestSpeechCorrelation:
    TRANSCRIPT = {
        "segments": [
            {"start_ms": 0, "end_ms": 4000, "text": "we are going to talk about lenses"},
            {"start_ms": 4000, "end_ms": 8000, "text": "and how they bend light"},
        ]
    }

    def test_a_caption_repeating_the_audio_tracks_speech(self):
        caption = item("we are going to talk about lenses", (0.1, 0.85, 0.6, 0.05), 0, 4000)
        assert tracks_speech(caption, self.TRANSCRIPT) is True

    def test_unrelated_on_screen_text_does_not(self):
        promo = item("link in the description", (0.1, 0.85, 0.4, 0.05), 0, 4000)
        assert tracks_speech(promo, self.TRANSCRIPT) is False

    def test_text_outside_the_spoken_span_does_not(self):
        late = item("we are going to talk about lenses", (0.1, 0.85, 0.6, 0.05), 60_000, 64_000)
        assert tracks_speech(late, self.TRANSCRIPT) is False

    def test_no_transcript_is_not_an_error(self):
        assert tracks_speech(item("anything", (0.1, 0.85, 0.4, 0.05)), None) is False


class TestConfidenceDiscipline:
    def test_low_confidence_text_is_retained_but_not_reportable(self):
        """Retained in artifacts so a reader can see what was read and
        rejected; never a finding on its own."""
        weak = item("m4yb3 w0rds", (0.1, 0.5, 0.3, 0.05), conf=0.40)
        assert weak.reportable is False
        assert weak.to_json()["text"] == "m4yb3 w0rds"

    def test_confident_text_is_reportable(self):
        assert item("clear words", (0.1, 0.5, 0.3, 0.05), conf=0.91).reportable is True

    def test_the_threshold_is_the_boundary(self):
        assert item("t", (0, 0, 0.1, 0.1), conf=MIN_FINDING_CONFIDENCE).reportable is True


class TestReport:
    def test_caption_coverage_counts_only_captions(self):
        report = OcrReport(items=[
            item("spoken words", (0.1, 0.85, 0.5, 0.05), 0, 300_000),
            item("CHANNEL", (0.85, 0.05, 0.1, 0.05), 0, 600_000),
        ])
        report.items[0].role = "burned_in_caption"
        report.items[1].role = "watermark"
        assert report.caption_coverage_ratio(600_000) == pytest.approx(0.5)
        assert report.has_burned_in_captions is True

    def test_coverage_of_a_zero_length_video_is_zero_not_a_crash(self):
        assert OcrReport().caption_coverage_ratio(0) == 0.0

    def test_coverage_never_exceeds_one(self):
        report = OcrReport(items=[item("x", (0.1, 0.85, 0.5, 0.05), 0, 900_000)])
        report.items[0].role = "burned_in_caption"
        assert report.caption_coverage_ratio(600_000) == 1.0


class FakeResult:
    def __init__(self, value, ok=True, reason=""):
        self.value = value
        self.ok = ok
        self.reason = reason
        self.calls = 0


class FakeRegistry:
    """Serves a fixed word list for every frame."""

    def __init__(self, words, ok=True, reason=""):
        self.words = words
        self.ok = ok
        self.reason = reason
        self.invocations = 0

    def invoke(self, capability, **kwargs):
        self.invocations += 1
        if not self.ok:
            return FakeResult(None, ok=False, reason=self.reason)
        return FakeResult({"words": self.words, "text": ""})


def keyframes(count: int, step_ms: int = 1000) -> list[Keyframe]:
    return [
        Keyframe(index=i, ts_ms=i * step_ms, path=Path(f"frame_{i:03d}.jpg"))
        for i in range(count)
    ]


CAPTION_WORDS = [
    {"text": "Follow", "box": (0.10, 0.80, 0.10, 0.05), "conf": 0.92, "line": (1, 1)},
    {"text": "for", "box": (0.21, 0.80, 0.05, 0.05), "conf": 0.90, "line": (1, 1)},
    {"text": "more", "box": (0.27, 0.80, 0.08, 0.05), "conf": 0.94, "line": (1, 1)},
]


class TestAgent:
    def test_one_cluster_produces_one_item_however_many_frames(self):
        """The specification's assertion, made executable."""
        registry = FakeRegistry(CAPTION_WORDS)
        result, report = analyse(keyframes(40), registry, duration_ms=40_000)
        assert result.status == "OK"
        assert report.raw_count == 40
        assert len(report.items) == 1
        assert result.artifacts["deduped_count"] == 1

    def test_no_provider_skips_and_the_run_continues(self):
        result, report = analyse(keyframes(5), None, duration_ms=5000)
        assert result.status == "SKIPPED"
        assert result.coverage == 0.0
        assert report.items == []

    def test_an_unavailable_provider_skips_with_its_reason(self):
        registry = FakeRegistry([], ok=False, reason="tesseract not on PATH")
        result, _ = analyse(keyframes(5), registry, duration_ms=5000)
        assert result.status == "SKIPPED"
        assert "tesseract" in (result.error or "")

    def test_no_keyframes_skips(self):
        result, _ = analyse([], FakeRegistry(CAPTION_WORDS), duration_ms=1000)
        assert result.status == "SKIPPED"

    def test_zero_text_found_is_a_valid_ok_result(self):
        """'This is a valid result, not an error.'"""
        result, report = analyse(keyframes(5), FakeRegistry([]), duration_ms=5000)
        assert result.status == "OK"
        assert report.items == []
        assert result.coverage == 1.0

    def test_uniformly_low_confidence_degrades_rather_than_reporting_guesses(self):
        garbage = [
            {"text": "l1I|", "box": (0.1, 0.5, 0.1, 0.05), "conf": 0.12, "line": (1, 1)}
        ]
        result, _ = analyse(keyframes(4), FakeRegistry(garbage), duration_ms=4000)
        assert result.status == "DEGRADED"
        assert result.coverage <= 0.5
        assert any("low OCR confidence" in line for line in result.log)

    def test_coverage_is_the_fraction_of_frames_actually_read(self):
        registry = FakeRegistry(CAPTION_WORDS)
        result, _ = analyse(keyframes(10), registry, duration_ms=10_000, budget=4)
        assert result.coverage == pytest.approx(0.4)

    def test_artifacts_carry_the_role_census(self):
        registry = FakeRegistry(CAPTION_WORDS)
        result, _ = analyse(keyframes(6), registry, duration_ms=600_000)
        assert "roles" in result.artifacts
        assert sum(result.artifacts["roles"].values()) == result.artifacts["deduped_count"]

    def test_evidence_is_the_ocr_output_verbatim_including_its_errors(self):
        """A downstream reader must be able to see what the model actually
        read, so the text is never cleaned up."""
        misread = [
            {"text": "Sub5cnbe", "box": (0.1, 0.8, 0.2, 0.05), "conf": 0.62, "line": (1, 1)}
        ]
        _, report = analyse(keyframes(3), FakeRegistry(misread), duration_ms=3000)
        assert report.items[0].text == "Sub5cnbe"

    def test_never_emits_a_verdict(self):
        registry = FakeRegistry(CAPTION_WORDS)
        result, _ = analyse(keyframes(4), registry, duration_ms=4000)
        blob = str(result.artifacts).upper()
        for forbidden in ("VIOLATION", "UNSAFE", "DEMONETIZ", "COPYRIGHT CLAIM"):
            assert forbidden not in blob

    def test_degraded_threshold_is_below_the_reporting_threshold(self):
        """Text can be too weak to report while still being a clean enough
        read to trust the run."""
        assert DEGRADED_CONFIDENCE < MIN_FINDING_CONFIDENCE
