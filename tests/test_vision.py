"""A03 — vision intelligence.

The tests worth reading are the containment ones. Any wrapper around a vision
model returns labels; the question is what happens when the model returns
`graphic violence`, invents an object that is not there, or names the same
knife four different ways across four frames. Each of those has a test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from preflight.ingest.frames import Keyframe
from preflight.perception.vision import (
    EMOTION_CEILING,
    SINGLETON_CEILING,
    Observation,
    VisionVocabulary,
    analyse,
    batches,
    build_tracks,
    parse_observations,
    select_frames,
    to_json,
)

VOCAB = VisionVocabulary()


def frame(index: int, ts_ms: int) -> Keyframe:
    return Keyframe(index=index, ts_ms=ts_ms, path=Path(f"f{index:05d}.jpg"))


def obs(label: str, ts_ms: int, confidence: float = 0.9, category: str = "weapon"):
    return Observation(label=label, category=category, confidence=confidence, ts_ms=ts_ms)


class TestVocabularyContainment:
    """The closed vocabulary is what stops a verdict entering as an observation."""

    @pytest.mark.parametrize(
        "label",
        [
            "graphic violence",
            "inappropriate content",
            "unsafe imagery",
            "policy violation",
            "explicit material",
            "nsfw content",
            "age-restricted scene",
            "shocking imagery",
        ],
    )
    def test_judgement_labels_are_rejected_outright(self, label):
        """There is no observation underneath to recover — the model skipped
        past what it can see to what it thinks the answer is."""
        assert VOCAB.is_judgment(label)
        assert VOCAB.normalise(label) is None

    def test_a_judgement_is_never_fuzzily_matched_to_something_plausible(self):
        """`graphic violence` contains no known label, but the check runs
        first so a future vocabulary addition cannot accidentally admit it."""
        assert VOCAB.normalise("graphic violence") is None

    def test_an_invented_label_is_dropped(self):
        assert VOCAB.normalise("interdimensional portal") is None

    def test_empty_and_whitespace_are_dropped(self):
        assert VOCAB.normalise("") is None
        assert VOCAB.normalise("   ") is None


class TestNormalisation:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("handgun", "gun"),
            ("pistol", "gun"),
            ("firearm", "gun"),
            ("assault rifle", "rifle"),
            ("machete", "knife"),
            ("blade", "knife"),
            ("bleeding", "blood"),
            ("motorbike", "motorcycle"),
            ("wine glass", "wine"),
            ("banknotes", "cash"),
        ],
    )
    def test_synonyms_collapse_to_one_canonical_label(self, raw, expected):
        """Four names for one object means four tracks and no deduplication."""
        resolved = VOCAB.normalise(raw)
        assert resolved is not None
        assert resolved[0] == expected

    def test_category_travels_with_the_label(self):
        assert VOCAB.normalise("knife")[1] == "weapon"
        assert VOCAB.normalise("blood")[1] == "injury"
        assert VOCAB.normalise("bedroom")[1] == "scene"

    def test_case_and_underscores_are_tolerated(self):
        assert VOCAB.normalise("KNIFE")[0] == "knife"
        assert VOCAB.normalise("police_car")[0] == "police_car"
        assert VOCAB.normalise("Police Car")[0] == "police_car"

    @pytest.mark.parametrize(
        "compound,expected_category",
        [
            ("person holding knife", "weapon"),
            ("man with a gun in the street", "weapon"),
            ("person with visible blood on arm", "injury"),
            ("woman holding a wine glass", "alcohol"),
        ],
    )
    def test_a_compound_yields_its_most_salient_label(self, compound, expected_category):
        """`person holding knife` contains both `person` and `knife`. Picking
        by string length returns `person`, because it is one character longer
        — which is exactly the wrong observation to surface."""
        resolved = VOCAB.normalise(compound)
        assert resolved is not None
        assert resolved[1] == expected_category

    def test_a_compound_with_only_a_generic_label_still_resolves(self):
        resolved = VOCAB.normalise("a person standing in a field")
        assert resolved is not None
        assert resolved[0] == "person"

    def test_injury_is_separate_from_weapons(self):
        """An injury with no weapon is AF-04, not AF-02. Conflating them sends
        the wrong clause to the adjudicator."""
        assert VOCAB.normalise("blood")[1] != VOCAB.normalise("knife")[1]


class TestResponseParsing:
    def test_accepts_the_documented_shape(self):
        payload = {"objects": [{"label": "knife", "confidence": 0.97}]}
        parsed, rejected = parse_observations(payload, frame(0, 1000), VOCAB)
        assert len(parsed) == 1
        assert parsed[0].label == "knife"
        assert rejected == []

    @pytest.mark.parametrize("key", ["objects", "observations", "labels", "detections"])
    def test_tolerates_the_wrapper_key_the_model_chose(self, key):
        parsed, _ = parse_observations(
            {key: [{"label": "gun", "confidence": 0.9}]}, frame(0, 1000), VOCAB
        )
        assert len(parsed) == 1

    def test_tolerates_a_bare_list(self):
        parsed, _ = parse_observations(
            [{"label": "knife", "confidence": 0.9}], frame(0, 1000), VOCAB
        )
        assert len(parsed) == 1

    def test_records_why_a_label_was_rejected(self):
        payload = {"objects": [
            {"label": "graphic violence", "confidence": 0.99},
            {"label": "knife", "confidence": 0.9},
        ]}
        parsed, rejected = parse_observations(payload, frame(0, 1000), VOCAB)
        assert len(parsed) == 1
        assert any("judgement" in r for r in rejected)

    def test_a_missing_confidence_becomes_zero_not_one(self):
        """Assuming certainty from an absent field is the wrong direction."""
        parsed, _ = parse_observations(
            {"objects": [{"label": "knife"}]}, frame(0, 1000), VOCAB
        )
        assert parsed[0].confidence == 0.0

    def test_a_nonsense_confidence_is_clamped(self):
        parsed, _ = parse_observations(
            {"objects": [{"label": "knife", "confidence": 7.4}]}, frame(0, 1000), VOCAB
        )
        assert parsed[0].confidence == 1.0

    def test_a_malformed_bbox_is_dropped_without_dropping_the_observation(self):
        parsed, _ = parse_observations(
            {"objects": [{"label": "knife", "confidence": 0.9, "bbox": [1, 2]}]},
            frame(0, 1000), VOCAB,
        )
        assert len(parsed) == 1
        assert parsed[0].bbox is None

    def test_garbage_yields_nothing_rather_than_raising(self):
        for payload in ("not json", 42, None, {}, []):
            parsed, _ = parse_observations(payload, frame(0, 1000), VOCAB)
            assert parsed == []

    def test_emotion_is_capped_at_parse_time(self):
        parsed, _ = parse_observations(
            {"objects": [{"label": "happy", "confidence": 0.99}]}, frame(0, 1000), VOCAB
        )
        assert parsed[0].confidence <= EMOTION_CEILING


class TestTemporalTracks:
    def test_one_object_across_many_frames_is_one_track(self):
        """A knife visible for eight seconds spans hundreds of frames.
        Reported per frame, every count downstream inflates."""
        observations = [obs("knife", ts) for ts in range(1000, 9001, 500)]
        tracks = build_tracks(observations)
        assert len(tracks) == 1
        assert tracks[0].frames == 17
        assert tracks[0].start_ms == 1000
        assert tracks[0].end_ms == 9000

    def test_a_gap_splits_one_label_into_two_appearances(self):
        observations = [obs("knife", 1000), obs("knife", 1500), obs("knife", 30_000)]
        tracks = build_tracks(observations)
        assert len(tracks) == 2

    def test_different_labels_never_merge(self):
        tracks = build_tracks([obs("knife", 1000), obs("gun", 1200)])
        assert {t.label for t in tracks} == {"knife", "gun"}

    def test_tracks_are_ordered_by_time(self):
        tracks = build_tracks([obs("gun", 5000), obs("knife", 1000)])
        assert [t.start_ms for t in tracks] == [1000, 5000]


class TestPersistenceAsCorroboration:
    """A single frame is where hallucination lives. A thing that is actually in
    the shot stays in the shot."""

    def test_a_singleton_cannot_reach_the_top_band(self):
        track = build_tracks([obs("knife", 1000, confidence=0.99)])[0]
        assert track.confidence <= SINGLETON_CEILING
        assert track.band != "VERY_HIGH"
        assert track.corroborated is False

    def test_persistence_raises_a_modest_detection_above_a_confident_singleton(self):
        singleton = build_tracks([obs("gun", 1000, confidence=0.94)])[0]
        persistent = build_tracks(
            [obs("knife", 1000 + i * 500, confidence=0.70) for i in range(5)]
        )[0]
        assert persistent.confidence > singleton.confidence
        assert persistent.corroborated is True

    def test_the_bonus_is_capped(self):
        """A long static shot must not manufacture certainty on its own."""
        long_run = build_tracks(
            [obs("knife", 1000 + i * 400, confidence=0.70) for i in range(40)]
        )[0]
        assert long_run.confidence <= 0.70 + 0.16 + 1e-9

    def test_peak_sets_the_floor_so_one_weak_frame_does_not_punish_a_real_object(self):
        """The mean would drag a genuine detection down for a single blurred
        frame; the max alone would let one confident hallucination through."""
        track = build_tracks([
            obs("knife", 1000, confidence=0.92),
            obs("knife", 1500, confidence=0.30),
            obs("knife", 2000, confidence=0.88),
        ])[0]
        assert track.confidence >= 0.92
        assert track.peak_confidence == 0.92

    def test_emotion_stays_capped_however_persistent(self):
        track = build_tracks(
            [obs("happy", 1000 + i * 400, 0.99, "emotion") for i in range(10)]
        )[0]
        assert track.confidence <= EMOTION_CEILING


class TestFrameGating:
    def test_flagged_spans_are_always_inspected(self):
        frames = [frame(i, i * 1000) for i in range(40)]
        selected = select_frames(frames, [(10_000, 14_000)], baseline=0)
        assert {f.ts_ms for f in selected} == {10_000, 11_000, 12_000, 13_000, 14_000}

    def test_a_baseline_sample_runs_even_with_no_flagged_spans(self):
        """A clean transcript with a visual problem must not be invisible."""
        frames = [frame(i, i * 1000) for i in range(80)]
        selected = select_frames(frames, [], baseline=8)
        assert 1 <= len(selected) <= 10

    def test_gating_is_a_large_reduction(self):
        frames = [frame(i, i * 1000) for i in range(90)]
        selected = select_frames(frames, [(5_000, 8_000)], baseline=8)
        assert len(selected) < len(frames) * 0.3

    def test_no_frame_is_sent_twice(self):
        frames = [frame(i, i * 1000) for i in range(40)]
        selected = select_frames(frames, [(0, 40_000)], baseline=8)
        assert len({f.ts_ms for f in selected}) == len(selected)

    def test_selection_is_ordered(self):
        frames = [frame(i, i * 1000) for i in range(30)]
        stamps = [f.ts_ms for f in select_frames(frames, [(20_000, 22_000)], baseline=4)]
        assert stamps == sorted(stamps)

    def test_budget_is_respected(self):
        frames = [frame(i, i * 1000) for i in range(60)]
        assert len(select_frames(frames, [(0, 60_000)], budget=12)) == 12

    def test_no_keyframes_selects_nothing(self):
        assert select_frames([], [(0, 1000)]) == []

    def test_batches_of_five(self):
        frames = [frame(i, i * 1000) for i in range(12)]
        sizes = [len(b) for b in batches(frames, 5)]
        assert sizes == [5, 5, 2]


class TestAgentContract:
    def test_no_keyframes_reports_skipped(self):
        result, tracks = analyse([], registry=object())
        assert result.status == "SKIPPED"
        assert tracks == []

    def test_no_provider_degrades_rather_than_failing(self):
        """Vision is optional by design. Without a provider the run continues
        and coverage states what was not inspected."""
        result, tracks = analyse([frame(0, 1000)], registry=None)
        assert result.status == "SKIPPED"
        assert "vision.describe" in (result.error or "")
        assert tracks == []

    def test_output_is_json_only(self):
        payload = to_json(build_tracks([obs("knife", 1000)]))
        assert set(payload) == {"visual_evidence"}
        assert isinstance(payload["visual_evidence"], list)

    def test_never_emits_a_verdict(self):
        """'Never write Unsafe. Never write Violation. Never write Graphic
        Violence.' Those belong to later reasoning agents."""
        tracks = build_tracks([
            obs("knife", 1000), obs("blood", 1200, category="injury"),
            obs("gun", 1400),
        ])
        blob = str(to_json(tracks)).upper()
        for forbidden in ("UNSAFE", "VIOLATION", "LIMITED ADS", "GRAPHIC"):
            assert forbidden not in blob

    def test_every_track_is_timestamped_and_scored(self):
        for track in build_tracks([obs("knife", 1000), obs("gun", 4000)]):
            assert track.end_ms >= track.start_ms
            assert 0.0 <= track.confidence <= 1.0
            assert track.band in {"VERY_HIGH", "HIGH", "MEDIUM", "LOW"}

    def test_low_confidence_observations_are_still_returned(self):
        """The specification requires they be returned and clearly marked."""
        track = build_tracks([obs("knife", 1000, confidence=0.31)])[0]
        assert track.band == "LOW"
        assert track.to_json()["band"] == "LOW"


class TestVocabularyFiles:
    def test_loads_every_file(self):
        assert len(VOCAB.loaded) >= 12
        assert VOCAB.size > 60
        assert len(VOCAB.synonyms) > 100

    def test_a_blocklist_is_present(self):
        assert len(VOCAB.blocklist) >= 10

    def test_every_vocabulary_states_it_is_not_a_verdict(self):
        import json

        for path in Path("data/vision").glob("*.json"):
            if path.name == "judgment_blocklist.json":
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert "never" in payload["_note"].lower(), path.name

    def test_no_vocabulary_label_is_itself_a_judgement(self):
        """A blocked term inside the whitelist would defeat the tripwire."""
        for label in VOCAB.canonical:
            assert not VOCAB.is_judgment(label), label
