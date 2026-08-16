"""Temporal coverage — where each modality actually looked.

The property under test is one sentence: **a stretch of video nothing
examined must never read as clean**. A scalar coverage figure cannot express
that, which is the whole reason this module exists — an agent can process
100% of its own samples and still have never looked at minutes nine through
twelve, and every absence claim over that stretch would be unsupported.
"""

from __future__ import annotations

import pytest

from preflight import coverage as cov


class Sample:
    """Duck-typed evidence: whatever a modality actually looked at."""

    def __init__(self, ts_ms: int) -> None:
        self.ts_ms = ts_ms


def evenly(count: int, duration_ms: int) -> list[Sample]:
    step = duration_ms / count
    return [Sample(int(i * step)) for i in range(count)]


FOURTEEN_MIN = 14 * 60_000


class TestBanding:
    def test_a_fourteen_minute_video_yields_fourteen_bands(self):
        result = cov.build(FOURTEEN_MIN, {"frames": evenly(336, FOURTEEN_MIN)})
        assert len(result.bands) == 14

    def test_bands_are_labelled_by_minute(self):
        result = cov.build(FOURTEEN_MIN, {"frames": evenly(336, FOURTEEN_MIN)})
        assert result.bands[0].label == "00–01"
        assert result.bands[13].label == "13–14"

    def test_a_partial_final_minute_still_gets_a_band(self):
        """90 seconds is two bands, not one — the second is short but real,
        and dropping it would leave the last 30 seconds unreported."""
        result = cov.build(90_000, {"frames": evenly(20, 90_000)})
        assert len(result.bands) == 2
        assert result.bands[1].end_ms == 90_000

    def test_a_zero_duration_video_produces_nothing(self):
        assert cov.build(0, {"frames": []}).bands == []


class TestBlindSpots:
    def test_a_gap_in_the_middle_is_reported_unexamined(self):
        """The failure this module exists to catch: even sampling everywhere
        except minutes 6–8, which a scalar would still call 86% coverage."""
        frames = [
            Sample(ts)
            for ts in range(0, FOURTEEN_MIN, 2_500)
            if not (6 * 60_000 <= ts < 8 * 60_000)
        ]
        result = cov.build(FOURTEEN_MIN, {"frames": frames})
        blind = [b.label for b in result.blind_spots("frames")]
        assert blind == ["06–07", "07–08"]

    def test_an_unexamined_band_is_never_reported_as_examined(self):
        frames = [Sample(ts) for ts in range(0, 6 * 60_000, 2_500)]
        result = cov.build(FOURTEEN_MIN, {"frames": frames})
        for band in result.bands[6:]:
            assert band.state_of("frames") == "UNEXAMINED"

    def test_full_even_coverage_has_no_blind_spots(self):
        result = cov.build(FOURTEEN_MIN, {"frames": evenly(336, FOURTEEN_MIN)})
        assert result.blind_spots("frames") == []
        assert result.share_examined("frames") == 1.0

    def test_share_examined_counts_bands_not_samples(self):
        """A modality that put every one of its samples in one minute has
        examined one minute, not the whole video — the exact distinction the
        scalar coverage figure cannot make."""
        crowded = [Sample(ts) for ts in range(0, 60_000, 200)]
        result = cov.build(FOURTEEN_MIN, {"frames": crowded})
        assert result.share_examined("frames") == pytest.approx(1 / 14, abs=0.01)


class TestThinBands:
    def test_a_barely_sampled_band_is_thin_not_examined(self):
        """One frame in a sixty-second window is not coverage of that
        window. Calling it examined is how a thin pass launders itself into
        a clean bill of health."""
        frames = [Sample(ts) for ts in range(0, FOURTEEN_MIN, 2_500)]
        frames = [f for f in frames if not (3 * 60_000 < f.ts_ms < 4 * 60_000)]
        frames.append(Sample(3 * 60_000 + 30_000))
        result = cov.build(FOURTEEN_MIN, {"frames": frames})
        assert result.bands[3].state_of("frames") == "THIN"

    def test_thin_is_distinct_from_unexamined(self):
        frames = [Sample(0), Sample(30_000)]
        result = cov.build(120_000, {"frames": frames})
        assert result.bands[0].state_of("frames") == "EXAMINED"
        assert result.bands[1].state_of("frames") == "UNEXAMINED"


class TestModalitiesAreIndependent:
    def test_each_modality_reports_its_own_timeline(self):
        """OCR reading every frame says nothing about whether speech was
        transcribed over the same minutes."""
        result = cov.build(
            180_000,
            {
                "ocr": [Sample(ts) for ts in range(0, 180_000, 2_500)],
                "speech": [Sample(ts) for ts in range(0, 60_000, 5_000)],
            },
        )
        assert result.blind_spots("ocr") == []
        assert [b.label for b in result.blind_spots("speech")] == ["01–02", "02–03"]

    def test_a_modality_that_did_not_run_is_absent_not_zero(self):
        """A row of zeroes reads as 'looked and saw nothing'. A modality that
        never ran has to be missing from the table instead."""
        result = cov.build(120_000, {"ocr": evenly(48, 120_000)})
        assert result.modalities == ["ocr"]
        assert "vision" not in result.modalities


class TestAbsenceGuard:
    def test_absence_is_supported_only_where_something_looked(self):
        frames = [
            Sample(ts)
            for ts in range(0, FOURTEEN_MIN, 2_500)
            if not (6 * 60_000 <= ts < 8 * 60_000)
        ]
        result = cov.build(FOURTEEN_MIN, {"frames": frames})
        # A span inside the examined region may claim absence.
        assert cov.absence_is_supported(result, "frames", 60_000, 120_000)
        # A span inside the hole may not.
        assert not cov.absence_is_supported(result, "frames", 6 * 60_000, 7 * 60_000)

    def test_a_span_crossing_a_blind_spot_cannot_claim_absence(self):
        """Partial coverage of a span is not coverage of it. This is what
        stops 'no secrets between 5:00 and 9:00' from being emitted when
        two of those minutes were never read."""
        frames = [
            Sample(ts)
            for ts in range(0, FOURTEEN_MIN, 2_500)
            if not (6 * 60_000 <= ts < 8 * 60_000)
        ]
        result = cov.build(FOURTEEN_MIN, {"frames": frames})
        assert not cov.absence_is_supported(result, "frames", 5 * 60_000, 9 * 60_000)

    def test_an_unknown_modality_cannot_support_absence(self):
        result = cov.build(120_000, {"ocr": evenly(48, 120_000)})
        assert not cov.absence_is_supported(result, "vision", 0, 60_000)

    def test_an_empty_timeline_cannot_support_absence(self):
        assert not cov.absence_is_supported(cov.build(0, {}), "ocr", 0, 1_000)


class TestReporting:
    def test_the_report_serialises(self):
        import json

        result = cov.build(FOURTEEN_MIN, {"frames": evenly(336, FOURTEEN_MIN)})
        payload = result.to_json()
        json.dumps(payload)
        assert len(payload["bands"]) == 14
        assert payload["shareExamined"]["frames"] == 1.0

    def test_blind_spots_are_named_in_the_payload(self):
        frames = [Sample(ts) for ts in range(0, 6 * 60_000, 2_500)]
        payload = cov.build(FOURTEEN_MIN, {"frames": frames}).to_json()
        assert "06–07" in payload["blindSpots"]["frames"]

    def test_the_table_marks_gaps_visibly(self):
        frames = [Sample(ts) for ts in range(0, 6 * 60_000, 2_500)]
        text = cov.build(FOURTEEN_MIN, {"frames": frames}).describe()
        assert "NONE" in text
        assert "13–14" in text
