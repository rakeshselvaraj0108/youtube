"""The what-if engine, reviewed adversarially.

A simulation that overstates what an edit buys is worse than no simulation:
the creator makes the edit, re-runs, and gets a number that does not match
the promise. So most of this file attacks the engine rather than
demonstrating it — double-counted repairs, edits credited for evidence they
cannot reach, and scenarios that claim improvement the real scorer would not
produce.

The strongest guarantee available is structural: the predicted score is
computed by `sub_scores` and `compute_readiness`, the same functions that
produced the current score. `test_the_prediction_is_the_real_scorer` is the
one that keeps that true.
"""

from __future__ import annotations

import pytest

from preflight.models import Adversarial, Evidence, Finding, PolicyRef
from preflight.scoring.readiness import compute_readiness, sub_scores
from preflight.scoring.simulation import (
    Edit,
    apply_edits,
    edit_for,
    explore,
    score,
    simulate,
)


def finding(
    fid: str = "f1",
    *,
    start: int = 10_000,
    end: int = 12_000,
    clause: str = "AF-01",
    category: str = "Language",
    modalities: dict[str, float] | None = None,
    severity: str = "HIGH",
    confidence: float = 0.9,
    fix: str = "BLEEP",
) -> Finding:
    return Finding(
        id=fid,
        clauseId=clause,
        category=category,
        title="t",
        description="d",
        startMs=start,
        endMs=end,
        severity=severity,  # type: ignore[arg-type]
        confidence=confidence,
        modalities=modalities if modalities is not None else {"speech": confidence},
        evidence=Evidence(transcript="x"),
        policy=PolicyRef(clause, "t", "s", "x"),
        adversarial=Adversarial(charge="c", rationale="r", confidence=confidence),
        suggestedFix=fix,  # type: ignore[arg-type]
    )


class TestEditsRemoveEvidenceNotFindings:
    """The claim the whole engine rests on."""

    def test_removing_the_only_evidence_removes_the_finding(self):
        survivors, removed, _ = apply_edits(
            [finding(modalities={"speech": 0.9})],
            [Edit("BLEEP", 10_000, 12_000)],
        )
        assert survivors == []
        assert removed == ["f1"]

    def test_a_corroborated_finding_survives_a_partial_edit(self):
        """Bleeping the word does not remove the picture. Promising that it
        does is the flattering error — the creator makes the edit and the
        finding is still there."""
        survivors, removed, weakened = apply_edits(
            [finding(modalities={"speech": 0.9, "vision": 0.8})],
            [Edit("BLEEP", 10_000, 12_000)],
        )
        assert removed == []
        assert weakened == ["f1"]
        assert set(survivors[0].modalities) == {"vision"}

    def test_a_weakened_finding_is_less_confident_than_before(self):
        original = finding(modalities={"speech": 0.9, "vision": 0.8})
        survivors, _, _ = apply_edits([original], [Edit("BLEEP", 10_000, 12_000)])
        before = original.fusedConfidence or original.confidence
        assert (survivors[0].fusedConfidence or 0) < before

    def test_an_edit_outside_the_span_changes_nothing(self):
        survivors, removed, weakened = apply_edits(
            [finding(start=10_000, end=12_000)],
            [Edit("BLEEP", 400_000, 402_000)],
        )
        assert removed == [] and weakened == []
        assert survivors[0].modalities == {"speech": 0.9}

    def test_a_blur_does_not_touch_the_audio(self):
        survivors, removed, _ = apply_edits(
            [finding(modalities={"speech": 0.9})],
            [Edit("BLUR_REGION", 10_000, 12_000)],
        )
        assert removed == []
        assert survivors[0].modalities == {"speech": 0.9}

    def test_a_cut_removes_everything_in_its_span(self):
        _, removed, _ = apply_edits(
            [finding(modalities={"speech": 0.9, "vision": 0.8, "ocr": 0.7})],
            [Edit("CUT", 10_000, 12_000)],
        )
        assert removed == ["f1"]

    def test_a_whole_file_edit_reaches_every_finding(self):
        findings = [finding("f1", start=1_000, end=2_000, modalities={"access": 0.9})]
        _, removed, _ = apply_edits(findings, [Edit("ADD_CAPTIONS", whole_file=True)])
        assert removed == ["f1"]


class TestNoDoubleCounting:
    """The single most likely way for a simulation to promise a score it
    cannot reach."""

    def test_two_identical_edits_are_worth_one(self):
        findings = [finding(modalities={"speech": 0.9, "vision": 0.8})]
        once = simulate(findings, [Edit("BLEEP", 10_000, 12_000)])
        twice = simulate(
            findings,
            [Edit("BLEEP", 10_000, 12_000), Edit("BLEEP", 10_000, 12_000)],
        )
        assert once.overall == twice.overall
        assert once.surviving == twice.surviving

    def test_overlapping_edits_removing_the_same_modality_do_not_stack(self):
        findings = [finding(modalities={"speech": 0.9, "vision": 0.8})]
        single = simulate(findings, [Edit("MUTE", 9_000, 13_000)])
        overlapping = simulate(
            findings,
            [Edit("MUTE", 9_000, 13_000), Edit("BLEEP", 10_000, 12_000)],
        )
        assert single.overall == overlapping.overall

    def test_applying_an_edit_never_mutates_the_original_findings(self):
        """A scenario that edited the real run would make every later
        scenario wrong, and the bug would look like drifting numbers."""
        original = finding(modalities={"speech": 0.9, "vision": 0.8})
        before = dict(original.modalities)
        apply_edits([original], [Edit("BLEEP", 10_000, 12_000)])
        assert original.modalities == before


class TestThePredictionIsTheRealScorer:
    """No second risk model. The guarantee that keeps a prediction honest."""

    def test_the_prediction_is_the_real_scorer(self):
        findings = [finding("f1"), finding("f2", start=50_000, end=52_000)]
        survivors, _, _ = apply_edits(findings, [Edit("BLEEP", 10_000, 12_000)])
        predicted = simulate(findings, [Edit("BLEEP", 10_000, 12_000)])
        truth = compute_readiness(sub_scores(survivors))
        assert predicted.overall == truth.overall
        assert predicted.verdict == truth.verdict

    def test_no_edits_reproduces_the_current_score(self):
        findings = [finding("f1"), finding("f2", start=50_000, end=52_000)]
        assert simulate(findings, []).overall == score(findings)[0]

    def test_the_saturating_combiner_is_respected(self):
        """Risk is not additive. Removing one of two identical CRITICAL
        findings must not halve the score, because the real combiner
        saturates — a simulation using linear subtraction would claim it
        does."""
        two = [
            finding("f1", severity="CRITICAL", confidence=0.95),
            finding("f2", start=50_000, end=52_000, severity="CRITICAL",
                    confidence=0.95),
        ]
        before = score(two)[0]
        after = simulate(two, [Edit("BLEEP", 10_000, 12_000)]).overall
        one_only = score([two[1]])[0]
        assert after == one_only
        assert after - before < 30, "linear subtraction would overstate this"


class TestScenariosAreCoherent:
    def test_fixing_everything_is_never_worse_than_fixing_one_thing(self):
        findings = [
            finding("f1", clause="AF-01"),
            finding("f2", start=50_000, end=52_000, clause="AF-02"),
            finding("f3", start=90_000, end=92_000, clause="AF-09"),
        ]
        report = explore(findings)
        combined = next(s for s in report.scenarios if s.name.startswith("apply every"))
        singles = [s for s in report.scenarios if len(s.edits) == 1]
        assert combined.overall >= max(s.overall for s in singles)

    def test_every_scenario_is_at_least_as_good_as_doing_nothing(self):
        """An edit that removes evidence cannot raise risk. A scenario
        scoring worse than the baseline would be a logical contradiction."""
        findings = [
            finding("f1", severity="CRITICAL"),
            finding("f2", start=50_000, end=52_000, severity="LOW",
                    modalities={"vision": 0.4}, fix="BLUR_REGION"),
            finding("f3", start=90_000, end=92_000, severity="MEDIUM"),
        ]
        report = explore(findings)
        for scenario in report.scenarios:
            assert scenario.overall >= report.baseline.overall, scenario.name

    def test_the_delta_matches_the_scores(self):
        findings = [finding("f1"), finding("f2", start=50_000, end=52_000)]
        report = explore(findings)
        for scenario in report.scenarios:
            assert scenario.delta == scenario.overall - report.baseline.overall

    def test_the_best_scenario_is_not_simply_the_most_destructive_one(self):
        """Caught by the adversarial pass on a realistic finding set: the
        first version ranked on score alone and recommended "silence every
        flagged span", which scored highest precisely because it destroyed
        the most audio. Technically the top number, terrible advice, and an
        engine that offers it has optimised the wrong objective."""
        findings = [
            finding("violence", start=454_000, end=457_000, clause="AF-04",
                    category="Violence", severity="CRITICAL",
                    modalities={"vision": 0.92, "speech": 0.88}, fix="BLUR_REGION"),
            finding("profanity", start=134_000, end=134_800, severity="HIGH",
                    modalities={"speech": 0.9}, fix="BLEEP"),
            finding("music", start=600_000, end=625_000, clause="COPY-01",
                    category="Copyright", severity="HIGH",
                    modalities={"music": 0.86}, fix="REPLACE_AUDIO"),
        ]
        report = explore(findings)
        assert "silence every" not in report.best.name
        assert report.best.impact <= report.highest_score.impact

    def test_the_highest_scoring_option_is_still_reported_separately(self):
        """Both questions are legitimate; neither should pretend to be the
        other."""
        findings = [finding("f1"), finding("f2", start=50_000, end=52_000)]
        report = explore(findings)
        assert report.highest_score.overall >= report.best.overall

    def test_an_edit_that_repairs_something_but_moves_nothing_says_why(self):
        """The clamp holds the overall at the weakest dimension, so
        repairing a healthier one shows +0. Reporting that without saying
        why reads as "this edit is worthless" — wrong, and the opposite of
        the advice a creator needs."""
        findings = [
            finding("captions", start=0, end=1_122_000, clause="ACC-02",
                    category="Accessibility", severity="HIGH", confidence=0.99,
                    modalities={"access": 0.99}, fix="NONE"),
            finding("music", start=600_000, end=625_000, clause="COPY-01",
                    category="Copyright", severity="HIGH",
                    modalities={"music": 0.86}, fix="REPLACE_AUDIO"),
        ]
        report = explore(findings)
        replaced = next(s for s in report.scenarios if "REPLACE_AUDIO" in s.name)
        assert replaced.removed_finding_ids == ("music",)
        if replaced.delta <= 0:
            assert replaced.gated_by == "accessibility"

    def test_viewer_impact_comes_from_the_compilers_own_table(self):
        """A simulation pricing a CUT differently from the compiler that
        renders it would recommend edits the compiler then refuses under its
        own impact ceiling."""
        from preflight.remediate.edl import COST
        from preflight.scoring.simulation import viewer_impact

        assert viewer_impact([Edit("CUT", 0, 1000)]) == pytest.approx(COST["CUT"][0])
        assert viewer_impact([Edit("BLEEP", 0, 1000)]) == pytest.approx(COST["BLEEP"][0])
        assert viewer_impact([]) == 0.0

    def test_a_clean_video_offers_no_edits(self):
        report = explore([])
        assert report.scenarios == []
        assert report.baseline.overall == 100

    def test_a_finding_with_no_recommended_fix_generates_no_scenario(self):
        findings = [finding("f1", fix="NONE")]
        named = {s.name for s in explore(findings).scenarios}
        assert not any("NONE" in n for n in named)
        assert edit_for(findings[0]) is None


class TestReportShape:
    def test_the_report_serialises(self):
        import json

        payload = explore([finding("f1"), finding("f2", start=50_000, end=52_000)]).to_json()
        json.dumps(payload)
        assert payload["baseline"]["name"] == "current"
        assert payload["best"]

    def test_every_scenario_names_what_it_removed(self):
        findings = [finding("f1", modalities={"speech": 0.9})]
        scenario = simulate(findings, [Edit("BLEEP", 10_000, 12_000)])
        assert scenario.removed_finding_ids == ("f1",)


class TestScaleAndProperties:
    @pytest.mark.parametrize("count", [1, 5, 20, 60])
    def test_exploration_stays_linear_in_findings(self, count):
        """Not the power set. Twelve findings would be four thousand
        scenarios, almost all of them nonsense."""
        findings = [
            finding(f"f{i}", start=i * 5_000, end=i * 5_000 + 800)
            for i in range(count)
        ]
        report = explore(findings)
        assert len(report.scenarios) <= count + 3

    def test_a_large_run_simulates_quickly(self):
        import time

        findings = [
            finding(f"f{i}", start=i * 3_000, end=i * 3_000 + 500)
            for i in range(300)
        ]
        started = time.perf_counter()
        explore(findings)
        assert time.perf_counter() - started < 2.0

    @pytest.mark.parametrize(
        "modalities",
        [
            {"speech": 0.9},
            {"vision": 0.8},
            {"speech": 0.9, "vision": 0.8},
            {"speech": 0.5, "vision": 0.5, "ocr": 0.5, "audio": 0.5},
        ],
    )
    def test_scores_stay_inside_bounds_whatever_the_evidence(self, modalities):
        findings = [finding("f1", modalities=modalities)]
        for scenario in explore(findings).scenarios:
            assert 0 <= scenario.overall <= 100
            for value in scenario.sub.values():
                assert 0.0 <= value <= 100.0
