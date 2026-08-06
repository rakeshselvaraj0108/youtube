"""Scoring, fusion and the remediation compiler."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from preflight.models import Adversarial, Evidence, Finding, PolicyRef
from preflight.perception.asr import Segment, Transcript, Word
from preflight.remediate.codegen import build_program
from preflight.remediate.edl import (
    COST,
    MAX_CUT_RATIO,
    STRATEGY_CEILING,
    InvalidEDL,
    candidates_for,
    choose,
    compile_edl,
    lower,
    optimise,
)
from preflight.scoring.fusion import apply_fusion, corroborate, fuse
from preflight.scoring.readiness import (
    SUB_SCORE_ORDER,
    compute_readiness,
    dimension_for,
    js_round,
    sub_scores,
    verdict_for,
)

VECTORS = Path("tests/fixtures/scoring_vectors.json")


def finding(
    clause="AF-01",
    severity="MEDIUM",
    confidence=0.9,
    start=30_000,
    end=32_000,
    fix="NONE",
    modalities=None,
    fid="f1",
) -> Finding:
    return Finding(
        id=fid,
        clauseId=clause,
        category="Language",
        title="Inappropriate language",
        description="d",
        startMs=start,
        endMs=end,
        severity=severity,  # type: ignore[arg-type]
        confidence=confidence,
        modalities=modalities or {"speech": confidence},
        evidence=Evidence(transcript="this is fucked"),
        policy=PolicyRef(clauseId=clause, title="t", section="s", text="x" * 80),
        adversarial=Adversarial(charge="c", rationale="r", confidence=confidence),
        suggestedFix=fix,  # type: ignore[arg-type]
    )


class TestJsRound:
    """Python rounds half to even, JavaScript rounds half up. One difference
    is enough to flip a verdict at a boundary."""

    @pytest.mark.parametrize(
        "value,expected",
        [(84.5, 85), (85.5, 86), (0.5, 1), (1.5, 2), (2.5, 3), (-0.5, 0), (84.4, 84)],
    )
    def test_rounds_half_up(self, value, expected):
        assert js_round(value) == expected

    def test_differs_from_python_round_where_it_matters(self):
        assert round(84.5) == 84  # banker's
        assert js_round(84.5) == 85  # JavaScript


class TestReadiness:
    def test_weights_sum_to_one(self):
        from preflight.scoring.readiness import WEIGHTS

        assert sum(WEIGHTS.values()) == pytest.approx(1.0)

    def test_clamp_stops_one_fatal_flaw_being_averaged_away(self):
        sub = {"policy": 95, "copyright": 19, "metadata": 95, "accessibility": 95, "audio": 95}
        result = compute_readiness(sub)
        assert result.weighted > 70  # a plain average would pass this video
        assert result.overall == 34
        assert result.capped is True
        assert result.verdict == "DO_NOT_PUBLISH"

    def test_clamp_does_not_bind_when_dimensions_agree(self):
        sub = dict.fromkeys(SUB_SCORE_ORDER, 90)
        result = compute_readiness(sub)
        assert result.overall == 90
        assert result.capped is False

    def test_is_monotonic(self):
        previous = -1
        for value in range(0, 101):
            overall = compute_readiness(dict.fromkeys(SUB_SCORE_ORDER, value)).overall
            assert overall >= previous
            previous = overall

    def test_weakest_ties_resolve_to_the_heaviest_dimension(self):
        sub = dict.fromkeys(SUB_SCORE_ORDER, 95)
        sub["policy"] = 40
        sub["audio"] = 40
        assert compute_readiness(sub).weakest == "policy"

    @pytest.mark.parametrize(
        "overall,worst,expected",
        [
            (85, 70, "READY_TO_PUBLISH"),
            (84, 70, "PUBLISH_WITH_FIXES"),
            (99, 69, "PUBLISH_WITH_FIXES"),
            (70, 50, "PUBLISH_WITH_FIXES"),
            (69, 50, "NOT_READY"),
            (80, 49, "NOT_READY"),
            (50, 0, "NOT_READY"),
            (49, 49, "DO_NOT_PUBLISH"),
        ],
    )
    def test_verdict_boundaries(self, overall, worst, expected):
        assert verdict_for(overall, worst) == expected

    def test_verdict_never_contradicts_the_displayed_score(self):
        for value in range(0, 101):
            result = compute_readiness(dict.fromkeys(SUB_SCORE_ORDER, value))
            assert result.verdict == verdict_for(result.overall, result.worst)


class TestSubScores:
    def test_clause_families_map_to_dimensions(self):
        assert dimension_for("AF-01") == "policy"
        assert dimension_for("CID-02") == "copyright"
        assert dimension_for("META-01") == "metadata"
        assert dimension_for("ACC-01") == "accessibility"
        assert dimension_for("AUD-03") == "audio"
        assert dimension_for("VID-02") == "accessibility"

    def test_unknown_clause_family_falls_back_to_policy(self):
        assert dimension_for("XX-99") == "policy"

    def test_a_black_or_frozen_frame_finding_does_not_touch_policy(self):
        """VID-* clauses share the retrieval SCOPES bug's exact failure mode:
        an unregistered prefix silently defaults to "policy". A VID-02 finding
        must dent "accessibility", the dimension its own findings.category
        already claims it belongs to, and leave "policy" untouched."""
        scores = sub_scores([finding(clause="VID-02", severity="LOW", confidence=0.75)])
        assert scores["policy"] == 100.0
        assert scores["accessibility"] < 100.0

    def test_no_findings_is_a_clean_sheet(self):
        assert all(v == 100.0 for v in sub_scores([]).values())

    def test_a_critical_finding_collapses_its_own_dimension_only(self):
        scores = sub_scores([finding(severity="CRITICAL", confidence=0.95)])
        assert scores["policy"] < 40
        assert scores["copyright"] == 100.0

    def test_many_small_findings_never_exceed_one_severe_one(self):
        """The saturating combiner. Otherwise a video full of advisories
        outranks one with a Content ID match."""
        small = [finding(severity="LOW", confidence=0.5, fid=f"f{i}") for i in range(12)]
        severe = [finding(severity="CRITICAL", confidence=0.95)]
        assert sub_scores(small)["policy"] > sub_scores(severe)["policy"]

    def test_early_findings_weigh_more(self):
        early = sub_scores([finding(start=5_000, end=7_000)])["policy"]
        late = sub_scores([finding(start=600_000, end=602_000)])["policy"]
        assert early < late


class TestFusion:
    def test_agreement_raises_confidence_above_either_source(self):
        assert fuse({"speech": 0.8, "vision": 0.8}) > 0.8

    def test_a_single_weak_modality_stays_weak(self):
        assert fuse({"vision": 0.4}) < 0.4

    def test_empty_modalities_is_zero(self):
        assert fuse({}) == 0.0

    def test_coverage_scales_a_modality_down(self):
        full = fuse({"vision": 0.9})
        degraded = fuse({"vision": 0.9}, coverage={"vision": 0.42})
        assert degraded < full

    def test_promote_when_modalities_agree_strongly(self):
        outcome = corroborate(
            finding(severity="MEDIUM", modalities={"speech": 0.94, "vision": 0.92})
        )
        assert outcome.rule == "PROMOTE"
        assert outcome.severity == "HIGH"

    def test_demote_a_lone_weak_vision_claim(self):
        """VLMs hallucinate objects; a single visual claim should not drive a
        demonetisation verdict."""
        outcome = corroborate(finding(severity="HIGH", modalities={"vision": 0.55}))
        assert outcome.rule == "DEMOTE"
        assert outcome.severity == "LOW"

    def test_contradiction_flags_for_review(self):
        outcome = corroborate(
            finding(severity="HIGH", modalities={"speech": 0.95, "vision": 0.10})
        )
        assert outcome.rule == "CONTRADICTION"
        assert outcome.requires_review is True

    def test_apply_fusion_records_what_fired(self):
        findings = [
            finding(severity="MEDIUM", modalities={"speech": 0.94, "vision": 0.92})
        ]
        log = apply_fusion(findings)
        assert log
        assert findings[0].fusedConfidence is not None
        assert findings[0].severity == "HIGH"


class TestEdlLowering:
    def test_only_findings_with_a_fix_produce_ops(self):
        edl = lower([finding(fix="NONE"), finding(fix="BLEEP", fid="f2")], "v.mp4", 60_000)
        assert len(edl.ops) == 1
        assert edl.ops[0].op == "BLEEP"

    def test_op_carries_its_source_finding(self):
        edl = lower([finding(fix="MUTE")], "v.mp4", 60_000)
        assert edl.ops[0].finding_id == "f1"


class TestCostAwareStrategy:
    """Not just where to fix something — which fix to use, under a budget on
    how much the file can visibly change in one pass. The header claim this
    exists to make true: 'chose BLEEP over CUT — same risk reduction, less
    viewer impact.'"""

    def test_candidates_for_an_audio_finding_include_the_audio_ops(self):
        assert set(candidates_for(finding(fix="MUTE", modalities={"speech": 0.9}))) == {
            "BLEEP", "MUTE", "REPLACE_AUDIO", "CUT",
        }

    def test_candidates_for_a_visual_finding_exclude_audio_only_ops(self):
        """Muting the audio does not remove a weapon from the picture."""
        candidates = candidates_for(
            finding(fix="BLUR_REGION", modalities={"vision": 0.9})
        )
        assert set(candidates) == {"BLUR_REGION", "CUT"}
        assert "MUTE" not in candidates
        assert "BLEEP" not in candidates

    def test_a_finding_with_no_fix_has_no_candidates(self):
        assert candidates_for(finding(fix="NONE")) == []

    def test_conservative_prefers_the_cheapest_viable_fix(self):
        """A single CRITICAL finding easily affords BLEEP; conservative must
        not reach for CUT when a cheaper op scores nearly as well."""
        ops, _ = choose(
            [finding(fix="CUT", severity="CRITICAL", modalities={"speech": 0.95})],
            "conservative",
        )
        assert ops[0].op != "CUT"

    def test_a_severity_ordered_budget_spends_on_the_worst_finding_first(self):
        """Two findings that together exceed the conservative ceiling: the
        higher-severity one must still get a real fix, not the lower one."""
        findings = [
            finding(fix="CUT", severity="LOW", start=10_000, end=11_000, fid="low"),
            finding(
                fix="CUT", severity="CRITICAL", start=50_000, end=51_000, fid="crit"
            ),
        ]
        ops, _ = choose(findings, "conservative")
        by_finding = {op.finding_id: op for op in ops}
        assert "crit" in by_finding

    def test_every_chosen_op_is_a_real_candidate_for_its_finding(self):
        f = finding(fix="CUT", modalities={"vision": 0.9})
        ops, _ = choose([f], "aggressive")
        assert ops[0].op in candidates_for(f)

    def test_cumulative_impact_never_exceeds_the_strategy_ceiling(self):
        findings = [
            finding(fix="CUT", start=i * 5_000, end=i * 5_000 + 1_000, fid=f"f{i}")
            for i in range(10)
        ]
        for strategy in ("conservative", "balanced", "aggressive"):
            ops, _ = choose(findings, strategy)
            spent = sum(COST[op.op][0] for op in ops)
            assert spent <= STRATEGY_CEILING[strategy] + 1e-9

    def test_a_finding_that_cannot_fit_the_budget_is_reported_not_dropped_silently(self):
        findings = [
            finding(fix="CUT", start=i * 10_000, end=i * 10_000 + 1_000, fid=f"f{i}")
            for i in range(8)
        ]
        ops, log = choose(findings, "conservative")
        if len(ops) < len(findings):
            assert any("skipped" in line for line in log)

    def test_aggressive_affords_more_cuts_than_conservative(self):
        findings = [
            finding(
                fix="CUT", severity="CRITICAL", start=i * 5_000, end=i * 5_000 + 1_000,
                fid=f"f{i}", modalities={"vision": 0.95},
            )
            for i in range(6)
        ]
        cons_ops, _ = choose(findings, "conservative")
        aggr_ops, _ = choose(findings, "aggressive")
        cons_cuts = sum(1 for op in cons_ops if op.op == "CUT")
        aggr_cuts = sum(1 for op in aggr_ops if op.op == "CUT")
        assert aggr_cuts >= cons_cuts

    def test_the_decision_log_explains_a_downgrade(self):
        ops, log = choose(
            [finding(fix="CUT", severity="CRITICAL", modalities={"speech": 0.95})],
            "conservative",
        )
        assert ops[0].op != "CUT"
        assert any("chose" in line and "CUT" in line for line in log)

    def test_lower_without_a_strategy_still_trusts_suggested_fix_directly(self):
        """Backward compatible: the default path is unchanged."""
        edl = lower([finding(fix="BLEEP")], "v.mp4", 60_000)
        assert edl.ops[0].op == "BLEEP"

    def test_lower_with_a_strategy_can_override_suggested_fix(self):
        edl = lower(
            [finding(fix="CUT", severity="CRITICAL", modalities={"speech": 0.95})],
            "v.mp4", 60_000, strategy="conservative",
        )
        assert edl.ops[0].op != "CUT"

    def test_compile_edl_accepts_a_strategy_end_to_end(self):
        edl = compile_edl(
            [finding(fix="CUT", severity="CRITICAL", modalities={"speech": 0.95})],
            "v.mp4", 60_000, strategy="conservative",
        )
        assert edl.ops
        assert edl.ops[0].op != "CUT"

    def test_every_strategy_name_has_a_ceiling(self):
        assert set(STRATEGY_CEILING) == {"conservative", "balanced", "aggressive"}
        assert (
            STRATEGY_CEILING["conservative"]
            < STRATEGY_CEILING["balanced"]
            < STRATEGY_CEILING["aggressive"]
        )

    def test_cost_table_covers_every_fix_kind_edl_can_emit(self):
        from preflight.remediate.edl import AUDIO_OPS, VIDEO_OPS

        assert (AUDIO_OPS | VIDEO_OPS) <= set(COST)


class TestOptimiserPasses:
    def _transcript(self) -> Transcript:
        words = [
            Word(w=f"w{i}", start_ms=29_000 + i * 500, end_ms=29_000 + i * 500 + 480, conf=0.9)
            for i in range(20)
        ]
        return Transcript(
            language="en", duration_ms=60_000, words=words,
            segments=[Segment(start_ms=29_000, end_ms=39_000, text="x")],
        )

    def test_pad_widens_audio_ops(self):
        edl = compile_edl([finding(fix="MUTE", start=10_000, end=12_000)], "v.mp4", 60_000)
        assert edl.ops[0].start_ms < 10_000
        assert edl.ops[0].end_ms > 12_000

    def test_snap_to_word_widens_to_boundaries(self):
        edl = compile_edl(
            [finding(fix="BLEEP", start=30_100, end=30_400)],
            "v.mp4", 60_000, self._transcript(),
        )
        # Snapped outward to the word, then padded.
        assert edl.ops[0].start_ms <= 30_000
        assert edl.ops[0].end_ms >= 30_400

    def test_coalesce_merges_adjacent_same_kind_ops(self):
        findings = [
            finding(fix="BLEEP", start=10_000, end=11_000, fid="a"),
            finding(fix="BLEEP", start=11_100, end=12_000, fid="b"),
        ]
        edl = compile_edl(findings, "v.mp4", 60_000)
        assert len(edl.ops) == 1

    def test_coalesce_leaves_distant_ops_alone(self):
        findings = [
            finding(fix="BLEEP", start=10_000, end=11_000, fid="a"),
            finding(fix="BLEEP", start=40_000, end=41_000, fid="b"),
        ]
        assert len(compile_edl(findings, "v.mp4", 60_000).ops) == 2

    def test_cut_subsumes_ops_inside_it(self):
        findings = [
            finding(fix="CUT", start=10_000, end=20_000, fid="a"),
            finding(fix="MUTE", start=12_000, end=13_000, fid="b"),
        ]
        edl = compile_edl(findings, "v.mp4", 600_000)
        assert [op.op for op in edl.ops] == ["CUT"]

    def test_replace_audio_absorbs_an_enclosed_mute(self):
        findings = [
            finding(fix="REPLACE_AUDIO", start=10_000, end=30_000, fid="a"),
            finding(fix="MUTE", start=15_000, end=16_000, fid="b"),
        ]
        edl = compile_edl(findings, "v.mp4", 600_000)
        assert [op.op for op in edl.ops] == ["REPLACE_AUDIO"]

    def test_cut_budget_demotes_rather_than_deleting_the_video(self):
        """Never silently remove a third of someone's footage."""
        duration = 100_000
        findings = [
            finding(fix="CUT", start=0, end=20_000, fid="a"),
            finding(fix="CUT", start=40_000, end=55_000, fid="b"),
        ]
        edl = compile_edl(findings, "v.mp4", duration)
        cut_total = sum(op.duration_ms for op in edl.ops if op.op == "CUT")
        assert cut_total <= duration * MAX_CUT_RATIO
        assert any("MUTE" == op.op for op in edl.ops)
        assert edl.warnings

    def test_ops_are_indexed_and_ordered(self):
        findings = [
            finding(fix="MUTE", start=40_000, end=41_000, fid="a"),
            finding(fix="BLEEP", start=10_000, end=11_000, fid="b"),
        ]
        edl = compile_edl(findings, "v.mp4", 60_000)
        assert [op.index for op in edl.ops] == [1, 2]
        assert edl.ops[0].start_ms < edl.ops[1].start_ms

    def test_validate_rejects_an_out_of_bounds_span(self):
        # A video op, because audio ops are clamped to the runtime by the pad
        # pass before validation ever sees them.
        edl = lower(
            [finding(fix="BLUR_REGION", start=50_000, end=90_000)], "v.mp4", 60_000
        )
        with pytest.raises(InvalidEDL, match="outside the runtime"):
            optimise(edl)

    def test_padding_clamps_audio_ops_to_the_runtime(self):
        """Not a validation failure — padding is allowed to bump the end of the
        file, it just must not run past it."""
        edl = compile_edl([finding(fix="MUTE", start=58_000, end=60_000)], "v.mp4", 60_000)
        assert edl.ops[0].end_ms == 60_000

    def test_validate_rejects_a_degenerate_span(self):
        edl = lower([finding(fix="BLUR_REGION", start=10_000, end=10_050)], "v.mp4", 60_000)
        with pytest.raises(InvalidEDL, match="below the"):
            optimise(edl)


class TestCodegen:
    def test_audio_only_edl_stream_copies_the_video(self):
        """The claim the UI makes, made true."""
        edl = compile_edl(
            [finding(fix="BLEEP", start=10_000, end=11_000)], "in.mp4", 60_000
        )
        program = build_program(edl, "in.mp4", "out.mp4")
        assert program.video_stream_copied is True
        assert "-c:v" in program.command
        assert program.command[program.command.index("-c:v") + 1] == "copy"
        assert "libx264" not in program.command

    def test_a_video_op_forces_a_re_encode(self):
        edl = compile_edl(
            [finding(fix="BLUR_REGION", start=10_000, end=12_000)], "in.mp4", 60_000
        )
        program = build_program(edl, "in.mp4", "out.mp4")
        assert program.video_stream_copied is False
        assert "libx264" in program.command

    def test_empty_edl_is_a_straight_copy(self):
        edl = compile_edl([], "in.mp4", 60_000)
        program = build_program(edl, "in.mp4", "out.mp4")
        assert program.command == ["ffmpeg", "-y", "-i", "in.mp4", "-c", "copy", "out.mp4"]

    def test_bleep_emits_a_matching_sine_and_delay(self):
        edl = compile_edl(
            [finding(fix="BLEEP", start=10_000, end=11_000)], "in.mp4", 60_000
        )
        program = build_program(edl, "in.mp4", "out.mp4")
        op = edl.ops[0]
        assert f"duration={op.duration_ms / 1000:.3f}" in " ".join(program.command)
        assert f"adelay={op.start_ms}|{op.start_ms}" in program.filter_graph

    def test_every_audio_op_silences_the_source_underneath(self):
        edl = compile_edl(
            [
                finding(fix="MUTE", start=10_000, end=12_000, fid="a"),
                finding(fix="BLEEP", start=40_000, end=41_000, fid="b"),
            ],
            "in.mp4", 60_000,
        )
        graph = build_program(edl, "in.mp4", "out.mp4").filter_graph
        assert graph.count("volume=enable=") == 2

    def test_blur_geometry_comes_from_the_box(self):
        edl = compile_edl(
            [finding(fix="BLUR_REGION", start=10_000, end=12_000)], "in.mp4", 60_000
        )
        graph = build_program(edl, "in.mp4", "out.mp4").filter_graph
        assert "crop=iw*0.42:ih*0.3:iw*0.29:ih*0.35" in graph
        assert "boxblur=20:2" in graph

    def test_command_is_generated_not_templated(self):
        a = build_program(
            compile_edl([finding(fix="MUTE", start=10_000, end=12_000)], "in.mp4", 60_000),
            "in.mp4", "out.mp4",
        )
        b = build_program(
            compile_edl([finding(fix="MUTE", start=30_000, end=32_000)], "in.mp4", 60_000),
            "in.mp4", "out.mp4",
        )
        assert a.filter_graph != b.filter_graph


@pytest.mark.skipif(not VECTORS.is_file(), reason="run scripts/emit_scoring_vectors.py")
class TestCrossLanguageContract:
    """Python and TypeScript must agree to the decimal.

    The page renders one implementation and the JSON carries the other. A
    disagreement is a report whose headline number contradicts its own data.
    """

    def test_python_reproduces_every_shared_vector(self):
        vectors = json.loads(VECTORS.read_text(encoding="utf-8"))
        for case in vectors["cases"]:
            result = compute_readiness(case["sub"])
            assert result.overall == case["overall"], case["sub"]
            assert result.verdict == case["verdict"], case["sub"]
            assert result.weakest == case["weakest"], case["sub"]
