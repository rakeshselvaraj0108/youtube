"""Correlation — one event seen by four agents, not four events.

The tests that matter here are the ones asserting things do NOT merge. A
correlator that merges aggressively looks impressive on the happy path and
destroys the report: two real problems disappear behind one label that
describes neither, and the count a creator acts on is wrong.

A false split is recoverable — the reader sees two findings that are really
one. A false merge hides a violation. The thresholds lean that way on
purpose.
"""

from __future__ import annotations

import pytest

from preflight.models import Adversarial, Evidence, Finding, PolicyRef
from preflight.scoring.incidents import (
    CONFIDENCE_CEILING,
    PROXIMITY_MS,
    build_graph,
    correlate,
    is_file_scoped,
)

DURATION = 600_000


def finding(
    fid: str,
    start: int,
    end: int,
    *,
    clause: str = "AF-01",
    category: str = "Language",
    modality: str = "speech",
    severity: str = "MEDIUM",
    confidence: float = 0.8,
    fix: str = "NONE",
) -> Finding:
    return Finding(
        id=fid,
        clauseId=clause,
        category=category,
        title=f"{category} finding",
        description="d",
        startMs=start,
        endMs=end,
        severity=severity,  # type: ignore[arg-type]
        confidence=confidence,
        modalities={modality: confidence},
        evidence=Evidence(transcript=""),
        policy=PolicyRef(clause, "t", "s", "x"),
        adversarial=Adversarial(charge="c", rationale="r", confidence=confidence),
        suggestedFix=fix,  # type: ignore[arg-type]
    )


class TestTheHeadlineCase:
    """Four agents at 02:14 have found one problem four times."""

    def test_four_agents_at_one_moment_become_one_incident(self):
        findings = [
            finding("f1", 134_000, 134_500, modality="speech", category="Substances"),
            finding("f2", 135_000, 135_500, modality="vision", category="Substances"),
            finding("f3", 135_000, 135_800, modality="ocr", category="Metadata"),
            finding("f4", 135_200, 136_000, modality="audio", category="Substances"),
        ]
        incidents = correlate(findings, DURATION)
        assert len(incidents) == 1
        assert set(incidents[0].agents) == {"speech", "vision", "ocr", "audio"}
        assert incidents[0].corroborated

    def test_the_incident_spans_every_observation(self):
        findings = [
            finding("f1", 134_000, 134_500, category="Substances"),
            finding("f2", 135_200, 136_000, modality="vision", category="Substances"),
        ]
        incident = correlate(findings, DURATION)[0]
        assert incident.start_ms == 134_000
        assert incident.end_ms == 136_000

    def test_it_names_every_clause_it_covers(self):
        findings = [
            finding("f1", 100, 500, clause="AF-01", category="Language"),
            finding("f2", 600, 900, clause="AF-09", modality="vision",
                    category="Controversial"),
        ]
        assert correlate(findings, DURATION)[0].clauses == ["AF-01", "AF-09"]

    def test_it_explains_itself(self):
        findings = [
            finding("f1", 100, 500, category="Substances"),
            finding("f2", 600, 900, modality="vision", category="Substances"),
        ]
        reasoning = correlate(findings, DURATION)[0].reasoning
        assert "2 findings" in reasoning
        assert "speech" in reasoning and "vision" in reasoning


class TestThingsThatMustNotMerge:
    """The half that decides whether this feature helps or harms."""

    def test_unrelated_problems_at_the_same_moment_stay_separate(self):
        """Profanity and a frozen frame at 02:14 are two problems that share
        a timestamp. One incident covering both would need a category that
        describes neither."""
        findings = [
            finding("f1", 134_000, 134_500, clause="AF-01", category="Language"),
            finding("f2", 134_000, 134_500, clause="VID-02", modality="vision",
                    category="Accessibility"),
        ]
        assert len(correlate(findings, DURATION)) == 2

    def test_the_same_violation_far_apart_stays_separate(self):
        findings = [
            finding("f1", 10_000, 10_500),
            finding("f2", 400_000, 400_500),
        ]
        assert len(correlate(findings, DURATION)) == 2

    def test_findings_just_outside_the_window_stay_separate(self):
        findings = [
            finding("f1", 10_000, 10_100),
            finding("f2", 10_100 + PROXIMITY_MS + 500, 10_600 + PROXIMITY_MS),
        ]
        assert len(correlate(findings, DURATION)) == 2

    def test_a_file_scoped_finding_does_not_swallow_the_timeline(self):
        """The trap the real corpus contains. "No caption track" spans the
        whole video, so any overlap rule merges it with every incident and
        produces one meaningless super-incident."""
        findings = [
            finding("whole", 0, DURATION, clause="ACC-02", modality="access",
                    category="Accessibility"),
            finding("a", 10_000, 10_500),
            finding("b", 300_000, 300_500),
        ]
        incidents = correlate(findings, DURATION)
        assert len(incidents) == 3
        for incident in incidents:
            assert len(incident.finding_ids) == 1

    def test_a_nearly_file_scoped_finding_is_also_excluded(self):
        """The corpus carries a frozen-video finding spanning 100ms to
        20000ms of a 20000ms file — file-scoped in everything but
        arithmetic."""
        almost = finding("almost", 100, DURATION - 100, category="Accessibility")
        assert is_file_scoped(almost, DURATION)

    def test_a_real_event_is_not_treated_as_file_scoped(self):
        assert not is_file_scoped(finding("f", 10_000, 12_000), DURATION)


class TestConfidenceIsNotInflated:
    """`fusion` already combined the modalities inside each finding. Running
    a noisy-or again here would count the same agreement twice."""

    def test_one_finding_keeps_its_own_confidence(self):
        incident = correlate([finding("f1", 100, 500, confidence=0.8)], DURATION)[0]
        assert incident.confidence == pytest.approx(0.8, abs=0.001)

    def test_corroboration_raises_confidence_only_a_little(self):
        alone = correlate([finding("f1", 100, 500, confidence=0.8)], DURATION)[0]
        together = correlate(
            [
                finding("f1", 100, 500, confidence=0.8, category="Substances"),
                finding("f2", 600, 900, confidence=0.8, modality="vision",
                        category="Substances"),
            ],
            DURATION,
        )[0]
        assert together.confidence > alone.confidence
        assert together.confidence - alone.confidence < 0.1

    def test_one_agent_repeating_itself_is_not_corroboration(self):
        """The failure mode that lets a single noisy detector talk itself
        into a near-certain incident."""
        repeated = correlate(
            [
                finding("f1", 100, 500, confidence=0.8, modality="speech"),
                finding("f2", 600, 900, confidence=0.8, modality="speech"),
                finding("f3", 1000, 1400, confidence=0.8, modality="speech"),
            ],
            DURATION,
        )[0]
        assert repeated.confidence == pytest.approx(0.8, abs=0.001)
        assert not repeated.corroborated

    def test_confidence_never_reaches_certainty(self):
        many = [
            finding(f"f{i}", i * 300, i * 300 + 200, confidence=0.95,
                    modality=m, category="Substances")
            for i, m in enumerate(["speech", "vision", "ocr", "audio", "music", "meta"])
        ]
        assert correlate(many, DURATION)[0].confidence <= CONFIDENCE_CEILING

    def test_an_incident_is_never_less_confident_than_its_best_evidence(self):
        incident = correlate(
            [
                finding("f1", 100, 500, confidence=0.4, category="Substances"),
                finding("f2", 600, 900, confidence=0.9, modality="vision",
                        category="Substances"),
            ],
            DURATION,
        )[0]
        assert incident.confidence >= 0.9


class TestSeverityAndFix:
    def test_the_incident_takes_the_worst_severity(self):
        incident = correlate(
            [
                finding("f1", 100, 500, severity="LOW", category="Substances"),
                finding("f2", 600, 900, severity="CRITICAL", modality="vision",
                        category="Substances"),
            ],
            DURATION,
        )[0]
        assert incident.severity == "CRITICAL"

    def test_the_incident_is_named_for_its_worst_member(self):
        incident = correlate(
            [
                finding("f1", 100, 500, severity="LOW", category="Substances"),
                finding("f2", 600, 900, severity="HIGH", modality="vision",
                        category="Substances"),
            ],
            DURATION,
        )[0]
        assert incident.category == "Substances"

    def test_a_fix_is_carried_forward(self):
        incident = correlate(
            [
                finding("f1", 100, 500, severity="HIGH", fix="BLEEP"),
                finding("f2", 600, 900, modality="vision", severity="LOW",
                        category="Language"),
            ],
            DURATION,
        )[0]
        assert incident.suggested_fix == "BLEEP"


class TestScaleAndShape:
    def test_no_findings_produces_no_incidents(self):
        assert correlate([], DURATION) == []

    def test_incidents_come_back_in_timeline_order(self):
        findings = [
            finding("late", 500_000, 500_500),
            finding("early", 1_000, 1_500),
            finding("mid", 250_000, 250_500),
        ]
        starts = [i.start_ms for i in correlate(findings, DURATION)]
        assert starts == sorted(starts)

    def test_every_finding_lands_in_exactly_one_incident(self):
        """No finding is lost, and none is counted twice."""
        findings = [
            finding(f"f{i}", i * 4_000, i * 4_000 + 500, modality="speech")
            for i in range(40)
        ]
        incidents = correlate(findings, DURATION)
        seen = [fid for incident in incidents for fid in incident.finding_ids]
        assert sorted(seen) == sorted(f.id for f in findings)
        assert len(seen) == len(set(seen))

    def test_a_long_video_with_many_findings_stays_fast(self):
        """A sixty-minute file with hundreds of findings must not become the
        quadratic scan the obvious implementation produces."""
        import time

        findings = [
            finding(f"f{i}", i * 5_000, i * 5_000 + 400, modality="speech")
            for i in range(800)
        ]
        started = time.perf_counter()
        incidents = correlate(findings, 3_600_000)
        assert time.perf_counter() - started < 1.0
        assert len(incidents) > 0

    def test_the_graph_serialises(self):
        import json

        graph = build_graph(
            [finding("f1", 100, 500), finding("f2", 400_000, 400_500)], DURATION
        )
        payload = graph.to_json()
        json.dumps(payload)
        assert payload["total"] == 2
