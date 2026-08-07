"""The reasoning chain, reviewed adversarially.

Written from the position the brief asks for: a YouTube reviewer trying to
disprove every incident. That reviewer does not attack the conclusion, they
attack the *provenance* — show me the observation, show me the clause, show
me what would have changed your mind. A chain that cannot answer those is
not auditable however confident it sounds.

So the assertions here are mostly about what the chain is forbidden to do:
state anything uncited, claim corroboration it did not have, treat an
agent's silence as evidence when that agent barely looked.
"""

from __future__ import annotations

import pytest

from preflight.models import Adversarial, Evidence, Finding, PolicyRef
from preflight.scoring.incidents import correlate
from preflight.scoring.reasoning import (
    STEPS,
    Claim,
    Source,
    UnsourcedClaim,
    explain,
    explain_all,
)

DURATION = 600_000


def finding(
    fid: str = "f1",
    *,
    start: int = 134_000,
    end: int = 134_800,
    clause: str = "AF-01",
    category: str = "Language",
    modality: str = "speech",
    severity: str = "MEDIUM",
    confidence: float = 0.85,
    transcript: str = "a quoted phrase",
    charge: str = "Strong profanity used as an intensifier.",
    defense: str | None = "Educational framing; the term is quoted, not directed.",
    rationale: str = "Upheld against AF-01; the exemption does not reach this use.",
    verdict: str = "UPHELD",
) -> Finding:
    return Finding(
        id=fid,
        clauseId=clause,
        category=category,
        title=f"{category} issue",
        description="d",
        startMs=start,
        endMs=end,
        severity=severity,  # type: ignore[arg-type]
        confidence=confidence,
        modalities={modality: confidence},
        evidence=Evidence(transcript=transcript),
        policy=PolicyRef(clause, "Inappropriate language", "§ 1.1", "clause text"),
        adversarial=Adversarial(
            charge=charge,
            rationale=rationale,
            confidence=confidence,
            defense=defense,
            defense_strength=0.4 if defense else 0.0,
            verdict=verdict,  # type: ignore[arg-type]
        ),
    )


def chain_for(findings, **kwargs):
    incidents = correlate(findings, DURATION)
    return explain(incidents[0], findings, **kwargs)


class TestNothingIsSaidWithoutACitation:
    """The rule the type system enforces, tested at the type."""

    def test_a_claim_cannot_be_built_without_a_source(self):
        with pytest.raises(UnsourcedClaim):
            Claim(step="policy", text="This violates policy.", source=Source("clause", ""))

    def test_an_empty_claim_is_rejected(self):
        with pytest.raises(UnsourcedClaim):
            Claim(step="policy", text="   ", source=Source("clause", "AF-01"))

    def test_an_unknown_step_is_rejected(self):
        """A step outside the chain would render nowhere and audit as
        nothing."""
        with pytest.raises(UnsourcedClaim):
            Claim(step="vibes", text="seems bad", source=Source("finding", "f1"))

    def test_every_claim_in_a_real_chain_cites_something(self):
        chain = chain_for([finding()])
        assert chain.claims
        for claim in chain.claims:
            assert claim.source.ref.strip(), f"uncited: {claim.text}"
            assert claim.step in STEPS

    def test_every_cited_finding_exists_in_the_run(self):
        """The reviewer's first probe: does this reference resolve?"""
        findings = [finding("f1"), finding("f2", start=400_000, end=400_500)]
        ids = {f.id for f in findings}
        for chain in explain_all(correlate(findings, DURATION), findings):
            for claim in chain.claims:
                if claim.source.kind == "finding":
                    assert claim.source.ref in ids

    def test_a_chain_cannot_be_built_for_findings_that_are_not_present(self):
        incidents = correlate([finding("f1")], DURATION)
        with pytest.raises(UnsourcedClaim):
            explain(incidents[0], [])


class TestBothSidesAreStated:
    def test_the_risk_argument_is_the_auditors_own_charge(self):
        """Rewriting it would be re-arguing a case already decided."""
        chain = chain_for([finding(charge="Profanity in the opening thirty seconds.")])
        assert any(
            "opening thirty seconds" in c.text for c in chain.step("risk_argument")
        )

    def test_the_counter_argument_is_the_advocates_own_defence(self):
        chain = chain_for([finding(defense="Quoted from a historical speech.")])
        assert any("historical speech" in c.text for c in chain.step("counter_argument"))

    def test_absence_of_a_defence_is_stated_not_omitted(self):
        """A missing section reads as a lost one. Saying no defence was
        offered is the auditable version."""
        chain = chain_for([finding(defense=None)])
        counter = chain.step("counter_argument")
        assert counter and "No defence" in counter[0].text

    def test_the_decision_carries_the_adjudicators_rationale(self):
        chain = chain_for([finding(rationale="Upheld; exemption does not apply.")])
        assert any("exemption does not apply" in c.text for c in chain.step("decision"))


class TestUncertaintyIsHonest:
    """The section a reviewer reads to find the weakness, which means it has
    to actually contain one when there is one."""

    def test_an_agent_that_never_ran_is_not_treated_as_disagreement(self):
        chain = chain_for(
            [finding()], coverage={"vision": 0.0}, known_agents=["speech", "vision"]
        )
        text = " ".join(c.text for c in chain.unresolved)
        assert "did not run" in text
        assert "neither supports nor contradicts" in text

    def test_a_barely_covered_agents_silence_is_not_absence(self):
        """Vision at 9% coverage reporting nothing has not established that
        nothing was there — it barely looked. Counting that as exculpatory
        is how a report talks itself into confidence it has not earned."""
        chain = chain_for(
            [finding()], coverage={"vision": 0.09}, known_agents=["speech", "vision"]
        )
        text = " ".join(c.text for c in chain.unresolved)
        assert "too little for its silence" in text

    def test_a_well_covered_agents_silence_is_recorded_as_evidence(self):
        chain = chain_for(
            [finding()], coverage={"vision": 0.95}, known_agents=["speech", "vision"]
        )
        text = " ".join(c.text for c in chain.unresolved)
        assert "found nothing supporting" in text

    def test_a_scheduler_is_never_cited_as_having_found_nothing(self):
        """Caught by the adversarial pass on real output, where the chain
        claimed "orchestrator agent examined 100% of the material and found
        nothing supporting this incident".

        The orchestrator schedules, ingest demuxes, score fuses. None of
        them look for content, so none of them can fail to find it. Listing
        them made the incident look checked by ten agents when three had no
        such capability — false corroboration dressed as thoroughness.
        """
        chain = chain_for(
            [finding()],
            coverage={"orchestrator": 1.0, "ingest": 1.0, "score": 1.0, "vision": 1.0},
            known_agents=["speech", "orchestrator", "ingest", "score", "vision"],
        )
        cited = {c.source.ref for c in chain.unresolved}
        assert not cited & {"orchestrator", "ingest", "score"}, (
            f"non-detector cited as evidence: {cited}"
        )
        assert "vision" in cited, "a real detector's silence should still be reported"

    def test_a_single_agent_incident_admits_it_is_uncorroborated(self):
        chain = chain_for([finding()])
        assert any("Nothing independent corroborates" in c.text for c in chain.unresolved)

    def test_a_corroborated_incident_does_not_claim_to_be_alone(self):
        findings = [
            finding("f1", modality="speech", category="Substances"),
            finding("f2", start=134_500, end=135_200, modality="vision",
                    category="Substances"),
        ]
        chain = chain_for(findings)
        assert not any("Nothing independent" in c.text for c in chain.unresolved)


class TestWhatWasRejected:
    def test_a_dismissed_charge_is_reported_with_its_reason(self):
        """"Which observations were ignored, and why" — a report that cannot
        answer looks like one that never considered the alternative."""
        findings = [
            finding("f1"),
            finding("f2", start=134_100, end=134_600, verdict="DISMISSED",
                    rationale="Attributed quotation; AF-01 exemption applies."),
        ]
        chain = chain_for(findings)
        assert chain.dismissed
        assert "exemption applies" in chain.dismissed[0].text

    def test_a_dismissed_charge_does_not_drive_the_decision(self):
        """It is recorded, not argued from."""
        findings = [
            finding("f1", charge="UPHELD CHARGE"),
            finding("f2", start=134_100, end=134_600, verdict="DISMISSED",
                    charge="DISMISSED CHARGE"),
        ]
        chain = chain_for(findings)
        risk = " ".join(c.text for c in chain.step("risk_argument"))
        assert "UPHELD CHARGE" in risk
        assert "DISMISSED CHARGE" not in risk

    def test_an_all_dismissed_incident_still_produces_a_chain(self):
        """It has no case to state, but it must still be explainable."""
        chain = chain_for([finding(verdict="DISMISSED")])
        assert chain.claims
        assert chain.dismissed


class TestAttribution:
    def test_the_producing_agent_is_named(self):
        chain = chain_for([finding(modality="ocr")])
        assert "ocr" in chain.agents_cited

    def test_the_clause_is_named(self):
        chain = chain_for([finding(clause="AF-14")])
        assert "AF-14" in chain.clauses_cited

    def test_a_measured_finding_says_so_rather_than_faking_a_quote(self):
        """Loudness has no quotable span. An empty evidence section would
        read as evidence that went missing."""
        chain = chain_for([finding(transcript="", clause="AUD-01")])
        evidence = chain.step("evidence")
        assert evidence and "measurement, not a classification" in evidence[0].text

    def test_the_chain_serialises_whole(self):
        import json

        payload = chain_for([finding()]).to_json()
        json.dumps(payload)
        assert payload["claims"] and payload["agentsCited"]


class TestEveryIncidentIsExplainable:
    """Property-style: whatever the shape of the run, no incident may be
    left without a chain, and no chain may contain an uncited claim."""

    @pytest.mark.parametrize("count", [1, 2, 5, 12, 40])
    def test_every_incident_gets_a_complete_chain(self, count):
        findings = [
            finding(f"f{i}", start=i * 20_000, end=i * 20_000 + 500)
            for i in range(count)
        ]
        incidents = correlate(findings, DURATION)
        chains = explain_all(incidents, findings, known_agents=["speech", "vision"])
        assert len(chains) == len(incidents)
        for chain in chains:
            assert chain.step("observation"), f"{chain.incident_id} states no observation"
            assert chain.step("policy"), f"{chain.incident_id} cites no clause"
            assert chain.step("decision"), f"{chain.incident_id} reaches no decision"
            for claim in chain.claims:
                assert claim.source.ref.strip()

    @pytest.mark.parametrize(
        "transcript,defense,rationale",
        [("", None, ""), ("x", None, ""), ("", "d", "r"), ("x", "d", "r")],
    )
    def test_sparse_findings_still_produce_a_valid_chain(
        self, transcript, defense, rationale
    ):
        """Deterministic agents fill these fields thinly, and a chain that
        only works for triad output would fail on most of a real report."""
        chain = chain_for(
            [finding(transcript=transcript, defense=defense, rationale=rationale)]
        )
        assert chain.step("evidence")
        assert chain.step("counter_argument")
        assert chain.step("decision")
