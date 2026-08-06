"""Conformance: the running pipeline must match its own specification.

`prompts/` declares twelve agents, their tiers, their dependencies and their
contracts. This asserts the implementation agrees. A spec that nothing checks
is a README with extra steps; these tests are what make it binding.

This is the same move PREFLIGHT makes on video — a specification compiled into
a linter — turned on the pipeline itself.
"""

from __future__ import annotations

import json

import pytest

from preflight.agents.roster import AgentSpec, load_roster, prompt_for
from preflight.pipeline import SURFACE_WEIGHT, TOPOLOGY

EXPECTED_AGENTS = 12
MODEL_DRIVEN = {"A03", "A09", "A10", "A11"}


@pytest.fixture(scope="module")
def roster():
    return load_roster("prompts")


class TestRosterLoads:
    def test_declares_twelve_agents(self, roster):
        assert len(roster.agents) == EXPECTED_AGENTS
        assert set(roster.agents) == {f"A{n:02d}" for n in range(1, 13)}

    def test_every_agent_has_a_codename_and_an_implementation(self, roster):
        for spec in roster.ordered:
            assert spec.codename and spec.codename.isupper(), spec.agent_id
            assert spec.implementation, spec.agent_id

    def test_codenames_are_unique(self, roster):
        names = [s.codename for s in roster.ordered]
        assert len(set(names)) == len(names)

    def test_digest_is_stable_across_loads(self, roster):
        load_roster.cache_clear()
        assert load_roster("prompts").digest == roster.digest

    def test_digest_changes_when_a_prompt_changes(self, roster, tmp_path):
        """Prompt text is hashed into the attestation. A finding produced under
        one adjudicator prompt is not the same evidence as one produced under
        another."""
        for path in (roster.agents["A11"].path,):
            copy_dir = tmp_path / "prompts"
            copy_dir.mkdir()
            for spec in roster.ordered:
                (copy_dir / spec.path.name).write_text(
                    spec.path.read_text(encoding="utf-8"), encoding="utf-8"
                )
            target = copy_dir / path.name
            target.write_text(
                target.read_text(encoding="utf-8") + "\nAn extra instruction.\n",
                encoding="utf-8",
            )
            assert load_roster(copy_dir).digest != roster.digest


class TestRosterIsWellFormed:
    def test_declares_a_valid_dag(self, roster):
        assert roster.validate() == []

    def test_orchestrator_has_no_parents(self, roster):
        assert roster["A01"].parents == ()
        assert roster["A01"].tier == 0

    def test_every_non_orchestrator_agent_has_a_parent(self, roster):
        for spec in roster.ordered:
            if spec.agent_id == "A01":
                continue
            assert spec.parents, f"{spec.agent_id} is unreachable"

    def test_parents_always_sit_in_a_lower_tier(self, roster):
        for spec in roster.ordered:
            for parent in spec.parents:
                assert roster[parent].tier < spec.tier, f"{spec.agent_id} -> {parent}"


class TestModelVersusDeterministic:
    def test_exactly_the_expected_agents_call_a_model(self, roster):
        """Only agents that talk to a model carry a prompt. Writing a system
        prompt for a function that computes spectral flatness would be text
        that is never sent anywhere."""
        assert {s.agent_id for s in roster.model_driven} == MODEL_DRIVEN

    def test_model_driven_agents_declare_a_capability_not_a_vendor(self, roster):
        for spec in roster.model_driven:
            assert spec.model_capability != "none"
            assert "." in spec.model_capability  # chat.reasoning, vision.describe
            # Agents request capabilities. A vendor name here would mean an
            # agent knows who serves it, which is the coupling the provider
            # layer exists to remove.
            for vendor in ("nvidia", "openai", "anthropic", "qdrant"):
                assert vendor not in spec.model_capability.lower()

    def test_deterministic_agents_request_no_capability(self, roster):
        for spec in roster.ordered:
            if spec.agent_id in MODEL_DRIVEN:
                continue
            assert spec.model_capability == "none", spec.agent_id
            assert spec.prompt == "", spec.agent_id

    def test_the_orchestrator_is_deterministic(self, roster):
        """Asked to orchestrate, a model would emit an agents_completed list
        indistinguishable from a real one whether or not the agents ran."""
        assert roster["A01"].kind == "deterministic"
        assert roster["A01"].model_capability == "none"


class TestPromptContent:
    @pytest.mark.parametrize("agent_id", sorted(MODEL_DRIVEN))
    def test_prompt_is_non_trivial(self, roster, agent_id):
        prompt = roster[agent_id].prompt
        assert len(prompt) > 300, agent_id

    @pytest.mark.parametrize("agent_id", sorted(MODEL_DRIVEN))
    def test_prompt_demands_json_without_fences(self, roster, agent_id):
        prompt = roster[agent_id].prompt.lower()
        assert "only valid json" in prompt, agent_id
        assert "fence" in prompt or "markdown" in prompt, agent_id

    @pytest.mark.parametrize("agent_id", ["A03", "A09"])
    def test_empty_result_is_explicitly_legitimate(self, roster, agent_id):
        """Without this, an auditor learns that finding something is always the
        expected behaviour and hallucinates violations in clean footage."""
        prompt = roster[agent_id].prompt.lower()
        assert "empty" in prompt or "nothing detected" in prompt, agent_id

    def test_advocate_is_told_to_concede(self, roster):
        prompt = roster["A10"].prompt.lower()
        assert "null" in prompt
        assert "fabricate" in prompt

    def test_auditor_is_confined_to_supplied_clauses(self, roster):
        prompt = roster["A09"].prompt.lower()
        assert "only cite clause_ids that appear" in prompt

    def test_adjudicator_matches_the_fix_to_the_evidence(self, roster):
        """A blur over spoken evidence is applied, reported as done, and
        changes nothing a classifier hears."""
        prompt = roster["A11"].prompt.lower()
        assert "audio evidence takes an audio fix" in prompt

    def test_no_prompt_leaks_a_vendor_or_model_name(self, roster):
        for spec in roster.model_driven:
            lowered = spec.prompt.lower()
            for token in ("nvidia", "llama", "gpt-", "claude", "nemotron"):
                assert token not in lowered, f"{spec.agent_id} names {token}"

    def test_prompt_for_returns_empty_for_a_deterministic_agent(self):
        assert prompt_for("A04", "prompts") == ""
        assert prompt_for("A09", "prompts") != ""


class TestImplementationConformance:
    """The specification and the running pipeline must not drift."""

    # prompts/ id -> the id the pipeline uses internally.
    IMPLEMENTED = {
        "A01": "orchestrator",
        "A02": "speech",
        "A03": "vision",
        "A04": "audio",
        "A05": "ocr",
        "A06": "meta",
        "A07": "policy",
        "A08": "score",
        "A12": "remedy",
    }

    def test_every_specified_agent_has_a_pipeline_counterpart(self, roster):
        for agent_id, pipeline_id in self.IMPLEMENTED.items():
            assert agent_id in roster.agents
            assert pipeline_id in TOPOLOGY, f"{agent_id} -> {pipeline_id}"

    def test_pipeline_agents_carry_a_coverage_weight(self):
        """An agent with no weight contributes nothing to coverage, which means
        it could fail silently without the report noticing."""
        for pipeline_id in TOPOLOGY:
            assert pipeline_id in SURFACE_WEIGHT, pipeline_id

    def test_coverage_weights_sum_to_one(self):
        assert sum(SURFACE_WEIGHT.values()) == pytest.approx(1.0)

    def test_the_triad_is_a_cascade_not_a_committee(self, roster):
        """A09 -> A10 -> A11 is sequential: the advocate answers the charge and
        the adjudicator reads both. Declaring them at one tier made the roster
        an invalid DAG, which the validator caught."""
        assert roster["A09"].tier < roster["A10"].tier < roster["A11"].tier
        assert roster["A10"].parents == ("A09",)
        assert roster["A11"].parents == ("A09", "A10")
        # Three prompts, one pipeline stage — the cascade runs inside `policy`.
        assert "policy" in TOPOLOGY

    def test_fusion_consumes_the_triad_rather_than_racing_it(self, roster):
        assert roster["A08"].tier > roster["A11"].tier
        assert roster["A12"].tier > roster["A08"].tier

    def test_implemented_agents_have_the_code_they_declare(self, roster):
        """An agent claiming to be implemented must point at real code. This
        caught A03 and A05 declaring modules that were never written."""
        from pathlib import Path

        for spec in roster.ordered:
            if not spec.implemented or not spec.implementation:
                continue
            assert Path(spec.implementation).exists(), (
                f"{spec.agent_id} claims implemented but {spec.implementation} "
                "is missing — mark it `status: unimplemented` or write it"
            )

    def test_the_roster_declares_what_is_built_in_both_directions(self, roster):
        """The roster is a status board, not a directory of files that all
        look finished. All twelve are now built; the assertion stays because
        the failure it guards against is a spec drifting away from its code,
        which is as easy to do downward as upward."""
        unbuilt = {s.agent_id for s in roster.ordered if not s.implemented}
        assert unbuilt == set()
        assert roster.validate() == []

    def test_a_spec_claiming_code_that_is_absent_is_a_validation_error(self):
        """The check that caught A03 and A05 declaring modules never written,
        now enforced by the library rather than only by this file."""
        from dataclasses import replace

        from preflight.agents.roster import Roster

        real = load_roster("prompts")["A05"]
        broken = Roster(agents={"A05": replace(real, implementation="preflight/nope.py")})
        problems = broken.validate()
        assert any("preflight/nope.py does not exist" in p for p in problems)

    def test_no_unimplemented_agent_is_required_by_an_implemented_one(self, roster):
        """A built agent depending on an unbuilt one is a pipeline that cannot
        run, which is worth knowing before the run rather than during it."""
        unbuilt = {s.agent_id for s in roster.ordered if not s.implemented}
        for spec in roster.ordered:
            if not spec.implemented:
                continue
            blocking = set(spec.parents) & unbuilt
            # A05 depends on A03; both unbuilt, so neither blocks anything
            # built. A07 lists A03/A05 as parents but degrades without them.
            if blocking and spec.agent_id not in {"A07"}:
                pytest.fail(f"{spec.agent_id} depends on unbuilt {sorted(blocking)}")


class TestSerialisation:
    def test_roster_json_is_serialisable_and_carries_no_prompt_text(self, roster):
        payload = json.dumps(roster.to_json())
        assert "You are AUDITOR" not in payload
        assert len(json.loads(payload)["agents"]) == EXPECTED_AGENTS

    def test_agent_json_records_the_hash(self, roster):
        entry = roster["A11"].to_json()
        assert len(entry["sha256"]) == 16
        assert entry["capability"] == "chat.reasoning"
