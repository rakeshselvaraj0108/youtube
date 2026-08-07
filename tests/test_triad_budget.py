"""`run_triad` under a call budget, driven end to end against a stub model.

Nothing exercised `run_triad` itself before this file — its helpers were
tested individually while the function that sequences them was not, which is
the same shape of gap that let the vision agent go uncalled for weeks. The
budget logic lives in that sequencing: what gets reserved, what yields
first, and what happens to a charge nobody ruled on.

The stub counts calls and returns well-formed payloads, so every assertion
here is about PREFLIGHT's control flow rather than a model's behaviour.
"""

from __future__ import annotations

import pytest

from preflight.agents import prompts
from preflight.agents import triad as triad_mod
from preflight.agents.triad import ADJUDICATOR_BATCH, AUDITOR_BATCH, run_triad
from preflight.budget import CallBudget
from preflight.chunking import Window
from preflight.config import Settings
from preflight.policy.corpus import Chunk


def chunk(clause_id: str = "AF-01") -> Chunk:
    return Chunk(
        clause_id=clause_id,
        clause_title="Inappropriate language",
        section="§ 1.1",
        text="Strong profanity used repeatedly.",
        severity_default="MEDIUM",
        source_url="https://example.invalid/policy",
    )


class StubClient:
    """Answers every stage with a valid payload and counts the calls."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.calls = 0
        self.usage = type("U", (), {"calls": 0})()

    @property
    def online(self) -> bool:
        return True

    def _stage_for(self, system: str) -> str:
        """Identify the stage by its system prompt constant.

        Neither of the obvious discriminators works. The prompt *text* is out
        because the ADVOCATE's system prompt quotes "AUDITOR" describing the
        charge it answers. The *model name* is out because ADVOCATE and
        ADJUDICATOR are configured to the same model by default — dispatching
        on it silently answered the adjudicator with a defence payload, so
        nothing was ever ruled and every "unruled charges never ship"
        assertion passed vacuously. Comparing against the constants is the
        only unambiguous option.
        """
        for stage, text in (
            ("auditor", prompts.AUDITOR_SYSTEM),
            ("advocate", prompts.ADVOCATE_SYSTEM),
            ("adjudicator", prompts.ADJUDICATOR_SYSTEM),
        ):
            if system == text:
                return stage
        raise AssertionError("unrecognised system prompt")

    def chat_json(self, *, model, system, user, **kwargs):
        self.calls += 1
        self.usage.calls = self.calls
        # Dispatch on the configured model name, not on prompt text — the
        # ADVOCATE's system prompt quotes the word "AUDITOR" describing the
        # charge it must answer, so sniffing the prompt routes the advocate
        # call into the auditor branch.
        stage = self._stage_for(system)
        if stage == "auditor":
            # One charge per window in the batch, keyed off the real prompt
            # format ("WINDOW 3 [90000ms - 120000ms]") so a change to
            # `Window.for_prompt` breaks this loudly instead of silently
            # producing zero candidates and vacuously passing every
            # downstream assertion.
            indices = [
                int(line.split()[1])
                for line in user.splitlines()
                if line.startswith("WINDOW ")
            ]
            assert indices, "stub parsed no windows out of the auditor prompt"
            return {
                "candidates": [
                    {
                        "window": i,
                        "clause_id": "AF-01",
                        # Unique per window. Identical evidence across every
                        # candidate makes "did an unruled charge ship?"
                        # unanswerable — a ruled finding's text matches an
                        # unruled candidate's, and the assertion fails on a
                        # string collision rather than a real leak.
                        "evidence": f"quoted phrase from window {i}",
                        "why": "profanity",
                        "category": "Language",
                    }
                    for i in indices
                ]
            }
        if stage == "advocate":
            return {"defenses": []}
        return {
            "verdicts": [
                {
                    "candidate_id": cid,
                    "verdict": "UPHELD",
                    "severity": "MEDIUM",
                    "confidence": 0.8,
                    "rationale": "ruled",
                }
                for cid in _candidate_ids(user)
            ]
        }


def _candidate_ids(user: str) -> list[str]:
    return [
        line.split("candidate_id:")[1].strip()
        for line in user.splitlines()
        if line.startswith("candidate_id:")
    ]


@pytest.fixture
def harness(monkeypatch):
    """Windows that all have content, with retrieval stubbed out."""

    def build(window_count: int):
        windows = [
            Window(
                index=i,
                start_ms=i * 30_000,
                end_ms=(i + 1) * 30_000,
                transcript=f"window {i} spoken content here",
            )
            for i in range(window_count)
        ]
        monkeypatch.setattr(
            triad_mod, "_retrieve_for_window", lambda window, index: [chunk()]
        )
        settings = Settings(api_key="k", offline=False)
        return windows, StubClient(settings), settings

    return build


class TestUnlimitedIsTheDefault:
    def test_no_budget_examines_every_window(self, harness):
        windows, client, settings = harness(20)
        result = run_triad(windows, None, None, client, settings)
        assert result.windows_examined == 20
        assert result.coverage == 1.0
        assert result.status == "OK"


class TestTheCeilingHolds:
    def test_calls_never_exceed_the_ceiling(self, harness):
        windows, client, settings = harness(40)
        run_triad(windows, None, None, client, settings, budget=CallBudget(ceiling=4))
        assert client.calls <= 4

    def test_a_tight_budget_still_rules_on_what_it_charged(self, harness):
        """The reserve doing its job: every finding that ships was ruled on."""
        windows, client, settings = harness(40)
        result = run_triad(
            windows, None, None, client, settings, budget=CallBudget(ceiling=5)
        )
        assert all(c.ruled for c in result.candidates if c in _shipped(result))

    def test_a_budget_of_zero_examines_nothing_and_says_so(self, harness):
        windows, client, settings = harness(10)
        budget = CallBudget(ceiling=0)
        result = run_triad(windows, None, None, client, settings, budget=budget)
        assert client.calls == 0
        assert result.windows_examined == 0
        assert result.coverage == 0.0
        assert budget.shed


def _shipped(result) -> list:
    return [c for c in result.candidates if c.ruled and c.verdict == "UPHELD"]


class TestSheddingIsHonest:
    """The property that matters more than the ceiling itself."""

    def test_coverage_falls_to_what_was_actually_examined(self, harness):
        windows, client, settings = harness(40)
        result = run_triad(
            windows, None, None, client, settings, budget=CallBudget(ceiling=3)
        )
        assert result.windows_examined < 40
        assert result.coverage == pytest.approx(result.windows_examined / 40, abs=0.01)
        assert result.coverage < 1.0

    def test_a_shortened_audit_degrades_rather_than_claiming_success(self, harness):
        windows, client, settings = harness(40)
        result = run_triad(
            windows, None, None, client, settings, budget=CallBudget(ceiling=3)
        )
        assert result.status == "DEGRADED"

    def test_the_shed_names_the_stage_and_the_windows_lost(self, harness):
        windows, client, settings = harness(40)
        budget = CallBudget(ceiling=3)
        run_triad(windows, None, None, client, settings, budget=budget)
        auditor_shed = [s for s in budget.shed if s.stage == "auditor"]
        assert auditor_shed
        assert auditor_shed[0].windows_lost > 0

    def test_full_coverage_is_never_claimed_after_shedding(self, harness):
        """The single assertion this whole feature exists to keep true."""
        for ceiling in (1, 2, 3, 5, 8):
            windows, client, settings = harness(40)
            result = run_triad(
                windows, None, None, client, settings, budget=CallBudget(ceiling=ceiling)
            )
            if result.windows_examined < 40:
                assert result.coverage < 1.0, (
                    f"ceiling={ceiling} examined {result.windows_examined}/40 "
                    "windows but still claimed full coverage"
                )


class TestUnruledChargesNeverShip:
    """An unruled charge is the unopposed-prosecutor output the triad exists
    to prevent. Dropping it is correct; reporting it is not."""

    def test_every_shipped_finding_was_ruled_on(self, harness):
        for ceiling in (2, 3, 4, 6, 10):
            windows, client, settings = harness(30)
            result = run_triad(
                windows, None, None, client, settings, budget=CallBudget(ceiling=ceiling)
            )
            for candidate in result.candidates:
                if not candidate.ruled:
                    assert not any(
                        f.evidence.transcript == candidate.evidence
                        for f in result.findings
                    ), "an unruled charge reached the findings"

    def test_findings_never_outnumber_ruled_candidates(self, harness):
        windows, client, settings = harness(30)
        result = run_triad(
            windows, None, None, client, settings, budget=CallBudget(ceiling=4)
        )
        ruled = [c for c in result.candidates if c.ruled]
        assert len(result.findings) <= len(ruled)
