"""Why PREFLIGHT decided what it decided — as a citable chain.

The goal is not to predict YouTube. It is to make PREFLIGHT's own conclusion
auditable: a reader who disagrees should be able to find the exact
observation, clause, or argument they disagree with, rather than being asked
to trust a number.

**The no-hallucination rule is enforced by the type, not by discipline.**
`Claim` cannot be constructed without a `source`, and every source is a
reference to something that already exists in the run — a finding id, a
clause id, an agent id. There is no code path that produces a sentence
without a citation attached, so "never invent evidence" is a property of the
representation rather than a rule someone has to remember. Prose assembled
from a template still has to name the finding it was assembled from.

Nothing here calls a model. Every step reads material the run already
produced: the AUDITOR's charge is the risk argument, the ADVOCATE's defence
is the counter-argument, the ADJUDICATOR's rationale is the ruling, and the
clause text is the policy. Generating new reasoning at report time would be
a fifth opinion nobody adjudicated.

The most valuable section is the one most reports omit: what was *not*
concluded, and why. Dismissed charges, agents that never observed the
moment, and confidence lost to partial coverage are all recorded, because a
reviewer's first question about any finding is what would have changed it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

from preflight.models import Finding
from preflight.scoring.incidents import Incident

SourceKind = Literal["finding", "clause", "agent", "measurement"]

# What each step of the chain is for. Ordered as a reader walks them.
STEPS = (
    "observation",
    "evidence",
    "policy",
    "risk_argument",
    "counter_argument",
    "decision",
    "uncertainty",
)

# Below this, a corroborating agent's own coverage is too thin for its
# silence to mean anything. An agent that saw 9% of the frames and reported
# nothing has not established absence — it barely looked.
SILENCE_IS_MEANINGFUL_ABOVE = 0.5

# Agents that can produce a finding. Only their silence carries information.
#
# The adversarial pass caught the first version claiming "orchestrator agent
# examined 100% of the material and found nothing supporting this incident".
# The orchestrator schedules, ingest demuxes, score fuses — none of them look
# for content at all, so none of them can fail to find something. Listing
# them made an incident appear checked by ten agents when three had no such
# capability, which is false corroboration dressed as thoroughness and
# exactly the inflation an auditor should catch.
DETECTOR_AGENTS = frozenset(
    {"speech", "vision", "ocr", "audio", "access", "meta", "policy", "music"}
)


class UnsourcedClaim(ValueError):
    """Raised when a statement is offered without something to cite."""


@dataclass(frozen=True)
class Source:
    """A pointer to something that exists in the run."""

    kind: SourceKind
    ref: str
    detail: str = ""

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "ref": self.ref, "detail": self.detail}


@dataclass(frozen=True)
class Claim:
    """One sentence, and the thing that entitles it to be said.

    Validated in `__post_init__` rather than trusted: a Claim carrying an
    empty ref is exactly the unsourced assertion this module exists to make
    impossible, and letting one through silently would defeat the whole
    design.
    """

    step: str
    text: str
    source: Source

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise UnsourcedClaim(f"empty claim in step {self.step!r}")
        if not self.source.ref.strip():
            raise UnsourcedClaim(f"claim in step {self.step!r} cites nothing: {self.text!r}")
        if self.step not in STEPS:
            raise UnsourcedClaim(f"unknown reasoning step {self.step!r}")

    def to_json(self) -> dict[str, Any]:
        return {"step": self.step, "text": self.text, "source": self.source.to_json()}


@dataclass(frozen=True)
class ReasoningChain:
    """The full argument for one incident, in reading order."""

    incident_id: str
    claims: list[Claim]
    decision: str
    confidence: float
    dismissed: list[Claim] = field(default_factory=list)
    unresolved: list[Claim] = field(default_factory=list)

    def step(self, name: str) -> list[Claim]:
        return [claim for claim in self.claims if claim.step == name]

    @property
    def agents_cited(self) -> list[str]:
        return sorted(
            {c.source.ref for c in self.claims if c.source.kind == "agent"}
        )

    @property
    def clauses_cited(self) -> list[str]:
        return sorted(
            {c.source.ref for c in self.claims if c.source.kind == "clause"}
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "incidentId": self.incident_id,
            "decision": self.decision,
            "confidence": self.confidence,
            "claims": [c.to_json() for c in self.claims],
            "dismissed": [c.to_json() for c in self.dismissed],
            "unresolved": [c.to_json() for c in self.unresolved],
            "agentsCited": self.agents_cited,
            "clausesCited": self.clauses_cited,
        }


def _observation(finding: Finding) -> list[Claim]:
    """Who saw what. Named per agent, because "which agent produced this"
    is the first question asked of any automated finding."""
    claims = []
    for agent in sorted(a for a, v in (finding.modalities or {}).items() if v > 0):
        claims.append(
            Claim(
                step="observation",
                text=f"{agent} agent reported: {finding.title}.",
                source=Source("agent", agent, f"finding {finding.id}"),
            )
        )
    if not claims:
        # A finding with no modality is still an observation by whichever
        # agent produced it; the clause family is the only honest attribution
        # available, and inventing an agent name would be worse.
        claims.append(
            Claim(
                step="observation",
                text=f"Reported: {finding.title}.",
                source=Source("finding", finding.id, finding.clauseId),
            )
        )
    return claims


def _evidence(finding: Finding) -> list[Claim]:
    claims = []
    quoted = (finding.evidence.transcript or "").strip()
    if quoted:
        claims.append(
            Claim(
                step="evidence",
                text=f'Evidence: "{quoted}"',
                source=Source("finding", finding.id, "transcript span"),
            )
        )
    frames = list(finding.evidence.frames or [])
    if frames:
        claims.append(
            Claim(
                step="evidence",
                text=f"{len(frames)} supporting frame(s) attached.",
                source=Source("finding", finding.id, "keyframes"),
            )
        )
    if not claims:
        # A measured finding — loudness, a black gap — has no quotable
        # evidence. Saying so is more honest than an empty section that
        # reads as though the evidence was lost.
        claims.append(
            Claim(
                step="evidence",
                text=(
                    f"Measured value over {finding.startMs}–{finding.endMs}ms; "
                    "no quotable span (this is a measurement, not a classification)."
                ),
                source=Source("measurement", finding.id, finding.clauseId),
            )
        )
    return claims


def _policy(finding: Finding) -> list[Claim]:
    reference = finding.policy
    return [
        Claim(
            step="policy",
            text=f"Judged under {reference.clauseId} — {reference.title} ({reference.section}).",
            source=Source("clause", reference.clauseId, reference.section),
        )
    ]


def _arguments(finding: Finding) -> tuple[list[Claim], list[Claim]]:
    """The charge and the defence, verbatim from the triad that produced
    them. Rewriting either would be re-arguing a case already decided."""
    adversarial = finding.adversarial
    risk = []
    if adversarial.charge.strip():
        risk.append(
            Claim(
                step="risk_argument",
                text=adversarial.charge.strip(),
                source=Source("finding", finding.id, "auditor charge"),
            )
        )

    counter = []
    defence = (adversarial.defense or "").strip()
    if defence:
        counter.append(
            Claim(
                step="counter_argument",
                text=defence,
                source=Source(
                    "finding",
                    finding.id,
                    f"advocate defence (strength {adversarial.defense_strength:.2f})",
                ),
            )
        )
    else:
        counter.append(
            Claim(
                step="counter_argument",
                text=(
                    "No defence was offered. For a measured value there is nothing "
                    "to argue with; for an adjudicated charge the advocate found "
                    "no exemption that applied."
                ),
                source=Source("finding", finding.id, "advocate returned no defence"),
            )
        )
    return risk, counter


def _uncertainty(
    incident: Incident,
    coverage: dict[str, float],
    all_agents: Iterable[str],
) -> list[Claim]:
    """What would change this conclusion.

    Two distinct kinds, and conflating them is the mistake worth avoiding:
    an agent that looked and saw nothing is weak counter-evidence, while an
    agent that never looked is no evidence at all. Only the first is
    silence; the second is a gap.
    """
    claims: list[Claim] = []
    participating = set(incident.agents)

    # Only detectors. A scheduler's silence is not evidence of anything.
    candidates = (set(all_agents) & DETECTOR_AGENTS) - participating

    for agent in sorted(candidates):
        seen = coverage.get(agent, 0.0)
        if seen <= 0.0:
            claims.append(
                Claim(
                    step="uncertainty",
                    text=(
                        f"{agent} agent did not run, so it neither supports nor "
                        "contradicts this incident."
                    ),
                    source=Source("agent", agent, "coverage 0%"),
                )
            )
        elif seen < SILENCE_IS_MEANINGFUL_ABOVE:
            claims.append(
                Claim(
                    step="uncertainty",
                    text=(
                        f"{agent} agent reported nothing here, but examined only "
                        f"{seen:.0%} of the material — too little for its silence "
                        "to count as absence."
                    ),
                    source=Source("agent", agent, f"coverage {seen:.0%}"),
                )
            )
        else:
            claims.append(
                Claim(
                    step="uncertainty",
                    text=(
                        f"{agent} agent examined {seen:.0%} of the material and "
                        "found nothing supporting this incident."
                    ),
                    source=Source("agent", agent, f"coverage {seen:.0%}"),
                )
            )

    if not incident.corroborated:
        claims.append(
            Claim(
                step="uncertainty",
                text=(
                    "Only one agent observed this. Nothing independent corroborates it, "
                    "so the confidence rests entirely on that single observation."
                ),
                source=Source("agent", incident.agents[0] if incident.agents else "unknown",
                              "sole observer"),
            )
        )
    return claims


def _dismissed(findings: list[Finding]) -> list[Claim]:
    """Charges that were raised and then rejected.

    "Which observations were ignored, and why" is the question a reviewer
    asks second, and a report that cannot answer it looks like one that
    never considered the alternative.
    """
    claims = []
    for finding in findings:
        if finding.adversarial.verdict == "DISMISSED":
            claims.append(
                Claim(
                    step="decision",
                    text=(
                        f"{finding.clauseId} was charged and dismissed: "
                        f"{finding.adversarial.rationale.strip() or 'no rationale recorded'}"
                    ),
                    source=Source("finding", finding.id, "adjudicator dismissal"),
                )
            )
    return claims


def explain(
    incident: Incident,
    findings: list[Finding],
    *,
    coverage: dict[str, float] | None = None,
    known_agents: Iterable[str] = (),
) -> ReasoningChain:
    """Build the chain for one incident from material the run produced."""
    members = [f for f in findings if f.id in set(incident.finding_ids)]
    if not members:
        raise UnsourcedClaim(
            f"{incident.id} cites findings that are not in the run: {incident.finding_ids}"
        )

    claims: list[Claim] = []
    upheld = [f for f in members if f.adversarial.verdict != "DISMISSED"]
    # An incident built entirely from dismissed charges has no case to state;
    # its members still appear under `dismissed`.
    basis = upheld or members

    for finding in basis:
        claims.extend(_observation(finding))
        claims.extend(_evidence(finding))
        claims.extend(_policy(finding))
        risk, counter = _arguments(finding)
        claims.extend(risk)
        claims.extend(counter)

    lead = max(basis, key=lambda f: f.fusedConfidence or f.confidence)
    claims.append(
        Claim(
            step="decision",
            text=(
                f"{incident.severity} risk. {lead.adversarial.rationale.strip()}"
                if lead.adversarial.rationale.strip()
                else f"{incident.severity} risk."
            ),
            source=Source("finding", lead.id, "adjudicator ruling"),
        )
    )

    unresolved = _uncertainty(incident, coverage or {}, known_agents)
    claims.extend(unresolved)

    return ReasoningChain(
        incident_id=incident.id,
        claims=claims,
        decision=f"{incident.severity} risk",
        confidence=incident.confidence,
        dismissed=_dismissed(members),
        unresolved=unresolved,
    )


def explain_all(
    incidents: list[Incident],
    findings: list[Finding],
    *,
    coverage: dict[str, float] | None = None,
    known_agents: Iterable[str] = (),
) -> list[ReasoningChain]:
    return [
        explain(
            incident, findings, coverage=coverage, known_agents=known_agents
        )
        for incident in incidents
    ]
