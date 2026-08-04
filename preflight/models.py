"""The wire contract, Python side.

These dataclasses mirror `src/types/analysis.ts` exactly. The generated schema
at `schema/analysis-report.schema.json` is the referee: every emitted report is
validated against it before it is written, so neither side can drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Severity = Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
Verdict = Literal[
    "READY_TO_PUBLISH", "PUBLISH_WITH_FIXES", "NOT_READY", "DO_NOT_PUBLISH"
]
AgentStatus = Literal["OK", "DEGRADED", "FAILED", "SKIPPED", "RUNNING", "PENDING"]
OpKind = Literal["MUTE", "BLEEP", "BLUR_REGION", "REPLACE_AUDIO", "CUT"]
FixKind = Literal["MUTE", "BLEEP", "BLUR_REGION", "REPLACE_AUDIO", "CUT", "NONE"]

SEVERITY_RANK: dict[str, int] = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


@dataclass
class PolicyRef:
    clauseId: str
    title: str
    section: str
    text: str

    def to_json(self) -> dict[str, Any]:
        return {
            "clauseId": self.clauseId,
            "title": self.title,
            "section": self.section,
            "text": self.text,
        }


@dataclass
class Evidence:
    transcript: str
    highlightSpan: tuple[int, int] = (0, 0)
    frames: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "transcript": self.transcript,
            "highlightSpan": [self.highlightSpan[0], self.highlightSpan[1]],
            "frames": list(self.frames),
        }

    @staticmethod
    def marking(text: str, needle: str, frames: list[str] | None = None) -> "Evidence":
        """Locate `needle` in `text` so highlight offsets cannot drift."""
        start = text.find(needle)
        span = (start, start + len(needle)) if start >= 0 else (0, 0)
        return Evidence(transcript=text, highlightSpan=span, frames=frames or [])


@dataclass
class Adversarial:
    """The three-agent record.

    Deterministic agents fill this too. A measured value still has a charge (the
    measurement), an absent defence (you cannot argue with a number), and a
    ruling — and populating it uniformly means the UI never has to special-case
    a finding that arrived without an argument.
    """

    charge: str
    rationale: str
    confidence: float
    defense: str | None = None
    defense_strength: float = 0.0
    verdict: Literal["UPHELD", "DISMISSED"] = "UPHELD"

    def to_json(self) -> dict[str, Any]:
        return {
            "auditor": {"charge": self.charge},
            "advocate": {
                "defense": self.defense,
                "strength": round(self.defense_strength, 3),
            },
            "adjudicator": {
                "verdict": self.verdict,
                "rationale": self.rationale,
                "confidence": round(self.confidence, 3),
            },
        }


@dataclass
class Finding:
    id: str
    clauseId: str
    category: str
    title: str
    description: str
    startMs: int
    endMs: int
    severity: Severity
    confidence: float
    modalities: dict[str, float]
    evidence: Evidence
    policy: PolicyRef
    adversarial: Adversarial
    suggestedFix: FixKind = "NONE"
    fusedConfidence: float | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "clauseId": self.clauseId,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "startMs": int(self.startMs),
            "endMs": int(self.endMs),
            "severity": self.severity,
            "confidence": round(self.confidence, 3),
            "modalities": {k: round(v, 3) for k, v in self.modalities.items()},
            "fusedConfidence": round(
                self.fusedConfidence if self.fusedConfidence is not None else self.confidence,
                3,
            ),
            "evidence": self.evidence.to_json(),
            "policy": self.policy.to_json(),
            "adversarial": self.adversarial.to_json(),
            "suggestedFix": self.suggestedFix,
        }


@dataclass
class AgentResult:
    """What every agent returns, whether it succeeded or not.

    `status` and `coverage` are as important as `findings`. An agent that saw
    42% of the keyframes must say so, because the alternative is a report that
    looks complete and is not.
    """

    agent_id: str
    name: str
    status: AgentStatus = "OK"
    findings: list[Finding] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    coverage: float = 1.0
    error: str | None = None
    elapsed_ms: int = 0
    calls: int = 0
    log: list[str] = field(default_factory=list)

    @property
    def detail(self) -> str:
        """One line for the terminal panel."""
        if self.error:
            return self.error
        return self.log[-1] if self.log else self.name

    def to_agent_run(self, tier: int, parents: list[str], ts_ms: int) -> dict[str, Any]:
        return {
            "id": self.agent_id,
            "name": self.name,
            "tier": tier,
            "parents": parents,
            "status": self.status,
            "detail": self.detail,
            "coverage": round(self.coverage, 4),
            "elapsedMs": int(self.elapsed_ms),
            "tsMs": int(ts_ms),
            "calls": int(self.calls),
        }

    @classmethod
    def skipped(cls, agent_id: str, name: str, reason: str) -> "AgentResult":
        return cls(
            agent_id=agent_id,
            name=name,
            status="SKIPPED",
            coverage=0.0,
            error=reason,
            log=[reason],
        )

    @classmethod
    def failed(cls, agent_id: str, name: str, reason: str) -> "AgentResult":
        return cls(
            agent_id=agent_id,
            name=name,
            status="FAILED",
            coverage=0.0,
            error=reason,
            log=[reason],
        )
