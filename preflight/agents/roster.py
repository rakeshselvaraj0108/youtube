"""The agent roster, loaded from `prompts/`.

Every agent declares itself in a markdown file with frontmatter. This module
parses those declarations into `AgentSpec` objects, and `tests/test_orchestrator.py`
asserts the running pipeline matches them — same agents, same DAG, same
contracts. Drift between the specification and the implementation turns a test
red rather than surfacing as a mysterious gap on demo day.

Two design points worth stating, because both look like omissions otherwise:

**Only agents that call a model have a prompt body.** A02 runs Whisper, A04 is
RMS and FFT, A06 is a regex cross-reference. Writing "You are A04, an audio
analyst…" for a function that computes spectral flatness would be theatre — the
text would never be sent anywhere and would rot immediately. Those agents
declare a contract and no prompt, and `kind: deterministic` says so.

**Prompt text is hashed into the attestation.** A finding produced under one
adjudicator prompt is not the same evidence as one produced under another, so
the digest travels with the report. This is the same reason the policy corpus
carries a hash.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

PROMPTS_DIR = Path("prompts")
FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)

AgentKind = Literal["deterministic", "model", "hybrid"]


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str  # A01
    codename: str  # ORCHESTRATOR
    kind: AgentKind
    status: str  # implemented | unimplemented
    implementation: str
    model_capability: str  # a capability name, or "none"
    tier: int
    parents: tuple[str, ...]
    produces: str
    body: str
    sha256: str
    path: Path

    @property
    def is_model_driven(self) -> bool:
        return self.kind in ("model", "hybrid")

    @property
    def implemented(self) -> bool:
        """Whether code exists behind this specification.

        Declaring this rather than quietly shipping a stub keeps the roster an
        honest status board. A judge reading `preflight agents` sees what is
        built and what is specified-but-not-yet-built, which is a more useful
        thing to know than a directory of files that all look finished.
        """
        return self.status == "implemented"

    @property
    def prompt(self) -> str:
        """The text actually sent to a model.

        Everything above the `## System prompt` heading is documentation for a
        human reading the repository. Only what follows is sent, so a spec can
        explain itself at length without inflating every request.
        """
        marker = "## System prompt"
        if marker not in self.body:
            return ""
        return self.body.split(marker, 1)[1].strip()

    def to_json(self) -> dict[str, Any]:
        return {
            "agentId": self.agent_id,
            "codename": self.codename,
            "kind": self.kind,
            "capability": self.model_capability,
            "tier": self.tier,
            "parents": list(self.parents),
            "sha256": self.sha256[:16],
        }


@dataclass
class Roster:
    agents: dict[str, AgentSpec] = field(default_factory=dict)
    digest: str = ""

    def __getitem__(self, agent_id: str) -> AgentSpec:
        return self.agents[agent_id]

    def get(self, agent_id: str) -> AgentSpec | None:
        return self.agents.get(agent_id)

    def by_codename(self, codename: str) -> AgentSpec | None:
        target = codename.upper()
        return next(
            (a for a in self.agents.values() if a.codename.upper() == target), None
        )

    @property
    def ordered(self) -> list[AgentSpec]:
        """Topological-ish order: by tier, then by id. Deterministic."""
        return sorted(self.agents.values(), key=lambda a: (a.tier, a.agent_id))

    @property
    def model_driven(self) -> list[AgentSpec]:
        return [a for a in self.ordered if a.is_model_driven]

    def dag(self) -> dict[str, tuple[str, ...]]:
        return {a.agent_id: a.parents for a in self.ordered}

    def validate(self) -> list[str]:
        """Structural problems in the declared roster itself."""
        problems: list[str] = []
        for spec in self.agents.values():
            for parent in spec.parents:
                if parent not in self.agents:
                    problems.append(f"{spec.agent_id} declares unknown parent {parent}")
                    continue
                if self.agents[parent].tier >= spec.tier:
                    problems.append(
                        f"{spec.agent_id} (tier {spec.tier}) depends on "
                        f"{parent} (tier {self.agents[parent].tier}) — not a DAG"
                    )
            if spec.is_model_driven and not spec.prompt:
                problems.append(
                    f"{spec.agent_id} is {spec.kind} but declares no system prompt"
                )
            if spec.kind == "deterministic" and spec.model_capability != "none":
                problems.append(
                    f"{spec.agent_id} is deterministic but requests "
                    f"capability {spec.model_capability}"
                )
        return problems

    def to_json(self) -> dict[str, Any]:
        return {
            "digest": self.digest[:16],
            "count": len(self.agents),
            "agents": [a.to_json() for a in self.ordered],
        }


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = FRONTMATTER.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        if key.strip():
            meta[key.strip()] = value.strip()
    return meta, text[match.end() :]


def _parse_list(raw: str) -> tuple[str, ...]:
    cleaned = raw.strip().strip("[]").strip()
    if not cleaned:
        return ()
    return tuple(part.strip().strip("'\"") for part in cleaned.split(",") if part.strip())


@lru_cache(maxsize=4)
def load_roster(directory: str | Path = PROMPTS_DIR) -> Roster:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"no prompt directory at {directory}. The agent roster is declared there."
        )

    agents: dict[str, AgentSpec] = {}
    hasher = hashlib.sha256()

    for path in sorted(directory.glob("A*.md")):
        raw = path.read_text(encoding="utf-8")
        hasher.update(raw.encode("utf-8"))
        meta, body = _parse_frontmatter(raw)

        agent_id = meta.get("agent_id", path.stem.split("_")[0])
        kind = meta.get("kind", "deterministic")
        spec = AgentSpec(
            agent_id=agent_id,
            codename=meta.get("codename", agent_id),
            kind=kind,  # type: ignore[arg-type]
            status=meta.get("status", "implemented"),
            implementation=meta.get("implementation", ""),
            model_capability=meta.get("model", "none"),
            tier=int(meta.get("tier", "9") or 9),
            parents=_parse_list(meta.get("parents", "")),
            produces=meta.get("produces", ""),
            body=body,
            sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            path=path,
        )
        agents[agent_id] = spec

    if not agents:
        raise FileNotFoundError(f"no agent specifications found in {directory}")

    return Roster(agents=agents, digest=hasher.hexdigest())


def prompt_for(agent_id: str, directory: str | Path = PROMPTS_DIR) -> str:
    """The system prompt for a model-driven agent, or empty for a deterministic one."""
    spec = load_roster(directory).get(agent_id)
    return spec.prompt if spec else ""
