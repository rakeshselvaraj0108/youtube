"""A6 — Metadata.

Rule-based, zero LLM calls, zero network.

The sleeper check is undisclosed paid promotion: the transcript says "this
video is sponsored by" and the description carries no disclosure line. That is
genuine regulatory exposure, creators forget it constantly, and both inputs are
already in hand — the transcript from A1 and the description from the sidecar.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from preflight.models import Adversarial, AgentResult, Evidence, Finding, PolicyRef
from preflight.perception.asr import Transcript

AGENT_ID = "meta"
AGENT_NAME = "Metadata Agent"

MIN_DESCRIPTION_CHARS = 200
MAX_TITLE_CHARS = 70
MAX_TAGS = 15

SPONSORSHIP = re.compile(
    r"\b("
    r"sponsored by|this video is sponsored|thanks to .{0,30} for sponsoring|"
    r"paid partnership|in partnership with|use my code|discount code|"
    r"affiliate link|link in the description below to get"
    r")\b",
    re.IGNORECASE,
)

DISCLOSURE = re.compile(
    r"\b("
    r"paid promotion|sponsored|#ad\b|\bad\b:|advertisement|affiliate|"
    r"commission|paid partnership|includes paid promotion"
    r")\b",
    re.IGNORECASE,
)

AFFILIATE_HOSTS = re.compile(
    r"\b(amzn\.to|amazon\.[a-z.]+/dp/|bit\.ly|geni\.us|shareasale|"
    r"impact\.com|awin1\.com|rstyle\.me|ltk\.app)\b",
    re.IGNORECASE,
)


@dataclass
class Sidecar:
    """Optional `<video>.meta.json` describing the intended upload."""

    title: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    category: str = ""
    declared_audience: str = ""

    @classmethod
    def load(cls, video: Path) -> "Sidecar | None":
        candidate = video.with_suffix("").with_suffix(".meta.json")
        if not candidate.is_file():
            candidate = video.parent / f"{video.stem}.meta.json"
        if not candidate.is_file():
            return None
        try:
            data: dict[str, Any] = json.loads(candidate.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return cls(
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            tags=[str(t) for t in data.get("tags", [])],
            category=str(data.get("category", "")),
            declared_audience=str(data.get("declared_audience", "")),
        )


def _clause(clause_id: str, title: str, section: str, text: str) -> PolicyRef:
    return PolicyRef(clauseId=clause_id, title=title, section=section, text=text)


def analyse(
    video: Path,
    duration_ms: int,
    transcript: Transcript | None,
    sidecar: Sidecar | None = None,
) -> AgentResult:
    started = time.perf_counter()
    log: list[str] = []
    findings: list[Finding] = []

    sidecar = sidecar or Sidecar.load(Path(video))
    if sidecar is None:
        return AgentResult(
            agent_id=AGENT_ID,
            name=AGENT_NAME,
            status="SKIPPED",
            coverage=0.0,
            error=f"no {Path(video).stem}.meta.json sidecar — metadata not linted",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            log=[f"no sidecar at {Path(video).stem}.meta.json"],
        )

    log.append(
        f"linted title ({len(sidecar.title)} chars), description "
        f"({len(sidecar.description)} chars), {len(sidecar.tags)} tags"
    )

    disclosure = _disclosure_finding(sidecar, transcript, duration_ms)
    if disclosure:
        findings.append(disclosure)
        # Name the trigger that actually fired — "sponsorship language in the
        # transcript" would be a lie when the affiliate link was the signal.
        trigger = (
            "spoken sponsorship language"
            if disclosure.modalities.get("speech", 0.0) > 0
            else "affiliate link in description"
        )
        log.append(f"{trigger} with no disclosure")

    findings.extend(_description_findings(sidecar, duration_ms))
    findings.extend(_title_findings(sidecar, duration_ms))
    findings.extend(_tag_findings(sidecar, duration_ms))

    if not findings:
        log.append("no metadata defects found")

    return AgentResult(
        agent_id=AGENT_ID,
        name=AGENT_NAME,
        status="OK",
        findings=findings,
        artifacts={
            "title": sidecar.title,
            "tags": len(sidecar.tags),
            "category": sidecar.category,
            "declared_audience": sidecar.declared_audience,
        },
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        log=log,
    )


def _disclosure_finding(
    sidecar: Sidecar, transcript: Transcript | None, duration_ms: int
) -> Finding | None:
    spoken = transcript.text if transcript else ""
    spoken_match = SPONSORSHIP.search(spoken)
    link_match = AFFILIATE_HOSTS.search(sidecar.description)

    if not spoken_match and not link_match:
        return None
    if DISCLOSURE.search(sidecar.description):
        return None

    if spoken_match:
        quote = spoken[max(0, spoken_match.start() - 60) : spoken_match.end() + 80].strip()
        charge = (
            f'Transcript contains sponsorship language ("{spoken_match.group(0)}") and the '
            "description carries no disclosure."
        )
        evidence = Evidence.marking(quote, spoken_match.group(0))
    else:
        assert link_match is not None
        quote = sidecar.description[
            max(0, link_match.start() - 60) : link_match.end() + 60
        ].strip()
        charge = (
            f"Description links a known affiliate host ({link_match.group(0)}) with no "
            "disclosure line."
        )
        evidence = Evidence.marking(quote, link_match.group(0))

    return Finding(
        id="m_disclosure",
        clauseId="META-01",
        category="Metadata",
        title="Undisclosed paid promotion",
        description="Sponsorship or affiliate content detected with no disclosure. This is "
        "a regulatory exposure, not a style note.",
        startMs=0,
        endMs=duration_ms,
        severity="HIGH",
        confidence=0.86,
        modalities={"meta": 0.86, "speech": 0.8 if spoken_match else 0.0},
        evidence=evidence,
        policy=_clause(
            "META-01",
            "Paid promotion disclosure",
            "PREFLIGHT metadata ruleset § 2.3",
            "Content containing paid promotion, sponsorship or affiliate links must "
            "disclose it. Disclosure obligations sit with the creator regardless of "
            "platform tooling.",
        ),
        adversarial=Adversarial(
            charge=charge,
            rationale="Cross-checked the spoken transcript against the description text. "
            "Adding a disclosure line clears this on a re-run.",
            confidence=0.86,
            defense="Disclosure may be present in an on-screen card the metadata pass "
            "cannot see.",
            defense_strength=0.35,
        ),
    )


def _description_findings(sidecar: Sidecar, duration_ms: int) -> list[Finding]:
    length = len(sidecar.description.strip())
    if length >= MIN_DESCRIPTION_CHARS:
        return []
    return [
        Finding(
            id="m_desc",
            clauseId="META-02",
            category="Metadata",
            title=f"Description is {length} characters",
            description=f"Under {MIN_DESCRIPTION_CHARS} characters. Little for search or "
            "recommendation to work with.",
            startMs=0,
            endMs=duration_ms,
            severity="LOW",
            confidence=0.99,
            modalities={"meta": 0.99},
            evidence=Evidence(transcript=sidecar.description or "[empty description]"),
            policy=_clause(
                "META-02",
                "Description depth",
                "PREFLIGHT metadata ruleset § 2.1",
                "A thin description gives search and recommendation systems little signal "
                "about what the video contains.",
            ),
            adversarial=Adversarial(
                charge=f"Description is {length} characters.",
                rationale="Character count. No interpretation involved.",
                confidence=0.99,
            ),
        )
    ]


def _title_findings(sidecar: Sidecar, duration_ms: int) -> list[Finding]:
    title = sidecar.title.strip()
    findings: list[Finding] = []

    if len(title) > MAX_TITLE_CHARS:
        findings.append(
            Finding(
                id="m_title_len",
                clauseId="META-03",
                category="Metadata",
                title=f"Title is {len(title)} characters",
                description=f"Over {MAX_TITLE_CHARS} characters, so it truncates in most "
                "surfaces.",
                startMs=0,
                endMs=duration_ms,
                severity="LOW",
                confidence=0.99,
                modalities={"meta": 0.99},
                evidence=Evidence(transcript=title),
                policy=_clause(
                    "META-03",
                    "Title length",
                    "PREFLIGHT metadata ruleset § 2.2",
                    "Titles beyond roughly 70 characters truncate on most surfaces, "
                    "hiding the end of the title from viewers.",
                ),
                adversarial=Adversarial(
                    charge=f"Title is {len(title)} characters.",
                    rationale="Character count.",
                    confidence=0.99,
                ),
            )
        )

    letters = [c for c in title if c.isalpha()]
    if len(letters) >= 12:
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_ratio > 0.7:
            findings.append(
                Finding(
                    id="m_title_caps",
                    clauseId="META-04",
                    category="Metadata",
                    title="Title is mostly uppercase",
                    description=f"{upper_ratio * 100:.0f}% uppercase. Reads as clickbait "
                    "and can suppress reach.",
                    startMs=0,
                    endMs=duration_ms,
                    severity="LOW",
                    confidence=0.9,
                    modalities={"meta": 0.9},
                    evidence=Evidence(transcript=title),
                    policy=_clause(
                        "META-04",
                        "Title presentation",
                        "PREFLIGHT metadata ruleset § 2.4",
                        "Titles in sustained uppercase are treated as a clickbait signal.",
                    ),
                    adversarial=Adversarial(
                        charge=f"{upper_ratio * 100:.0f}% of title letters are uppercase.",
                        rationale="Ratio over alphabetic characters.",
                        confidence=0.9,
                        defense="Acronym-heavy titles can legitimately skew uppercase.",
                        defense_strength=0.4,
                    ),
                )
            )
    return findings


def _tag_findings(sidecar: Sidecar, duration_ms: int) -> list[Finding]:
    if len(sidecar.tags) <= MAX_TAGS:
        return []
    return [
        Finding(
            id="m_tags",
            clauseId="META-05",
            category="Metadata",
            title=f"{len(sidecar.tags)} tags",
            description=f"Over {MAX_TAGS} tags reads as keyword stuffing and dilutes "
            "topical signal.",
            startMs=0,
            endMs=duration_ms,
            severity="LOW",
            confidence=0.95,
            modalities={"meta": 0.95},
            evidence=Evidence(transcript=", ".join(sidecar.tags[:30])),
            policy=_clause(
                "META-05",
                "Tag stuffing",
                "PREFLIGHT metadata ruleset § 2.5",
                "Excessive or irrelevant tags dilute topical signal and may be treated as "
                "metadata spam.",
            ),
            adversarial=Adversarial(
                charge=f"{len(sidecar.tags)} tags supplied.",
                rationale="Tag count.",
                confidence=0.95,
            ),
        )
    ]
