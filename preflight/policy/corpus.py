"""Policy corpus loading and chunking.

Clauses are chunked at heading level — Scope, Green, Yellow, Red, Exemptions —
rather than by token count. Retrieval that returns "the Yellow conditions of the
language clause" is directly usable as an argument; retrieval that returns
"tokens 200-400 of a markdown file" is not, and the adjudicator has to be told
verbatim text it can reason about.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
HEADING = re.compile(r"^##\s+(.+)$", re.MULTILINE)

# Sections that state the rule, and are therefore what a window is matched
# against.
# Retrieval scopes.
#
# One index over every clause means a transcript window about an avalanche can
# retrieve the paid-promotion clause, because "disclosure" and "casualty" share
# no tokens but their embeddings are not orthogonal either. Measured on this
# corpus, a general index put META-01 or COPY-01 in the top three for roughly a
# third of policy queries — retrievals that could never be correct, occupying
# slots the adjudicator needed.
#
# Scoping is derived from the clause id prefix, so adding a clause puts it in
# the right index without a second place to update.
SCOPES: dict[str, str] = {
    "AF": "policy",
    "COPY": "copyright",
    "META": "metadata",
    "ACC": "accessibility",
    "AUD": "audio",
}
DEFAULT_SCOPE = "policy"


def scope_for(clause_id: str) -> str:
    return SCOPES.get(clause_id.split("-")[0].upper(), DEFAULT_SCOPE)


NORMATIVE_SECTIONS = {
    "Scope",
    "Fully monetized when",
    "Limited ads when",
    "No ads when",
    "Documented exemptions",
}

# Sections that guide an agent once a clause has already been selected. They are
# carried on the clause and handed to the adjudicator, but they are deliberately
# not retrievable: "Remediation guidance" is instruction for the compiler, and
# indexing it means a query about an avalanche can match the sentence explaining
# which fix to apply. Measured on the 17-clause corpus, indexing them put a
# guidance chunk in the top three for roughly a third of probe queries.
ADVISORY_SECTIONS = {
    "Signals that distinguish this clause from neighbours",
    "Remediation guidance",
}


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage of one clause."""

    clause_id: str
    clause_title: str
    section: str
    text: str
    severity_default: str
    source_url: str

    @property
    def id(self) -> str:
        return f"{self.clause_id}::{self.section}"

    @property
    def scope(self) -> str:
        return scope_for(self.clause_id)

    @property
    def citation(self) -> str:
        return f"Advertiser-friendly guidelines § {self.clause_id} — {self.section}"

    def for_prompt(self) -> str:
        return f"[{self.clause_id}] {self.clause_title} — {self.section}\n{self.text}"


@dataclass
class Clause:
    clause_id: str
    title: str
    severity_default: str
    version: str
    source_url: str
    fetched_at: str
    sections: dict[str, str]
    sha256: str

    @property
    def scope(self) -> str:
        return self.sections.get("Scope", "")

    @property
    def exemptions(self) -> str:
        """What the ADVOCATE is permitted to argue from."""
        return self.sections.get("Documented exemptions", "")

    @property
    def distinguishing_signals(self) -> str:
        """What the ADJUDICATOR uses to pick between neighbouring clauses.

        Retrieval routinely surfaces three adjacent clauses for one window.
        This is the text that separates them, handed to the adjudicator
        directly rather than left to be matched against.
        """
        return self.sections.get(
            "Signals that distinguish this clause from neighbours", ""
        )

    @property
    def remediation_guidance(self) -> str:
        return self.sections.get("Remediation guidance", "")

    @property
    def preferred_fix(self) -> str:
        for line in self.remediation_guidance.splitlines():
            if "Preferred fix:" in line:
                return line.split("Preferred fix:", 1)[1].strip()
        return "NONE"


@dataclass
class Corpus:
    clauses: list[Clause]
    chunks: list[Chunk]
    version: str
    digest: str

    def clause(self, clause_id: str) -> Clause | None:
        return next((c for c in self.clauses if c.clause_id == clause_id), None)

    def scoped(self, scope: str) -> "Corpus":
        """A sub-corpus containing only clauses in one retrieval scope.

        The digest is scope-specific, so each index caches independently and
        editing a metadata clause does not invalidate the policy index.
        """
        clauses = [c for c in self.clauses if scope_for(c.clause_id) == scope]
        chunks = [c for c in self.chunks if c.scope == scope]
        digest = hashlib.sha256(
            (scope + "|" + "|".join(c.sha256 for c in clauses)).encode("utf-8")
        ).hexdigest()
        return Corpus(clauses=clauses, chunks=chunks, version=self.version, digest=digest)

    @property
    def scopes(self) -> list[str]:
        return sorted({scope_for(c.clause_id) for c in self.clauses})

    def chunk(self, chunk_id: str) -> Chunk | None:
        return next((c for c in self.chunks if c.id == chunk_id), None)

    @property
    def clause_ids(self) -> list[str]:
        return [c.clause_id for c in self.clauses]


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


def _split_sections(body: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    matches = list(HEADING.finditer(body))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        # Normalise the heading: "Green (fully monetized)" -> "Green".
        name = match.group(1).strip()
        short = name.split("(")[0].strip()
        sections[short] = body[start:end].strip()
    return sections


def load_corpus(directory: Path = Path("data/policy")) -> Corpus:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"policy corpus not found at {directory}. Run: python scripts/build_corpus.py"
        )

    files = sorted(p for p in directory.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"no clause files in {directory}")

    clauses: list[Clause] = []
    chunks: list[Chunk] = []
    hasher = hashlib.sha256()

    for path in files:
        raw = path.read_text(encoding="utf-8")
        hasher.update(raw.encode("utf-8"))
        meta, body = _parse_frontmatter(raw)
        sections = _split_sections(body)

        clause = Clause(
            clause_id=meta.get("clause_id", path.stem),
            title=meta.get("title", path.stem),
            severity_default=meta.get("severity_default", "LIMITING"),
            version=meta.get("version", "unknown"),
            source_url=meta.get("source_url", ""),
            fetched_at=meta.get("fetched_at", ""),
            sections=sections,
            sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        )
        clauses.append(clause)

        for section, text in sections.items():
            if not text.strip() or section in ADVISORY_SECTIONS:
                continue
            chunks.append(
                Chunk(
                    clause_id=clause.clause_id,
                    clause_title=clause.title,
                    section=section,
                    text=text.strip(),
                    severity_default=clause.severity_default,
                    source_url=clause.source_url,
                )
            )

    version = clauses[0].version if clauses else "unknown"
    return Corpus(
        clauses=clauses, chunks=chunks, version=version, digest=hasher.hexdigest()
    )


def load_manifest(directory: Path = Path("data/policy")) -> dict:
    path = Path(directory) / "manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
