"""The Policy Drift Watcher.

    Your back catalogue was compliant when you uploaded it. The rules changed.

Everything else in PREFLIGHT answers "is this video safe to publish now". This
answers "which of the videos I already published stopped being safe", which is
the same shift that took security scanning from "run it before release" to
"watch the dependency graph forever".

The mechanism:

1. Snapshot the corpus — a SHA-256 per clause, which `manifest.json` already
   records.
2. Diff against the previous snapshot: added, removed, modified.
3. For modified clauses, embed both texts and measure semantic delta. A
   reworded sentence and a tightened rule produce very different deltas, and
   only one of them is worth re-linting a back catalogue over.
4. Query the archive for videos whose findings cite the changed clauses, or
   whose retrieval considered them and fell short.
5. Re-lint only those. Selective invalidation is the whole point: re-linting
   214 videos because one clause moved is not monitoring, it is a bill.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from preflight.archive import Archive, ArchivedVideo
from preflight.policy.corpus import Corpus, load_corpus

ChangeKind = Literal["ADDED", "REMOVED", "MODIFIED"]

# Below this, a clause changed wording without changing meaning — a typo fix,
# a reordered list. Re-linting a back catalogue over it wastes quota.
SEMANTIC_FLOOR = 0.05


@dataclass
class ClauseChange:
    clause_id: str
    title: str
    kind: ChangeKind
    semantic_delta: float = 0.0
    old_sha: str | None = None
    new_sha: str | None = None
    sections_changed: list[str] = field(default_factory=list)
    diff: list[str] = field(default_factory=list)

    @property
    def material(self) -> bool:
        """Worth re-linting over."""
        return self.kind in ("ADDED", "REMOVED") or self.semantic_delta >= SEMANTIC_FLOOR

    def to_json(self) -> dict[str, Any]:
        return {
            "clauseId": self.clause_id,
            "title": self.title,
            "kind": self.kind,
            "semanticDelta": round(self.semantic_delta, 4),
            "material": self.material,
            "sectionsChanged": list(self.sections_changed),
            "oldSha256": self.old_sha,
            "newSha256": self.new_sha,
            "diff": self.diff[:40],
        }


@dataclass
class DriftReport:
    detected_at: str
    from_version: str
    to_version: str
    changes: list[ClauseChange] = field(default_factory=list)
    affected: list[ArchivedVideo] = field(default_factory=list)
    archive_size: int = 0

    @property
    def material_changes(self) -> list[ClauseChange]:
        return [c for c in self.changes if c.material]

    @property
    def changed_clause_ids(self) -> set[str]:
        return {c.clause_id for c in self.material_changes}

    def to_json(self) -> dict[str, Any]:
        return {
            "detectedAt": self.detected_at,
            "fromVersion": self.from_version,
            "toVersion": self.to_version,
            "changes": [c.to_json() for c in self.changes],
            "affected": [
                {
                    "filename": v.filename,
                    "videoHash": v.video_hash,
                    "overall": v.overall,
                    "verdict": v.verdict,
                    "clauses": list(v.clauses),
                    "nearMiss": list(v.near_miss_clauses),
                }
                for v in self.affected
            ],
            "archiveSize": self.archive_size,
            "selectivity": (
                round(len(self.affected) / self.archive_size, 4)
                if self.archive_size
                else 0.0
            ),
        }


def snapshot(corpus: Corpus) -> dict[str, dict[str, Any]]:
    """A comparable record of the corpus.

    Sections are stored individually as well as concatenated. Comparing whole
    clauses hides the change that matters: moving one condition from Yellow to
    Red rewrites a fifth of the text, and a whole-clause embedding of that
    scores ~0.02 — indistinguishable from a typo fix, and dismissed as
    cosmetic. Section-level comparison sees it.
    """
    return {
        clause.clause_id: {
            "sha256": clause.sha256,
            "title": clause.title,
            "text": "\n".join(
                f"## {name}\n{body}" for name, body in clause.sections.items()
            ),
            "sections": dict(clause.sections),
            "version": clause.version,
        }
        for clause in corpus.clauses
    }


def corpus_version(corpus: Corpus, policy_dir: Path | None = None) -> str:
    """The corpus version.

    Not clause 01's version. A policy update touches a handful of clauses, so
    reading the first one reports the corpus as unchanged while three of its
    rules have moved. The manifest is authoritative; failing that, the newest
    version any clause declares.
    """
    if policy_dir is not None:
        manifest = Path(policy_dir) / "manifest.json"
        if manifest.is_file():
            try:
                declared = json.loads(manifest.read_text(encoding="utf-8")).get("version")
                if declared:
                    return str(declared)
            except json.JSONDecodeError:
                pass
    versions = {c.version for c in corpus.clauses if c.version}
    return max(versions) if versions else "unknown"


def write_snapshot(corpus: Corpus, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "version": corpus.version,
                "digest": corpus.digest,
                "capturedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "clauses": snapshot(corpus),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return destination


def read_snapshot(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _semantic_delta(old: str, new: str, embed) -> float:
    """1 − cosine between the two clause texts.

    Falls back to a character-level ratio when no embedder is available, which
    is coarser but still separates a typo from a rewrite — and never silently
    reports zero drift because a key was missing.
    """
    if embed is not None:
        vectors = embed([old, new])
        if vectors is not None and len(vectors) == 2:
            a, b = np.asarray(vectors[0]), np.asarray(vectors[1])
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denominator > 0:
                return float(max(0.0, 1.0 - float(a @ b) / denominator))
    return 1.0 - difflib.SequenceMatcher(None, old, new).ratio()


def _changed_sections(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    old_sections: dict[str, str] = old.get("sections") or {}
    new_sections: dict[str, str] = new.get("sections") or {}
    names = set(old_sections) | set(new_sections)
    return sorted(n for n in names if old_sections.get(n) != new_sections.get(n))


def _clause_delta(old: dict[str, Any], new: dict[str, Any], embed) -> float:
    """Largest semantic movement across the sections that actually changed.

    Max rather than mean, deliberately. A clause where Red was rewritten and
    four other sections were untouched has moved, and averaging that against
    four zeroes reports it as stable — which is precisely the change a creator
    needs to know about.
    """
    old_sections: dict[str, str] = old.get("sections") or {}
    new_sections: dict[str, str] = new.get("sections") or {}

    # Snapshots taken before section-level capture only carry the full text.
    if not old_sections or not new_sections:
        return _semantic_delta(old.get("text", ""), new.get("text", ""), embed)

    changed = _changed_sections(old, new)
    if not changed:
        return 0.0

    deltas = [
        _semantic_delta(old_sections.get(name, ""), new_sections.get(name, ""), embed)
        for name in changed
    ]
    return max(deltas) if deltas else 0.0


def diff_corpus(
    previous: dict[str, Any],
    current: Corpus,
    *,
    embed=None,
) -> list[ClauseChange]:
    old_clauses: dict[str, dict[str, str]] = previous.get("clauses", {})
    new_clauses = snapshot(current)

    changes: list[ClauseChange] = []

    for clause_id, new in new_clauses.items():
        old = old_clauses.get(clause_id)
        if old is None:
            changes.append(
                ClauseChange(
                    clause_id=clause_id,
                    title=new["title"],
                    kind="ADDED",
                    semantic_delta=1.0,
                    new_sha=new["sha256"],
                )
            )
            continue

        if old["sha256"] == new["sha256"]:
            continue

        changes.append(
            ClauseChange(
                clause_id=clause_id,
                title=new["title"],
                kind="MODIFIED",
                semantic_delta=_clause_delta(old, new, embed),
                old_sha=old["sha256"],
                new_sha=new["sha256"],
                sections_changed=_changed_sections(old, new),
                diff=[
                    line
                    for line in difflib.unified_diff(
                        old["text"].splitlines(),
                        new["text"].splitlines(),
                        lineterm="",
                        n=1,
                    )
                    if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
                ],
            )
        )

    for clause_id, old in old_clauses.items():
        if clause_id not in new_clauses:
            changes.append(
                ClauseChange(
                    clause_id=clause_id,
                    title=old["title"],
                    kind="REMOVED",
                    semantic_delta=1.0,
                    old_sha=old["sha256"],
                )
            )

    order = {"ADDED": 0, "MODIFIED": 1, "REMOVED": 2}
    return sorted(changes, key=lambda c: (order[c.kind], -c.semantic_delta))


def detect(
    snapshot_path: Path,
    policy_dir: Path,
    archive: Archive,
    *,
    embed=None,
) -> DriftReport:
    current = load_corpus(policy_dir)
    previous = read_snapshot(snapshot_path)
    changes = diff_corpus(previous, current, embed=embed)

    report = DriftReport(
        detected_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        from_version=str(previous.get("version", "unknown")),
        to_version=corpus_version(current, policy_dir),
        changes=changes,
    )

    everything = archive.latest_reports()
    report.archive_size = len(everything)
    if report.changed_clause_ids:
        report.affected = archive.affected_by(report.changed_clause_ids)
    return report
