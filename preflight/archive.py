"""The findings archive.

A SQLite record of every report ever produced, and — crucially — which policy
clauses each video's findings cited, plus which clauses were *retrieved* for a
window but did not fire.

That second set is what makes the Drift Watcher precise rather than paranoid.
When a clause tightens, the videos at risk are not only the ones it already
flagged; they are also the ones where it was considered and fell just short.
Re-linting only the former misses exactly the videos a tightening would newly
catch, and re-linting everything defeats the purpose of selective invalidation.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_hash   TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    duration_ms  INTEGER NOT NULL,
    first_seen   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reports (
    report_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    video_hash     TEXT NOT NULL REFERENCES videos(video_hash),
    analyzed_at    TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    policy_digest  TEXT NOT NULL,
    engine_version TEXT NOT NULL,
    overall        INTEGER NOT NULL,
    verdict        TEXT NOT NULL,
    coverage       REAL NOT NULL,
    report_json    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS citations (
    report_id  INTEGER NOT NULL REFERENCES reports(report_id) ON DELETE CASCADE,
    clause_id  TEXT NOT NULL,
    severity   TEXT NOT NULL,
    confidence REAL NOT NULL,
    fired      INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_citations_clause ON citations(clause_id);
CREATE INDEX IF NOT EXISTS idx_reports_video ON reports(video_hash);
"""


@dataclass(frozen=True)
class ArchivedVideo:
    video_hash: str
    filename: str
    duration_ms: int
    overall: int
    verdict: str
    analyzed_at: str
    policy_version: str
    clauses: tuple[str, ...]
    near_miss_clauses: tuple[str, ...]


class Archive:
    """Read/write access to the findings database."""

    def __init__(self, path: Path | str = ".preflight/archive.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def record(
        self,
        report: dict[str, Any],
        video_hash: str,
        policy_digest: str,
        *,
        considered: Iterable[str] = (),
    ) -> int:
        """Store a report. `considered` is clauses retrieved but not fired."""
        video = report["video"]
        meta = report["meta"]
        scores = report["scores"]
        fired = {f["clauseId"] for f in report["findings"]}
        near_miss = {c for c in considered if c not in fired}

        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO videos VALUES (?,?,?,?)",
                (
                    video_hash,
                    video["filename"],
                    video["durationMs"],
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )
            cursor = conn.execute(
                "INSERT INTO reports "
                "(video_hash, analyzed_at, policy_version, policy_digest, "
                " engine_version, overall, verdict, coverage, report_json) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    video_hash,
                    meta["analyzedAt"],
                    meta["policyVersion"],
                    policy_digest,
                    meta["engineVersion"],
                    scores["overall"],
                    scores["verdict"],
                    meta["coverage"],
                    json.dumps(report, separators=(",", ":")),
                ),
            )
            report_id = int(cursor.lastrowid or 0)

            rows = [
                (report_id, f["clauseId"], f["severity"], f["confidence"], 1)
                for f in report["findings"]
            ]
            rows += [(report_id, clause, "NONE", 0.0, 0) for clause in sorted(near_miss)]
            conn.executemany("INSERT INTO citations VALUES (?,?,?,?,?)", rows)
            conn.commit()
        return report_id

    def latest_reports(self) -> list[ArchivedVideo]:
        """The most recent report per video."""
        query = """
        SELECT r.report_id, r.video_hash, v.filename, v.duration_ms,
               r.overall, r.verdict, r.analyzed_at, r.policy_version
        FROM reports r
        JOIN videos v ON v.video_hash = r.video_hash
        WHERE r.report_id IN (
            SELECT MAX(report_id) FROM reports GROUP BY video_hash
        )
        ORDER BY r.overall ASC
        """
        out: list[ArchivedVideo] = []
        with closing(self._connect()) as conn:
            for row in conn.execute(query):
                cites = conn.execute(
                    "SELECT clause_id, fired FROM citations WHERE report_id = ?",
                    (row["report_id"],),
                ).fetchall()
                out.append(
                    ArchivedVideo(
                        video_hash=row["video_hash"],
                        filename=row["filename"],
                        duration_ms=row["duration_ms"],
                        overall=row["overall"],
                        verdict=row["verdict"],
                        analyzed_at=row["analyzed_at"],
                        policy_version=row["policy_version"],
                        clauses=tuple(
                            sorted({c["clause_id"] for c in cites if c["fired"]})
                        ),
                        near_miss_clauses=tuple(
                            sorted({c["clause_id"] for c in cites if not c["fired"]})
                        ),
                    )
                )
        return out

    def affected_by(self, clause_ids: set[str]) -> list[ArchivedVideo]:
        """Videos whose latest report cites, or nearly cited, any of these.

        Near misses are included deliberately. A clause that tightens will
        newly catch the videos where it was already in contention.
        """
        return [
            video
            for video in self.latest_reports()
            if clause_ids & (set(video.clauses) | set(video.near_miss_clauses))
        ]

    def score_history(self, video_hash: str) -> list[tuple[str, int, str]]:
        with closing(self._connect()) as conn:
            return [
                (row["analyzed_at"], row["overall"], row["policy_version"])
                for row in conn.execute(
                    "SELECT analyzed_at, overall, policy_version FROM reports "
                    "WHERE video_hash = ? ORDER BY report_id",
                    (video_hash,),
                )
            ]

    def stats(self) -> dict[str, int]:
        with closing(self._connect()) as conn:
            return {
                "videos": conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0],
                "reports": conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0],
                "citations": conn.execute("SELECT COUNT(*) FROM citations").fetchone()[0],
            }
