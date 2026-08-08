"""The run graph — what came from what, durably.

Everything the verification loop produces is derived from something else. A
remediation derives from an analysis; a rendered artifact derives from a
remediation; a verification run derives from that artifact; a verdict derives
from comparing the verification run against the original. Held in memory,
that chain lasts exactly as long as the process, which means a verdict
survives a browser refresh but not a restart — and a verdict you cannot trace
back to the evidence that produced it is an assertion, not a finding.

So the chain is a table. Three rules shaped the schema:

**References, never copies.** A run row carries the *path* to its report, not
the report. Reports are megabytes with embedded frames; duplicating one into
a lineage row would double the storage for every run and, worse, create a
second copy that can disagree with the first.

**Artifacts are content-addressed.** `artifact_id` is derived from the file's
hash, so re-rendering identical output reuses the row rather than accruing a
new one per attempt, and a row can be checked against the file it claims to
describe. A remediated artifact whose bytes changed underneath the record is
detectable rather than silently trusted.

**Transitions are append-only and validated.** Every lifecycle move goes
through `preflight.lifecycle`, so an impossible history cannot be written
even by a caller that wants to. The table is the audit trail; a table that
accepts anything is not one.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from preflight import lifecycle

SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id  TEXT PRIMARY KEY,
    path         TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    size_bytes   INTEGER NOT NULL,
    duration_ms  INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    parent_run_id TEXT REFERENCES runs(run_id),
    role          TEXT NOT NULL,
    artifact_id   TEXT REFERENCES artifacts(artifact_id),
    video_path    TEXT NOT NULL,
    video_hash    TEXT NOT NULL,
    report_path   TEXT,
    overall       INTEGER,
    verdict       TEXT,
    duration_ms   INTEGER NOT NULL DEFAULT 0,
    findings      INTEGER NOT NULL DEFAULT 0,
    incidents     INTEGER NOT NULL DEFAULT 0,
    coverage      REAL NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS simulations (
    simulation_id TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL REFERENCES runs(run_id),
    best          TEXT,
    baseline      INTEGER,
    predicted     INTEGER,
    scenarios     INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS remediations (
    remediation_id      TEXT PRIMARY KEY,
    source_run_id       TEXT NOT NULL REFERENCES runs(run_id),
    simulation_id       TEXT REFERENCES simulations(simulation_id),
    verification_run_id TEXT REFERENCES runs(run_id),
    verification_id     TEXT,
    artifact_id         TEXT REFERENCES artifacts(artifact_id),
    source_path         TEXT NOT NULL,
    output_path         TEXT,
    finding_ids         TEXT NOT NULL DEFAULT '[]',
    incident_ids        TEXT NOT NULL DEFAULT '[]',
    edl_json            TEXT NOT NULL DEFAULT '[]',
    state               TEXT NOT NULL,
    previous_state      TEXT,
    verdict             TEXT,
    error               TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transitions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    remediation_id TEXT NOT NULL REFERENCES remediations(remediation_id)
                   ON DELETE CASCADE,
    from_state     TEXT NOT NULL,
    to_state       TEXT NOT NULL,
    at             TEXT NOT NULL,
    detail         TEXT NOT NULL DEFAULT '',
    error          TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS verifications (
    verification_id     TEXT PRIMARY KEY,
    remediation_id      TEXT NOT NULL REFERENCES remediations(remediation_id),
    original_run_id     TEXT NOT NULL REFERENCES runs(run_id),
    verification_run_id TEXT REFERENCES runs(run_id),
    verdict             TEXT NOT NULL,
    comparison_json     TEXT NOT NULL,
    telemetry_json      TEXT NOT NULL DEFAULT '{}',
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS certificates (
    certificate_id   TEXT PRIMARY KEY,
    verification_id  TEXT NOT NULL REFERENCES verifications(verification_id),
    certificate_hash TEXT NOT NULL,
    payload_json     TEXT NOT NULL,
    created_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_parent ON runs(parent_run_id);
CREATE INDEX IF NOT EXISTS idx_rem_source ON remediations(source_run_id);
CREATE INDEX IF NOT EXISTS idx_trans_rem ON transitions(remediation_id);
CREATE INDEX IF NOT EXISTS idx_cert_ver ON certificates(verification_id);
"""

# What a run is, in the graph. `ORIGINAL` is the analysis a reader started
# from; `VERIFICATION` is the re-analysis of a rendered artifact. The
# distinction matters because a verification run must never be offered as a
# starting point for a new remediation without the reader knowing it is one.
Role = str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    path: str
    content_hash: str
    size_bytes: int
    duration_ms: int
    created_at: str

    @property
    def exists(self) -> bool:
        return Path(self.path).is_file()

    def still_matches(self) -> bool:
        """Do the bytes on disk still hash to what this row claims?

        A resumed remediation must not reuse an artifact that changed after it
        was recorded — that is the one way persistence could make the system
        *less* correct than the ephemeral version it replaces.
        """
        from preflight import cas

        path = Path(self.path)
        if not path.is_file():
            return False
        if path.stat().st_size != self.size_bytes:
            return False
        return cas.hash_file(path) == self.content_hash

    def to_json(self) -> dict[str, Any]:
        return {
            "artifactId": self.artifact_id,
            "path": self.path,
            "contentHash": self.content_hash,
            "sizeBytes": self.size_bytes,
            "durationMs": self.duration_ms,
            "createdAt": self.created_at,
        }


@dataclass(frozen=True)
class RunNode:
    run_id: str
    parent_run_id: str | None
    role: Role
    artifact_id: str | None
    video_path: str
    video_hash: str
    report_path: str | None
    overall: int | None
    verdict: str | None
    duration_ms: int
    findings: int
    incidents: int
    coverage: float
    created_at: str

    def to_json(self) -> dict[str, Any]:
        return {
            "runId": self.run_id,
            "parentRunId": self.parent_run_id,
            "role": self.role,
            "artifactId": self.artifact_id,
            "videoPath": self.video_path,
            "videoHash": self.video_hash,
            "reportPath": self.report_path,
            "overall": self.overall,
            "verdict": self.verdict,
            "durationMs": self.duration_ms,
            "findings": self.findings,
            "incidents": self.incidents,
            "coverage": self.coverage,
            "createdAt": self.created_at,
        }


@dataclass(frozen=True)
class RemediationRecord:
    remediation_id: str
    source_run_id: str
    simulation_id: str | None
    verification_run_id: str | None
    verification_id: str | None
    artifact_id: str | None
    source_path: str
    output_path: str | None
    finding_ids: tuple[str, ...]
    incident_ids: tuple[str, ...]
    ops: tuple[dict[str, Any], ...]
    state: str
    previous_state: str | None
    verdict: str | None
    error: str | None
    created_at: str
    updated_at: str
    transitions: tuple[lifecycle.Transition, ...] = ()

    @property
    def terminal(self) -> bool:
        return lifecycle.is_terminal(self.state)

    @property
    def interrupted(self) -> bool:
        """Non-terminal, so a process stopped without finishing it."""
        return not self.terminal

    def resume_state(self) -> str | None:
        return lifecycle.resume_state(self.state)

    def describe(self) -> str:
        if self.terminal:
            return f"{self.remediation_id} finished in {self.state}"
        return f"{self.remediation_id} was interrupted during {self.state}"

    def to_json(self) -> dict[str, Any]:
        return {
            "remediationId": self.remediation_id,
            "sourceRunId": self.source_run_id,
            "simulationId": self.simulation_id,
            "verificationRunId": self.verification_run_id,
            "verificationId": self.verification_id,
            "artifactId": self.artifact_id,
            "sourcePath": self.source_path,
            "outputPath": self.output_path,
            "findingIds": list(self.finding_ids),
            "incidentIds": list(self.incident_ids),
            "ops": [dict(op) for op in self.ops],
            "state": self.state,
            "previousState": self.previous_state,
            "stateDetail": lifecycle.describe(self.state),
            "terminal": self.terminal,
            "verdict": self.verdict,
            "error": self.error,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "transitions": [t.to_json() for t in self.transitions],
        }


@dataclass
class Graph:
    """One original run and everything derived from it."""

    root: RunNode | None = None
    simulations: list[dict[str, Any]] = field(default_factory=list)
    remediations: list[RemediationRecord] = field(default_factory=list)
    verification_runs: list[RunNode] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    certificates: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "root": self.root.to_json() if self.root else None,
            "simulations": list(self.simulations),
            "remediations": [r.to_json() for r in self.remediations],
            "verificationRuns": [r.to_json() for r in self.verification_runs],
            "artifacts": [a.to_json() for a in self.artifacts],
            "certificates": list(self.certificates),
        }


class Lineage:
    """Read/write access to the run graph."""

    def __init__(self, path: Path | str = ".preflight/lineage.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ---- identifiers --------------------------------------------------- #

    def _next_id(self, conn: sqlite3.Connection, table: str, column: str,
                 prefix: str) -> str:
        """Monotonic per-prefix, derived from what is already stored.

        Read from the table rather than kept in a counter file: a counter that
        can disagree with the rows it numbers is a source of duplicate ids
        after any partial restore, and the table is the thing that has to be
        right.
        """
        rows = conn.execute(f"SELECT {column} FROM {table}").fetchall()
        highest = 0
        for row in rows:
            value = str(row[0] or "")
            if value.startswith(f"{prefix}-"):
                tail = value[len(prefix) + 1:]
                if tail.isdigit():
                    highest = max(highest, int(tail))
        return f"{prefix}-{highest + 1:04d}"

    # ---- artifacts ------------------------------------------------------ #

    def record_artifact(
        self, path: Path | str, *, duration_ms: int = 0
    ) -> Artifact:
        """Hash a file on disk and record it. Idempotent by content.

        The id is the hash prefix, so the same bytes always produce the same
        id. Re-rendering identical output does not create a second row, and a
        row always names bytes that were actually measured.
        """
        from preflight import cas

        file = Path(path)
        if not file.is_file():
            raise FileNotFoundError(f"no artifact at {file}")
        digest = cas.hash_file(file)
        artifact_id = f"ART-{digest[:12]}"
        row = Artifact(
            artifact_id=artifact_id,
            path=str(file),
            content_hash=cas.prefixed(digest),
            size_bytes=file.stat().st_size,
            duration_ms=int(duration_ms),
            created_at=_now(),
        )
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO artifacts VALUES (?,?,?,?,?,"
                " COALESCE((SELECT created_at FROM artifacts WHERE artifact_id=?), ?))",
                (
                    row.artifact_id,
                    row.path,
                    row.content_hash,
                    row.size_bytes,
                    row.duration_ms,
                    row.artifact_id,
                    row.created_at,
                ),
            )
            conn.commit()
        return row

    def artifact(self, artifact_id: str) -> Artifact | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)
            ).fetchone()
        return _artifact(row) if row else None

    # ---- runs ----------------------------------------------------------- #

    def record_run(
        self,
        run_id: str,
        report: dict[str, Any],
        *,
        role: Role = "ORIGINAL",
        parent_run_id: str | None = None,
        video_path: str = "",
        video_hash: str = "",
        report_path: str | None = None,
        artifact_id: str | None = None,
    ) -> RunNode:
        """Record a run by reference. The report stays on disk."""
        scores = report.get("scores", {})
        meta = report.get("meta", {})
        video = report.get("video", {})
        node = RunNode(
            run_id=run_id,
            parent_run_id=parent_run_id,
            role=role,
            artifact_id=artifact_id,
            video_path=video_path or str(video.get("filename", "")),
            video_hash=video_hash or str(meta.get("videoHash", "")),
            report_path=report_path,
            overall=scores.get("overall"),
            verdict=scores.get("verdict"),
            duration_ms=int(video.get("durationMs", 0) or 0),
            findings=len(report.get("findings", [])),
            incidents=len((report.get("incidents") or [])),
            coverage=float(meta.get("coverage", 0.0) or 0.0),
            created_at=_now(),
        )
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    node.run_id,
                    node.parent_run_id,
                    node.role,
                    node.artifact_id,
                    node.video_path,
                    node.video_hash,
                    node.report_path,
                    node.overall,
                    node.verdict,
                    node.duration_ms,
                    node.findings,
                    node.incidents,
                    node.coverage,
                    node.created_at,
                ),
            )
            conn.commit()
        return node

    def run(self, run_id: str) -> RunNode | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return _run(row) if row else None

    def children_of(self, run_id: str) -> list[RunNode]:
        with closing(self._connect()) as conn:
            return [
                _run(row)
                for row in conn.execute(
                    "SELECT * FROM runs WHERE parent_run_id = ? ORDER BY created_at",
                    (run_id,),
                )
            ]

    # ---- simulations ---------------------------------------------------- #

    def record_simulation(
        self, run_id: str, simulation: dict[str, Any]
    ) -> str:
        """Store the simulation's summary and return its durable id.

        Scenarios are not copied — they are already in the run's report, and
        the row carries the numbers a verification later needs to compare
        against: which scenario was best and what it predicted.
        """
        best = simulation.get("best")
        predicted = None
        for scenario in simulation.get("scenarios", []):
            if scenario.get("name") == best:
                predicted = int(scenario.get("overall", 0))
                break
        with closing(self._connect()) as conn:
            simulation_id = self._next_id(
                conn, "simulations", "simulation_id", "SIM"
            )
            conn.execute(
                "INSERT INTO simulations VALUES (?,?,?,?,?,?,?)",
                (
                    simulation_id,
                    run_id,
                    best,
                    int((simulation.get("baseline") or {}).get("overall", 0)),
                    predicted,
                    len(simulation.get("scenarios", [])),
                    _now(),
                ),
            )
            conn.commit()
        return simulation_id

    def simulation(self, simulation_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM simulations WHERE simulation_id = ?", (simulation_id,)
            ).fetchone()
        return dict(row) if row else None

    def simulations_for(self, run_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM simulations WHERE run_id = ? ORDER BY created_at",
                    (run_id,),
                )
            ]

    # ---- remediations --------------------------------------------------- #

    def open_remediation(
        self,
        source_run_id: str,
        *,
        source_path: str,
        simulation_id: str | None = None,
        finding_ids: Iterable[str] = (),
        incident_ids: Iterable[str] = (),
    ) -> RemediationRecord:
        """Allocate REM-000N and enter REMEDIATION_REQUESTED.

        The record exists before ffmpeg is invoked, on purpose: a remediation
        that dies during the render must still be findable afterwards, and one
        that only becomes durable on success can only ever record successes.
        """
        now = _now()
        with closing(self._connect()) as conn:
            remediation_id = self._next_id(
                conn, "remediations", "remediation_id", "REM"
            )
            conn.execute(
                "INSERT INTO remediations "
                "(remediation_id, source_run_id, simulation_id, source_path, "
                " finding_ids, incident_ids, edl_json, state, previous_state, "
                " created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    remediation_id,
                    source_run_id,
                    simulation_id,
                    source_path,
                    json.dumps(sorted(set(finding_ids))),
                    json.dumps(sorted(set(incident_ids))),
                    "[]",
                    "REMEDIATION_REQUESTED",
                    "ANALYSIS_COMPLETE",
                    now,
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO transitions "
                "(remediation_id, from_state, to_state, at, detail) VALUES (?,?,?,?,?)",
                (
                    remediation_id,
                    "ANALYSIS_COMPLETE",
                    "REMEDIATION_REQUESTED",
                    now,
                    "remediation opened",
                ),
            )
            conn.commit()
        record = self.remediation(remediation_id)
        assert record is not None
        return record

    def transition(
        self,
        remediation_id: str,
        to_state: str,
        *,
        detail: str = "",
        error: str = "",
        **fields: Any,
    ) -> RemediationRecord:
        """Move to `to_state`, refusing edges the graph does not have.

        `fields` sets columns atomically with the transition — the artifact id
        lands in the same write as RENDERED, so there is no window where a
        record claims to be rendered without naming what it rendered.
        """
        allowed = {
            "artifact_id",
            "output_path",
            "edl_json",
            "verdict",
            "verification_run_id",
            "verification_id",
            "simulation_id",
            "finding_ids",
            "incident_ids",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"unknown remediation columns: {sorted(unknown)}")

        now = _now()
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT state FROM remediations WHERE remediation_id = ?",
                (remediation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"no remediation {remediation_id}")
            current = str(row["state"])
            lifecycle.check(current, to_state)

            assignments = ", ".join(f"{name} = ?" for name in fields)
            prefix = f"{assignments}, " if assignments else ""
            conn.execute(
                f"UPDATE remediations SET {prefix}state = ?, previous_state = ?, "
                "error = ?, updated_at = ? WHERE remediation_id = ?",
                (
                    *fields.values(),
                    to_state,
                    current,
                    error or None,
                    now,
                    remediation_id,
                ),
            )
            conn.execute(
                "INSERT INTO transitions "
                "(remediation_id, from_state, to_state, at, detail, error) "
                "VALUES (?,?,?,?,?,?)",
                (remediation_id, current, to_state, now, detail, error),
            )
            conn.commit()
        record = self.remediation(remediation_id)
        assert record is not None
        return record

    def fail(self, remediation_id: str, error: str) -> RemediationRecord:
        return self.transition(
            remediation_id, "FAILED", detail="remediation failed", error=error
        )

    def remediation(self, remediation_id: str) -> RemediationRecord | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM remediations WHERE remediation_id = ?",
                (remediation_id,),
            ).fetchone()
            if row is None:
                return None
            history = conn.execute(
                "SELECT * FROM transitions WHERE remediation_id = ? ORDER BY id",
                (remediation_id,),
            ).fetchall()
        return _remediation(row, history)

    def remediations(self, *, source_run_id: str | None = None) -> list[RemediationRecord]:
        query = "SELECT * FROM remediations"
        args: tuple[Any, ...] = ()
        if source_run_id:
            query += " WHERE source_run_id = ?"
            args = (source_run_id,)
        query += " ORDER BY remediation_id"
        with closing(self._connect()) as conn:
            rows = conn.execute(query, args).fetchall()
            out = []
            for row in rows:
                history = conn.execute(
                    "SELECT * FROM transitions WHERE remediation_id = ? ORDER BY id",
                    (row["remediation_id"],),
                ).fetchall()
                out.append(_remediation(row, history))
        return out

    def interrupted(self) -> list[RemediationRecord]:
        """Remediations a process left non-terminal.

        This is what makes a restart informative rather than lossy: the answer
        to "what was happening when it died" is a row, so the system can say
        "REM-0001 was interrupted during REANALYSING" instead of forgetting
        the operation existed.
        """
        return [r for r in self.remediations() if r.interrupted]

    # ---- verifications and certificates --------------------------------- #

    def record_verification(
        self,
        remediation_id: str,
        *,
        original_run_id: str,
        verification_run_id: str | None,
        comparison: dict[str, Any],
        telemetry: dict[str, Any] | None = None,
    ) -> str:
        with closing(self._connect()) as conn:
            verification_id = self._next_id(
                conn, "verifications", "verification_id", "VER"
            )
            conn.execute(
                "INSERT INTO verifications VALUES (?,?,?,?,?,?,?,?)",
                (
                    verification_id,
                    remediation_id,
                    original_run_id,
                    verification_run_id,
                    str(comparison.get("verdict", "INCONCLUSIVE")),
                    json.dumps(comparison, separators=(",", ":")),
                    json.dumps(telemetry or {}, separators=(",", ":")),
                    _now(),
                ),
            )
            conn.commit()
        return verification_id

    def verification(self, verification_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM verifications WHERE verification_id = ?",
                (verification_id,),
            ).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["comparison"] = json.loads(out.pop("comparison_json"))
        out["telemetry"] = json.loads(out.pop("telemetry_json"))
        return out

    def record_certificate(
        self, verification_id: str, payload: dict[str, Any], certificate_hash: str
    ) -> str:
        with closing(self._connect()) as conn:
            certificate_id = self._next_id(
                conn, "certificates", "certificate_id", "CERT"
            )
            conn.execute(
                "INSERT INTO certificates VALUES (?,?,?,?,?)",
                (
                    certificate_id,
                    verification_id,
                    certificate_hash,
                    json.dumps(payload, separators=(",", ":"), sort_keys=True),
                    _now(),
                ),
            )
            conn.commit()
        return certificate_id

    def certificate(self, certificate_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM certificates WHERE certificate_id = ?",
                (certificate_id,),
            ).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["payload"] = json.loads(out.pop("payload_json"))
        return out

    def certificate_for(self, verification_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT certificate_id FROM certificates WHERE verification_id = ? "
                "ORDER BY certificate_id DESC LIMIT 1",
                (verification_id,),
            ).fetchone()
        return self.certificate(row["certificate_id"]) if row else None

    # ---- the graph ------------------------------------------------------ #

    def graph(self, run_id: str) -> Graph:
        """One original run and everything derived from it, resolved."""
        root = self.run(run_id)
        remediations = self.remediations(source_run_id=run_id)
        verification_runs = [
            node
            for node in self.children_of(run_id)
            if node.role == "VERIFICATION"
        ]
        artifacts = [
            artifact
            for artifact in (
                self.artifact(r.artifact_id) for r in remediations if r.artifact_id
            )
            if artifact is not None
        ]
        certificates = [
            certificate
            for certificate in (
                self.certificate_for(r.verification_id)
                for r in remediations
                if r.verification_id
            )
            if certificate is not None
        ]
        return Graph(
            root=root,
            simulations=self.simulations_for(run_id),
            remediations=remediations,
            verification_runs=verification_runs,
            artifacts=artifacts,
            certificates=certificates,
        )

    def stats(self) -> dict[str, int]:
        with closing(self._connect()) as conn:
            return {
                table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in (
                    "runs",
                    "artifacts",
                    "simulations",
                    "remediations",
                    "transitions",
                    "verifications",
                    "certificates",
                )
            }


def _artifact(row: sqlite3.Row) -> Artifact:
    return Artifact(
        artifact_id=row["artifact_id"],
        path=row["path"],
        content_hash=row["content_hash"],
        size_bytes=row["size_bytes"],
        duration_ms=row["duration_ms"],
        created_at=row["created_at"],
    )


def _run(row: sqlite3.Row) -> RunNode:
    return RunNode(
        run_id=row["run_id"],
        parent_run_id=row["parent_run_id"],
        role=row["role"],
        artifact_id=row["artifact_id"],
        video_path=row["video_path"],
        video_hash=row["video_hash"],
        report_path=row["report_path"],
        overall=row["overall"],
        verdict=row["verdict"],
        duration_ms=row["duration_ms"],
        findings=row["findings"],
        incidents=row["incidents"],
        coverage=row["coverage"],
        created_at=row["created_at"],
    )


def _remediation(
    row: sqlite3.Row, history: Iterable[sqlite3.Row]
) -> RemediationRecord:
    return RemediationRecord(
        remediation_id=row["remediation_id"],
        source_run_id=row["source_run_id"],
        simulation_id=row["simulation_id"],
        verification_run_id=row["verification_run_id"],
        verification_id=row["verification_id"],
        artifact_id=row["artifact_id"],
        source_path=row["source_path"],
        output_path=row["output_path"],
        finding_ids=tuple(json.loads(row["finding_ids"] or "[]")),
        incident_ids=tuple(json.loads(row["incident_ids"] or "[]")),
        ops=tuple(json.loads(row["edl_json"] or "[]")),
        state=row["state"],
        previous_state=row["previous_state"],
        verdict=row["verdict"],
        error=row["error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        transitions=tuple(
            lifecycle.Transition(
                from_state=t["from_state"],
                to_state=t["to_state"],
                at=t["at"],
                detail=t["detail"],
                error=t["error"],
            )
            for t in history
        ),
    )
