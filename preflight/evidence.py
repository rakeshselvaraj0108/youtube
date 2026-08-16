"""Before / after evidence, extracted from the artifacts that actually exist.

The single rule this module exists to enforce: **an "after" frame comes out of
the rendered file, or there is no after frame.**

The tempting shortcut is to show the original frame under an "after" label —
it is already extracted, already embedded in the report, and for a
stream-copied audio edit it even looks right. It is also the exact failure
this whole verification loop was built to prevent. A reader looking at a
before/after pair is being told "this is what the fix did"; showing them the
same frame twice is a claim about the output made from the input.

Two consequences follow, and both are visible in the output shape.

**A cut has no after frame.** The evidence was removed, which is a real and
good outcome, but there is nothing to photograph. `after` is None and
`removed_by_remediation` is True, so the deck can say EVIDENCE REMOVED BY
REMEDIATION rather than seeking to a timestamp that no longer exists.

**Extraction can fail.** A frame at 00:59.8 of a file that is 59.9 seconds
long may not decode. That is `NOT_MEASURED`, not a blank image and not the
nearest frame silently substituted — a substituted frame is a wrong answer
wearing the costume of a right one.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from preflight import ffmpeg
from preflight.verify import TimeMap

# Evidence frames are wider than the report's embedded keyframes: a reader
# comparing two stills is looking for a difference between them, and 640px is
# where a blur or a masked region stops being legible.
EVIDENCE_WIDTH = 720

# Accurate seeking, cheaply. Input seeking (`-ss` before `-i`) jumps to the
# nearest keyframe and is fast but can land a second early; output seeking is
# exact but decodes from zero. Seeking the bulk on input and the remainder on
# output gets exactness at roughly the cost of the fast path — which matters
# because a verification pass extracts two frames per finding.
PREROLL_MS = 2_000


@dataclass(frozen=True)
class EvidenceFrame:
    """One still, and where it genuinely came from."""

    ts_ms: int
    path: str
    source: str  # "original" | "remediated"
    run_id: str | None = None
    width: int = EVIDENCE_WIDTH

    def data_uri(self) -> str:
        payload = base64.b64encode(Path(self.path).read_bytes()).decode("ascii")
        return f"data:image/jpeg;base64,{payload}"

    def to_json(self, *, embed: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "tsMs": self.ts_ms,
            "source": self.source,
            "runId": self.run_id,
            "width": self.width,
        }
        if embed and Path(self.path).is_file():
            out["image"] = self.data_uri()
        return out


@dataclass(frozen=True)
class RemediationStep:
    """What the edit did to this span — the middle panel of the triptych."""

    remediation_id: str
    op: str
    start_ms: int
    end_ms: int

    def to_json(self) -> dict[str, Any]:
        return {
            "remediationId": self.remediation_id,
            "op": self.op,
            "startMs": self.start_ms,
            "endMs": self.end_ms,
        }


@dataclass
class EvidencePair:
    """One finding, before and after, with the edit between them."""

    finding_id: str
    clause_id: str
    category: str
    severity: str
    status: str
    incident_id: str | None = None

    before_run_id: str | None = None
    before_ts_ms: int = 0
    before: EvidenceFrame | None = None
    transcript: str = ""
    highlight_span: tuple[int, int] = (0, 0)
    confidence: float = 0.0
    coverage: float | None = None

    remediation: RemediationStep | None = None

    after_run_id: str | None = None
    after_ts_ms: int | None = None
    after: EvidenceFrame | None = None
    removed_by_remediation: bool = False
    after_unavailable: str = ""

    notes: list[str] = field(default_factory=list)

    def to_json(self, *, embed: bool = True) -> dict[str, Any]:
        return {
            "findingId": self.finding_id,
            "incidentId": self.incident_id,
            "clauseId": self.clause_id,
            "category": self.category,
            "severity": self.severity,
            "status": self.status,
            "before": {
                "runId": self.before_run_id,
                "tsMs": self.before_ts_ms,
                "frame": self.before.to_json(embed=embed) if self.before else None,
                "transcript": self.transcript,
                "highlightSpan": list(self.highlight_span),
                "confidence": self.confidence,
                # None, not 0. A coverage the engine did not record is not a
                # coverage of zero, and rendering it as 0% would be a
                # measurement claim nobody made.
                "coverage": self.coverage,
            },
            "remediation": self.remediation.to_json() if self.remediation else None,
            "after": {
                "runId": self.after_run_id,
                "tsMs": self.after_ts_ms,
                "frame": self.after.to_json(embed=embed) if self.after else None,
                "removedByRemediation": self.removed_by_remediation,
                "unavailable": self.after_unavailable,
            },
            "notes": list(self.notes),
        }


def extract_frame(
    source: Path,
    ts_ms: int,
    destination: Path,
    *,
    width: int = EVIDENCE_WIDTH,
) -> Path | None:
    """One frame at one instant, or None.

    None rather than a substitute. Every caller here is building a claim about
    a specific moment, and the nearest decodable frame is a different moment —
    close enough to look right and wrong enough to mislead.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    ts_ms = max(0, int(ts_ms))
    preroll_s = max(0.0, (ts_ms - PREROLL_MS) / 1000.0)
    remainder_s = (ts_ms / 1000.0) - preroll_s

    try:
        ffmpeg.run(
            [
                "-y",
                "-ss",
                f"{preroll_s:.3f}",
                "-i",
                str(source),
                "-ss",
                f"{remainder_s:.3f}",
                "-frames:v",
                "1",
                "-q:v",
                "3",
                "-vf",
                f"scale={width}:-2",
                str(destination),
            ]
        )
    except (ffmpeg.FfmpegFailed, ffmpeg.FfmpegMissing):
        destination.unlink(missing_ok=True)
        return None

    if not destination.is_file() or destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        return None
    return destination


def _op_covering(ops: list[Any], start_ms: int, end_ms: int) -> Any | None:
    """The edit operation that acts on this span, if any."""
    for op in ops:
        op_start, op_end = int(getattr(op, "start_ms", 0)), int(getattr(op, "end_ms", 0))
        if op_start < end_ms and op_end > start_ms:
            return op
    return None


def build_pairs(
    changes: list[Any],
    original_findings: list[dict[str, Any]],
    remediated_findings: list[dict[str, Any]],
    *,
    original_path: Path,
    remediated_path: Path | None,
    ops: list[Any],
    remediation_id: str,
    original_run_id: str,
    verification_run_id: str | None,
    out_dir: Path,
    coverage: dict[str, float] | None = None,
    incidents: list[dict[str, Any]] | None = None,
    limit: int = 12,
) -> list[EvidencePair]:
    """Before / after evidence for the findings that carry the verdict.

    Bounded by `limit` because each pair costs two ffmpeg invocations and a
    verdict is carried by a handful of findings, not by all of them. The
    ordering puts the ones a reader will ask about first: what appeared, what
    survived, then what was fixed.
    """
    time_map = TimeMap.from_ops(ops)
    by_id = {str(f.get("id")): f for f in original_findings}
    after_by_id = {str(f.get("id")): f for f in remediated_findings}
    incident_of: dict[str, str] = {}
    for incident in incidents or []:
        for fid in incident.get("findingIds", []):
            incident_of[str(fid)] = str(incident.get("id"))

    rank = {"NEW": 0, "PERSISTING": 1, "CHANGED": 1, "INCONCLUSIVE": 2, "RESOLVED": 3}
    ordered = sorted(changes, key=lambda c: rank.get(c.status, 4))[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs: list[EvidencePair] = []

    for index, change in enumerate(ordered):
        original = by_id.get(str(change.original_id or ""))
        after_finding = after_by_id.get(str(change.remediated_id or ""))
        anchor = original or after_finding
        if anchor is None:
            continue

        start_ms = int(anchor.get("startMs", 0))
        end_ms = int(anchor.get("endMs", start_ms))
        midpoint = (start_ms + end_ms) // 2
        evidence = anchor.get("evidence") or {}
        observers = [
            name for name, value in (anchor.get("modalities") or {}).items() if value > 0
        ]
        measured = [
            (coverage or {}).get(name) for name in observers
            if (coverage or {}).get(name) is not None
        ]

        pair = EvidencePair(
            finding_id=str(change.original_id or change.remediated_id or ""),
            incident_id=incident_of.get(str(change.original_id or "")),
            clause_id=change.clause_id,
            category=change.category,
            severity=change.severity,
            status=change.status,
            transcript=str(evidence.get("transcript", "")),
            highlight_span=tuple(evidence.get("highlightSpan", (0, 0)))[:2],  # type: ignore[arg-type]
            confidence=float(anchor.get("confidence", 0.0) or 0.0),
            coverage=max(measured) if measured else None,
        )

        # BEFORE — from the original file, at the original timestamp. Skipped
        # for a NEW finding, which by definition has no "before": it did not
        # exist in the input, and inventing a before frame for it would assert
        # it did.
        if original is not None:
            pair.before_run_id = original_run_id
            pair.before_ts_ms = midpoint
            frame = extract_frame(
                original_path, midpoint, out_dir / f"before_{index:02d}.jpg"
            )
            if frame:
                pair.before = EvidenceFrame(
                    ts_ms=midpoint,
                    path=str(frame),
                    source="original",
                    run_id=original_run_id,
                )
            else:
                pair.notes.append("before frame NOT MEASURED — extraction failed")

        acting = _op_covering(ops, start_ms, end_ms)
        if acting is not None:
            pair.remediation = RemediationStep(
                remediation_id=remediation_id,
                op=str(getattr(acting, "op", "")),
                start_ms=int(getattr(acting, "start_ms", 0)),
                end_ms=int(getattr(acting, "end_ms", 0)),
            )

        # AFTER — from the rendered file, at the mapped timestamp. Never from
        # the original, under any circumstances.
        mapped = time_map.to_remediated(midpoint)
        if remediated_path is None or not Path(remediated_path).is_file():
            pair.after_unavailable = "NOT MEASURED — no rendered artifact"
        elif mapped is None:
            pair.removed_by_remediation = True
            pair.after_unavailable = "EVIDENCE REMOVED BY REMEDIATION"
        else:
            pair.after_run_id = verification_run_id
            pair.after_ts_ms = mapped
            frame = extract_frame(
                Path(remediated_path), mapped, out_dir / f"after_{index:02d}.jpg"
            )
            if frame:
                pair.after = EvidenceFrame(
                    ts_ms=mapped,
                    path=str(frame),
                    source="remediated",
                    run_id=verification_run_id,
                )
            else:
                pair.after_unavailable = (
                    "NOT MEASURED — the rendered file did not decode a frame here"
                )

        pairs.append(pair)

    return pairs


def summarise(pairs: list[EvidencePair]) -> dict[str, Any]:
    """Counts a reader can check the evidence layer against.

    `afterFramesExtracted` is the number that matters: it is how many after
    frames genuinely came out of the rendered file, so a mismatch against
    `pairs` is visible rather than hidden behind a full-looking gallery.
    """
    return {
        "pairs": len(pairs),
        "beforeFramesExtracted": sum(1 for p in pairs if p.before is not None),
        "afterFramesExtracted": sum(1 for p in pairs if p.after is not None),
        "removedByRemediation": sum(1 for p in pairs if p.removed_by_remediation),
        "afterUnavailable": sum(
            1 for p in pairs if p.after is None and not p.removed_by_remediation
        ),
    }
