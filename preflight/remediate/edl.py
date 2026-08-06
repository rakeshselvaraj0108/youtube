"""The Edit Decision List — lowering and optimisation.

This is a compiler. Findings lower into a typed IR, seven passes run over it in
a fixed order, and the result is validated before codegen touches it. Each pass
exists because of a specific way naive remediation produces something a creator
would not ship.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from preflight.models import Finding
from preflight.perception.asr import Transcript

OpKind = Literal["MUTE", "BLEEP", "BLUR_REGION", "REPLACE_AUDIO", "CUT"]

AUDIO_OPS = {"MUTE", "BLEEP", "REPLACE_AUDIO"}
VIDEO_OPS = {"BLUR_REGION", "CUT"}

COALESCE_GAP_MS = 400
PAD_HEAD_MS = 60
PAD_TAIL_MS = 80
MAX_CUT_RATIO = 0.08
MIN_SPAN_MS = 120

DEFAULT_BOX = (0.29, 0.35, 0.42, 0.30)
DEFAULT_BLEEP_HZ = 1000
DEFAULT_BED = "assets/cc_music/glacier_calm.mp3"

Strategy = Literal["conservative", "balanced", "aggressive"]

# (viewer_impact, forces_reencode, risk_reduction). Impact and risk reduction
# are both 0-1 and independently authored — they are not derived from each
# other, because the whole point of the table is that a fix can score high on
# one and low on the other. REPLACE_AUDIO scores the least destructive of the
# audio ops: something plays where something else did, rather than a gap.
# CUT scores full risk reduction because nothing survives to re-offend, at
# the highest cost because it is the one op a viewer always notices.
COST: dict[str, tuple[float, bool, float]] = {
    "BLEEP": (0.15, False, 0.90),
    "MUTE": (0.25, False, 0.90),
    "REPLACE_AUDIO": (0.10, False, 0.95),
    "BLUR_REGION": (0.20, True, 0.85),
    "CUT": (0.80, True, 1.00),
}

# Cumulative viewer-impact budget for one file, by strategy. Aggressive still
# has a ceiling — "aggressive" means willing to cut, not willing to gut the
# video, and a ceiling below 1.0 is what keeps that true under a finding list
# long enough that summing every fix's impact would otherwise exceed it.
STRATEGY_CEILING: dict[str, float] = {
    "conservative": 0.25,
    "balanced": 0.45,
    "aggressive": 0.80,
}

SEVERITY_SCORE = {"CRITICAL": 1.00, "HIGH": 0.76, "MEDIUM": 0.52, "LOW": 0.28}


@dataclass
class Op:
    op: OpKind
    start_ms: int
    end_ms: int
    finding_id: str
    reason: str
    details: str = ""
    box: tuple[float, float, float, float] | None = None
    asset: str | None = None
    freq_hz: int | None = None
    index: int = 0

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    @property
    def is_audio(self) -> bool:
        return self.op in AUDIO_OPS

    @property
    def is_video(self) -> bool:
        return self.op in VIDEO_OPS

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "index": self.index,
            "op": self.op,
            "startMs": int(self.start_ms),
            "endMs": int(self.end_ms),
            "details": self.details or self.reason,
            "findingId": self.finding_id,
        }
        if self.box is not None:
            payload["box"] = [round(v, 4) for v in self.box]
        if self.asset is not None:
            payload["asset"] = self.asset
        if self.freq_hz is not None:
            payload["freqHz"] = self.freq_hz
        return payload


@dataclass
class EDL:
    source: str
    duration_ms: int
    ops: list[Op] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_video_ops(self) -> bool:
        return any(op.is_video for op in self.ops)

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "durationMs": self.duration_ms,
            "ops": [op.to_json() for op in self.ops],
            "warnings": list(self.warnings),
        }


class InvalidEDL(ValueError):
    """The optimised EDL would produce a corrupt render."""


# ------------------------------------------------------------------ #
# Lowering                                                            #
# ------------------------------------------------------------------ #


def _make_op(finding: Finding, kind: str) -> Op:
    return Op(
        op=kind,  # type: ignore[arg-type]
        start_ms=int(finding.startMs),
        end_ms=int(finding.endMs),
        finding_id=finding.id,
        reason=finding.clauseId,
        details=f"{finding.title} · {finding.clauseId}",
        box=DEFAULT_BOX if kind == "BLUR_REGION" else None,
        asset=DEFAULT_BED if kind == "REPLACE_AUDIO" else None,
        freq_hz=DEFAULT_BLEEP_HZ if kind == "BLEEP" else None,
    )


def lower(
    findings: list[Finding],
    source: str,
    duration_ms: int,
    strategy: Strategy | None = None,
) -> EDL:
    """Findings -> typed ops. Only findings carrying a fix produce one.

    Without a strategy, each finding's own `suggestedFix` — the ADJUDICATOR's
    single least-destructive pick — is trusted directly, which is this
    function's original and still-default behaviour. With one, `choose()`
    decides instead: not just where to fix something, but which fix to use,
    under a budget on how much the file can visibly change in one pass.
    """
    if strategy is not None:
        ops, log = choose(findings, strategy)
        edl = EDL(source=source, duration_ms=duration_ms, ops=ops)
        edl.log.extend(log)
        return edl

    ops = []
    for finding in findings:
        kind = finding.suggestedFix
        if kind == "NONE" or kind not in AUDIO_OPS | VIDEO_OPS:
            continue
        ops.append(_make_op(finding, kind))

    return EDL(source=source, duration_ms=duration_ms, ops=ops)


# ------------------------------------------------------------------ #
# Cost-aware strategy selection                                       #
# ------------------------------------------------------------------ #


def candidates_for(finding: Finding) -> list[str]:
    """Which fix kinds could plausibly resolve this finding.

    Inferred from modality rather than from the finding's own `suggestedFix`,
    so a downgrade under budget pressure has real alternatives to downgrade
    TO. A visual finding cannot be fixed by muting audio, so BLUR_REGION and
    CUT are its only candidates; an audio finding has the full audio set plus
    CUT as the last resort common to both.
    """
    if finding.suggestedFix == "NONE":
        return []
    if finding.modalities.get("vision", 0.0) >= max(
        (v for k, v in finding.modalities.items() if k != "vision"), default=0.0
    ):
        return ["BLUR_REGION", "CUT"]
    return ["BLEEP", "MUTE", "REPLACE_AUDIO", "CUT"]


def choose(
    findings: list[Finding], strategy: Strategy = "balanced"
) -> tuple[list[Op], list[str]]:
    """Decide HOW to fix each finding, not just where.

    Findings are processed most-severe-and-most-confident first, so if the
    budget runs out it runs out on the findings that matter least. For each,
    every candidate fix is scored as risk reduction minus a discounted
    viewer-impact cost minus a re-encode penalty — the same shape of tradeoff
    an editor makes by hand, made explicit and consistent across every
    finding in the file rather than improvised per clip.

    The budget is the total viewer impact this pass may spend, by strategy.
    A finding whose only candidates would blow the remaining budget is
    skipped rather than forced through over budget — reported, not silently
    dropped, so a reader knows the plan is partial and why.
    """
    ceiling = STRATEGY_CEILING[strategy]
    ordered = sorted(
        findings,
        key=lambda f: -(SEVERITY_SCORE.get(f.severity, 0.0) * f.confidence),
    )

    ops: list[Op] = []
    log: list[str] = []
    spent = 0.0

    for finding in ordered:
        candidates = candidates_for(finding)
        if not candidates:
            continue

        best: tuple[float, str, float] | None = None  # (score, kind, impact)
        for kind in candidates:
            impact, reencode, risk_reduction = COST[kind]
            if spent + impact > ceiling:
                continue
            score = risk_reduction - impact * 0.6 - (0.25 if reencode else 0.0)
            if best is None or score > best[0]:
                best = (score, kind, impact)

        if best is None:
            log.append(
                f"skipped {finding.clauseId} at {finding.startMs}ms — every "
                f"candidate fix would exceed the {strategy} budget"
            )
            continue

        _, chosen, impact = best
        spent += impact
        ops.append(_make_op(finding, chosen))

        original = finding.suggestedFix
        if chosen != original and original in COST:
            orig_impact, _, orig_reduction = COST[original]
            delta_reduction = COST[chosen][2] - orig_reduction
            delta_impact = orig_impact - impact
            comparison = (
                "same risk reduction"
                if abs(delta_reduction) < 0.01
                else f"{delta_reduction:+.2f} risk reduction"
            )
            log.append(
                f"chose {chosen} over {original} at {finding.startMs}ms — "
                f"{comparison}, {delta_impact:+.2f} viewer impact"
            )

    return ops, log


# ------------------------------------------------------------------ #
# Optimiser passes                                                    #
# ------------------------------------------------------------------ #


def optimise(edl: EDL, transcript: Transcript | None = None) -> EDL:
    """Seven passes, in order. Order matters — snapping before coalescing
    would merge spans that word boundaries had not yet widened."""
    edl.ops.sort(key=lambda o: (o.start_ms, o.end_ms))

    _snap_to_words(edl, transcript)
    _pad(edl)
    _coalesce(edl)
    _dominance(edl)
    _resolve_conflicts(edl)
    _cut_budget(edl)
    _validate(edl)

    for index, op in enumerate(edl.ops, start=1):
        op.index = index
    return edl


def _snap_to_words(edl: EDL, transcript: Transcript | None) -> None:
    """Pass 1 — expand audio ops to whole word boundaries.

    Clipping mid-syllable is audible and immediately reads as amateur. The
    transcript already knows where every word starts and ends.
    """
    if transcript is None or not transcript.words:
        return
    snapped = 0
    for op in edl.ops:
        if not op.is_audio:
            continue
        start, end = transcript.snap_to_words(op.start_ms, op.end_ms)
        if (start, end) != (op.start_ms, op.end_ms):
            op.start_ms, op.end_ms = start, end
            snapped += 1
    if snapped:
        edl.log.append(f"snap-to-word: widened {snapped} audio op(s)")


def _pad(edl: EDL) -> None:
    """Pass 2 — head/tail padding on audio ops.

    ASR timestamps drift by a few tens of milliseconds. Without padding the
    first consonant of a bleeped word survives, which is worse than no bleep at
    all because it draws attention.
    """
    for op in edl.ops:
        if op.is_audio:
            op.start_ms = max(0, op.start_ms - PAD_HEAD_MS)
            op.end_ms = min(edl.duration_ms, op.end_ms + PAD_TAIL_MS)
    edl.log.append(f"pad: {PAD_HEAD_MS}ms head / {PAD_TAIL_MS}ms tail on audio ops")


def _coalesce(edl: EDL) -> None:
    """Pass 3 — merge same-kind ops separated by less than 400ms.

    Two bleeps 200ms apart produce an audible stutter between them; one bleep
    across both is what an editor would actually do.
    """
    merged: list[Op] = []
    for op in sorted(edl.ops, key=lambda o: (o.op, o.start_ms)):
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and previous.op == op.op
            and op.start_ms - previous.end_ms < COALESCE_GAP_MS
            and op.start_ms >= previous.start_ms
        ):
            previous.end_ms = max(previous.end_ms, op.end_ms)
            previous.details = f"{previous.details} + {op.details}"
            continue
        merged.append(op)

    if len(merged) != len(edl.ops):
        edl.log.append(f"coalesce: {len(edl.ops)} -> {len(merged)} op(s)")
    edl.ops = sorted(merged, key=lambda o: o.start_ms)


def _dominance(edl: EDL) -> None:
    """Pass 4 — a CUT subsumes anything inside its span.

    Muting audio that is about to be deleted is wasted work and makes the
    filter graph harder to read.
    """
    cuts = [op for op in edl.ops if op.op == "CUT"]
    if not cuts:
        return
    survivors: list[Op] = []
    dropped = 0
    for op in edl.ops:
        if op.op != "CUT" and any(
            cut.start_ms <= op.start_ms and op.end_ms <= cut.end_ms for cut in cuts
        ):
            dropped += 1
            continue
        survivors.append(op)
    if dropped:
        edl.log.append(f"dominance: dropped {dropped} op(s) inside a CUT")
    edl.ops = survivors


def _resolve_conflicts(edl: EDL) -> None:
    """Pass 5 — REPLACE_AUDIO beats MUTE on any intersection.

    Both silence the source; only one puts something back. Muting the span the
    replacement bed occupies would silence the replacement too.
    """
    replaces = [op for op in edl.ops if op.op == "REPLACE_AUDIO"]
    if not replaces:
        return
    survivors: list[Op] = []
    trimmed = 0
    for op in edl.ops:
        if op.op != "MUTE":
            survivors.append(op)
            continue
        covered = any(
            r.start_ms <= op.start_ms and op.end_ms <= r.end_ms for r in replaces
        )
        if covered:
            trimmed += 1
            continue
        survivors.append(op)
    if trimmed:
        edl.log.append(f"conflict: REPLACE_AUDIO absorbed {trimmed} MUTE op(s)")
    edl.ops = survivors


def _cut_budget(edl: EDL) -> None:
    """Pass 6 — demote cuts beyond the budget rather than deleting the video.

    Nobody wants a tool that silently removes a third of their footage. Past
    the budget the lowest-confidence cuts become mutes and the report says so.
    """
    cuts = [op for op in edl.ops if op.op == "CUT"]
    if not cuts or edl.duration_ms <= 0:
        return

    budget = edl.duration_ms * MAX_CUT_RATIO
    total = sum(op.duration_ms for op in cuts)
    if total <= budget:
        return

    # Demote longest-first: removing the biggest offenders recovers the budget
    # in the fewest demotions.
    demoted = 0
    for op in sorted(cuts, key=lambda o: -o.duration_ms):
        if total <= budget:
            break
        op.op = "MUTE"  # type: ignore[assignment]
        op.details = f"{op.details} (demoted from CUT — budget)"
        total -= op.duration_ms
        demoted += 1

    warning = (
        f"cut budget: {demoted} cut(s) demoted to MUTE — total cuts exceeded "
        f"{MAX_CUT_RATIO:.0%} of runtime"
    )
    edl.log.append(warning)
    edl.warnings.append(warning)


def _validate(edl: EDL) -> None:
    """Pass 7 — fail loudly rather than render something corrupt."""
    for op in edl.ops:
        if op.end_ms <= op.start_ms:
            raise InvalidEDL(f"{op.op} has a non-positive span: {op.start_ms}-{op.end_ms}")
        if op.duration_ms < MIN_SPAN_MS:
            raise InvalidEDL(
                f"{op.op} span {op.duration_ms}ms is below the {MIN_SPAN_MS}ms floor"
            )
        if op.start_ms < 0 or op.end_ms > edl.duration_ms:
            raise InvalidEDL(
                f"{op.op} span {op.start_ms}-{op.end_ms} falls outside the runtime "
                f"0-{edl.duration_ms}"
            )
        if op.op == "BLUR_REGION" and op.box is None:
            raise InvalidEDL("BLUR_REGION without a region box")
        if op.op == "REPLACE_AUDIO" and not op.asset:
            raise InvalidEDL("REPLACE_AUDIO without a replacement asset")

    starts = [op.start_ms for op in edl.ops]
    if starts != sorted(starts):
        raise InvalidEDL("ops are not in monotonic order after optimisation")


def compile_edl(
    findings: list[Finding],
    source: str,
    duration_ms: int,
    transcript: Transcript | None = None,
    strategy: Strategy | None = None,
) -> EDL:
    return optimise(lower(findings, source, duration_ms, strategy), transcript)
