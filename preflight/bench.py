"""Measure the pipeline against the golden corpus, layer by layer.

Thirty clips in fifteen VIOLATION/CLEAN pairs. Each pair shares its surface
content and differs only in context: the same word inside an attributed
quotation, the same dangerous act with an explicit warning, the same sponsor
read with the disclosure present. Ground truth is exact by construction — the
profanity was inserted at 4,200ms, so it IS at 4,200ms, and nobody eyeballed
a timestamp.

TWO NUMBERS MATTER, AND THE SECOND ONE IS THE POINT.

Clip accuracy is the ordinary metric and it is close to worthless here, since
a system can reach 50% by answering VIOLATION to everything and another 50% by
answering CLEAN to everything.

PAIR accuracy is the honest one: a pair counts only when BOTH twins are
correct. Firing on the profanity AND staying quiet when the same word appears
inside a quotation is the whole claim of this project, and it is the only way
to score above chance. A keyword filter gets exactly 0% pair accuracy on this
corpus by construction — it cannot tell the twins apart, because at the level
of the words present, they are identical.

THE ABLATION harvests one run at three depths rather than running three times.

    lexicon     deterministic agents only — what a keyword filter sees
    auditor     + retrieval-grounded charges, before any defence
    triad       + advocate and adjudicator, the shipped system

Because all three read the same run, the difference between them is
attributable to the stage rather than to two runs disagreeing with each other.
The interesting column is the middle one: the AUDITOR is deliberately
over-sensitive, so it should score high recall and poor pair accuracy, and the
gap between `auditor` and `triad` is precisely what the ADVOCATE is worth.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from preflight import cas
from preflight.config import Settings
from preflight.models import Finding
from preflight.pipeline import PipelineResult, run_perception

LABELS = Path("data/corpus/labels.jsonl")
CLIPS = Path("data/corpus/clips")

# Depths at which the same run is harvested, shallowest first.
STAGES = ("lexicon", "auditor", "triad")

# A predicted span counts as located when it overlaps the constructed one by
# this much. Generous on purpose: ASR word boundaries drift by a syllable, and
# this metric is about whether the system found the right MOMENT, not whether
# it agrees with ffmpeg about where a word starts.
SPAN_IOU_FLOOR = 0.30


@dataclass(frozen=True)
class Label:
    clip: str
    label: str  # VIOLATION | CLEAN
    clause: str | None
    span_ms: tuple[int, int] | None
    twin_of: str | None
    note: str

    @property
    def is_violation(self) -> bool:
        return self.label == "VIOLATION"


@dataclass
class Prediction:
    """What one depth of one run claimed about one clip."""

    clip: str
    stage: str
    clauses: set[str] = field(default_factory=set)
    spans: list[tuple[int, int]] = field(default_factory=list)

    @property
    def fired(self) -> bool:
        return bool(self.clauses)


@dataclass
class Outcome:
    label: Label
    prediction: Prediction
    # For a CLEAN clip, the clause its twin violates — the one thing this clip
    # was built to NOT be.
    target_clause: str | None = None

    @property
    def correct(self) -> bool:
        """Right answer on this clip.

        A VIOLATION is right when the expected clause was cited.

        A CLEAN twin is right when the clause its twin violates is ABSENT —
        not when nothing fired at all. The distinction matters: every
        generated clip lacks a caption track, so scoring a clean twin against
        total silence marks it wrong for a house-rule finding that is
        perfectly true and has nothing to do with what the pair tests. The
        pair asks one question — same word, exempting context, does the
        system still charge it? — and that is the question scored.

        A CLEAN clip with no twin has nothing specific to be clean OF, so it
        falls back to the strict reading.
        """
        if self.label.is_violation:
            return self.label.clause in self.prediction.clauses
        if self.target_clause:
            return self.target_clause not in self.prediction.clauses
        return not self.prediction.fired

    @property
    def located(self) -> bool | None:
        """Did a correct VIOLATION also land on the right moment?

        None where the question does not apply — a CLEAN clip has no span to
        find, and a missed violation has nothing to locate.
        """
        if not self.label.is_violation or not self.correct:
            return None
        if self.label.span_ms is None:
            return None
        return any(
            span_iou(span, self.label.span_ms) >= SPAN_IOU_FLOOR
            for span in self.prediction.spans
        )


@dataclass
class Metrics:
    stage: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    pairs_total: int = 0
    pairs_correct: int = 0
    located: int = 0
    locatable: int = 0
    calls: int = 0
    elapsed_ms: int = 0

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    @property
    def pair_accuracy(self) -> float:
        return self.pairs_correct / self.pairs_total if self.pairs_total else 0.0

    @property
    def span_accuracy(self) -> float:
        return self.located / self.locatable if self.locatable else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "f1": round(self.f1, 3),
            "pairAccuracy": round(self.pair_accuracy, 3),
            "spanAccuracy": round(self.span_accuracy, 3),
            "truePositives": self.true_positives,
            "falsePositives": self.false_positives,
            "falseNegatives": self.false_negatives,
            "trueNegatives": self.true_negatives,
            "pairsCorrect": self.pairs_correct,
            "pairsTotal": self.pairs_total,
            "calls": self.calls,
            "elapsedMs": self.elapsed_ms,
        }


# ------------------------------------------------------------------ #
# Ground truth                                                        #
# ------------------------------------------------------------------ #


def load_labels(path: Path = LABELS) -> list[Label]:
    path = Path(path)
    if not path.is_file():
        return []
    out: list[Label] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        span = row.get("span_ms")
        out.append(
            Label(
                clip=row["clip"],
                label=row["label"],
                clause=row.get("clause"),
                span_ms=(int(span[0]), int(span[1])) if span else None,
                twin_of=row.get("twin_of"),
                note=row.get("note", ""),
            )
        )
    return out


def twin_clause(labels: Iterable[Label]) -> dict[str, str]:
    """For each CLEAN clip, the clause its VIOLATION twin was built to breach."""
    by_clip = {label.clip: label for label in labels}
    out: dict[str, str] = {}
    for label in labels:
        if label.is_violation or not label.twin_of:
            continue
        twin = by_clip.get(label.twin_of) or by_clip.get(f"{label.twin_of}.mp4")
        if twin is not None and twin.clause:
            out[label.clip] = twin.clause
    return out


def pairs(labels: Iterable[Label]) -> list[tuple[Label, Label]]:
    """VIOLATION/CLEAN twins, matched by the `twin_of` back-reference."""
    by_clip = {label.clip: label for label in labels}
    out: list[tuple[Label, Label]] = []
    for label in labels:
        if not label.twin_of:
            continue
        twin = by_clip.get(label.twin_of) or by_clip.get(f"{label.twin_of}.mp4")
        if twin is None:
            continue
        violation, clean = (twin, label) if label.label == "CLEAN" else (label, twin)
        out.append((violation, clean))
    return out


def span_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    left = max(a[0], b[0])
    right = min(a[1], b[1])
    if right <= left:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return (right - left) / union if union > 0 else 0.0


# ------------------------------------------------------------------ #
# Harvesting one run at three depths                                  #
# ------------------------------------------------------------------ #


def harvest(result: PipelineResult, clip: str, stage: str) -> Prediction:
    """What this depth of the pipeline claimed, from a single run.

    `lexicon` reads the deterministic agents only — the same evidence a
    keyword filter would act on, before anything read a clause.

    `auditor` adds the charges the AUDITOR brought. Charges, not findings:
    the AUDITOR is instructed to be over-sensitive and to leave exemptions to
    the ADVOCATE, so this depth is deliberately the noisy one.

    `triad` is the shipped answer — upheld verdicts after the defence was
    argued and ruled on.
    """
    prediction = Prediction(clip=clip, stage=stage)

    deterministic = [
        finding
        for agent in result.agents
        if agent.agent_id != "policy"
        for finding in agent.findings
    ]
    _absorb(prediction, deterministic)

    if stage == "lexicon":
        return prediction

    policy = result.agent("policy")
    if policy is None:
        return prediction

    if stage == "auditor":
        for candidate in policy.artifacts.get("candidates", []):
            prediction.clauses.add(str(candidate.get("clause_id", "")))
            prediction.spans.append(
                (int(candidate.get("start_ms", 0)), int(candidate.get("end_ms", 0)))
            )
        return prediction

    _absorb(prediction, policy.findings)
    return prediction


def _absorb(prediction: Prediction, findings: list[Finding]) -> None:
    for finding in findings:
        prediction.clauses.add(finding.clauseId)
        prediction.spans.append((finding.startMs, finding.endMs))


# ------------------------------------------------------------------ #
# Scoring                                                             #
# ------------------------------------------------------------------ #


def score(outcomes: list[Outcome], stage: str, all_labels: list[Label]) -> Metrics:
    metrics = Metrics(stage=stage)
    by_clip = {outcome.label.clip: outcome for outcome in outcomes}

    for outcome in outcomes:
        if outcome.label.is_violation:
            if outcome.correct:
                metrics.true_positives += 1
            else:
                metrics.false_negatives += 1
        else:
            if outcome.correct:
                metrics.true_negatives += 1
            else:
                metrics.false_positives += 1

        located = outcome.located
        if located is not None:
            metrics.locatable += 1
            metrics.located += int(located)

    for violation, clean in pairs(all_labels):
        left = by_clip.get(violation.clip)
        right = by_clip.get(clean.clip)
        if left is None or right is None:
            continue
        metrics.pairs_total += 1
        metrics.pairs_correct += int(left.correct and right.correct)

    return metrics


# ------------------------------------------------------------------ #
# Running                                                             #
# ------------------------------------------------------------------ #


def run_bench(
    labels: list[Label],
    *,
    clips_dir: Path = CLIPS,
    settings: Settings | None = None,
    store: cas.Store | None = None,
    stages: tuple[str, ...] = STAGES,
    progress=None,
) -> tuple[dict[str, Metrics], list[dict[str, Any]]]:
    """One pipeline run per clip, harvested at every requested depth.

    Running once and harvesting three times is what makes this an ablation
    rather than three experiments. It is also what makes it affordable: the
    triad is the expensive stage, and running it three times per clip to
    measure the contribution of its own sub-stages would be paying three times
    for information already present in the first run.
    """
    settings = settings or Settings.load()
    store = store or cas.Store(Path(settings.cache_dir))
    clips_dir = Path(clips_dir)

    outcomes: dict[str, list[Outcome]] = {stage: [] for stage in stages}
    per_clip: list[dict[str, Any]] = []
    totals: dict[str, list[int]] = {stage: [0, 0] for stage in stages}
    targets = twin_clause(labels)

    for label in labels:
        source = clips_dir / label.clip
        if not source.is_file():
            continue

        started = time.perf_counter()
        result = run_perception(source, store, settings=settings)
        elapsed = int((time.perf_counter() - started) * 1000)
        calls = result.total_calls

        row: dict[str, Any] = {
            "clip": label.clip,
            "expected": label.label,
            "clause": label.clause,
            "calls": calls,
            "elapsedMs": elapsed,
        }
        for stage in stages:
            prediction = harvest(result, label.clip, stage)
            outcome = Outcome(
                label=label,
                prediction=prediction,
                target_clause=targets.get(label.clip),
            )
            outcomes[stage].append(outcome)
            totals[stage][0] += calls
            totals[stage][1] += elapsed
            row[stage] = {
                "fired": prediction.fired,
                "clauses": sorted(c for c in prediction.clauses if c),
                "correct": outcome.correct,
                "located": outcome.located,
                "mustNotCite": targets.get(label.clip),
            }
        per_clip.append(row)
        if progress is not None:
            progress(row)

    scored: dict[str, Metrics] = {}
    for stage in stages:
        metrics = score(outcomes[stage], stage, labels)
        metrics.calls, metrics.elapsed_ms = totals[stage]
        scored[stage] = metrics
    return scored, per_clip


def to_json(
    scored: dict[str, Metrics], per_clip: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "corpus": {
            "clips": len(per_clip),
            "pairs": scored[next(iter(scored))].pairs_total if scored else 0,
        },
        "spanIouFloor": SPAN_IOU_FLOOR,
        "ablation": [metrics.to_json() for metrics in scored.values()],
        "clips": per_clip,
    }
