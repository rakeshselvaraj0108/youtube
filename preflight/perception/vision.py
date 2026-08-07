"""A03 — VISION INTELLIGENCE.

Turns scene keyframes into structured visual evidence. Reports what is visibly
present and never what it means: `knife` is an observation, `graphic violence`
is A11's conclusion, and A03 emitting the second would put a verdict into the
pipeline three agents early with no clause attached and no advocate to contest
it.

Three things carry this module, and all three exist because the underlying
model is the least reliable component in the system.

**A closed vocabulary.** The model may only speak in labels the vocabulary
knows. Anything else is normalised through a synonym map or discarded. This is
what stops a judgement arriving disguised as an observation, and it is enforced
rather than requested — the prompt asks for containment, the normaliser
guarantees it.

**Temporal tracks, not frame hits.** A knife visible for eight seconds spans
hundreds of frames. Reported per frame that is hundreds of observations, and
every count downstream inflates. Observations are merged into tracks with one
span, exactly as OCR text is.

**Persistence as corroboration.** A label seen once at 0.90 is weaker evidence
than the same label seen across five consecutive keyframes at 0.70. A single
frame is where hallucination lives; a thing that is actually in the shot stays
in the shot. Track confidence therefore rises with persistence and is capped
below the ceiling for singletons — the visual analogue of the cross-modal
agreement rule in the fusion layer.
"""

from __future__ import annotations

import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from preflight.ingest.frames import Keyframe
from preflight.models import AgentResult

AGENT_ID = "vision"
AGENT_NAME = "Vision Agent"
VOCAB_DIR = Path("data/vision")

# Concurrent frame descriptions. Four is latency hiding, not throughput —
# the vendor governor's token bucket is the real ceiling and it is enforced
# across threads, so raising this buys queueing rather than speed.
VISION_WORKERS = 4

# Confidence bands, per the specification.
BAND_VERY_HIGH = 0.95
BAND_HIGH = 0.90
BAND_MEDIUM = 0.75

# A single-frame observation cannot exceed this, however confident the model
# claims to be. One frame is precisely where a hallucination lives, and a
# model's self-reported certainty is not evidence about its own reliability.
SINGLETON_CEILING = 0.80

# Each additional consecutive frame is corroboration. Capped so a long static
# shot cannot manufacture certainty on its own.
PERSISTENCE_BONUS = 0.04
PERSISTENCE_CAP = 0.16

# Emotion inference is unreliable across cultures, ages and camera angles.
EMOTION_CEILING = 0.60

# Frames further apart than this are separate appearances, not one track.
TRACK_GAP_MS = 4_000

# Which category wins when a model-built compound contains several known
# labels. "person holding knife" contains both `person` and `knife`; the
# weapon is the observation worth surfacing, and picking by string length
# would return `person` because it happens to be one character longer.
#
# Ordered most to least consequential. Generic presence labels sit last
# deliberately: `person` appears in almost every frame and carries no signal on
# its own.
CATEGORY_SALIENCE = [
    "weapon", "injury", "substance", "activity", "money", "alcohol",
    "fire", "attire", "animal", "vehicle", "brand", "scene", "person",
    "emotion",
]

# Vision is the most expensive layer. Batching and gating together cut calls by
# roughly eighty percent while keeping recall on the segments that matter.
FRAMES_PER_CALL = 5
BASELINE_FRAMES = 8


@dataclass(frozen=True)
class Observation:
    """One label in one frame. Never a conclusion."""

    label: str
    category: str
    confidence: float
    ts_ms: int
    bbox: tuple[float, float, float, float] | None = None
    raw_label: str = ""

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": self.label,
            "category": self.category,
            "confidence": round(self.confidence, 3),
            "ts_ms": self.ts_ms,
        }
        if self.bbox:
            payload["bbox"] = [round(v, 4) for v in self.bbox]
        if self.raw_label and self.raw_label != self.label:
            payload["normalised_from"] = self.raw_label
        return payload


@dataclass
class Track:
    """One thing, seen across one or more frames."""

    label: str
    category: str
    start_ms: int
    end_ms: int
    frames: int
    peak_confidence: float
    confidence: float
    bbox: tuple[float, float, float, float] | None = None

    @property
    def band(self) -> str:
        if self.confidence > BAND_VERY_HIGH:
            return "VERY_HIGH"
        if self.confidence >= BAND_HIGH:
            return "HIGH"
        if self.confidence >= BAND_MEDIUM:
            return "MEDIUM"
        return "LOW"

    @property
    def corroborated(self) -> bool:
        """Seen in more than one frame. A singleton is the hallucination case."""
        return self.frames > 1

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": self.label,
            "category": self.category,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "frames": self.frames,
            "confidence": round(self.confidence, 3),
            "peak_confidence": round(self.peak_confidence, 3),
            "band": self.band,
            "corroborated": self.corroborated,
        }
        if self.bbox:
            payload["bbox"] = [round(v, 4) for v in self.bbox]
        return payload


@dataclass
class FrameFailure:
    scene: str
    ts_ms: int
    reason: str

    def to_json(self) -> dict[str, Any]:
        return {
            "status": "FAILED",
            "scene": self.scene,
            "ts_ms": self.ts_ms,
            "reason": self.reason,
        }


# ------------------------------------------------------------------ #
# Vocabulary                                                          #
# ------------------------------------------------------------------ #


class VisionVocabulary:
    """Closed label set with synonym normalisation and a judgement tripwire."""

    def __init__(self, directory: Path = VOCAB_DIR) -> None:
        self.directory = Path(directory)
        self.canonical: dict[str, str] = {}  # label -> category
        self.synonyms: dict[str, str] = {}  # alias -> canonical label
        self.blocklist: list[str] = []
        self.loaded: list[str] = []
        self._load()

    def _load(self) -> None:
        if not self.directory.is_dir():
            return
        for path in sorted(self.directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if path.name == "judgment_blocklist.json":
                self.blocklist = [t.lower() for t in payload.get("terms", [])]
                continue
            category = payload.get("_category", path.stem)
            self.loaded.append(path.name)
            for label in payload.get("labels", []):
                self.canonical[label.lower()] = category
            for alias, target in payload.get("synonyms", {}).items():
                self.synonyms[alias.lower()] = target.lower()

    @property
    def size(self) -> int:
        return len(self.canonical)

    def is_judgment(self, label: str) -> bool:
        lowered = label.lower()
        return any(term in lowered for term in self.blocklist)

    def normalise(self, label: str) -> tuple[str, str] | None:
        """Canonical label and category, or None if it cannot be admitted.

        Order matters: the judgement check runs first, so `graphic violence`
        is rejected rather than fuzzily matched onto something plausible.
        """
        if not label or not label.strip():
            return None

        cleaned = " ".join(label.lower().replace("_", " ").split())
        if self.is_judgment(cleaned):
            return None

        underscored = cleaned.replace(" ", "_")
        for candidate in (underscored, cleaned):
            if candidate in self.canonical:
                return candidate, self.canonical[candidate]
            if candidate in self.synonyms:
                target = self.synonyms[candidate]
                return target, self.canonical.get(target, "unknown")

        # A compound the model built itself — "person holding knife". Rather
        # than discard the frame's evidence, take the most salient known label
        # inside it: the weapon, not the person holding it.
        matches = [
            (known, category)
            for known, category in self.canonical.items()
            if known.replace("_", " ") in cleaned
        ]
        if matches:
            return min(matches, key=lambda item: self._salience(item))
        return None

    @staticmethod
    def _salience(match: tuple[str, str]) -> tuple[int, int]:
        """Sort key: category rank first, then longer label as a tie-break."""
        label, category = match
        rank = (
            CATEGORY_SALIENCE.index(category)
            if category in CATEGORY_SALIENCE
            else len(CATEGORY_SALIENCE)
        )
        return (rank, -len(label))


# ------------------------------------------------------------------ #
# Frame gating                                                        #
# ------------------------------------------------------------------ #


def select_frames(
    keyframes: list[Keyframe],
    flagged_spans: Iterable[tuple[int, int]] = (),
    *,
    baseline: int = BASELINE_FRAMES,
    budget: int | None = None,
) -> list[Keyframe]:
    """Which frames are worth paying for.

    Every frame inside a span the text layer already flagged, plus a uniform
    baseline so a clean transcript with a visual problem is not invisible.
    Roughly an eighty percent reduction against sending everything, and the
    frames it drops are the ones nothing else pointed at.
    """
    if not keyframes:
        return []

    spans = list(flagged_spans)
    chosen: dict[int, Keyframe] = {}

    for frame in keyframes:
        if any(start <= frame.ts_ms <= end for start, end in spans):
            chosen[frame.ts_ms] = frame

    if baseline > 0:
        step = max(1, len(keyframes) // baseline)
        for frame in keyframes[::step][:baseline]:
            chosen.setdefault(frame.ts_ms, frame)

    ordered = sorted(chosen.values(), key=lambda f: f.ts_ms)
    return ordered[:budget] if budget else ordered


def batches(frames: list[Keyframe], size: int = FRAMES_PER_CALL):
    for start in range(0, len(frames), size):
        yield frames[start : start + size]


# ------------------------------------------------------------------ #
# Response parsing                                                    #
# ------------------------------------------------------------------ #


def parse_observations(
    payload: Any, frame: Keyframe, vocab: VisionVocabulary
) -> tuple[list[Observation], list[str]]:
    """Strict validation of one frame's response.

    Anything that does not conform is dropped with a reason rather than
    partially trusted. A half-understood detection is worse than none: it
    carries a confidence the pipeline will weigh.
    """
    rejected: list[str] = []
    records: list[dict] = []

    if isinstance(payload, dict):
        for key in ("objects", "observations", "labels", "detections"):
            value = payload.get(key)
            if isinstance(value, list):
                records = [r for r in value if isinstance(r, dict)]
                break
        else:
            if "label" in payload:
                records = [payload]
    elif isinstance(payload, list):
        records = [r for r in payload if isinstance(r, dict)]

    observations: list[Observation] = []
    for record in records:
        raw = str(record.get("label") or record.get("object") or "").strip()
        if not raw:
            rejected.append("record with no label")
            continue

        resolved = vocab.normalise(raw)
        if resolved is None:
            reason = "judgement" if vocab.is_judgment(raw) else "unknown label"
            rejected.append(f"{raw!r} ({reason})")
            continue

        label, category = resolved
        try:
            confidence = float(record.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        if category == "emotion":
            confidence = min(confidence, EMOTION_CEILING)

        bbox = record.get("bbox")
        box: tuple[float, float, float, float] | None = None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                box = tuple(float(v) for v in bbox)  # type: ignore[assignment]
            except (TypeError, ValueError):
                box = None

        observations.append(
            Observation(
                label=label,
                category=category,
                confidence=confidence,
                ts_ms=frame.ts_ms,
                bbox=box,
                raw_label=raw.lower(),
            )
        )

    return observations, rejected


# ------------------------------------------------------------------ #
# Temporal merge                                                      #
# ------------------------------------------------------------------ #


def build_tracks(
    observations: list[Observation], gap_ms: int = TRACK_GAP_MS
) -> list[Track]:
    """Merge per-frame observations into tracks.

    Confidence is not the mean and not the max. The mean punishes a real object
    for one weak frame; the max lets a single confident hallucination through.
    Instead the peak sets the floor and persistence adds to it, capped — so
    corroboration raises certainty and a singleton cannot reach the top band.
    """
    by_label: dict[str, list[Observation]] = {}
    for observation in observations:
        by_label.setdefault(observation.label, []).append(observation)

    tracks: list[Track] = []
    for label, group in by_label.items():
        group.sort(key=lambda o: o.ts_ms)
        run: list[Observation] = [group[0]]

        for observation in group[1:]:
            if observation.ts_ms - run[-1].ts_ms <= gap_ms:
                run.append(observation)
                continue
            tracks.append(_track_from(label, run))
            run = [observation]
        tracks.append(_track_from(label, run))

    tracks.sort(key=lambda t: (t.start_ms, t.label))
    return tracks


def _track_from(label: str, run: list[Observation]) -> Track:
    peak = max(o.confidence for o in run)
    frames = len(run)

    if frames == 1:
        confidence = min(peak, SINGLETON_CEILING)
    else:
        bonus = min(PERSISTENCE_CAP, PERSISTENCE_BONUS * (frames - 1))
        confidence = min(0.99, peak + bonus)

    if run[0].category == "emotion":
        confidence = min(confidence, EMOTION_CEILING)

    widest = max(run, key=lambda o: o.confidence)
    return Track(
        label=label,
        category=run[0].category,
        start_ms=run[0].ts_ms,
        end_ms=run[-1].ts_ms,
        frames=frames,
        peak_confidence=peak,
        confidence=confidence,
        bbox=widest.bbox,
    )


# ------------------------------------------------------------------ #
# Agent entry point                                                   #
# ------------------------------------------------------------------ #


def encode_frame(frame: Keyframe) -> str:
    return base64.b64encode(frame.path.read_bytes()).decode("ascii")


def analyse(
    keyframes: list[Keyframe],
    registry: Any = None,
    *,
    flagged_spans: Iterable[tuple[int, int]] = (),
    vocab: VisionVocabulary | None = None,
    budget: int | None = None,
) -> tuple[AgentResult, list[Track]]:
    """Run the vision agent, degrading rather than failing.

    Without a `vision.describe` provider this reports SKIPPED with a reason and
    coverage zero. Coverage is the fraction of available keyframes actually
    inspected, so a gated or partially-failed run states plainly how much of
    the picture it saw.
    """
    started = time.perf_counter()
    vocab = vocab or VisionVocabulary()

    if not keyframes:
        return (
            AgentResult.skipped(AGENT_ID, AGENT_NAME, "no keyframes extracted"),
            [],
        )

    if registry is None:
        return (
            AgentResult.skipped(
                AGENT_ID, AGENT_NAME,
                "no vision.describe provider — visual findings unavailable",
            ),
            [],
        )

    from preflight.agents.roster import prompt_for
    from preflight.providers.registry import VISION_DESCRIBE

    selected = select_frames(keyframes, flagged_spans, budget=budget)
    prompt = prompt_for("A03") or "Describe what is visibly present. JSON only."

    observations: list[Observation] = []
    failures: list[FrameFailure] = []
    rejected: list[str] = []
    calls = 0

    # Encode first: reading a JPEG off disk is microseconds and must not be
    # done inside a worker where its failure would look like a provider one.
    encoded: list[tuple[Keyframe, str]] = []
    for frame in selected:
        try:
            encoded.append((frame, encode_frame(frame)))
        except OSError as exc:
            failures.append(FrameFailure(f"S{frame.index:03d}", frame.ts_ms, str(exc)))

    # Frames are independent, and each hosted call spends about half a minute
    # waiting on a socket. Run sequentially they dominated everything else:
    # 220s of a 228s run, one agent, 96% of the wall clock while nine others
    # finished in eight seconds between them.
    #
    # Safe to overlap because the vendor governor holds a real lock around
    # its token bucket, so the rate limit is enforced across threads rather
    # than per-thread — concurrency changes how long the run waits, never how
    # fast it calls. Kept deliberately small: the ceiling here is latency
    # hiding, not throughput, and the bucket is the actual limit.
    workers = min(VISION_WORKERS, max(1, len(encoded)))

    def describe(item: tuple[Keyframe, str]):
        frame, image_b64 = item
        return frame, registry.invoke(
            VISION_DESCRIBE, prompt=prompt, image_b64=image_b64, max_tokens=512
        )

    if workers == 1:
        results = [describe(item) for item in encoded]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # `map` preserves input order, so observations are assembled in
            # timeline order regardless of which frame answered first — the
            # run stays deterministic even though the calls do not.
            results = list(pool.map(describe, encoded))

    for frame, result in results:
        calls += 1
        if not result:
            failures.append(
                FrameFailure(f"S{frame.index:03d}", frame.ts_ms, result.reason)
            )
            continue

        parsed, dropped = parse_observations(result.value, frame, vocab)
        observations.extend(parsed)
        rejected.extend(dropped)

    tracks = build_tracks(observations)

    inspected = len(selected) - len(failures)
    coverage = inspected / len(keyframes) if keyframes else 0.0

    # Coverage and status answer different questions, and deriving one from
    # the other conflated them for every run this agent has ever done.
    #
    # Coverage is "how much of the video did vision actually see", and
    # gating means that is deliberately less than all of it — fusion scales
    # a vision claim by exactly this number, which is the honest thing to do
    # for a modality that sampled.
    #
    # Status is "did this agent do what it set out to do". Skipping a frame
    # no other modality pointed at is the cost optimisation working, not a
    # failure, and reporting it as DEGRADED meant a perfectly healthy vision
    # agent showed amber on every single run — which trains a reader to
    # ignore the one signal that is supposed to mean something is wrong.
    #
    # So: degraded when frames were *attempted and lost*, not when they were
    # never attempted by design.
    attempted = len(selected)
    status = "OK" if not failures else "DEGRADED"
    if inspected == 0:
        # Nothing was inspected, and the two reasons for that are not the same
        # thing. If every frame was refused for the same reason and none was
        # ever read, the capability is absent — running offline, or with no
        # key — and that is a SKIP with a reason. FAILED is for a provider
        # that was there and broke, and reporting an unavailable optional
        # capability in red reads to a judge as a broken tool rather than as
        # the honest degradation this pipeline is built around.
        reasons = {f.reason for f in failures}
        if failures and len(reasons) == 1:
            return (
                AgentResult.skipped(AGENT_ID, AGENT_NAME, failures[0].reason),
                [],
            )
        status = "FAILED"

    log = [
        f"{vocab.size} labels, {len(vocab.synonyms)} synonyms, "
        f"{len(vocab.blocklist)} blocked terms",
        f"gated {len(selected)}/{len(keyframes)} keyframes into {calls} call(s)",
        f"{len(observations)} observation(s) merged into {len(tracks)} track(s)",
    ]
    if rejected:
        log.append(f"rejected {len(rejected)} label(s): {', '.join(rejected[:4])}")
    if failures:
        log.append(f"{len(failures)}/{attempted} frame(s) unreadable or refused")

    return (
        AgentResult(
            agent_id=AGENT_ID,
            name=AGENT_NAME,
            status=status,
            coverage=round(coverage, 4),
            calls=calls,
            artifacts={
                "tracks": [t.to_json() for t in tracks],
                "failures": [f.to_json() for f in failures],
                "rejected_labels": rejected[:32],
            },
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            log=log,
        ),
        tracks,
    )


def to_json(tracks: Iterable[Track]) -> dict[str, Any]:
    """The A03 output contract. JSON only — never a paragraph."""
    return {"visual_evidence": [track.to_json() for track in tracks]}
