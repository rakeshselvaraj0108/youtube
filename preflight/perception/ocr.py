"""A05 — OCR.

Text burned into the picture: captions, memes, chyrons, watermarks. The
extraction is the easy half and tesseract does it. The half that matters is
what happens to the extraction afterwards.

TWO PROBLEMS, both of which quietly wreck a naive implementation.

The first is counting. A lower-third caption persists across forty keyframes.
Reported per frame, that is forty findings for one caption, and every count
downstream — finding totals, clause frequencies, the risk score itself —
inflates by the frame rate of the sampler rather than by anything about the
video. Text is therefore clustered into one item spanning first sighting to
last, by string similarity AND box overlap together: the same words in a
different corner is a different element, and similar words in the same corner
across consecutive frames is one element flickering through OCR noise.

The second is that on-screen text is not one kind of thing. A burned-in
caption duplicating the narration, a persistent corner watermark, and meme
text across the top of the frame are three different signals that happen to
share an extraction method. The caption is a duplicate of something A02
already found and should not be counted twice. The watermark is a third-party
footage cue, which points at COPY-01 and not at the language clauses. The meme
text is where profanity actually lives. Handing all three to the adjudicator
as "on-screen text" throws away the only information that tells them apart —
so each cluster is classified by position, persistence and whether it tracks
the transcript, and the role travels with it.

What this module does NOT do is decide anything. A role is a routing hint. The
triad rules, using the clause text.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from preflight.ingest.frames import Keyframe
from preflight.models import AgentResult

AGENT_ID = "ocr"
AGENT_NAME = "OCR Agent"
ROLE_PATTERNS = Path("data/lexicons/ocr_role_patterns.json")

# Clustering. Similarity is on normalised text; IoU is on the box. Both must
# agree, because either alone produces a wrong merge: the same caption template
# with different words occupies the same box, and a repeated slogan moves.
SIMILARITY_THRESHOLD = 0.85
IOU_THRESHOLD = 0.60

# The proportional threshold alone splits short captions over a couple of
# misread characters. See `similar_enough`.
ABSOLUTE_EDIT_ALLOWANCE = 2
ABSOLUTE_ALLOWANCE_MIN_LEN = 8

# A cluster whose sightings are minutes apart is two appearances of the same
# text, not one persistent element. Watermarks are exempt — a mark in the
# corner of every shot IS one element, and that is what the ratio measures.
CLUSTER_GAP_MS = 5_000

# Concurrent tesseract subprocesses. Each spends its time in native code with
# the GIL released, so this is close to linear; kept modest so a forensic run
# does not saturate every core on the machine it is sharing.
OCR_WORKERS = 4

# Below this, tesseract is guessing at the pixels. Retained in artifacts so a
# reader can see what was read and rejected, never promoted to a finding.
MIN_FINDING_CONFIDENCE = 0.55

# All-round low confidence means the OCR did not work on this footage —
# stylised type, heavy compression, text over motion. Better to say so than to
# emit a page of garbage at 0.2.
DEGRADED_CONFIDENCE = 0.30

CORNER_MARGIN = 0.25

ROLES = (
    "watermark",
    "burned_in_caption",
    "chyron",
    "lower_third",
    "meme_text",
    "unclassified",
)


# ------------------------------------------------------------------ #
# Geometry and text                                                   #
# ------------------------------------------------------------------ #


Box = tuple[float, float, float, float]  # x, y, w, h — all normalised


def box_iou(a: Box, b: Box) -> float:
    """Intersection over union of two normalised boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    union = aw * ah + bw * bh - overlap
    return overlap / union if union > 0 else 0.0


def merge_boxes(boxes: Iterable[Box]) -> Box:
    boxes = list(boxes)
    if not boxes:
        return (0.0, 0.0, 0.0, 0.0)
    left = min(b[0] for b in boxes)
    top = min(b[1] for b in boxes)
    right = max(b[0] + b[2] for b in boxes)
    bottom = max(b[1] + b[3] for b in boxes)
    return (left, top, right - left, bottom - top)


def normalise(text: str) -> str:
    """Casefold and collapse whitespace, nothing else.

    Deliberately not spell-correcting or stripping punctuation. Two sightings
    of the same caption differ by OCR noise, and the similarity metric is what
    absorbs that. Cleaning the text here would also clean it in the evidence,
    and the evidence has to show what the model actually read.
    """
    return " ".join(text.split()).casefold()


def edit_distance(left: str, right: str) -> int:
    """Levenshtein distance.

    Written out rather than imported. `rapidfuzz` would be a dependency for
    one function on strings that are rarely longer than a caption line, and
    this project's rule is that a full report must be producible with numpy
    and the standard library alone.
    """
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, lchar in enumerate(left, start=1):
        current = [i]
        for j, rchar in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (lchar != rchar),  # substitution
                )
            )
        previous = current
    return previous[-1]


def similarity(left: str, right: str) -> float:
    """Normalised Levenshtein similarity, 0 to 1."""
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    return 1.0 - edit_distance(left, right) / max(len(left), len(right))


def similar_enough(
    left: str, right: str, threshold: float = SIMILARITY_THRESHOLD
) -> bool:
    """Are these two readings of the same text?

    A purely proportional threshold gets the short end wrong. OCR noise is
    per-character, so two bad characters is two bad characters whether the
    line is twelve characters or sixty — but at 0.85, twelve characters
    tolerates 1.8 edits and sixty tolerates 9. Measured on a real pair:
    "subscribe now" against "subscnbe now" is two edits and scores 0.846,
    which splits one caption into two items over a dropped `i` and an `r`
    read as an `n`.

    So a small absolute allowance sits alongside the ratio. It applies only to
    strings long enough for a couple of characters not to change the word —
    at four characters, two edits turns `cat.` into `dog.` — and clustering
    additionally requires the boxes to overlap and the sightings to be close
    in time, so a wrong merge needs all three to agree.
    """
    if similarity(left, right) >= threshold:
        return True
    if min(len(left), len(right)) >= ABSOLUTE_ALLOWANCE_MIN_LEN:
        return edit_distance(left, right) <= ABSOLUTE_EDIT_ALLOWANCE
    return False


# ------------------------------------------------------------------ #
# Data model                                                          #
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class TextSighting:
    """One line of text in one frame. Never a finding."""

    text: str
    box: Box
    conf: float
    ts_ms: int

    def to_json(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "box": [round(v, 4) for v in self.box],
            "conf": round(self.conf, 3),
            "ts_ms": self.ts_ms,
        }


@dataclass
class TextItem:
    """One on-screen element, across however many frames showed it."""

    id: str
    text: str
    box: Box
    start_ms: int
    end_ms: int
    frames: int
    conf: float
    role: str = "unclassified"
    tracks_speech: bool = False

    @property
    def persist_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    @property
    def reportable(self) -> bool:
        """Confident enough to become a finding on its own."""
        return self.conf >= MIN_FINDING_CONFIDENCE

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "box": [round(v, 4) for v in self.box],
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "persist_ms": self.persist_ms,
            "frames": self.frames,
            "conf": round(self.conf, 3),
            "role": self.role,
            "tracks_speech": self.tracks_speech,
            "reportable": self.reportable,
        }


# ------------------------------------------------------------------ #
# Role rules                                                          #
# ------------------------------------------------------------------ #


class RoleRules:
    """Position and persistence thresholds, loaded from the lexicon.

    In the data file rather than in code because the numbers are editorial —
    where a lower third sits is a convention, not a law of nature — and a
    reviewer should be able to see and change them without reading Python.
    """

    def __init__(self, path: Path = ROLE_PATTERNS) -> None:
        self.path = Path(path)
        self.roles: dict[str, dict[str, Any]] = {}
        if self.path.is_file():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.roles = payload.get("roles", {})

    def get(self, role: str) -> dict[str, Any]:
        return self.roles.get(role, {})

    @property
    def loaded(self) -> bool:
        return bool(self.roles)


def is_corner(box: Box) -> bool:
    x, y, w, h = box
    near_left = x < CORNER_MARGIN
    near_right = (x + w) > 1.0 - CORNER_MARGIN
    near_top = y < CORNER_MARGIN
    near_bottom = (y + h) > 1.0 - CORNER_MARGIN
    return (near_left or near_right) and (near_top or near_bottom)


def tracks_speech(item: TextItem, transcript: Any) -> bool:
    """Does this text repeat what is being said while it is on screen?

    A burned-in caption is a duplicate of the audio. Detecting that lets the
    fusion layer avoid counting one utterance twice, once from A02 and once
    from here — which would otherwise look like cross-modal corroboration and
    promote the severity of a finding that has only one real source.
    """
    if not transcript:
        return False

    spoken = _spoken_between(transcript, item.start_ms, item.end_ms)
    if not spoken:
        return False

    words = set(normalise(item.text).split())
    if not words:
        return False
    overlap = words & set(normalise(spoken).split())
    return len(overlap) / len(words) >= 0.6


def _spoken_between(transcript: Any, start_ms: int, end_ms: int) -> str:
    """Transcript text overlapping a span, from either shape A02 emits."""
    segments = transcript
    if isinstance(transcript, dict):
        segments = transcript.get("segments") or transcript.get("words") or []

    parts: list[str] = []
    for segment in segments or []:
        if not isinstance(segment, dict):
            continue
        seg_start = int(segment.get("start_ms", segment.get("startMs", 0)))
        seg_end = int(segment.get("end_ms", segment.get("endMs", seg_start)))
        if seg_end < start_ms or seg_start > end_ms:
            continue
        parts.append(str(segment.get("text", segment.get("w", ""))))
    return " ".join(parts)


def classify_role(
    item: TextItem,
    duration_ms: int,
    transcript: Any = None,
    rules: RoleRules | None = None,
) -> str:
    """Assign a role from position, persistence and speech correlation.

    Ordered most specific first. A watermark is tested before anything else
    because the corner-plus-persistence signature is unambiguous and its
    downstream consequence — a third-party footage cue pointing at COPY-01
    rather than at a language clause — is the one that would be most wrong to
    miss.
    """
    rules = rules or RoleRules()
    x, y, w, h = item.box
    ratio = item.persist_ms / max(duration_ms, 1)

    watermark = rules.get("watermark")
    if is_corner(item.box) and ratio >= watermark.get("persist_ratio", 0.6):
        return "watermark"

    caption = rules.get("burned_in_caption")
    low, high = caption.get("y_range", [0.72, 1.0])
    if low <= y <= high and item.tracks_speech:
        return "burned_in_caption"

    chyron = rules.get("chyron")
    c_low, c_high = chyron.get("y_range", [0.8, 1.0])
    if c_low <= y <= c_high and w >= 0.8:
        return "chyron"

    third = rules.get("lower_third")
    t_low, t_high = third.get("y_range", [0.62, 0.92])
    if (
        t_low <= y <= t_high
        and item.persist_ms >= third.get("min_persist_ms", 2000)
        and len(item.text.split()) <= third.get("max_words", 8)
    ):
        return "lower_third"

    meme = rules.get("meme_text")
    m_low, m_high = meme.get("y_range", [0.0, 0.35])
    if (
        m_low <= y <= m_high
        and h >= meme.get("font_height_ratio", 0.06)
        and item.persist_ms <= meme.get("max_persist_ms", 4000)
    ):
        return "meme_text"

    return "unclassified"


# ------------------------------------------------------------------ #
# Clustering                                                          #
# ------------------------------------------------------------------ #


def group_lines(words: list[dict[str, Any]], ts_ms: int) -> list[TextSighting]:
    """Words into lines, using tesseract's own block and line numbering.

    Per-word sightings would fragment a caption into six clusters that each
    look like a separate element. Tesseract already knows which words share a
    line; using that is more reliable than inferring it from box geometry.
    """
    lines: dict[Any, list[dict[str, Any]]] = {}
    for index, word in enumerate(words):
        key = word.get("line", index)
        lines.setdefault(key, []).append(word)

    sightings: list[TextSighting] = []
    for key in sorted(lines, key=lambda k: (str(k))):
        group = lines[key]
        text = " ".join(str(w["text"]) for w in group).strip()
        if not text:
            continue
        confidences = [float(w.get("conf", 0.0)) for w in group]
        sightings.append(
            TextSighting(
                text=text,
                box=merge_boxes(tuple(w["box"]) for w in group),
                # The weakest word in a line caps the line. A caption is only
                # as trustworthy as the word most likely to be misread, and
                # averaging lets one confident article carry five guesses.
                conf=min(confidences) if confidences else 0.0,
                ts_ms=ts_ms,
            )
        )
    return sightings


def cluster(
    sightings: list[TextSighting],
    *,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD,
    gap_ms: int = CLUSTER_GAP_MS,
) -> list[TextItem]:
    """Temporal dedup. Forty frames of one caption become one item.

    Matching requires text similarity AND box overlap AND temporal proximity.
    Each guard removes a specific wrong merge: without similarity, every
    caption in the same template merges into one; without IoU, a slogan
    repeated in a different corner merges with its first appearance; without
    the gap, a title card at 0:03 and the same title card at 14:00 become a
    single fourteen-minute element.
    """
    clusters: list[dict[str, Any]] = []

    for sighting in sorted(sightings, key=lambda s: (s.ts_ms, s.text)):
        key = normalise(sighting.text)
        for existing in clusters:
            if sighting.ts_ms - existing["end_ms"] > gap_ms:
                continue
            if not similar_enough(existing["key"], key, similarity_threshold):
                continue
            if box_iou(existing["box"], sighting.box) < iou_threshold:
                continue
            existing["end_ms"] = max(existing["end_ms"], sighting.ts_ms)
            existing["frames"] += 1
            existing["boxes"].append(sighting.box)
            existing["box"] = merge_boxes(existing["boxes"])
            # Best sighting wins: OCR failures are noise on one frame, and a
            # word read cleanly once was there the whole time.
            if sighting.conf > existing["conf"]:
                existing["conf"] = sighting.conf
                existing["text"] = sighting.text
                existing["key"] = key
            break
        else:
            clusters.append(
                {
                    "key": key,
                    "text": sighting.text,
                    "box": sighting.box,
                    "boxes": [sighting.box],
                    "start_ms": sighting.ts_ms,
                    "end_ms": sighting.ts_ms,
                    "frames": 1,
                    "conf": sighting.conf,
                }
            )

    items: list[TextItem] = []
    for index, entry in enumerate(sorted(clusters, key=lambda c: c["start_ms"])):
        items.append(
            TextItem(
                id=f"ocr_{index:03d}",
                text=entry["text"],
                box=entry["box"],
                start_ms=entry["start_ms"],
                end_ms=entry["end_ms"],
                frames=entry["frames"],
                conf=entry["conf"],
            )
        )
    return items


# ------------------------------------------------------------------ #
# Agent entry point                                                   #
# ------------------------------------------------------------------ #


@dataclass
class OcrReport:
    items: list[TextItem] = field(default_factory=list)
    raw_count: int = 0
    frames_read: int = 0
    frames_failed: int = 0

    @property
    def has_burned_in_captions(self) -> bool:
        return any(i.role == "burned_in_caption" for i in self.items)

    def caption_coverage_ratio(self, duration_ms: int) -> float:
        """Fraction of the runtime carrying a burned-in caption.

        Feeds the accessibility dimension. A video with burned-in captions
        across most of its runtime is not the same accessibility case as one
        with none, and scoring it as though it were would be wrong in the
        direction that penalises a creator who did the work.
        """
        if duration_ms <= 0:
            return 0.0
        covered = sum(
            i.persist_ms for i in self.items if i.role == "burned_in_caption"
        )
        return min(1.0, covered / duration_ms)

    def to_json(self, duration_ms: int = 0) -> dict[str, Any]:
        return {
            "items": [i.to_json() for i in self.items],
            "raw_count": self.raw_count,
            "deduped_count": len(self.items),
            "has_burned_in_captions": self.has_burned_in_captions,
            "caption_coverage_ratio": round(
                self.caption_coverage_ratio(duration_ms), 3
            ),
            "roles": {
                role: sum(1 for i in self.items if i.role == role)
                for role in ROLES
                if any(i.role == role for i in self.items)
            },
        }


def analyse(
    keyframes: list[Keyframe],
    registry: Any = None,
    *,
    duration_ms: int = 0,
    transcript: Any = None,
    rules: RoleRules | None = None,
    budget: int | None = None,
) -> tuple[AgentResult, OcrReport]:
    """Read the picture, degrading rather than failing.

    Optional by design. Without tesseract this reports SKIPPED with coverage
    zero and the run continues — a report that omits on-screen text and says
    so is more useful than no report.
    """
    # Local, matching vision.py: importing the registry at module scope would
    # close an import cycle back through the provider package.
    from preflight.providers.registry import OCR_IMAGE

    started = time.perf_counter()
    rules = rules or RoleRules()
    report = OcrReport()

    if not keyframes:
        return (
            AgentResult.skipped(AGENT_ID, AGENT_NAME, "no keyframes extracted"),
            report,
        )
    if registry is None:
        return (
            AgentResult.skipped(AGENT_ID, AGENT_NAME, "no ocr.image provider"),
            report,
        )

    # Spread across the timeline, never the first N. A head slice is how a
    # budgeted run ends up having read only the opening of the video and
    # reporting the remainder as clean — the same bug adaptive sampling
    # already had to fix once. Evenly spaced frames keep coverage
    # proportional across every minute of the runtime.
    if budget and budget < len(keyframes):
        step = len(keyframes) / budget
        frames = [keyframes[int(i * step)] for i in range(budget)]
    else:
        frames = keyframes
    sightings: list[TextSighting] = []
    log: list[str] = []
    calls = 0
    unavailable_reason = ""

    # Tesseract is a subprocess per frame, so the GIL is released while it
    # runs and overlapping them is close to linear speedup. This matters at
    # length: a fourteen-minute video now yields several hundred frames at
    # half a second each, which is minutes of wall clock serially and well
    # under one overlapped. `map` preserves input order, so sightings stay in
    # timeline order regardless of which frame finishes first.
    workers = min(OCR_WORKERS, max(1, len(frames)))
    if workers == 1:
        results = [(f, registry.invoke(OCR_IMAGE, image=f.path)) for f in frames]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(
                pool.map(
                    lambda f: (f, registry.invoke(OCR_IMAGE, image=f.path)), frames
                )
            )

    for frame, result in results:
        if not getattr(result, "ok", False):
            reason = getattr(result, "reason", "provider unavailable")
            if not report.frames_read:
                # Nothing has succeeded yet, so this is a missing capability
                # rather than one bad frame.
                unavailable_reason = reason
                break
            report.frames_failed += 1
            continue

        calls += getattr(result, "calls", 0) or 0
        payload = result.value
        words = payload.get("words", []) if isinstance(payload, dict) else []
        report.frames_read += 1
        sightings.extend(group_lines(words, frame.ts_ms))

    if unavailable_reason:
        return (
            AgentResult.skipped(AGENT_ID, AGENT_NAME, unavailable_reason),
            report,
        )

    report.raw_count = len(sightings)
    report.items = cluster(sightings)

    for item in report.items:
        item.tracks_speech = tracks_speech(item, transcript)
        item.role = classify_role(item, duration_ms, transcript, rules)

    inspected = report.frames_read
    coverage = inspected / len(keyframes) if keyframes else 0.0

    status = "OK"
    if report.items and all(i.conf < DEGRADED_CONFIDENCE for i in report.items):
        # Not a failure and not a clean read. Stylised type over motion does
        # this, and reporting it as OK would present guesses as text.
        status = "DEGRADED"
        coverage = min(coverage, 0.5)
        log.append("low OCR confidence throughout — text may be misread")
    elif report.frames_failed:
        status = "DEGRADED"

    log.insert(
        0,
        f"{report.raw_count} text element(s) → {len(report.items)} after temporal dedup",
    )
    if report.items:
        counts = ", ".join(
            f"{n} {role}"
            for role, n in sorted(
                ((r, sum(1 for i in report.items if i.role == r)) for r in ROLES),
                key=lambda pair: -pair[1],
            )
            if n
        )
        log.append(f"roles: {counts}")
    if report.frames_failed:
        log.append(f"{report.frames_failed} frame(s) unreadable")
    if not rules.loaded:
        log.append("role patterns missing — every item is unclassified")

    return (
        AgentResult(
            agent_id=AGENT_ID,
            name=AGENT_NAME,
            status=status,
            coverage=round(coverage, 3),
            artifacts=report.to_json(duration_ms),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            calls=calls,
            log=log,
        ),
        report,
    )
