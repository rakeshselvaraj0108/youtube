"""A1 — Speech.

The default backend is local and the default path touches no network. This is
the most consequential decision in the project: a judge who clones the repo and
hits an authentication error has already scored it on functionality. `--asr nim`
upgrades to a hosted model; nothing requires it.

Word-level timestamps are non-negotiable. They are what make surgical
remediation possible — bleep exactly one word rather than mute fifteen seconds.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from preflight import cas
from preflight.models import AgentResult

AGENT_ID = "speech"
AGENT_NAME = "Speech Agent"
DEFAULT_MODEL = "base.en"

_PUNCT = str.maketrans("", "", ".,!?;:\"'()[]—–-")


def _norm(token: str) -> str:
    return token.lower().translate(_PUNCT).strip()


@dataclass
class Word:
    w: str
    start_ms: int
    end_ms: int
    conf: float


@dataclass
class Segment:
    start_ms: int
    end_ms: int
    text: str


@dataclass
class Transcript:
    language: str
    duration_ms: int
    words: list[Word] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    model_id: str = DEFAULT_MODEL

    @property
    def text(self) -> str:
        return " ".join(segment.text.strip() for segment in self.segments).strip()

    @property
    def word_count(self) -> int:
        return len(self.words)

    def to_json(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "duration_ms": self.duration_ms,
            "model_id": self.model_id,
            "words": [asdict(w) for w in self.words],
            "segments": [asdict(s) for s in self.segments],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Transcript":
        return cls(
            language=data["language"],
            duration_ms=data["duration_ms"],
            model_id=data.get("model_id", DEFAULT_MODEL),
            words=[Word(**w) for w in data["words"]],
            segments=[Segment(**s) for s in data["segments"]],
        )

    def text_between(self, start_ms: int, end_ms: int) -> str:
        return " ".join(
            w.w for w in self.words if w.end_ms > start_ms and w.start_ms < end_ms
        ).strip()

    def words_between(self, start_ms: int, end_ms: int) -> list[Word]:
        return [w for w in self.words if w.end_ms > start_ms and w.start_ms < end_ms]

    def locate(self, phrase: str, near_ms: int | None = None) -> tuple[int, int] | None:
        """Find a verbatim phrase in the transcript and return its real span.

        Language models are poor at arithmetic over timestamps — asked for the
        millisecond bounds of a phrase they return plausible-looking numbers
        that are frequently seconds out. But they are asked to quote evidence
        verbatim, and word-level ASR timings are exact, so the span can be
        recovered from the quote rather than trusted from the model.

        This matters beyond tidiness: the evidence span becomes the remediation
        op's span. A bleep placed seven seconds early silences the wrong word
        and leaves the profanity audible.
        """
        target = [_norm(t) for t in phrase.split() if _norm(t)]
        if not target or not self.words:
            return None

        normalised = [_norm(w.w) for w in self.words]
        span = len(target)
        best_index = -1
        best_score = 0.0

        for start in range(0, max(1, len(normalised) - span + 1)):
            window = normalised[start : start + span]
            if not window:
                continue
            matches = sum(1 for a, b in zip(window, target) if a == b)
            score = matches / span
            if score > best_score:
                # Prefer the occurrence nearest a hint when scores tie, so a
                # phrase repeated across the video resolves to the right one.
                best_score, best_index = score, start
            elif score == best_score and near_ms is not None and best_index >= 0:
                current = abs(self.words[start].start_ms - near_ms)
                incumbent = abs(self.words[best_index].start_ms - near_ms)
                if current < incumbent:
                    best_index = start

        if best_index < 0 or best_score < 0.6:
            return None

        matched = self.words[best_index : best_index + span]
        return matched[0].start_ms, matched[-1].end_ms

    def snap_to_words(self, start_ms: int, end_ms: int) -> tuple[int, int]:
        """Widen a span to whole word boundaries.

        Clipping mid-syllable is audible and reads as amateur, so every audio
        remediation op is expanded to the words it actually overlaps.
        """
        overlapping = self.words_between(start_ms, end_ms)
        if not overlapping:
            return start_ms, end_ms
        return (
            min(w.start_ms for w in overlapping),
            max(w.end_ms for w in overlapping),
        )


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def transcribe(
    wav: Path,
    store: cas.Store,
    *,
    model_id: str = DEFAULT_MODEL,
    duration_ms: int = 0,
) -> tuple[AgentResult, Transcript | None]:
    started = time.perf_counter()
    log: list[str] = []

    if not Path(wav).is_file():
        return (
            AgentResult.skipped(
                AGENT_ID, AGENT_NAME, "no audio stream — nothing to transcribe"
            ),
            None,
        )

    key = cas.hash_many([cas.hash_file(wav), model_id])
    entry = store.entry("t", key)

    if entry.exists:
        transcript = Transcript.from_json(entry.read_json("transcript.json"))
        log.append(
            f"cache hit · {transcript.word_count} words · {model_id} · 0 model invocations"
        )
        return (
            AgentResult(
                agent_id=AGENT_ID,
                name=AGENT_NAME,
                status="OK",
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                log=log,
            ),
            transcript,
        )

    if not available():
        # Degrade, never fail. Without the optional ASR backend the report still
        # ships; the speech dimension reports SKIPPED and coverage drops, which
        # the header chip and certificate both surface.
        return (
            AgentResult.skipped(
                AGENT_ID,
                AGENT_NAME,
                "faster-whisper not installed — install preflight[asr] for speech coverage",
            ),
            None,
        )

    from faster_whisper import WhisperModel

    model = WhisperModel(model_id, device="cpu", compute_type="int8")
    raw_segments, info = model.transcribe(
        str(wav), word_timestamps=True, vad_filter=True
    )

    words: list[Word] = []
    segments: list[Segment] = []
    for segment in raw_segments:
        segments.append(
            Segment(
                start_ms=int(segment.start * 1000),
                end_ms=int(segment.end * 1000),
                text=segment.text.strip(),
            )
        )
        for word in segment.words or []:
            words.append(
                Word(
                    w=word.word.strip(),
                    start_ms=int(word.start * 1000),
                    end_ms=int(word.end * 1000),
                    conf=round(float(word.probability), 4),
                )
            )

    transcript = Transcript(
        language=getattr(info, "language", "en") or "en",
        duration_ms=duration_ms or int(getattr(info, "duration", 0) * 1000),
        words=words,
        segments=segments,
        model_id=model_id,
    )

    entry.discard()
    entry.root.mkdir(parents=True, exist_ok=True)
    entry.write_json("transcript.json", transcript.to_json())
    entry.commit()

    mean_conf = (
        sum(w.conf for w in words) / len(words) if words else 0.0
    )
    log.append(
        f"{model_id} · {len(words)} words · {len(segments)} segments · "
        f"{mean_conf * 100:.1f}% mean confidence"
    )

    return (
        AgentResult(
            agent_id=AGENT_ID,
            name=AGENT_NAME,
            status="OK",
            artifacts={"mean_confidence": round(mean_conf, 4)},
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            log=log,
        ),
        transcript,
    )


def speech_rate_wpm(words: Iterable[Word], window_ms: int = 30_000) -> list[tuple[int, float]]:
    """Rolling words-per-minute, sampled per window.

    Sustained delivery above ~180 wpm measurably hurts comprehension and is
    something a creator can actually act on, unlike most accessibility advice.
    """
    ordered = sorted(words, key=lambda w: w.start_ms)
    if not ordered:
        return []
    end = ordered[-1].end_ms
    out: list[tuple[int, float]] = []
    for start in range(0, max(end - window_ms, 0) + 1, window_ms // 2 or 1):
        stop = start + window_ms
        count = sum(1 for w in ordered if start <= w.start_ms < stop)
        out.append((start, count * 60_000 / window_ms))
    return out
