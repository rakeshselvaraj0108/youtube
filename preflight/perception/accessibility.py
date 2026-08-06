"""A5 — Accessibility.

The headline check is photosensitive flash detection. Roughly twenty lines of
numpy, genuine safety value, and no mainstream creator tool does it. A sequence
flashing three or more times per second can trigger seizures; a creator has no
way to know their montage does that until someone tells them.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from preflight.models import Adversarial, AgentResult, Evidence, Finding, PolicyRef
from preflight.perception import signal as sig
from preflight.perception.asr import Transcript, speech_rate_wpm

AGENT_ID = "access"
AGENT_NAME = "Accessibility Agent"

# Nyquist. Sampling a strobe at the strobe's own rate lands on the same phase
# every time and reports a flat luminance series — a 10Hz strobe sampled at
# 10fps measures as ZERO flashes, which is the most dangerous possible failure
# for this particular check. Caught by clip g010 in the synthetic corpus, which
# constructs exactly that case.
#
# 30fps resolves flashes up to ~15Hz, comfortably past the range that matters.
# The cost is a greyscale 64x36 frame every 33ms, which is nothing.
SAMPLE_FPS = 30
FLASH_DELTA = 0.10 * 255  # 10% of full luminance range between adjacent samples
FLASH_HIGH = 3            # flashes per second — the widely used seizure threshold
FLASH_MODERATE = 2
WPM_LIMIT = 180.0
CHAPTERS_REQUIRED_ABOVE_MS = 8 * 60 * 1000
LONG_RUN_MS = 5 * 60 * 1000


def _clause(clause_id: str, title: str, section: str, text: str) -> PolicyRef:
    return PolicyRef(clauseId=clause_id, title=title, section=section, text=text)


CAPTION_SUFFIXES = (".vtt", ".srt")


def find_caption_track(source: Path) -> Path | None:
    """A sidecar caption file next to the video.

    `preflight fix` writes one of these from the transcript, so a re-check
    genuinely sees the repair rather than being told about it.
    """
    source = Path(source)
    for suffix in CAPTION_SUFFIXES:
        candidate = source.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def analyse(
    source: Path,
    duration_ms: int,
    transcript: Transcript | None,
    *,
    has_captions: bool | None = None,
    has_chapters: bool = False,
) -> AgentResult:
    started = time.perf_counter()
    log: list[str] = []
    findings: list[Finding] = []
    coverage = 1.0

    if has_captions is None:
        track = find_caption_track(source)
        has_captions = track is not None
        if track is not None:
            log.append(f"caption track found: {track.name}")

    flash, flash_log = _flash_analysis(source)
    log.extend(flash_log)
    if flash is None:
        coverage -= 0.4
    elif flash["risk"] != "LOW":
        findings.append(_flash_finding(flash))

    if not has_captions:
        findings.append(_caption_finding(duration_ms, transcript))

    if transcript is not None:
        findings.extend(_speech_rate_findings(transcript))
    else:
        coverage -= 0.3
        log.append("no transcript — speech-rate and caption-quality checks skipped")

    if duration_ms > CHAPTERS_REQUIRED_ABOVE_MS and not has_chapters:
        findings.append(_chapter_finding(duration_ms))

    status = "OK" if coverage >= 0.999 else "DEGRADED"
    return AgentResult(
        agent_id=AGENT_ID,
        name=AGENT_NAME,
        status=status,
        findings=findings,
        artifacts={"flash": flash} if flash else {},
        coverage=max(0.0, coverage),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        log=log,
    )


def flash_risk(luminance: np.ndarray, fps: int = SAMPLE_FPS) -> dict[str, object]:
    """Peak flashes per second across the file.

    A **flash** is a light-dark-light cycle, which is what the seizure guidance
    counts. It is not a single luminance transition: counting every delta
    double-counts, because each cycle produces one rise and one fall.

    So only rising edges are counted. A 1Hz strobe then measures 1 flash/s
    rather than 2, and a 10Hz strobe measures 10.

    Getting this wrong is not academic. Counting transitions put a clean 1Hz
    strobe at exactly 3/s — the harm threshold — purely because the window
    happened to span two rises and one fall, so the detector condemned footage
    that is demonstrably safe. Corpus clip g011 exists to pin this.
    """
    if luminance.size < 2:
        return {"max_flashes_per_second": 0, "risk": "LOW", "worst_ts_ms": None}

    delta = np.diff(luminance)
    flashes = (delta > FLASH_DELTA).astype(np.int32)

    window = max(int(fps), 1)
    if flashes.size < window:
        peak, at = int(flashes.sum()), 0
    else:
        counts = np.convolve(flashes, np.ones(window, dtype=np.int32), mode="valid")
        peak, at = int(counts.max()), int(counts.argmax())

    risk = "HIGH" if peak >= FLASH_HIGH else "MODERATE" if peak >= FLASH_MODERATE else "LOW"
    return {
        "max_flashes_per_second": peak,
        "risk": risk,
        "worst_ts_ms": int(at / fps * 1000),
    }


def _flash_analysis(source: Path) -> tuple[dict[str, object] | None, list[str]]:
    try:
        luminance = sig.luminance_series(source, fps=SAMPLE_FPS)
    except Exception as exc:  # noqa: BLE001 - degrade, never fail
        return None, [f"flash analysis unavailable: {exc}"]

    if luminance.size == 0:
        return None, ["flash analysis produced no frames"]

    result = flash_risk(luminance, SAMPLE_FPS)
    return result, [
        f"photosensitivity: peak {result['max_flashes_per_second']} flashes/s "
        f"({result['risk']}) over {luminance.size} samples at {SAMPLE_FPS}fps"
    ]


def _flash_finding(flash: dict[str, object]) -> Finding:
    peak = int(flash["max_flashes_per_second"])  # type: ignore[arg-type]
    worst = int(flash["worst_ts_ms"] or 0)  # type: ignore[arg-type]
    high = flash["risk"] == "HIGH"
    return Finding(
        id="x_flash",
        clauseId="ACC-01",
        category="Accessibility",
        title=f"Photosensitive flash risk — {peak} flashes/s",
        description=(
            "Rapid luminance changes in this range can trigger seizures in "
            "photosensitive viewers. Add a warning card or reduce the flash rate."
        ),
        startMs=max(0, worst - 1000),
        endMs=worst + 1000,
        severity="HIGH" if high else "MEDIUM",
        confidence=0.88 if high else 0.7,
        modalities={"vision": 0.88 if high else 0.7},
        evidence=Evidence(
            transcript=f"[peak {peak} luminance transitions per second at "
            f"{worst // 1000}s, sampled at {SAMPLE_FPS}fps]"
        ),
        policy=_clause(
            "ACC-01",
            "Photosensitive content",
            "PREFLIGHT accessibility ruleset § 1.4",
            "Content flashing more than three times per second, or containing rapid "
            "transitions to and from saturated red, presents a seizure risk to "
            "photosensitive viewers and should carry a warning.",
        ),
        adversarial=Adversarial(
            charge=f"Peak of {peak} luminance transitions exceeding 10% of full range "
            f"within a one-second window, at {worst // 1000}s.",
            rationale="Measured from a 10fps luminance series across the whole file. "
            "Scene-cut frames would have missed this entirely — a strobe lives "
            "between cuts.",
            confidence=0.88 if high else 0.7,
            defense=None if high else "Two flashes per second sits below the widely "
            "cited three-per-second threshold.",
            defense_strength=0.0 if high else 0.5,
        ),
    )


def _caption_finding(duration_ms: int, transcript: Transcript | None) -> Finding:
    have_words = transcript is not None and transcript.word_count > 0
    return Finding(
        id="x_captions",
        clauseId="ACC-02",
        category="Accessibility",
        title="No caption track present",
        description=(
            "Word-level timings already exist in this run — captions can be emitted "
            "directly from them."
            if have_words
            else "No caption track and no transcript available to generate one."
        ),
        startMs=0,
        endMs=duration_ms,
        severity="HIGH",
        confidence=0.99,
        modalities={"access": 0.99},
        evidence=Evidence(transcript="[file-scoped · no timed-text stream in container]"),
        policy=_clause(
            "ACC-02",
            "Caption availability",
            "PREFLIGHT accessibility ruleset § 1.1",
            "A published video should ship with a caption track. Automatic captions are "
            "not a substitute where the audio contains technical vocabulary, accents, or "
            "background noise.",
        ),
        adversarial=Adversarial(
            charge="No sidecar caption track and no embedded timed-text stream.",
            rationale="Deterministic check, not a judgement call."
            + (
                " The speech agent already produced word-level timings, so the "
                "remediation cost is effectively zero."
                if have_words
                else ""
            ),
            confidence=0.99,
        ),
    )


def _speech_rate_findings(transcript: Transcript) -> list[Finding]:
    samples = speech_rate_wpm(transcript.words)
    if not samples:
        return []

    hot = [(start, wpm) for start, wpm in samples if wpm > WPM_LIMIT]
    if not hot:
        return []

    worst_start, worst_wpm = max(hot, key=lambda item: item[1])
    return [
        Finding(
            id="x_wpm",
            clauseId="ACC-03",
            category="Accessibility",
            title=f"Speech rate peaks at {worst_wpm:.0f} wpm",
            description=f"{len(hot)} window(s) above {WPM_LIMIT:.0f} wpm. Sustained fast "
            "delivery measurably reduces comprehension and caption readability.",
            startMs=worst_start,
            endMs=worst_start + 30_000,
            severity="LOW",
            confidence=0.8,
            modalities={"speech": 0.8},
            evidence=Evidence(
                transcript=transcript.text_between(worst_start, worst_start + 30_000)[:400]
                or "[no words in window]"
            ),
            policy=_clause(
                "ACC-03",
                "Speech rate",
                "PREFLIGHT accessibility ruleset § 1.2",
                "Delivery sustained above roughly 180 words per minute reduces "
                "comprehension for non-native speakers and makes captions hard to follow.",
            ),
            adversarial=Adversarial(
                charge=f"Rolling 30s window peaks at {worst_wpm:.0f} wpm.",
                rationale="Measured from word-level ASR timings.",
                confidence=0.8,
                defense="Fast delivery can be a deliberate stylistic choice and is not a "
                "policy violation.",
                defense_strength=0.6,
            ),
        )
    ]


def _chapter_finding(duration_ms: int) -> Finding:
    minutes = duration_ms / 60_000
    return Finding(
        id="x_chapters",
        clauseId="ACC-04",
        category="Accessibility",
        title="No chapter markers",
        description=f"{minutes:.0f} minutes with no chapters. Viewers cannot navigate and "
        "retention on Suggested surfaces suffers.",
        startMs=0,
        endMs=duration_ms,
        severity="LOW",
        confidence=0.99,
        modalities={"meta": 0.99},
        evidence=Evidence(transcript="[file-scoped · no chapter markers in container]"),
        policy=_clause(
            "ACC-04",
            "Chapter markers",
            "PREFLIGHT accessibility ruleset § 1.3",
            "Videos over roughly eight minutes without chapter markers lose navigability "
            "and retention.",
        ),
        adversarial=Adversarial(
            charge=f"Runtime {minutes:.0f} minutes, zero chapter markers found.",
            rationale="Deterministic container check.",
            confidence=0.99,
        ),
    )
