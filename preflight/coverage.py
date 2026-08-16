"""Where on the timeline each modality actually looked.

Coverage has always been a single number per agent — "vision 0.62" — and
that number answers the wrong question for a long video. An agent can
examine 62% of a fourteen-minute upload while never once looking at minutes
nine through twelve, and the scalar cannot tell you so. Every downstream
claim then inherits the ambiguity: "no secrets found" might mean the video
is clean, or it might mean nothing looked where the secret was.

So this projects evidence back onto the timeline and reports, band by band,
what was examined and what was not. The rule it exists to enforce is the one
the rest of the engine already follows on the confidence axis, applied to
the time axis instead:

    A section nothing examined is UNEXAMINED, never clean.

That distinction is the whole point. `UNEXAMINED` is not a smaller kind of
safe — it is an absence of information, and a reader deciding whether to
publish needs to see it as a hole rather than as a pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

# One band per minute. Matches how a reader reasons about a long video ("what
# happens around eight minutes in") and keeps the report readable — a
# fourteen-minute upload becomes fourteen rows, not eight hundred.
DEFAULT_BAND_MS = 60_000

# Below this share of a band's expected samples, the band was touched but not
# meaningfully covered. Deliberately not zero: one frame in a sixty-second
# window is not coverage of that window, and calling it so is how a thin pass
# launders itself into a clean bill of health.
THIN_BAND_RATIO = 0.5

# What a fully covered band looks like, in samples per band, per modality.
# Frame-driven modalities are compared against the frames that actually landed
# in the band; this is the floor below which a band cannot be called examined
# at all.
MIN_SAMPLES_FOR_EXAMINED = 1

State = str  # EXAMINED | THIN | UNEXAMINED


@dataclass(frozen=True)
class Band:
    """One slice of the timeline, and what looked at it."""

    index: int
    start_ms: int
    end_ms: int
    samples: dict[str, int] = field(default_factory=dict)
    states: dict[str, State] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.start_ms // 60_000:02d}–{self.end_ms // 60_000:02d}"

    def state_of(self, modality: str) -> State:
        return self.states.get(modality, "UNEXAMINED")

    @property
    def examined_modalities(self) -> list[str]:
        return sorted(m for m, s in self.states.items() if s == "EXAMINED")

    @property
    def unexamined_modalities(self) -> list[str]:
        return sorted(m for m, s in self.states.items() if s == "UNEXAMINED")

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "samples": dict(self.samples),
            "states": dict(self.states),
        }


@dataclass
class TemporalCoverage:
    """Per-band coverage for every modality that reported evidence."""

    duration_ms: int
    band_ms: int
    bands: list[Band] = field(default_factory=list)
    modalities: list[str] = field(default_factory=list)

    def blind_spots(self, modality: str) -> list[Band]:
        """Bands nothing of this modality examined. The honest answer to
        "is the whole video clean?" starts here."""
        return [b for b in self.bands if b.state_of(modality) == "UNEXAMINED"]

    def thin(self, modality: str) -> list[Band]:
        return [b for b in self.bands if b.state_of(modality) == "THIN"]

    def share_examined(self, modality: str) -> float:
        """Fraction of the timeline this modality genuinely examined.

        Distinct from the agent's own coverage figure, which measures how
        much of *its own sample set* it processed. A modality can process
        every frame it was given and still have examined only half the
        runtime, if that is where the frames were.
        """
        if not self.bands:
            return 0.0
        good = sum(1 for b in self.bands if b.state_of(modality) == "EXAMINED")
        return good / len(self.bands)

    def to_json(self) -> dict[str, Any]:
        return {
            "durationMs": self.duration_ms,
            "bandMs": self.band_ms,
            "modalities": list(self.modalities),
            "bands": [b.to_json() for b in self.bands],
            "shareExamined": {
                m: round(self.share_examined(m), 4) for m in self.modalities
            },
            "blindSpots": {
                m: [b.label for b in self.blind_spots(m)] for m in self.modalities
            },
        }

    def describe(self) -> str:
        """The table a reader checks before trusting an absence claim."""
        if not self.bands:
            return "no timeline to report"
        width = max(len(m) for m in self.modalities) if self.modalities else 8
        head = "  band   " + "  ".join(m[:10].ljust(10) for m in self.modalities)
        rows = [head, "  " + "-" * (len(head) - 2)]
        for band in self.bands:
            cells = []
            for m in self.modalities:
                state = band.state_of(m)
                mark = {"EXAMINED": "ok", "THIN": "thin", "UNEXAMINED": "NONE"}[state]
                cells.append(f"{mark}({band.samples.get(m, 0)})".ljust(10))
            rows.append(f"  {band.label}  " + "  ".join(cells))
        _ = width
        return "\n".join(rows)


def _timestamps(items: Iterable[Any]) -> list[int]:
    """Pull a millisecond timestamp off whatever shape the caller passed.

    Duck-typed on purpose: keyframes carry `ts_ms`, OCR items and transcript
    segments carry `start_ms`, and report dicts carry `startMs`. Accepting
    all three keeps this module from importing — and coupling to — every
    perception module that has evidence to contribute.
    """
    out: list[int] = []
    for item in items or ():
        for attr in ("ts_ms", "start_ms", "startMs"):
            value = (
                item.get(attr)
                if isinstance(item, dict)
                else getattr(item, attr, None)
            )
            if value is not None:
                out.append(int(value))
                break
    return out


def build(
    duration_ms: int,
    evidence: dict[str, Iterable[Any]],
    *,
    band_ms: int = DEFAULT_BAND_MS,
) -> TemporalCoverage:
    """Project each modality's evidence onto the timeline.

    `evidence` maps a modality name to whatever that modality actually
    looked at — keyframes for vision and OCR, segments for speech and audio.
    A modality absent from the mapping is absent from the report rather than
    being shown as zero: "did not run" and "ran and found nothing" are
    different facts and must not share a row.
    """
    if duration_ms <= 0 or band_ms <= 0:
        return TemporalCoverage(duration_ms=max(0, duration_ms), band_ms=band_ms)

    count = max(1, -(-duration_ms // band_ms))  # ceil
    modalities = sorted(evidence)

    # Expected samples per band, per modality — derived from what that
    # modality actually produced rather than from a fixed target, so a
    # modality that legitimately samples sparsely is not marked thin for it.
    expected: dict[str, float] = {}
    for name, items in evidence.items():
        total = len(_timestamps(items))
        expected[name] = (total / count) if count else 0.0

    bands: list[Band] = []
    for index in range(count):
        start = index * band_ms
        end = min(duration_ms, start + band_ms)
        samples: dict[str, int] = {}
        states: dict[str, State] = {}

        for name, items in evidence.items():
            hits = sum(1 for ts in _timestamps(items) if start <= ts < end)
            samples[name] = hits
            floor = max(
                MIN_SAMPLES_FOR_EXAMINED, expected.get(name, 0.0) * THIN_BAND_RATIO
            )
            if hits == 0:
                states[name] = "UNEXAMINED"
            elif hits < floor:
                states[name] = "THIN"
            else:
                states[name] = "EXAMINED"

        bands.append(
            Band(index=index, start_ms=start, end_ms=end, samples=samples, states=states)
        )

    return TemporalCoverage(
        duration_ms=duration_ms,
        band_ms=band_ms,
        bands=bands,
        modalities=modalities,
    )


def absence_is_supported(
    coverage: TemporalCoverage, modality: str, start_ms: int, end_ms: int
) -> bool:
    """May a caller claim this modality found nothing in this span?

    Only when every band the span touches was genuinely examined. This is the
    guard that stops "no secrets detected" from being emitted over a stretch
    of video that nothing read — the temporal twin of the coverage floor the
    verification comparison already applies to absence claims.
    """
    if not coverage.bands:
        return False
    touched = [
        b for b in coverage.bands if b.start_ms < end_ms and b.end_ms > start_ms
    ]
    if not touched:
        return False
    return all(b.state_of(modality) == "EXAMINED" for b in touched)
