"""Wires scripts/verify_corpus_truth.py into the suite.

The corpus's whole claim is that ground truth is exact by construction — the
defect was inserted at 4,200ms, so it IS at 4,200ms. That claim held for
every check except one: the loudness pair's RELATIVE check ("hot master is
louder than normalised") passed while the real production detector, run
against the actual rendered files, fired on neither clip. g052 measured -12.4
LUFS against the -14+-2 target used in `preflight/perception/audio.py` — 0.4
LUFS inside tolerance — and the clip built to be a loudness VIOLATION did not
violate. A relative check cannot catch that, because it never asks the
question the detector actually asks.

Skipped, not failed, when the corpus has not been generated — clips are
gitignored, and a clone that has not run `make corpus` yet is a setup gap,
not a truth-verification failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import verify_corpus_truth as vct  # noqa: E402

CORPUS_PRESENT = vct.CLIPS.is_dir() and any(vct.CLIPS.glob("*.mp4"))

pytestmark = pytest.mark.skipif(
    not CORPUS_PRESENT,
    reason="corpus not generated — run `make corpus` or `python data/corpus/generate.py`",
)

CHECKS = [vct.check_pairs, vct.check_photosensitive, vct.check_channels, vct.check_loudness]


@pytest.mark.parametrize("check", CHECKS, ids=lambda c: c.__name__)
def test_corpus_ground_truth_check(check):
    failures = check()
    assert not failures, "\n".join(failures)
