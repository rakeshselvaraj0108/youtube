"""Wires scripts/verify_data.py into the suite so its checks run on every CI
build rather than only when someone remembers to invoke the script by hand.

The checks themselves — and the reasoning behind each one — live in
scripts/verify_data.py, next to a `python scripts/verify_data.py` entry point
a human can run directly. This module just makes pytest one of the callers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import verify_data  # noqa: E402


@pytest.mark.parametrize("build", verify_data.CHECKS, ids=lambda c: c.__name__)
def test_data_provenance_check(build):
    result = build()
    assert result.ok, "\n".join(result.problems)
