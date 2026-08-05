"""Emit shared scoring vectors for the cross-language contract test.

The page renders the TypeScript `computeReadiness`; the JSON carries the Python
one. If they disagree, the report's headline number contradicts its own data.
This writes deterministic vectors that BOTH test suites assert against, so a
drift in either implementation turns a build red.

    python scripts/emit_scoring_vectors.py
    npm test          # TypeScript side
    pytest            # Python side
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from preflight.scoring.readiness import SUB_SCORE_ORDER, compute_readiness

OUT = Path("tests/fixtures/scoring_vectors.json")


def cases() -> list[dict]:
    out: list[dict] = []

    # Deterministic edge cases first — the boundaries are where the two
    # implementations are most likely to part company.
    handpicked = [
        dict.fromkeys(SUB_SCORE_ORDER, 0),
        dict.fromkeys(SUB_SCORE_ORDER, 100),
        dict.fromkeys(SUB_SCORE_ORDER, 50),
        {"policy": 31, "copyright": 19, "metadata": 78, "accessibility": 62, "audio": 88},
        {"policy": 96, "copyright": 100, "metadata": 94, "accessibility": 88, "audio": 95},
        {"policy": 95, "copyright": 95, "metadata": 95, "accessibility": 70, "audio": 95},
        {"policy": 95, "copyright": 95, "metadata": 95, "accessibility": 69, "audio": 95},
        {"policy": 95, "copyright": 95, "metadata": 95, "accessibility": 55, "audio": 95},
        {"policy": 95, "copyright": 95, "metadata": 95, "accessibility": 45, "audio": 95},
        # Ties, to pin `weakest` resolution.
        {"policy": 40, "copyright": 95, "metadata": 95, "accessibility": 95, "audio": 40},
        # Half-integer weighted means, to pin rounding direction.
        {"policy": 85, "copyright": 84, "metadata": 84, "accessibility": 84, "audio": 84},
        {"policy": 70, "copyright": 70, "metadata": 71, "accessibility": 70, "audio": 70},
    ]
    out.extend(handpicked)

    rng = random.Random(20260805)
    for _ in range(40):
        out.append({key: rng.randint(0, 100) for key in SUB_SCORE_ORDER})

    return out


def main() -> int:
    payload = {
        "note": (
            "Shared vectors for the Python/TypeScript scoring contract. "
            "Regenerate with scripts/emit_scoring_vectors.py."
        ),
        "cases": [],
    }
    for sub in cases():
        result = compute_readiness(sub)
        payload["cases"].append(
            {
                "sub": sub,
                "overall": result.overall,
                "verdict": result.verdict,
                "weakest": result.weakest,
                "capped": result.capped,
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(payload['cases'])} vectors to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
