"""Simulate a policy update, so the Drift Watcher can be demonstrated.

The Watcher's real deployment is a scheduled job that refetches the published
guidelines. That cannot be shown on camera in thirty seconds, and waiting for
YouTube to actually change a rule is not a demo plan.

This applies a realistic edit to the local corpus instead: two clauses tighten
and one new clause appears. The mechanism being demonstrated — snapshot, diff,
semantic delta, selective re-lint — is identical either way; only the trigger
differs, and the README says so plainly.

    python scripts/build_corpus.py                    # baseline
    preflight snapshot --out data/policy-snapshots/2026-08.json
    python scripts/simulate_drift.py                  # the rules move
    preflight drift --against data/policy-snapshots/2026-08.json

Undo with `python scripts/build_corpus.py`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

POLICY = Path("data/policy")
NEW_VERSION = "2026-09"

# AF-08: firearms tightens. Range and demonstration content moves from Yellow
# to Red, which is the kind of change that quietly demonetises a back
# catalogue of gun-adjacent videos overnight.
FIREARMS_YELLOW = """- Firearm safety instruction with no discharge shown
- Brief historical reference"""

FIREARMS_RED = """- Instructions for manufacturing firearms or modifying them to fire
  automatically
- Facilitating sale or transfer
- Firearm demonstration, range or discharge content of any kind
- Detailed discussion of firearm capability or lethality"""

# AF-10: sensitive events tightens slightly. Casualty figures now require
# explicit framing rather than merely avoiding sensationalism.
SENSITIVE_YELLOW = """- Casualty figures stated without explicit contextual framing
- Any discussion of a tragedy within twelve months of the event
- Extended discussion of a recent tragedy"""

# AF-15 is new. Synthetic media disclosure is exactly the sort of clause that
# appears between one policy revision and the next.
AI_CONTENT = """---
clause_id: AF-15
title: Synthetic and AI-generated content
severity_default: LIMITING
version: 2026-09
source_url: https://support.google.com/youtube/answer/6162278
fetched_at: 2026-09-01
---

## Scope

Content that is synthetically generated or materially altered by AI, including
synthesised voice, face replacement and generated footage presented as real.

## Green (fully monetized)

- Clearly labelled synthetic content with visible disclosure
- AI used for production assistance without altering factual claims

## Yellow (limited ads)

- Synthetic voice or likeness without disclosure
- Generated imagery presented without context

## Red (no ads)

- Synthetic media depicting a real person saying or doing something they did not
- Generated footage of real events presented as authentic

## Documented exemptions

- Educational, documentary, scientific or artistic (EDSA) framing where the
  context is clear from the content itself
- News reporting on a matter of public interest
- Non-graphic reference rather than depiction
- Clearly fictional or scripted context
- Quotation or condemnation of a third party rather than endorsement
"""


def replace_section(text: str, heading: str, body: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    inside = False
    for line in lines:
        if line.startswith("## "):
            if inside:
                inside = False
            if line.strip().startswith(f"## {heading}"):
                inside = True
                out.append(line)
                out.append("")
                out.extend(body.splitlines())
                out.append("")
                continue
        if not inside:
            out.append(line)
    return "\n".join(out) + "\n"


def bump_version(text: str) -> str:
    return text.replace("version: 2026-08", f"version: {NEW_VERSION}")


def main() -> int:
    if not POLICY.is_dir():
        print("no corpus — run python scripts/build_corpus.py first")
        return 2

    touched: list[str] = []

    firearms = POLICY / "08_firearms.md"
    text = bump_version(firearms.read_text(encoding="utf-8"))
    text = replace_section(text, "Yellow", FIREARMS_YELLOW)
    text = replace_section(text, "Red", FIREARMS_RED)
    firearms.write_text(text, encoding="utf-8")
    touched.append("AF-08 tightened (range content Yellow -> Red)")

    sensitive = POLICY / "10_sensitive_events.md"
    text = bump_version(sensitive.read_text(encoding="utf-8"))
    text = replace_section(text, "Yellow", SENSITIVE_YELLOW)
    sensitive.write_text(text, encoding="utf-8")
    touched.append("AF-10 tightened (casualty figures require framing)")

    (POLICY / "15_ai_generated_content.md").write_text(AI_CONTENT, encoding="utf-8")
    touched.append("AF-15 added (synthetic and AI-generated content)")

    # Keep the manifest honest — it is what records provenance per clause.
    manifest_path = POLICY / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["version"] = NEW_VERSION
        by_id = {c["clause_id"]: c for c in manifest["clauses"]}
        for path in sorted(POLICY.glob("*.md")):
            body = path.read_text(encoding="utf-8")
            clause_id = next(
                (
                    line.split(":", 1)[1].strip()
                    for line in body.splitlines()
                    if line.startswith("clause_id:")
                ),
                path.stem,
            )
            entry = by_id.setdefault(
                clause_id,
                {
                    "clause_id": clause_id,
                    "file": path.name,
                    "title": clause_id,
                    "severity_default": "LIMITING",
                    "source_url": manifest.get("source_url", ""),
                },
            )
            entry["sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
            entry["fetched_at"] = "2026-09-01"
        manifest["clauses"] = sorted(by_id.values(), key=lambda c: c["clause_id"])
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"corpus moved to {NEW_VERSION}:")
    for line in touched:
        print(f"  - {line}")
    print("\nrun:  preflight drift --against data/policy-snapshots/2026-08.json")
    print("undo: python scripts/build_corpus.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
