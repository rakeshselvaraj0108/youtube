"""Author the policy corpus.

The corpus is a **structured restatement** of publicly published platform
guidance, stored with source URLs and fetch timestamps, used for
retrieval-grounded classification. It is not a copy of the guidelines and does
not claim to be authoritative — it is the rule set PREFLIGHT lints against, and
every finding cites the clause it was judged under so a human can check it.

This script is the authoring surface. It emits `data/policy/*.md` plus a
manifest recording a SHA-256 per clause, which is what makes a report
reproducible against a known snapshot and what the Drift Watcher diffs.

    python scripts/build_corpus.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUT = Path("data/policy")
SOURCE = "https://support.google.com/youtube/answer/6162278"
FETCHED = "2026-08-05"
VERSION = "2026-08"

EDSA = (
    "- Educational, documentary, scientific or artistic (EDSA) framing where the "
    "context is clear from the content itself\n"
    "- News reporting on a matter of public interest\n"
    "- Non-graphic reference rather than depiction\n"
    "- Clearly fictional or scripted context\n"
    "- Quotation or condemnation of a third party rather than endorsement"
)

CLAUSES: list[dict] = [
    {
        "file": "01_inappropriate_language.md",
        "id": "AF-01",
        "title": "Inappropriate language",
        "severity": "LIMITING",
        "scope": (
            "Profanity and vulgar language in speech, on-screen text, titles, "
            "thumbnails and metadata. On-screen text is assessed the same way as "
            "spoken language; masked or censored terms are weighted lower but are "
            "still assessed."
        ),
        "green": [
            "No profanity, or only mild language used infrequently",
            "Strong profanity that is fully bleeped or muted",
            "Isolated moderate profanity outside the opening",
        ],
        "yellow": [
            "Strong profanity used more than occasionally through the video",
            "Any strong profanity in the first seven seconds",
            "Profanity in the title, thumbnail or on-screen text",
        ],
        "red": [
            "Strong profanity used continuously or as the focus of the content",
            "Slurs directed at a person or group",
        ],
    },
    {
        "file": "02_violence.md",
        "id": "AF-02",
        "title": "Violence",
        "severity": "DEMONETIZING",
        "scope": (
            "Real or dramatised violence, physical injury, blood and its aftermath. "
            "Assessment turns on whether the injury is the subject of the shot and "
            "whether it is dwelt upon, not merely on whether it appears."
        ),
        "green": [
            "Violence that is implied rather than shown",
            "Non-graphic depictions in a clearly fictional or gaming context",
            "Brief, incidental injury that is not the focus of the frame",
        ],
        "yellow": [
            "Real injury shown briefly with visible blood",
            "Dramatised violence held on screen without graphic detail",
        ],
        "red": [
            "Graphic real injury, mutilation or death held in focus",
            "Violence presented for shock value with no contextual purpose",
        ],
    },
    {
        "file": "03_adult_content.md",
        "id": "AF-03",
        "title": "Adult content",
        "severity": "DEMONETIZING",
        "scope": (
            "Sexually gratifying content, nudity, and sexually suggestive framing "
            "including camera emphasis on body parts."
        ),
        "green": [
            "Non-sexual nudity in an artistic, medical or documentary context",
            "Romantic content without sexual emphasis",
        ],
        "yellow": [
            "Sexually suggestive framing or dancing without explicit content",
            "Discussion of sexual topics in an educational register",
        ],
        "red": [
            "Sexual acts, whether real, simulated or animated",
            "Nudity presented for sexual gratification",
        ],
    },
    {
        "file": "04_shocking_content.md",
        "id": "AF-04",
        "title": "Shocking content",
        "severity": "DEMONETIZING",
        "scope": (
            "Content intended to shock or disgust, including gore divorced from "
            "narrative purpose, bodily fluids, and graphic medical procedures."
        ),
        "green": [
            "Medical or scientific footage with clear educational framing",
            "Mild shock content with a stated warning",
        ],
        "yellow": [
            "Graphic medical procedure shown in detail",
            "Content likely to disturb a general audience",
        ],
        "red": [
            "Gore or bodily harm presented for its own sake",
        ],
    },
    {
        "file": "05_harmful_dangerous_acts.md",
        "id": "AF-05",
        "title": "Harmful or dangerous acts",
        "severity": "DEMONETIZING",
        "scope": (
            "Acts a viewer could imitate and be seriously injured by, including "
            "dangerous stunts, challenges, and unsafe technique presented approvingly. "
            "The documentary exemption requires an explicit warning against imitation."
        ),
        "green": [
            "Professional stunt work with visible safety measures and a warning",
            "Discussion of a dangerous act without demonstrating it",
        ],
        "yellow": [
            "Risky activity by trained subjects with no warning against imitation",
            "Minor self-harm-adjacent challenge content",
        ],
        "red": [
            "Instructional content for a dangerous act",
            "Content encouraging viewers to attempt serious risk",
        ],
    },
    {
        "file": "06_hateful_derogatory.md",
        "id": "AF-06",
        "title": "Hateful and derogatory content",
        "severity": "DEMONETIZING",
        "scope": (
            "Content promoting hatred, demeaning an individual or group on the basis "
            "of a protected attribute, or degrading a person on the basis of an "
            "immutable characteristic."
        ),
        "green": [
            "Neutral discussion of discrimination as a subject",
            "Reporting on a hate incident without repeating slurs gratuitously",
        ],
        "yellow": [
            "Insulting content targeting an individual without a protected attribute",
        ],
        "red": [
            "Slurs or dehumanising language aimed at a protected group",
            "Content promoting or justifying hatred",
        ],
    },
    {
        "file": "07_recreational_drugs.md",
        "id": "AF-07",
        "title": "Recreational drugs and drug-related content",
        "severity": "LIMITING",
        "scope": (
            "Depiction, promotion or discussion of recreational drugs, including "
            "sale, manufacture and use."
        ),
        "green": [
            "Educational or recovery-focused discussion",
            "Passing reference without depiction",
        ],
        "yellow": [
            "Drug use shown without promotion",
            "Discussion of drug culture in a non-educational register",
        ],
        "red": [
            "Promotion or sale of drugs",
            "Instructions for manufacture or acquisition",
        ],
    },
    {
        "file": "08_firearms.md",
        "id": "AF-08",
        "title": "Firearms",
        "severity": "LIMITING",
        "scope": (
            "Firearms and firearm-adjacent content, including handling, modification "
            "and sale."
        ),
        "green": [
            "Firearms appearing incidentally in a scene",
            "Historical or documentary treatment",
        ],
        "yellow": [
            "Firearm demonstration or range content",
            "Detailed discussion of firearm capability",
        ],
        "red": [
            "Instructions for manufacturing firearms or modifying them to fire "
            "automatically",
            "Facilitating sale or transfer",
        ],
    },
    {
        "file": "09_controversial_issues.md",
        "id": "AF-09",
        "title": "Controversial issues",
        "severity": "LIMITING",
        "scope": (
            "Contentious political and social topics, including armed conflict, "
            "abortion, immigration and civil unrest. Assessment turns on whether the "
            "treatment is inflammatory or sourced."
        ),
        "green": [
            "Balanced, sourced reporting",
            "Passing mention without argument",
        ],
        "yellow": [
            "One-sided treatment of a contentious issue",
            "Unsourced allegation against a named person or institution",
        ],
        "red": [
            "Inflammatory content likely to incite",
        ],
    },
    {
        "file": "10_sensitive_events.md",
        "id": "AF-10",
        "title": "Sensitive events",
        "severity": "DEMONETIZING",
        "scope": (
            "Tragedies, disasters, deaths and violent incidents affecting real "
            "people. Assessment turns on whether the treatment dwells on the loss or "
            "sensationalises it."
        ),
        "green": [
            "Factual reference in passing, with framing",
            "Memorial or tribute content",
        ],
        "yellow": [
            "Casualty figures stated without surrounding framing",
            "Extended discussion of a recent tragedy",
        ],
        "red": [
            "Graphic discussion or footage of a tragedy",
            "Content exploiting a tragedy for engagement",
        ],
    },
    {
        "file": "11_enabling_dishonest_behavior.md",
        "id": "AF-11",
        "title": "Enabling dishonest behaviour",
        "severity": "DEMONETIZING",
        "scope": (
            "Content facilitating dishonesty, including academic cheating, hacking "
            "for harm, forged documents and circumvention of payment."
        ),
        "green": [
            "Security research with a defensive framing",
            "Discussion of fraud as a subject",
        ],
        "yellow": [
            "Demonstration of a circumvention technique without instruction",
        ],
        "red": [
            "Step-by-step instructions enabling fraud or unauthorised access",
        ],
    },
    {
        "file": "12_tobacco.md",
        "id": "AF-12",
        "title": "Tobacco, vaping and alcohol",
        "severity": "LIMITING",
        "scope": (
            "Regulated goods: tobacco, vaping products and alcohol. Incidental adult "
            "references are generally eligible; promotion and excess are not."
        ),
        "green": [
            "Incidental reference in adult-audience content",
            "Cessation or harm-reduction content",
        ],
        "yellow": [
            "Consumption shown on camera without promotion",
            "Product review of regulated goods",
        ],
        "red": [
            "Promotion of tobacco or vaping products",
            "Content promoting excessive consumption",
            "Any depiction in content made for kids",
        ],
    },
    {
        "file": "13_adult_themes_in_family_content.md",
        "id": "AF-13",
        "title": "Adult themes in family content",
        "severity": "DEMONETIZING",
        "scope": (
            "Mature themes presented in content whose format, characters or framing "
            "would lead a viewer to expect it is made for children."
        ),
        "green": [
            "Family content with no mature themes",
            "Mature content clearly framed for an adult audience",
        ],
        "yellow": [
            "Mild mature themes in family-coded formats",
        ],
        "red": [
            "Violence, sexual themes or profanity in content resembling children's "
            "programming",
        ],
    },
    {
        "file": "14_gambling.md",
        "id": "AF-14",
        "title": "Gambling",
        "severity": "LIMITING",
        "scope": (
            "Gambling, betting, casino content and simulated gambling including loot "
            "boxes."
        ),
        "green": [
            "Discussion of gambling harm or regulation",
            "Passing reference",
        ],
        "yellow": [
            "Gameplay of casino-style games",
            "Discussion of betting strategy",
        ],
        "red": [
            "Promotion of unlicensed gambling sites",
            "Content directing viewers to place bets",
        ],
    },
]

TEMPLATE = """---
clause_id: {id}
title: {title}
severity_default: {severity}
version: {version}
source_url: {source}
fetched_at: {fetched}
---

## Scope

{scope}

## Green (fully monetized)

{green}

## Yellow (limited ads)

{yellow}

## Red (no ads)

{red}

## Documented exemptions

{exemptions}
"""


def bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # This script owns the directory. Without removing clause files it did not
    # write, a clause added by a drift simulation survives a rebuild and ends
    # up in the next baseline snapshot — so the very change being demonstrated
    # is already present before the demonstration starts.
    expected = {clause["file"] for clause in CLAUSES}
    for stale in OUT.glob("*.md"):
        if stale.name not in expected:
            stale.unlink()
            print(f"removed stale clause {stale.name}")

    manifest: list[dict] = []

    for clause in CLAUSES:
        body = TEMPLATE.format(
            id=clause["id"],
            title=clause["title"],
            severity=clause["severity"],
            version=VERSION,
            source=SOURCE,
            fetched=FETCHED,
            scope=clause["scope"],
            green=bullets(clause["green"]),
            yellow=bullets(clause["yellow"]),
            red=bullets(clause["red"]),
            exemptions=EDSA,
        )
        path = OUT / clause["file"]
        path.write_text(body, encoding="utf-8")
        manifest.append(
            {
                "clause_id": clause["id"],
                "file": clause["file"],
                "title": clause["title"],
                "severity_default": clause["severity"],
                "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                "source_url": SOURCE,
                "fetched_at": FETCHED,
            }
        )

    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "version": VERSION,
                "source_url": SOURCE,
                "fetched_at": FETCHED,
                "note": (
                    "Structured restatement of publicly published platform guidance, "
                    "used for retrieval-grounded classification. Not authoritative and "
                    "not affiliated with YouTube."
                ),
                "clauses": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"wrote {len(manifest)} clauses + manifest to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
