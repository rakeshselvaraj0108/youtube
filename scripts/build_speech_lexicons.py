"""Author the A02 speech lexicons.

Thirteen event types, loaded into RAM at startup. Never indexed, never
embedded, never retrieved — a dict lookup answers in nanoseconds and a vector
search would answer the same question worse and slower.

Two authoring rules run through every file:

**A term is a cue, not a verdict.** A02 emits `PROFANITY` with a span; it never
emits `VIOLATION`. Whether a term in context breaches a clause is A11's
decision, and A02 would make it worse by guessing.

**Slurs are not enumerated.** AF-06 needs no list to work: the clause text
describes the category and the adjudicator recognises instances, which it does
better than a list because a list cannot tell a slur being used from one being
condemned. A slur list in a public repository is a liability with no upside.
`HATE` therefore ships structural cues — protected-attribute nouns plus
derogatory frames — rather than terms.

    python scripts/build_speech_lexicons.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("data/lexicons/speech")

NEVER_A_VERDICT = (
    "A hit here is EVIDENCE, never a verdict. A02 reports what was said and "
    "where. Whether it breaches a clause is decided downstream by the "
    "adversarial triad, which can see context this file cannot."
)


def lex(
    event: str,
    *,
    note: str,
    terms: list[dict],
    phrases: list[str] | None = None,
    severity_default: str = "MEDIUM",
) -> dict:
    return {
        "_event": event,
        "_note": NEVER_A_VERDICT,
        "_purpose": note,
        "severity_default": severity_default,
        "terms": terms,
        "phrases": phrases or [],
    }


def t(term: str, severity: str = "MEDIUM", variants: list[str] | None = None) -> dict:
    return {"term": term, "severity": severity, "variants": variants or []}


LEXICONS = {
    # ── PROFANITY ────────────────────────────────────────────────────
    # Variants are explicit rather than generated. A generated obfuscation
    # matcher that maps 4->a and 1->i will happily read "1nformat10n" as a hit;
    # listing the forms that actually occur is both safer and easier to audit.
    "profanity.json": lex(
        "PROFANITY",
        note="Vulgar language. Strength and position matter more than presence.",
        severity_default="MEDIUM",
        terms=[
            t("fuck", "HIGH", ["f*ck", "f**k", "fck", "fuk", "phuck", "fuq", "f u c k"]),
            t("fucking", "HIGH", ["f*cking", "fkin", "fukin", "effing"]),
            t("motherfucker", "HIGH", ["mf", "m*therf*cker"]),
            t("shit", "MEDIUM", ["sh*t", "sh1t", "shyt", "s h i t"]),
            t("bullshit", "MEDIUM", ["bs", "bullsh*t"]),
            t("bitch", "MEDIUM", ["b*tch", "b1tch", "biatch"]),
            t("asshole", "MEDIUM", ["a**hole", "arsehole", "a-hole"]),
            t("bastard", "MEDIUM"),
            t("dick", "MEDIUM", ["d*ck"]),
            t("piss", "LOW", ["p*ss"]),
            t("damn", "LOW", ["d*mn", "goddamn", "dammit"]),
            t("hell", "LOW"),
            t("crap", "LOW"),
            t("bloody", "LOW"),
            t("bollocks", "MEDIUM"),
            t("wanker", "MEDIUM"),
        ],
    ),
    # ── HATE — structural cues only, deliberately no term list ───────
    "hate.json": lex(
        "HATE",
        note=(
            "No slur enumeration. A list cannot distinguish use from mention, "
            "and shipping one is a liability with no upside. These are the "
            "structural signals that a protected attribute is being targeted: "
            "an attribute noun inside a derogatory frame. The triad decides."
        ),
        severity_default="HIGH",
        terms=[],
        phrases=[
            "all of them are",
            "those people always",
            "they should go back",
            "do not belong here",
            "shouldn't be allowed to",
            "are ruining this country",
            "typical of that lot",
        ],
    ),
    # ── VIOLENCE ─────────────────────────────────────────────────────
    "violence.json": lex(
        "VIOLENCE",
        note=(
            "Verbs, weapons and conflict terms. Enormously polysemous — "
            "'shoot' in basketball, 'kill' in a game review, 'attack' in chess. "
            "This is precisely why a hit is not a finding."
        ),
        severity_default="MEDIUM",
        terms=[
            t("kill", "MEDIUM"), t("killed", "MEDIUM"), t("murder", "HIGH"),
            t("stab", "HIGH"), t("stabbed", "HIGH"), t("shoot", "MEDIUM"),
            t("shot", "MEDIUM"), t("shooting", "MEDIUM"), t("gun", "MEDIUM"),
            t("rifle", "MEDIUM"), t("pistol", "MEDIUM"), t("knife", "MEDIUM"),
            t("bomb", "HIGH"), t("explosion", "MEDIUM"), t("blood", "MEDIUM"),
            t("bleeding", "MEDIUM"), t("wound", "MEDIUM"), t("injury", "LOW"),
            t("assault", "HIGH"), t("beat", "LOW"), t("punch", "LOW"),
            t("strangle", "HIGH"), t("execute", "HIGH"), t("massacre", "HIGH"),
        ],
    ),
    # ── DRUGS ────────────────────────────────────────────────────────
    "drugs.json": lex(
        "DRUGS",
        note="Common names and widely used slang. Not exhaustive by design.",
        severity_default="MEDIUM",
        terms=[
            t("cocaine", "HIGH", ["coke", "blow"]), t("heroin", "HIGH", ["smack"]),
            t("methamphetamine", "HIGH", ["meth", "crystal meth"]),
            t("mdma", "MEDIUM", ["ecstasy", "molly"]), t("ketamine", "MEDIUM"),
            t("fentanyl", "HIGH"), t("lsd", "MEDIUM", ["acid tab"]),
            t("marijuana", "LOW", ["weed", "cannabis", "pot"]),
            t("edible", "LOW"), t("bong", "LOW"), t("dab", "LOW"),
            t("psilocybin", "MEDIUM", ["shrooms"]),
        ],
    ),
    "alcohol.json": lex(
        "ALCOHOL",
        note="Incidental adult reference is generally fine; promotion is not.",
        severity_default="LOW",
        terms=[
            t("beer", "LOW"), t("wine", "LOW"), t("vodka", "LOW"),
            t("whiskey", "LOW", ["whisky"]), t("tequila", "LOW"), t("gin", "LOW"),
            t("rum", "LOW"), t("bourbon", "LOW"), t("shots", "LOW"),
            t("pint", "LOW"), t("drunk", "LOW"), t("hangover", "LOW"),
        ],
    ),
    "gambling.json": lex(
        "GAMBLING",
        note="Betting, casino and simulated gambling including loot boxes.",
        severity_default="MEDIUM",
        terms=[
            t("casino", "MEDIUM"), t("roulette", "MEDIUM"), t("blackjack", "MEDIUM"),
            t("poker", "LOW"), t("lottery", "LOW"), t("jackpot", "MEDIUM"),
            t("sportsbook", "MEDIUM"), t("wager", "MEDIUM"), t("bookmaker", "MEDIUM"),
            t("loot box", "LOW"), t("slots", "MEDIUM"),
        ],
        phrases=["place a bet", "odds on", "free spins", "deposit bonus"],
    ),
    # ── FINANCIAL — phrases carry the signal, not single words ───────
    "financial.json": lex(
        "FINANCIAL_CLAIM",
        note=(
            "Single words are useless here: 'profit' and 'return' are ordinary "
            "business vocabulary. The scam signal lives in the guarantee, so "
            "this file is almost entirely phrases."
        ),
        severity_default="HIGH",
        terms=[],
        phrases=[
            "double your money", "triple your money", "guaranteed return",
            "guaranteed returns", "guaranteed profit", "risk free investment",
            "no risk investment", "crypto giveaway", "send me one", "get rich quick",
            "passive income guaranteed", "financial freedom in", "insider tip",
            "cannot lose", "can't lose", "secret method the banks",
        ],
    ),
    "medical.json": lex(
        "MEDICAL_CLAIM",
        note=(
            "Also phrase-driven. 'Cancer' is not a claim; 'cures cancer' is. "
            "The harm is in the assertion of efficacy, not the condition named."
        ),
        severity_default="HIGH",
        terms=[],
        phrases=[
            "cures cancer", "cure cancer", "miracle cure", "miracle treatment",
            "guaranteed treatment", "guaranteed cure", "big pharma doesn't want",
            "reverses diabetes", "detox your", "boosts your immune system instantly",
            "stop taking your medication", "doctors hate this",
            "no need for a doctor", "natural alternative to chemotherapy",
        ],
    ),
    "sponsor.json": lex(
        "SPONSOR",
        note=(
            "Detection here is half of META-01: spoken sponsorship plus no "
            "disclosure in the description is real regulatory exposure."
        ),
        severity_default="MEDIUM",
        terms=[],
        phrases=[
            "this video is sponsored by", "this episode is sponsored by",
            "sponsored by", "brought to you by", "paid partnership",
            "in partnership with", "for sponsoring this video", "our sponsor",
            "use code", "use my code", "with my code", "discount code",
            "promo code", "coupon code", "affiliate link", "affiliate links",
            "i earn a commission", "at no extra cost to you",
            "as an amazon associate", "link in the description",
        ],
    ),
    "sensitive.json": lex(
        "SENSITIVE_EVENT",
        note="An identifiable tragedy. Casualty patterns live in the regex layer.",
        severity_default="MEDIUM",
        terms=[
            t("terrorism", "HIGH"), t("terrorist", "HIGH"), t("massacre", "HIGH"),
            t("genocide", "HIGH"), t("earthquake", "MEDIUM"), t("tsunami", "MEDIUM"),
            t("hurricane", "LOW"), t("wildfire", "LOW"), t("pandemic", "LOW"),
            t("outbreak", "LOW"), t("hostage", "HIGH"), t("evacuation", "LOW"),
            t("avalanche", "MEDIUM"), t("derailment", "MEDIUM"),
        ],
        phrases=["school shooting", "mass shooting", "suicide bombing", "war crime"],
    ),
    "copyright.json": lex(
        "COPYRIGHT_MENTION",
        note=(
            "Naming a rights holder is not infringement — this is a weak "
            "corroborating signal for COPY-01, nothing more. A review that "
            "says 'Netflix' has not used anything belonging to Netflix."
        ),
        severity_default="LOW",
        terms=[
            t("disney", "LOW"), t("netflix", "LOW"), t("marvel", "LOW"),
            t("pixar", "LOW"), t("sony", "LOW"), t("universal", "LOW"),
            t("warner", "LOW"), t("hbo", "LOW"), t("nintendo", "LOW"),
            t("paramount", "LOW"),
        ],
        phrases=["copyrighted music", "owned by", "all rights reserved"],
    ),
    "ai_disclosure.json": lex(
        "AI_DISCLOSURE",
        note=(
            "Presence of a disclosure is a POSITIVE signal — it satisfies a "
            "synthetic-media clause rather than breaching one. Recorded as "
            "evidence so the advocate can cite it."
        ),
        severity_default="LOW",
        terms=[t("deepfake", "MEDIUM"), t("deepfakes", "MEDIUM")],
        phrases=[
            "generated by ai", "ai generated", "ai-generated", "synthetic voice",
            "voice clone", "cloned voice", "made with ai", "created using ai",
            "this is not a real person",
        ],
    ),
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, payload in LEXICONS.items():
        path = OUT / name
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        count = len(payload["terms"]) + len(payload["phrases"])
        print(f"  {path.name:<22} {payload['_event']:<18} {count:>3} entries")
    print(f"\nwrote {len(LEXICONS)} speech lexicons to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
