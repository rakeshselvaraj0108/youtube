"""Author the deterministic lexicons.

These are cue lists, not verdict lists. Every one of them does exactly one job:
raise the retrieval priority of a clause for a window so that clause reaches the
adjudicator. **A lexicon hit is never a finding.**

That distinction is the whole reason this project exists. Word-list matching is
the naive approach that produces roughly nine false positives in ten — "shoot"
in a basketball video, "kill" in a game review, a slur being quoted in order to
condemn it. The lexicons are cheap recall; the triad supplies precision.

Written here in one place rather than scattered as literals through the agents,
so a judge can read what the system keys on and `verify_data.py` can check it.

    python scripts/build_lexicons.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("data/lexicons")

NEVER_A_VERDICT = (
    "A hit in this file is NEVER a finding. It raises the retrieval priority of "
    "the named clause for that window and nothing else. The adversarial triad "
    "decides, using the clause text. Treating a cue hit as a violation is the "
    "naive keyword approach this project exists to replace."
)


# --------------------------------------------------------------------------
# Profanity, tiered.
#
# Tier 1 mild, 2 moderate, 3 strong, 4 slur. Tier 4 is deliberately EMPTY: a
# slur list in a public repository is a liability with no upside, and AF-06
# needs no enumeration to work. The clause text describes the category and the
# triad recognises instances — which it does better than a list would, because a
# list cannot tell quotation from endorsement.
# --------------------------------------------------------------------------
PROFANITY = [
    {"term": "fuck", "tier": 3, "clause": "AF-01",
     "variants": ["f*ck", "f**k", "fck", "fuk", "f u c k", "phuck", "fuq"]},
    {"term": "shit", "tier": 2, "clause": "AF-01",
     "variants": ["sh*t", "sh1t", "shyt", "s h i t"]},
    {"term": "bitch", "tier": 2, "clause": "AF-01",
     "variants": ["b*tch", "b1tch", "biatch"]},
    {"term": "asshole", "tier": 2, "clause": "AF-01", "variants": ["a**hole", "arsehole"]},
    {"term": "bastard", "tier": 2, "clause": "AF-01", "variants": []},
    {"term": "dick", "tier": 2, "clause": "AF-01", "variants": ["d*ck"]},
    {"term": "piss", "tier": 1, "clause": "AF-01", "variants": ["p*ss"]},
    {"term": "damn", "tier": 1, "clause": "AF-01", "variants": ["d*mn", "dam"]},
    {"term": "hell", "tier": 1, "clause": "AF-01", "variants": []},
    {"term": "crap", "tier": 1, "clause": "AF-01", "variants": []},
    {"term": "bloody", "tier": 1, "clause": "AF-01", "variants": []},
    {"term": "bollocks", "tier": 2, "clause": "AF-01", "variants": []},
]

ATTRIBUTION_CUES = {
    "_note": NEVER_A_VERDICT,
    "_purpose": (
        "Quotation of a third party is a documented exemption in several "
        "clauses. Detecting attribution lets the ADVOCATE argue it rather than "
        "invent it."
    ),
    "reporting_verbs": [
        "said", "says", "wrote", "posted", "tweeted", "claimed", "told",
        "argued", "insisted", "replied", "commented", "stated", "remarked",
        "put it", "described it as",
    ],
    "quote_markers": [
        "quote", "end quote", "unquote", "in his words", "in her words",
        "in their words", "according to", "as they put it", "and I quote",
    ],
    # Quoting AND condemning is a materially stronger exemption than quoting
    # alone. These are surfaced to the ADVOCATE as a separate signal.
    "condemnation_markers": [
        "which is disgusting", "which is appalling", "that is unacceptable",
        "i strongly disagree", "this is exactly the problem", "obviously wrong",
        "should never have said", "i want to be clear that",
    ],
    "sentence_terminators": [".", "?", "!"],
    "max_quote_span_ms": 20000,
}

SPONSORSHIP_CUES = {
    "_note": NEVER_A_VERDICT,
    "_rule": (
        "explicit OR code_offers present in transcript, AND no "
        "disclosure_cues.description_markers present in the description -> raise "
        "META-01. `soft` alone is advisory only."
    ),
    "explicit": [
        "this video is sponsored by", "this episode is sponsored by",
        "sponsored by", "brought to you by", "paid partnership",
        "in partnership with", "thanks to * for sponsoring",
        "for sponsoring this video", "our sponsor",
    ],
    "code_offers": [
        "use code", "use my code", "with my code", "discount code",
        "promo code", "coupon code", "link in the description",
        "link in the bio", "first * customers", "percent off with",
    ],
    "affiliate": [
        "affiliate link", "affiliate links", "i earn a commission",
        "commission from", "at no extra cost to you", "as an amazon associate",
    ],
    "soft": [
        "check them out at", "sign up at", "head over to", "go to * dot com",
    ],
}

DISCLOSURE_CUES = {
    "_note": NEVER_A_VERDICT,
    "_purpose": (
        "The absence of these in a description, when sponsorship cues are "
        "present in the transcript, is what makes META-01 fire."
    ),
    "description_markers": [
        "paid promotion", "includes paid promotion", "contains paid promotion",
        "#ad", "ad:", "advertisement", "sponsored", "sponsorship",
        "affiliate disclosure", "affiliate link", "commission",
        "paid partnership", "gifted",
    ],
    "spoken_markers": [
        "this is a paid promotion", "this video contains paid promotion",
        "this is sponsored content", "disclosure",
    ],
    "platform_setting": (
        "YouTube Studio provides a 'contains paid promotion' checkbox that "
        "surfaces a disclosure for the first ten seconds. A metadata pass cannot "
        "observe that setting, so a META-01 finding is always rebuttable."
    ),
}

EDSA_FRAMING_CUES = {
    "_note": NEVER_A_VERDICT,
    "_purpose": (
        "This is the file that most directly lowers the false-positive rate. "
        "The ADVOCATE may only argue exemptions the clause documents; these cues "
        "tell it which exemption is plausibly in play so it can check the clause "
        "for it rather than inventing one."
    ),
    "educational": [
        "in this lesson", "let me explain", "the reason this happens",
        "historically", "the research shows", "step one", "for context",
        "what this means is", "to understand why", "in other words",
    ],
    "documentary": [
        "this footage shows", "survivors described", "at the time", "archival",
        "the report found", "years later", "we spoke to",
    ],
    "news": [
        "reported earlier today", "according to officials", "breaking",
        "the statement said", "sources confirm", "a spokesperson",
    ],
    "scientific": [
        "the study", "peer reviewed", "the data indicates", "control group",
        "sample size", "the mechanism is",
    ],
    "artistic": [
        "in the film", "the character", "scripted", "performance",
        "the lyrics", "in the novel",
    ],
    # Distance matters here, not just presence. A warning three minutes after
    # the demonstration is not a warning.
    "harm_reduction": [
        "do not attempt", "do not try this", "this is dangerous", "warning",
        "never do this", "please don't", "if you or someone you know",
        "seek professional", "i am not recommending", "for your safety",
    ],
    "harm_reduction_max_distance_ms": 15000,
}

SENSITIVE_EVENT_PATTERNS = {
    "_note": NEVER_A_VERDICT,
    "_purpose": (
        "Patterns rather than a list of events. A hardcoded event list goes "
        "stale the week after it is written and is a maintenance trap."
    ),
    "casualty_patterns": [
        r"\b\d+\s+(people\s+)?(were\s+)?(killed|dead|died|injured|wounded|lost)\b",
        r"\bdeath toll\b",
        r"\bcasualt(y|ies)\b",
        r"\b(none|no one|nobody)\s+(survived|made it)\b",
        r"\bdidn'?t\s+(come|make it)\s+back\b",
    ],
    "event_nouns": [
        "shooting", "bombing", "attack", "massacre", "earthquake", "tsunami",
        "hurricane", "wildfire", "crash", "collision", "derailment", "outbreak",
        "pandemic", "hostage", "evacuation", "avalanche", "landslide",
        "explosion", "collapse", "flood",
    ],
    "recency_markers": [
        "yesterday", "this morning", "last night", "this week", "breaking",
        "just in", "developing", "earlier today", "overnight",
    ],
    "rule": (
        "event_noun AND (casualty_pattern OR recency_marker) -> raise AF-10 "
        "retrieval priority"
    ),
}

OCR_ROLE_PATTERNS = {
    "_note": NEVER_A_VERDICT,
    "_purpose": (
        "On-screen text means different things depending on where it sits and "
        "how long it persists. A burned-in caption tracking speech is not the "
        "same finding as meme text, and a persistent corner mark is a "
        "third-party footage cue rather than a language issue."
    ),
    "roles": {
        "burned_in_caption": {
            "y_range": [0.72, 1.0], "min_persist_ms": 800, "tracks_speech": True,
            "note": "duplicate of the audio; dedupe against the speech finding",
        },
        "lower_third": {
            "y_range": [0.62, 0.92], "min_persist_ms": 2000, "max_words": 8,
        },
        "meme_text": {
            "y_range": [0.0, 0.35], "font_height_ratio": 0.06,
            "max_persist_ms": 4000,
        },
        "watermark": {
            "corner": True, "persist_ratio": 0.6,
            "note": "third-party footage cue -> COPY-01, not AF-01",
        },
        "chyron": {"y_range": [0.8, 1.0], "full_width": True,
                   "note": "news framing signal; supports an EDSA defence"},
    },
    "dedupe_rule": (
        "Text persisting across N frames is ONE finding, not N. Merge by "
        "normalised string plus overlapping span."
    ),
}

DANGEROUS_ACTS_CUES = {
    "_note": NEVER_A_VERDICT,
    "imitable_verbs": [
        "bypass", "short out", "disable the", "override", "remove the guard",
        "jump the", "hotwire", "defeat the", "you can just",
    ],
    "risk_nouns": [
        "safety cutout", "interlock", "breaker", "live wire", "mains",
        "harness", "belay", "anchor", "rope", "airway", "dosage",
    ],
    "unprotected_markers": [
        "no rope", "no harness", "without a spotter", "free solo", "no gloves",
        "no mask", "barehanded", "unsupervised",
    ],
    "rule": "imitable_verb NEAR risk_noun, or unprotected_marker -> raise AF-05",
}

REGULATED_GOODS_CUES = {
    "_note": NEVER_A_VERDICT,
    "alcohol": ["whisky", "whiskey", "vodka", "beer", "wine", "tequila", "gin",
                "rum", "shots", "pint", "cracked open a"],
    "tobacco": ["cigarette", "cigar", "vape", "vaping", "e-cigarette", "nicotine",
                "juul", "smoking a"],
    "gambling": ["casino", "betting", "odds on", "place a bet", "slots",
                 "roulette", "sportsbook", "loot box", "wager"],
    "firearms": ["rifle", "handgun", "pistol", "shotgun", "magazine", "caliber",
                 "calibre", "muzzle", "trigger", "ammo", "ammunition"],
    "drugs": ["cocaine", "heroin", "meth", "mdma", "ketamine", "edible",
              "dab", "bong"],
    "clause_routing": {
        "alcohol": "AF-12", "tobacco": "AF-12", "gambling": "AF-14",
        "firearms": "AF-08", "drugs": "AF-07",
    },
    "_caveat": (
        "Surface terms only. 'Shot' in basketball, 'trigger' in software, "
        "'magazine' in publishing. This is exactly why a hit is not a finding."
    ),
}

CLICKBAIT_PATTERNS = {
    "_note": NEVER_A_VERDICT,
    "_purpose": "Title/content mismatch signals for the metadata agent.",
    "title_patterns": [
        r"\bYOU WON'?T BELIEVE\b", r"\bGONE WRONG\b", r"\bSHOCKING\b",
        r"\b(\d+)\s+THINGS\b", r"\bTHE TRUTH ABOUT\b", r"\bEXPOSED\b",
        r"!{2,}", r"\?{2,}",
    ],
    "uppercase_ratio_warn": 0.7,
    "min_letters_for_case_check": 12,
    "mismatch_rule": (
        "Title asserts a claim the transcript never supports -> advisory. "
        "Requires semantic comparison, not pattern matching alone."
    ),
}


def write(name: str, payload) -> Path:
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # JSONL, one term per line, with the discipline note as the first record.
    lines = [json.dumps({"_note": NEVER_A_VERDICT, "_tiers": {
        "1": "mild", "2": "moderate", "3": "strong",
        "4": "slur — deliberately not enumerated; AF-06 clause text handles it",
    }})]
    for entry in PROFANITY:
        lines.append(json.dumps({**entry, "note": {1: "mild", 2: "moderate", 3: "strong"}[entry["tier"]]}))
    path = OUT / "profanity.tiered.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(path)

    written.append(write("attribution_cues.json", ATTRIBUTION_CUES))
    written.append(write("sponsorship_cues.json", SPONSORSHIP_CUES))
    written.append(write("disclosure_cues.json", DISCLOSURE_CUES))
    written.append(write("edsa_framing_cues.json", EDSA_FRAMING_CUES))
    written.append(write("sensitive_event_patterns.json", SENSITIVE_EVENT_PATTERNS))
    written.append(write("ocr_role_patterns.json", OCR_ROLE_PATTERNS))
    written.append(write("dangerous_acts_cues.json", DANGEROUS_ACTS_CUES))
    written.append(write("regulated_goods_cues.json", REGULATED_GOODS_CUES))
    written.append(write("clickbait_patterns.json", CLICKBAIT_PATTERNS))

    for path in written:
        print(f"  {path}  ({path.stat().st_size:,} bytes)")
    print(f"wrote {len(written)} lexicons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
