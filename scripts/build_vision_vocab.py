"""Author the A03 vision vocabulary.

Reference mappings, not training data. Two jobs, and the second is the one that
matters:

**Normalisation.** A vision model will call the same object a handgun, a
pistol, a firearm and a weapon across four adjacent frames. Without a canonical
form those are four objects, temporal deduplication never fires, and one knife
visible for eight seconds becomes two hundred and forty observations.

**Containment.** The vocabulary is CLOSED. A label the model invents is
rejected rather than passed through, which is what stops a verdict entering the
pipeline disguised as an observation. A model asked for objects will happily
return "graphic violence" or "inappropriate content for minors" — those are
A11's conclusions, arriving three agents early with no clause attached and no
advocate to contest them. The blocklist below is the tripwire.

    python scripts/build_vision_vocab.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("data/vision")

NOT_A_VERDICT = (
    "Labels here are OBSERVATIONS. A03 reports what is visibly present and "
    "never what it means. 'knife' is an observation; 'graphic violence' is a "
    "conclusion belonging to A11, which has the clause text and an advocate."
)

# Verdict-shaped tokens. A label containing any of these is rejected outright
# rather than normalised, because there is no observation underneath to
# recover — the model has skipped to the answer.
JUDGMENT_BLOCKLIST = [
    "violation", "violates", "unsafe", "inappropriate", "graphic",
    "obscene", "illegal", "prohibited", "banned", "nsfw", "explicit",
    "disturbing", "offensive", "harmful", "limited ads", "demonetiz",
    "not advertiser", "advertiser friendly", "policy", "should be removed",
    "age restricted", "age-restricted", "shocking",
]


def vocab(category: str, note: str, canon: dict[str, list[str]]) -> dict:
    return {
        "_category": category,
        "_note": NOT_A_VERDICT,
        "_purpose": note,
        "labels": sorted(canon),
        "synonyms": {
            alias: canonical
            for canonical, aliases in canon.items()
            for alias in aliases
        },
    }


FILES = {
    "objects.json": vocab(
        "person",
        "Presence and count. Never identity, never protected characteristics.",
        {
            "person": ["human", "man", "woman", "individual", "figure", "adult"],
            "child": ["kid", "minor", "toddler", "infant", "baby"],
            "crowd": ["group of people", "audience", "crowd of people", "many people"],
            "face": ["human face", "close-up face"],
            "hand": ["hands", "human hand"],
        },
    ),
    "weapons.json": vocab(
        "weapon",
        (
            "Presence only. 'weapon visible' is the observation; whether it "
            "breaches AF-08 or AF-02 depends on the environment, the handling "
            "and the framing, none of which a single frame settles."
        ),
        {
            "knife": ["blade", "dagger", "machete", "kitchen knife", "combat knife"],
            "gun": ["handgun", "pistol", "revolver", "firearm", "sidearm"],
            "rifle": ["assault rifle", "shotgun", "long gun", "carbine"],
            "sword": ["katana", "sabre", "saber", "broadsword"],
            "explosive": ["bomb", "grenade", "dynamite", "ied"],
            "arrow": ["bow and arrow", "crossbow", "archery bow"],
            "ammunition": ["bullets", "magazine", "cartridge", "ammo"],
        },
    ),
    "injury.json": vocab(
        "injury",
        (
            "Blood and wounds. Deliberately separate from weapons: an injury "
            "with no weapon is AF-04 rather than AF-02, and conflating them "
            "sends the wrong clause to the adjudicator."
        ),
        {
            "blood": ["bleeding", "bloodstain", "blood spatter"],
            "wound": ["open wound", "laceration", "cut", "gash"],
            "injury": ["injured person", "bodily injury", "hurt person"],
            "bandage": ["dressing", "first aid", "gauze"],
        },
    ),
    "fire.json": vocab(
        "fire",
        "Fire, smoke and explosion. Common in cooking and pyrotechnics alike.",
        {
            "fire": ["flames", "flame", "burning", "bonfire"],
            "smoke": ["smoke cloud", "smoking"],
            "explosion": ["blast", "fireball", "detonation"],
            "firework": ["fireworks", "pyrotechnics", "sparkler"],
        },
    ),
    "vehicles.json": vocab(
        "vehicle",
        "Vehicles, including emergency and military, which carry news context.",
        {
            "car": ["automobile", "sedan", "suv", "hatchback"],
            "motorcycle": ["motorbike", "bike", "scooter"],
            "police_car": ["police vehicle", "patrol car", "squad car"],
            "ambulance": ["emergency vehicle", "paramedic vehicle"],
            "fire_truck": ["fire engine"],
            "tank": ["armoured vehicle", "armored vehicle", "military tank"],
            "boat": ["ship", "vessel", "yacht", "dinghy"],
            "aircraft": ["airplane", "plane", "helicopter", "jet", "drone"],
            "truck": ["lorry", "van", "pickup"],
        },
    ),
    "animals.json": vocab(
        "animal",
        "Animal presence. Distress or harm is A04/AF-04, not this file.",
        {
            "dog": ["puppy", "canine"],
            "cat": ["kitten", "feline"],
            "snake": ["serpent", "python", "cobra"],
            "bear": ["grizzly", "black bear"],
            "horse": ["pony", "stallion"],
            "big_cat": ["lion", "tiger", "leopard", "jaguar", "cheetah"],
            "bird": ["eagle", "parrot", "hawk"],
            "livestock": ["cow", "sheep", "pig", "goat", "cattle"],
        },
    ),
    "dangerous_actions.json": vocab(
        "activity",
        (
            "Activity labels, phrased as observations. 'cliff diving' describes "
            "what is happening; 'dangerous stunt' is already an assessment, and "
            "AF-05 turns on whether a warning against imitation is present, "
            "which a frame cannot show."
        ),
        {
            "climbing_unroped": ["free solo", "climbing without rope", "unroped climb"],
            "roof_jump": ["rooftop jump", "parkour jump", "building jump"],
            "cliff_diving": ["cliff jump", "cliff dive", "high dive"],
            "handling_firearm": ["holding gun", "aiming firearm", "person with gun"],
            "handling_knife": ["holding knife", "person with knife"],
            "street_racing": ["racing on road", "illegal race", "drag race"],
            "fire_stunt": ["fire breathing", "playing with fire", "fire stunt"],
            "height_exposure": ["standing on ledge", "on rooftop edge", "near cliff edge"],
        },
    ),
    "substances.json": vocab(
        "substance",
        (
            "Objects, not use. A syringe appears in medical, diabetic and "
            "veterinary content far more often than in drug content, which is "
            "exactly why this is evidence rather than a finding."
        ),
        {
            "cigarette": ["smoking", "cigar", "vape", "e-cigarette", "vaping device"],
            "pills": ["tablets", "medication", "capsules", "pill bottle"],
            "syringe": ["needle", "injection", "hypodermic"],
            "powder": ["white powder", "powder line"],
            "smoking_pipe": ["bong", "pipe", "water pipe"],
            "rolled_cigarette": ["joint", "blunt", "rolled paper"],
        },
    ),
    "alcohol.json": vocab(
        "alcohol",
        "Containers and glassware. Presence is not consumption.",
        {
            "beer": ["beer bottle", "beer can", "pint glass"],
            "wine": ["wine glass", "wine bottle"],
            "spirits": ["whiskey bottle", "whisky bottle", "vodka bottle", "liquor bottle"],
            "cocktail": ["mixed drink", "cocktail glass"],
            "bar": ["bar counter", "pub interior"],
        },
    ),
    "money.json": vocab(
        "money",
        "Financial props. Strong corroboration for a financial-claim event.",
        {
            "cash": ["banknotes", "money", "currency", "bills", "stack of cash"],
            "credit_card": ["bank card", "payment card"],
            "cheque": ["check", "bank cheque"],
            "lottery_ticket": ["scratch card", "lottery"],
            "casino_chips": ["poker chips", "gambling chips"],
        },
    ),
    "brands.json": vocab(
        "brand",
        (
            "Visible marks. A logo is a weak corroborating signal for COPY-01, "
            "nothing more — a laptop in shot is not infringement."
        ),
        {
            "apple": ["apple logo", "macbook logo"],
            "netflix": ["netflix logo"],
            "disney": ["disney logo", "mickey mouse"],
            "marvel": ["marvel logo"],
            "nike": ["nike swoosh", "nike logo"],
            "tesla": ["tesla logo"],
            "youtube": ["youtube logo", "play button logo"],
            "broadcast_bug": ["station logo", "channel bug", "network watermark"],
        },
    ),
    "scene_labels.json": vocab(
        "scene",
        (
            "Setting. Bedroom plus swimwear reads differently from beach plus "
            "swimwear, and the setting is what carries that difference."
        ),
        {
            "indoor": ["inside", "interior", "room"],
            "outdoor": ["outside", "exterior"],
            "bedroom": ["bed", "bedroom scene"],
            "bathroom": ["shower", "bathtub"],
            "kitchen": ["cooking area"],
            "studio": ["recording studio", "set"],
            "street": ["road", "pavement", "sidewalk"],
            "beach": ["seaside", "shore", "pool"],
            "mountain": ["cliff", "summit", "alpine"],
            "vehicle_interior": ["car interior", "dashboard view"],
            "crowd_venue": ["stadium", "concert", "arena"],
        },
    ),
    "attire.json": vocab(
        "attire",
        (
            "Clothing state as an OBSERVATION. This file is the one most likely "
            "to be misused, so: 'swimwear' is a garment, and a garment on a "
            "beach is not a finding. AF-03 turns on sexualised framing, which "
            "is a judgement A11 makes with the clause in front of it."
        ),
        {
            "swimwear": ["bikini", "swimsuit", "trunks", "bathing suit"],
            "underwear": ["lingerie", "undergarment"],
            "shirtless": ["bare chest", "topless male", "no shirt"],
            "formal_wear": ["suit", "dress", "gown"],
            "costume": ["cosplay", "fancy dress", "uniform"],
        },
    ),
    "emotion.json": vocab(
        "emotion",
        (
            "Supporting evidence only, and the weakest signal A03 produces. "
            "Facial-expression inference is unreliable across cultures, ages "
            "and camera angles, so these carry a confidence ceiling and never "
            "corroborate on their own."
        ),
        {
            "happy": ["smiling", "joy", "laughing"],
            "neutral": ["calm", "expressionless"],
            "fear": ["afraid", "scared", "frightened"],
            "anger": ["angry", "furious", "enraged"],
            "sadness": ["sad", "crying", "upset"],
            "surprise": ["shocked", "astonished"],
        },
    ),
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    for name, payload in FILES.items():
        path = OUT / name
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(
            f"  {path.name:<24} {payload['_category']:<10} "
            f"{len(payload['labels']):>2} labels  "
            f"{len(payload['synonyms']):>3} synonyms"
        )

    blocklist = {
        "_purpose": (
            "A label containing any of these is REJECTED, not normalised. There "
            "is no observation underneath to recover: the model has skipped "
            "past what it can see to what it thinks the answer is."
        ),
        "terms": sorted(JUDGMENT_BLOCKLIST),
    }
    path = OUT / "judgment_blocklist.json"
    path.write_text(json.dumps(blocklist, indent=2) + "\n", encoding="utf-8")
    print(f"  {path.name:<24} {'blocklist':<10} {len(JUDGMENT_BLOCKLIST):>2} terms")

    print(f"\nwrote {len(FILES) + 1} vision vocabulary files to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
