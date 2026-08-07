"""Author the policy corpus.

Every clause here is a **structured restatement in our own words** of publicly
published guidance, recorded with the page it was read from and the date it was
read. It is not a copy of anyone's guidelines and does not claim to be
authoritative — it is the rule set PREFLIGHT lints against, and every finding
cites the clause it was judged under so a human can check the machine.

Restating rather than copying is also the better engineering choice: these are
chunked at heading level and embedded, and prose written for retrieval
outperforms prose written for a help centre.

Two sections carry disproportionate weight:

**Documented exemptions** is what the ADVOCATE argues from. The advocate prompt
forbids inventing exemptions, so a thin exemptions section leaves it nothing to
work with and the false-positive rate stays high.

**Distinguishing signals** is what the ADJUDICATOR rules with. Retrieval
routinely surfaces three neighbouring clauses for one window; the discriminating
language is what lets it pick correctly rather than plausibly.

    python scripts/build_corpus.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUT = Path("data/policy")
FETCHED = "2026-08-05"
VERSION = "2026-08"

# Sources actually read, not assumed. Verified reachable 2026-08-05.
SRC_ADVERTISER = "https://support.google.com/youtube/answer/6162278"
SRC_DISCLOSURE = "https://support.google.com/youtube/answer/154235"
SRC_CONTENT_ID = "https://support.google.com/youtube/answer/2797370"
SRC_WCAG = "https://www.w3.org/WAI/WCAG22/Understanding/three-flashes-or-below-threshold.html"

DERIVATION = "structured restatement in own words; not a verbatim copy"

BASE_EXEMPTIONS = [
    "Educational, documentary, scientific or artistic (EDSA) framing where the "
    "context is evident from the content itself rather than asserted in metadata",
    "News reporting on a matter of public interest",
    "Non-graphic verbal reference without depiction",
    "Clearly fictional or scripted context, signposted as such",
    "Quotation of a third party, particularly where it is also condemned",
]

SRC_HOUSE = "PREFLIGHT engineering ruleset"
DERIVATION_HOUSE = (
    "PREFLIGHT's own production rule, not a restatement of any platform policy"
)

# House exemptions. Deliberately thinner than the policy ones: a measurement
# has fewer honest defences than a judgement. You cannot argue that -6 LUFS is
# contextually appropriate the way you can argue that a quoted slur is
# educational.
HOUSE_EXEMPTIONS = [
    "Deliberate artistic choice where the content itself makes the intent "
    "evident rather than the metadata asserting it",
    "A measurement taken over a span too short to characterise the file",
]


def house(
    *,
    file: str,
    clause_id: str,
    title: str,
    severity: str,
    scope: str,
    green: list[str],
    yellow: list[str],
    red: list[str],
    signals: list[str],
    fix: str,
    span_note: str,
) -> dict:
    """A PREFLIGHT house rule in the same shape as a policy clause.

    These are the tool's own engineering thresholds — loudness targets, caption
    availability, tag stuffing — and they are NOT platform policy. They carry a
    different source and a different derivation so nothing in the report can
    imply that a -14 LUFS target was published by YouTube. The distinction is
    load-bearing: this project's entire claim is that a finding cites the
    clause it was judged under, and a citation is worthless if the reader
    cannot tell whose rule it is.
    """
    return {
        "file": file,
        "id": clause_id,
        "title": title,
        "severity": severity,
        "source": SRC_HOUSE,
        "derivation": DERIVATION_HOUSE,
        "scope": scope,
        "green": green,
        "yellow": yellow,
        "red": red,
        "exemptions": HOUSE_EXEMPTIONS,
        "signals": signals,
        "fix": fix,
        "span_note": span_note,
    }


HOUSE_RULES: list[dict] = [
    house(
        file="ACC-02_captions.md",
        clause_id="ACC-02",
        title="Caption availability",
        severity="LIMITING",
        scope=(
            "Whether the file ships a timed-text track. File-scoped: this is a "
            "property of the container, not of any moment in the video."
        ),
        green=["A caption track is present and covers the spoken content"],
        yellow=[
            "No caption track, but word-level timings exist in this run and "
            "captions can be emitted directly from them",
        ],
        red=[
            "No caption track and no transcript available to generate one",
            "Audio containing technical vocabulary, strong accents or heavy "
            "background noise, where automatic captions are least reliable",
        ],
        signals=[
            "vs ACC-01 (photosensitive content): ACC-01 is a harm to a viewer "
            "with a medical condition and is scoped to a span. This is an "
            "access gap scoped to the whole file. They were once the same "
            "clause id, which meant a caption finding cited a seizure-risk "
            "policy.",
        ],
        fix="NONE",
        span_note="File-scoped. Start 0, end at duration.",
    ),
    house(
        file="ACC-03_speech_rate.md",
        clause_id="ACC-03",
        title="Speech rate",
        severity="ADVISORY",
        scope=(
            "Sustained words per minute over a rolling window. Fast delivery "
            "reduces comprehension for non-native speakers and for anyone "
            "relying on automatic captions, which degrade as rate rises."
        ),
        green=["Sustained rate within a comfortable listening range"],
        yellow=["Sustained rate well above conversational pace"],
        red=["Rate high enough that automatic captions are unlikely to track it"],
        signals=[
            "Advisory only. A fast talker is not a policy problem, and this "
            "clause exists to inform rather than to gate.",
        ],
        fix="NONE",
        span_note="The window that exceeded the threshold, not the whole file.",
    ),
    house(
        file="ACC-04_chapters.md",
        clause_id="ACC-04",
        title="Chapter markers",
        severity="ADVISORY",
        scope=(
            "Whether a long video carries chapter markers. Navigation aid; "
            "matters more the longer the runtime."
        ),
        green=["Chapters present, or a runtime short enough not to need them"],
        yellow=["Long runtime with no chapter markers"],
        red=[],
        signals=[
            "Advisory. Absence of chapters is never a monetization risk and "
            "must never be scored as one.",
        ],
        fix="NONE",
        span_note="File-scoped.",
    ),
    house(
        file="AUD-01_loudness.md",
        clause_id="AUD-01",
        title="Loudness normalisation",
        severity="ADVISORY",
        scope=(
            "Integrated loudness against the platform's normalisation target, "
            "measured to EBU R128. Content far from target is turned down on "
            "playback, and a mix built loud loses its dynamics in the process."
        ),
        green=["Integrated loudness within tolerance of the target"],
        yellow=["Loudness outside tolerance in either direction"],
        red=["Loudness far enough from target that playback normalisation will "
             "materially change the mix"],
        signals=[
            "vs AUD-02 (clipping): loudness is where the whole file sits; "
            "clipping is samples destroyed at individual peaks. A quiet file "
            "can clip and a loud one need not.",
        ],
        fix="REPLACE_AUDIO",
        span_note="File-scoped — an integrated measurement has no span.",
    ),
    house(
        file="AUD-02_clipping.md",
        clause_id="AUD-02",
        title="Clipping",
        severity="LIMITING",
        scope=(
            "Samples at or beyond full scale. Clipping is destroyed signal: the "
            "waveform above the ceiling is gone and no later processing "
            "restores it."
        ),
        green=["No samples at full scale"],
        yellow=["Isolated clipped samples, likely inaudible"],
        red=["Sustained clipping across a span, audible as distortion"],
        signals=[
            "vs AUD-01 (loudness): clipping is a defect in the recording. "
            "Turning the file down does not repair it.",
        ],
        fix="REPLACE_AUDIO",
        span_note="The clipped region, padded to the nearest zero crossing.",
    ),
    house(
        file="AUD-03_dead_air.md",
        clause_id="AUD-03",
        title="Dead air",
        severity="ADVISORY",
        scope=(
            "A sustained span with RMS below the noise floor. Usually an "
            "editing error — a muted track, a dropped clip, a gap left in the "
            "timeline."
        ),
        green=["No silent span longer than a natural pause"],
        yellow=["A silent span long enough to read as a mistake"],
        red=["Extended silence where content was clearly intended"],
        signals=[
            "A deliberate pause for effect is short. This clause is scoped to "
            "spans long enough that a viewer checks whether their audio broke.",
        ],
        fix="CUT",
        span_note="The silent span itself.",
    ),
    house(
        file="AUD-04_channel_balance.md",
        clause_id="AUD-04",
        title="Channel balance",
        severity="LIMITING",
        scope=(
            "Per-channel RMS difference. A recording where one channel sits far "
            "below the other is a dead microphone — real, common, and expensive, "
            "because nobody notices until the video is live and half the "
            "audience is hearing silence."
        ),
        green=["Channels within a few dB of each other, or genuinely mono"],
        yellow=["Noticeable imbalance between channels"],
        red=["One channel effectively silent — a dead mic"],
        signals=[
            "Intentional hard-panning exists but is rare outside music, and a "
            "channel that is silent for the whole runtime is not a pan.",
        ],
        fix="REPLACE_AUDIO",
        span_note="File-scoped — measured across the whole track.",
    ),
    house(
        file="AUD-05_phase.md",
        clause_id="AUD-05",
        title="Mono compatibility",
        severity="LIMITING",
        scope=(
            "Correlation between the left and right channels. Content recorded "
            "or mixed out of phase collapses toward silence when the two "
            "channels are summed — which happens on a phone speaker, a laptop, "
            "a single earbud, or any playback system that is not true stereo, "
            "which is most of a video platform's audience most of the time."
        ),
        green=["Channels positively correlated — safe when summed to mono"],
        yellow=["Correlation near zero — no consistent phase relationship"],
        red=["Channels negatively correlated — audibly hollow or silent in mono"],
        signals=[
            "vs AUD-04 (channel balance): balance is a LEVEL difference between "
            "channels; phase is a TIMING/POLARITY relationship. A file can fail "
            "either independently of the other — balanced channels can still be "
            "out of phase, and a dead channel has no phase relationship to measure.",
        ],
        fix="NONE",
        span_note="File-scoped — measured across the whole track.",
    ),
    house(
        file="VID-01_black_frames.md",
        clause_id="VID-01",
        title="Black frames",
        severity="ADVISORY",
        scope=(
            "A sustained span with no picture — a black gap from a bad edit, a "
            "missing clip, or a render that failed to composite."
        ),
        green=["No sustained black span outside a deliberate transition"],
        yellow=["A black span long enough to read as a mistake"],
        red=["Extended black where content was clearly intended"],
        signals=[
            "vs AUD-03 (dead air): this is the visual sibling of the same class "
            "of editing accident, measured on the picture rather than the sound.",
        ],
        fix="NONE",
        span_note="The black span itself.",
    ),
    house(
        file="DISC-01_credential_on_screen.md",
        clause_id="DISC-01",
        title="Credential visible on screen",
        severity="LIMITING",
        scope=(
            "An API key, access token, private key or labelled password "
            "legible in the picture — a terminal left open behind a demo, an "
            "editor tab, a .env file scrolled past during a screen recording."
        ),
        green=["No credential-shaped text anywhere on screen"],
        yellow=["A labelled secret whose value is partly obscured"],
        red=["A complete vendor-issued key or private key block legible"],
        signals=[
            "Matched on published vendor prefixes with a length floor rather "
            "than on entropy: entropy flags base64 thumbnails and minified "
            "JavaScript, and a detector that cries wolf is muted before it "
            "ever catches the real one.",
            "The consequence is not a policy one. A leaked key is charged to "
            "the creator's account within hours of the upload going public, "
            "and unlike a policy strike there is no appeal.",
        ],
        fix="BLUR_REGION",
        span_note="The span the text was legible for, merged across frames.",
    ),
    house(
        file="DISC-02_personal_data_on_screen.md",
        clause_id="DISC-02",
        title="Personal data visible on screen",
        severity="ADVISORY",
        scope=(
            "A phone number, email address or payment card legible in the "
            "picture — a notification banner, a browser autofill, a document "
            "left open in shot."
        ),
        green=["No personal data legible on screen"],
        yellow=["An email or phone number visible briefly"],
        red=["A payment card number legible, checksum-valid"],
        signals=[
            "Card numbers are Luhn-checked. Sixteen digits that fail the "
            "checksum are an order number, not a card, and reporting them as "
            "one teaches a creator to ignore this finding.",
            "Phone numbers require a separator or a country code: a bare run "
            "of ten digits is far more often a timestamp or a score.",
        ],
        fix="BLUR_REGION",
        span_note="The span the text was legible for, merged across frames.",
    ),
    house(
        file="VID-02_frozen_frames.md",
        clause_id="VID-02",
        title="Frozen frames",
        severity="LIMITING",
        scope=(
            "A sustained span with no motion between frames — a stalled screen "
            "recording, a render that silently dropped frames, or a capture "
            "device that disconnected mid-record while audio kept rolling."
        ),
        green=["No sustained freeze outside a deliberate still frame"],
        yellow=["A freeze long enough to read as a technical fault"],
        red=["Extended freeze — the upload is functionally broken for its "
             "duration"],
        signals=[
            "A deliberate still frame under narration is short and usually "
            "intentional; this clause is scoped to freezes long enough that a "
            "viewer checks whether playback stalled.",
        ],
        fix="NONE",
        span_note="The frozen span itself.",
    ),
    house(
        file="META-02_description.md",
        clause_id="META-02",
        title="Description depth",
        severity="ADVISORY",
        scope="Length and substance of the description field.",
        green=["A description that describes the content"],
        yellow=["A description too short to describe anything"],
        red=[],
        signals=["Advisory. PREFLIGHT verifies metadata and never writes it."],
        fix="NONE",
        span_note="File-scoped.",
    ),
    house(
        file="META-03_title_length.md",
        clause_id="META-03",
        title="Title length",
        severity="ADVISORY",
        scope="Title length against what survives truncation in a results list.",
        green=["A title that reads fully in search and on mobile"],
        yellow=["A title long enough to be truncated where it matters"],
        red=[],
        signals=["Advisory. Never a monetization risk."],
        fix="NONE",
        span_note="File-scoped.",
    ),
    house(
        file="META-04_title_presentation.md",
        clause_id="META-04",
        title="Title presentation",
        severity="ADVISORY",
        scope=(
            "Presentation of the title — sustained capitals, runs of "
            "punctuation, and other patterns associated with clickbait."
        ),
        green=["Ordinary sentence presentation"],
        yellow=["Sustained capitals or repeated punctuation"],
        red=[],
        signals=[
            "vs META-01 (paid promotion): META-01 is a disclosure obligation "
            "with real consequences. This is presentation, and advisory.",
        ],
        fix="REFRAME",
        span_note="File-scoped.",
    ),
    house(
        file="META-05_tags.md",
        clause_id="META-05",
        title="Tag stuffing",
        severity="ADVISORY",
        scope=(
            "Tag count, duplication, and tags with no support anywhere in the "
            "content."
        ),
        green=["A tag set that reflects the content"],
        yellow=["Excessive, duplicated, or unsupported tags"],
        red=[],
        signals=["Advisory. Reported so a human can decide."],
        fix="NONE",
        span_note="File-scoped.",
    ),
]

CLAUSES: list[dict] = [
    {
        "file": "AF-01_language.md",
        "id": "AF-01",
        "title": "Inappropriate language",
        "severity": "LIMITING",
        "source": SRC_ADVERTISER,
        "scope": (
            "Profanity and vulgar language wherever it appears — spoken, in "
            "on-screen text, in the title, in the thumbnail, or in metadata. "
            "Assessment turns on strength, frequency, and position rather than "
            "on mere presence. On-screen text is judged the same way as speech; "
            "obscured or bleeped terms are weighted lower than uncensored ones."
        ),
        "green": [
            "No profanity, or mild language used infrequently",
            "Strong profanity that is fully bleeped, muted or obscured",
            "Moderate profanity in a music or stand-up comedy performance",
        ],
        "yellow": [
            "Moderate profanity in the title or thumbnail",
            "Strong profanity used repeatedly through the body of the video",
            "Profanity in on-screen text held long enough to read",
        ],
        "red": [
            "Strong profanity in the title or thumbnail",
            "Slurs targeting a protected group, anywhere in the content",
        ],
        "exemptions": BASE_EXEMPTIONS
        + [
            "Musical performance and stand-up comedy, where the guidelines treat "
            "moderate profanity as expected within the form",
        ],
        "signals": [
            "vs AF-06 (hateful & derogatory): AF-01 covers vulgarity as such. The "
            "moment a term targets a protected attribute it leaves this clause "
            "entirely and becomes AF-06, where no strength threshold applies.",
            "vs AF-14 (incendiary and demeaning): AF-01 is about the word; AF-14 "
            "is about sustained hostility toward a person. Insults with no "
            "profanity still fall under AF-14.",
        ],
        "fix": "BLEEP",
        "span_note": (
            "Usually a single word, 300-900ms. Snap to word boundaries and pad, "
            "or the leading consonant survives the bleep and draws attention."
        ),
    },
    {
        "file": "AF-02_violence.md",
        "id": "AF-02",
        "title": "Violence",
        "severity": "DEMONETIZING",
        "source": SRC_ADVERTISER,
        "scope": (
            "Real or dramatised physical violence, injury, blood and its "
            "aftermath. What matters is whether the injury is the subject of the "
            "shot and whether the content dwells on it — not whether it appears "
            "at all. Gameplay violence is treated differently from real footage."
        ),
        "green": [
            "Violence implied rather than shown",
            "Unedited gameplay violence outside the opening seconds",
            "Mild violence with minimal blood, not held in focus",
            "Law enforcement or emergency response footage in a news frame",
        ],
        "yellow": [
            "Real injury shown briefly with visible blood",
            "Bodies with visible injury in an educational or news setting",
            "Graphic game violence in the thumbnail or the first fifteen seconds",
        ],
        "red": [
            "Graphic real injury, mutilation or death held in focus",
            "Violence glorified or presented approvingly",
            "Graphic footage produced by or promoting a violent organisation",
        ],
        "exemptions": BASE_EXEMPTIONS,
        "signals": [
            "vs AF-04 (shocking content): AF-02 requires an act of violence. "
            "Gore with no violent act — a surgical procedure, an accident "
            "aftermath, decomposition — is AF-04.",
            "vs AF-08 (firearms): a firearm present is AF-08. A firearm used "
            "against a person is AF-02, and AF-02 governs.",
            "vs AF-09 (controversial issues): AF-02 is depiction. Discussion of "
            "abuse or self-harm without depiction is AF-09.",
        ],
        "fix": "BLUR_REGION",
        "span_note": (
            "Typically 2-8s. Blur the region rather than cutting: cutting "
            "re-encodes and usually removes narrative the creator needs."
        ),
    },
    {
        "file": "AF-03_adult_content.md",
        "id": "AF-03",
        "title": "Adult content",
        "severity": "DEMONETIZING",
        "source": SRC_ADVERTISER,
        "scope": (
            "Sexually gratifying content, nudity, and framing that sexualises a "
            "subject — including camera emphasis on body parts independent of "
            "what is being worn."
        ),
        "green": [
            "Romance and kissing without sexual emphasis",
            "Obscured or non-sexual nudity in an artistic or medical context",
            "Non-graphic sex education",
            "Professional dance performance",
        ],
        "yellow": [
            "Sexualised framing in the thumbnail",
            "Educational sexual content with explicit description",
            "Classical artwork depicting discernible sexual activity",
        ],
        "red": [
            "Exposed sexual body parts",
            "Real or simulated sexual acts",
            "Content whose purpose is sexual gratification",
        ],
        "exemptions": BASE_EXEMPTIONS
        + [
            "Non-graphic sex education presented in an instructional register",
            "Musical performance where the guidelines allow wider latitude",
        ],
        "signals": [
            "vs AF-13 (kids and families): identical material is judged far more "
            "strictly when the format, characters or framing signal a child "
            "audience. AF-13 governs when both could apply.",
            "vs AF-04 (shocking content): AF-03 is sexual in purpose. Nudity that "
            "is disturbing rather than sexual — medical, forensic — is AF-04.",
        ],
        "fix": "BLUR_REGION",
        "span_note": "Variable. Prefer blur or reframe over cut.",
    },
    {
        "file": "AF-04_shocking_content.md",
        "id": "AF-04",
        "title": "Shocking content",
        "severity": "DEMONETIZING",
        "source": SRC_ADVERTISER,
        "scope": (
            "Content intended to disgust or disturb: gore divorced from narrative "
            "purpose, bodily fluids, graphic medical procedure, and graphic "
            "treatment of animals. The distinguishing question is whether the "
            "shock is the point."
        ),
        "green": [
            "Mild shocking content inside an educational or documentary frame",
            "Unsensational food preparation involving animals",
        ],
        "yellow": [
            "Graphic human or animal body parts shown unobscured but with context",
            "Detailed medical procedure",
        ],
        "red": [
            "Gore or bodily harm presented for its own sake",
            "Graphic mistreatment of animals",
        ],
        "exemptions": BASE_EXEMPTIONS
        + [
            "Scientific or medical instruction where the graphic element is the "
            "subject being taught",
        ],
        "signals": [
            "vs AF-02 (violence): AF-04 needs no violent act. If a person caused "
            "the harm to another person, it is AF-02.",
            "vs AF-09 (controversial issues): AF-04 is visual. AF-09 covers the "
            "same subject matter discussed rather than shown.",
        ],
        "fix": "BLUR_REGION",
        "span_note": "Usually short, 1-5s, and often a single shot.",
    },
    {
        "file": "AF-05_harmful_acts.md",
        "id": "AF-05",
        "title": "Harmful acts and unreliable content",
        "severity": "DEMONETIZING",
        "source": SRC_ADVERTISER,
        "scope": (
            "Two related things: acts a viewer could imitate and be seriously "
            "hurt by, and claims that could cause harm if believed. Both turn on "
            "whether the content warns against the behaviour or encourages it."
        ),
        "green": [
            "Professional stunt work in a controlled environment",
            "Fail compilations without graphic injury",
            "Discussion of a dangerous act without demonstrating it",
        ],
        "yellow": [
            "High-risk activity shown without an explicit warning against imitation",
            "Graphic injury resulting from a failed attempt",
            "Prank content causing severe distress to its subject",
        ],
        "red": [
            "Instructional content enabling a dangerous act",
            "Glorification of self-endangering behaviour",
            "Health claims contradicting established consensus in a way that "
            "could lead a viewer to refuse care",
            "Challenge content involving ingestion of harmful substances",
        ],
        "exemptions": BASE_EXEMPTIONS
        + [
            "Explicit harm-reduction framing: a stated warning against imitation, "
            "close to the depiction rather than buried in a description",
            "Public service or awareness content addressing the behaviour",
        ],
        "signals": [
            "vs AF-09 (controversial issues): AF-05 is imitable physical risk. "
            "Self-harm and suicide are AF-09, which has its own handling.",
            "vs AF-11 (dishonest behaviour): AF-05 risks injury; AF-11 risks "
            "someone else's property, money or access.",
        ],
        "fix": "NONE",
        "span_note": (
            "Often unfixable by editing — the remedy is a disclaimer card, which "
            "is why this clause frequently produces an advisory rather than an op."
        ),
    },
    {
        "file": "AF-06_hateful_derogatory.md",
        "id": "AF-06",
        "title": "Hateful and derogatory content",
        "severity": "DEMONETIZING",
        "source": SRC_ADVERTISER,
        "scope": (
            "Content promoting hatred against, or demeaning, a person or group on "
            "the basis of a protected attribute. This clause has no strength "
            "threshold: a single slur is sufficient, and no frequency argument "
            "applies."
        ),
        "green": [
            "Criticism of ideas, policies or actions rather than people",
            "Satire whose target is the prejudice itself",
            "Educational content about discrimination as a subject",
        ],
        "yellow": [
            "Offensive language reproduced inside an educational, news or "
            "documentary treatment",
        ],
        "red": [
            "Statements disparaging a group on the basis of a protected attribute",
            "Content promoting or justifying hatred",
            "Malicious personal attack on an individual",
        ],
        "exemptions": [
            "Educational or documentary treatment where the material is the "
            "subject of study rather than the message",
            "News reporting that quotes in order to report",
            "Satire whose evident target is the prejudice, not the group — note "
            "that satire is evaluated carefully and the framing must be legible "
            "from the content itself",
            "Quotation accompanied by explicit condemnation",
        ],
        "signals": [
            "vs AF-01 (language): a slur is never AF-01. Protected-attribute "
            "targeting moves it here regardless of how mild the term sounds.",
            "vs AF-14 (incendiary and demeaning): AF-06 requires a protected "
            "attribute. Sustained hostility toward someone without one is AF-14.",
        ],
        "fix": "NONE",
        "span_note": (
            "Muting rarely resolves this: the surrounding argument usually "
            "carries the same message. Escalate for human review."
        ),
    },
    {
        "file": "AF-07_recreational_drugs.md",
        "id": "AF-07",
        "title": "Recreational drugs and drug-related content",
        "severity": "LIMITING",
        "source": SRC_ADVERTISER,
        "scope": (
            "Depiction, promotion, sale or facilitation of recreational drugs, "
            "and content relating to organisations that traffic them."
        ),
        "green": [
            "Educational or recovery-focused discussion",
            "Fleeting depiction in a musical performance",
            "Documentary treatment of drug policy or its consequences",
        ],
        "yellow": [
            "Dramatised drug use",
            "Educational content about trafficking organisations that includes "
            "violent elements",
        ],
        "red": [
            "Promotion of drug use, sale or manufacture",
            "Instructions for producing or acquiring drugs",
            "Content glorifying or recruiting for a trafficking organisation",
        ],
        "exemptions": BASE_EXEMPTIONS
        + [
            "Recovery, harm-reduction and addiction-support content",
        ],
        "signals": [
            "vs AF-12 (tobacco): tobacco, vaping and alcohol have their own "
            "clause and a more permissive baseline for incidental adult use.",
            "vs AF-05 (harmful acts): AF-07 is the substance; AF-05 is the "
            "imitable act. Ingestion challenges sit in AF-05.",
        ],
        "fix": "MUTE",
        "span_note": "Usually a sentence, 3-10s.",
    },
    {
        "file": "AF-08_firearms.md",
        "id": "AF-08",
        "title": "Firearms-related content",
        "severity": "LIMITING",
        "source": SRC_ADVERTISER,
        "scope": (
            "Firearms and firearm-adjacent content: handling, demonstration, "
            "modification, sale, and the environment in which any of it happens. "
            "Whether the setting is controlled is the pivotal fact."
        ),
        "green": [
            "Non-automatic and semi-automatic firearms handled in a controlled "
            "environment such as a supervised range",
            "Repair, maintenance and cleaning",
            "Responsible airsoft or replica use",
            "Discussion of firearms legislation",
        ],
        "yellow": [
            "Firearms used outside a controlled environment",
            "Detailed discussion of capability or lethality",
        ],
        "red": [
            "Instructions for manufacturing a firearm or converting one to fire "
            "automatically",
            "Facilitating a sale or transfer",
            "Fully automatic weapons",
            "Minors handling firearms without evident supervision",
        ],
        "exemptions": BASE_EXEMPTIONS
        + [
            "Legislative, historical and policy discussion",
            "Safety instruction conducted in a controlled setting",
        ],
        "signals": [
            "vs AF-02 (violence): presence and handling are AF-08. A firearm used "
            "against a person is AF-02.",
            "vs AF-11 (dishonest behaviour): manufacturing instructions are AF-08 "
            "specifically; AF-11 covers circumvention of other systems.",
        ],
        "fix": "MUTE",
        "span_note": "Variable, often the whole segment. Consider REFRAME.",
    },
    {
        "file": "AF-09_controversial_issues.md",
        "id": "AF-09",
        "title": "Controversial issues",
        "severity": "LIMITING",
        "source": SRC_ADVERTISER,
        "scope": (
            "Difficult subject matter discussed rather than depicted: abuse, "
            "harassment, self-harm, suicide, eating disorders, domestic violence "
            "and abortion. Treatment is what is judged — whether the content is "
            "graphic, and whether it dwells."
        ),
        "green": [
            "Prevention and support-focused treatment",
            "Fleeting, non-graphic mention",
            "Dramatised treatment that is descriptive but not graphic",
        ],
        "yellow": [
            "Educational, artistic or documentary representation of the subject",
            "Graphic thumbnail",
            "Non-graphic child-abuse content as the primary topic",
        ],
        "red": [
            "Graphic depiction of any of these subjects",
            "Descriptive child-abuse content",
            "Eating-disorder content containing behaviour a viewer could copy",
        ],
        "exemptions": BASE_EXEMPTIONS
        + [
            "Support, prevention and crisis-resource content",
            "Survivor testimony presented without graphic detail",
        ],
        "signals": [
            "vs AF-10 (sensitive events): AF-09 is a category of harm in general. "
            "AF-10 attaches to a specific identifiable event.",
            "vs AF-02 and AF-04: those are depiction. AF-09 is discussion.",
        ],
        "fix": "NONE",
        "span_note": "Usually requires framing rather than editing.",
    },
    {
        "file": "AF-10_sensitive_events.md",
        "id": "AF-10",
        "title": "Sensitive events",
        "severity": "DEMONETIZING",
        "source": SRC_ADVERTISER,
        "scope": (
            "A specific, identifiable tragedy, disaster, death or violent "
            "incident affecting real people. The question is whether the content "
            "engages with the event or exploits it for attention."
        ),
        "green": [
            "Discussion of loss or tragedy that is not exploitative",
            "Memorial and tribute content",
            "Factual reference with surrounding context",
        ],
        "yellow": [
            "Casualty figures stated without framing",
            "Extended discussion of a recent event",
        ],
        "red": [
            "Content exploiting an event for traffic, including keyword-stuffed "
            "titles referencing a tragedy",
            "Graphic footage of the event",
        ],
        "exemptions": BASE_EXEMPTIONS
        + [
            "Journalism and factual reporting on the event",
            "Commemoration, memorial and tribute",
            "First-hand testimony from those affected",
        ],
        "signals": [
            "vs AF-09 (controversial issues): AF-10 needs a named or otherwise "
            "identifiable event. Discussing suicide as a subject is AF-09; "
            "discussing a specific person's death is AF-10.",
            "vs AF-02 (violence): AF-10 covers the discussion. Footage of the "
            "event itself is AF-02 or AF-04.",
        ],
        "fix": "MUTE",
        "span_note": "Usually one or two sentences, 4-12s.",
    },
    {
        "file": "AF-11_dishonest_behavior.md",
        "id": "AF-11",
        "title": "Enabling dishonest behaviour",
        "severity": "DEMONETIZING",
        "source": SRC_ADVERTISER,
        "scope": (
            "Content that equips a viewer to deceive, defraud or gain "
            "unauthorised access: account compromise, circumvention of payment, "
            "forged documents, academic dishonesty and trespass."
        ),
        "green": [
            "Educational or humorous reference without instruction",
            "Journalistic reporting on fraud or its victims",
        ],
        "yellow": [
            "Demonstration of a circumvention technique without a complete method",
        ],
        "red": [
            "Step-by-step instructions enabling unauthorised access or fraud",
            "Promotion of essay mills or academic cheating services",
            "Glorification of trespass",
        ],
        "exemptions": BASE_EXEMPTIONS
        + [
            "Security research with a defensive framing, including authorised "
            "penetration testing and bug-bounty work",
            "Consumer-protection content teaching recognition of a scam",
        ],
        "signals": [
            "vs AF-05 (harmful acts): AF-11 risks property, money or access. "
            "AF-05 risks a body.",
            "The defensive framing exemption is narrow: recognising a scam is "
            "exempt, reproducing it end to end is not.",
        ],
        "fix": "MUTE",
        "span_note": "Often the whole instructional passage. CUT may be warranted.",
    },
    {
        "file": "AF-12_tobacco.md",
        "id": "AF-12",
        "title": "Tobacco-related content",
        "severity": "LIMITING",
        "source": SRC_ADVERTISER,
        "scope": (
            "Tobacco, vaping and related products, and by extension other "
            "regulated goods such as alcohol where the same promotion logic "
            "applies. Incidental adult reference is treated far more leniently "
            "than promotion."
        ),
        "green": [
            "Incidental reference in content addressed to adults",
            "Cessation, harm-reduction and health-consequence content",
        ],
        "yellow": [
            "Consumption shown on camera without promotion",
            "Product review of a regulated good",
        ],
        "red": [
            "Promotion of tobacco or vaping products",
            "Content encouraging excessive consumption",
            "Any depiction inside content made for kids",
        ],
        "exemptions": BASE_EXEMPTIONS
        + [
            "Cessation support and public-health messaging",
        ],
        "signals": [
            "vs AF-07 (recreational drugs): legality is the divider. Tobacco, "
            "vaping and alcohol are AF-12 and start from a permissive baseline.",
            "vs AF-13 (kids and families): the permissive baseline disappears "
            "entirely if the content reads as made for children.",
        ],
        "fix": "NONE",
        "span_note": "Usually advisory; a passing mention rarely warrants an edit.",
    },
    {
        "file": "AF-13_kids_and_families.md",
        "id": "AF-13",
        "title": "Inappropriate content for kids and families",
        "severity": "DEMONETIZING",
        "source": SRC_ADVERTISER,
        "scope": (
            "Mature themes inside content whose format, characters, animation "
            "style or framing would lead a viewer to expect it is made for "
            "children. This clause raises the severity of material that would be "
            "unremarkable elsewhere."
        ),
        "green": [
            "Educational content modelling positive behaviour",
            "Safe do-it-yourself projects and gentle pranks",
            "Age-appropriate fitness and activity content",
        ],
        "yellow": [
            "Mild mature themes inside a family-coded format",
        ],
        "red": [
            "Violence, sexual themes, profanity or regulated goods in content "
            "resembling children's programming",
            "Content encouraging cheating, bullying or unsafe imitation",
            "Horror content built to frighten children",
            "Risky do-it-yourself activity presented to a child audience",
        ],
        "exemptions": [
            "Educational content addressing negative behaviour in order to "
            "discourage it",
            "Public service and safety messaging aimed at families",
            "Content clearly and consistently framed for an adult audience "
            "despite superficially family-adjacent subject matter",
        ],
        "signals": [
            "This clause modifies others rather than replacing them. When AF-01, "
            "AF-02, AF-03, AF-07 or AF-12 fires inside kid-coded content, AF-13 "
            "governs and the severity rises.",
            "Format signals — animation style, character design, toys, nursery "
            "colour palettes, simplified narration — carry more weight here than "
            "the declared audience setting.",
        ],
        "fix": "NONE",
        "span_note": (
            "Rarely fixable by editing. The remedy is usually the audience "
            "setting or a different edit entirely."
        ),
    },
    {
        "file": "AF-14_incendiary_demeaning.md",
        "id": "AF-14",
        "title": "Incendiary and demeaning content",
        "severity": "LIMITING",
        "source": SRC_ADVERTISER,
        "scope": (
            "Content whose purpose is to shame, insult or harass a person or "
            "group, including denial of a well-documented tragedy. Unlike AF-06 "
            "this does not require a protected attribute — sustained hostility is "
            "enough."
        ),
        "green": [
            "Criticism directed at work, ideas or public conduct",
            "Robust disagreement without personal degradation",
        ],
        "yellow": [
            "Sustained mockery of an individual",
            "Content shaming a group without reference to a protected attribute",
        ],
        "red": [
            "Harassment campaigns or calls to pile on",
            "Denial that a well-documented tragic event occurred",
            "Malicious personal attack whose evident purpose is degradation",
        ],
        "exemptions": BASE_EXEMPTIONS
        + [
            "Accountability journalism concerning a public figure's public conduct",
            "Review and criticism of published work",
        ],
        "signals": [
            "vs AF-06 (hateful & derogatory): the protected attribute is the "
            "divider. Present, it is AF-06; absent, AF-14.",
            "vs AF-09 (controversial issues): AF-14 is directed at someone. AF-09 "
            "is a subject discussed.",
        ],
        "fix": "MUTE",
        "span_note": "Usually a passage rather than a phrase.",
    },
    {
        "file": "META-01_disclosure.md",
        "id": "META-01",
        "title": "Paid promotion disclosure",
        "severity": "DEMONETIZING",
        "source": SRC_DISCLOSURE,
        "scope": (
            "Paid product placement, sponsorship and endorsement must be "
            "disclosed. YouTube provides a checkbox in Studio that surfaces a "
            "disclosure to viewers for the first ten seconds, and creators remain "
            "separately responsible for whatever their own jurisdiction requires "
            "— the FTC in the United States, the ASA in the United Kingdom, and "
            "equivalents elsewhere."
        ),
        "green": [
            "Paid promotion disclosed both in the platform setting and visibly to "
            "viewers",
            "Content with no commercial relationship to disclose",
        ],
        "yellow": [
            "Disclosure present but buried below the description fold",
        ],
        "red": [
            "Sponsorship, affiliate or endorsement content with no disclosure",
            "Promotion of a category that may not be promoted at all, including "
            "unreviewed gambling services, prescription pharmaceuticals, essay "
            "mills, counterfeit goods and hacking tools",
        ],
        "exemptions": [
            "No commercial relationship exists, so there is nothing to disclose",
            "Disclosure is made on screen rather than in the description, and is "
            "legible and early — a metadata-only check cannot see this, so a "
            "finding here is rebuttable",
        ],
        "signals": [
            "This is a deterministic cross-check rather than a judgement: "
            "sponsorship language in the transcript, no disclosure marker in the "
            "description. Both inputs are already in hand.",
            "Non-disclosure carries removal and strike consequences rather than "
            "only demonetisation, which is why it is rated higher than its "
            "apparent severity.",
        ],
        "fix": "NONE",
        "span_note": (
            "File-scoped. The fix is a description edit and the Studio checkbox, "
            "not an edit to the video."
        ),
    },
    {
        "file": "ACC-01_photosensitive.md",
        "id": "ACC-01",
        "title": "Photosensitive content",
        "severity": "LIMITING",
        "source": SRC_WCAG,
        "scope": (
            "Rapid luminance change that can provoke a seizure in photosensitive "
            "viewers. The widely used threshold is three flashes within any one "
            "second, with transitions to and from saturated red treated more "
            "strictly. This is a safety property, measurable directly, and it is "
            "not a monetisation rule — it is included because a creator has no "
            "other way to discover it."
        ),
        "green": [
            "No sequence exceeding two flashes per second",
            "Gradual transitions and cross-fades",
        ],
        "yellow": [
            "Two flashes per second sustained across a sequence",
            "Rapid cuts producing large luminance swings without a warning card",
        ],
        "red": [
            "Three or more flashes within any one second",
            "Rapid transitions to and from saturated red",
        ],
        "exemptions": [
            "An explicit photosensitivity warning shown before the sequence — "
            "this mitigates the harm but does not remove it, so it lowers "
            "severity rather than dismissing the finding",
            "The flashing area occupies a small enough proportion of the frame "
            "that it falls under the small-area exception in the underlying "
            "guidance",
            "Luminance change stays below the general flash threshold even where "
            "transitions are frequent — rapid but low-contrast cutting is not a "
            "flash",
            "The sequence is a single transition rather than a repeating pattern",
        ],
        "signals": [
            "Measured, not inferred: a luminance series sampled at 10fps or "
            "better, differenced, and counted in a one-second sliding window.",
            "Scene-cut keyframes cannot detect this. A strobe lives entirely "
            "between two cuts, so this check requires its own sampling rate.",
        ],
        "fix": "NONE",
        "span_note": (
            "The remedy is a warning card or a re-edit of the sequence; a filter "
            "cannot make a strobe safe."
        ),
    },
    {
        "file": "COPY-01_content_id.md",
        "id": "COPY-01",
        "title": "Third-party content and Content ID",
        "severity": "DEMONETIZING",
        "source": SRC_CONTENT_ID,
        "scope": (
            "Third-party copyrighted material, principally music. Content ID "
            "scans uploads against a database of reference files submitted by "
            "rights holders. A match lets the claimant block the video, take its "
            "revenue, or track it, and the outcome can differ by territory."
        ),
        "green": [
            "Original material, or material licensed for this use",
            "Public-domain or CC0 material with the licence recorded",
        ],
        "yellow": [
            "Music present under speech with licensing unverified",
            "Third-party footage cues such as station bugs or letterboxing",
        ],
        "red": [
            "Commercially released recording used without a licence",
            "Substantial third-party footage without a licence",
        ],
        "exemptions": [
            "A licence exists for this use — the tool cannot see licences and "
            "this finding is therefore always rebuttable by the creator",
            "The material is public domain or CC0",
            "Use qualifies as fair use or fair dealing, which is a legal "
            "determination this tool does not and cannot make",
        ],
        "signals": [
            "The reference database is private and is not published, so no "
            "pre-upload check can be authoritative. PREFLIGHT reports "
            "CLAIM_LIKELY on a public fingerprint match and MUSIC_BED_PRESENT on "
            "unidentified tonal content. It never reports SAFE.",
            "Claims are applied automatically at upload, and there is no "
            "published mechanism for previewing them beforehand — which is "
            "precisely the gap this clause exists to narrow.",
            "vs AF-* clauses: COPY-01 is about ownership, not about content "
            "suitability. A perfectly advertiser-friendly video can be claimed.",
        ],
        "fix": "REPLACE_AUDIO",
        "span_note": (
            "Usually the full extent of the bed, 15-60s. Replace rather than "
            "mute so the segment keeps its pacing."
        ),
    },
]

TEMPLATE = """---
clause_id: {id}
title: {title}
severity_default: {severity}
version: {version}
source_url: {source}
fetched_at: {fetched}
derivation: {derivation}
---

## Scope

{scope}

## Fully monetized when

{green}

## Limited ads when

{yellow}

## No ads when

{red}

## Documented exemptions

{exemptions}

## Signals that distinguish this clause from neighbours

{signals}

## Remediation guidance

- Preferred fix: {fix}
- Typical span: {span_note}
"""


def bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # This script owns the directory. A clause left behind by a drift
    # simulation would otherwise survive a rebuild and end up in the next
    # baseline snapshot, so the change being demonstrated is already present
    # before the demonstration starts.
    every_clause = CLAUSES + HOUSE_RULES
    expected = {clause["file"] for clause in every_clause}
    for stale in OUT.glob("*.md"):
        if stale.name not in expected:
            stale.unlink()
            print(f"removed stale clause {stale.name}")

    manifest: list[dict] = []
    for clause in every_clause:
        body = TEMPLATE.format(
            id=clause["id"],
            title=clause["title"],
            severity=clause["severity"],
            version=VERSION,
            source=clause["source"],
            fetched=FETCHED,
            derivation=clause.get("derivation", DERIVATION),
            scope=clause["scope"],
            green=bullets(clause["green"]),
            yellow=bullets(clause["yellow"]),
            red=bullets(clause["red"]),
            exemptions=bullets(clause["exemptions"]),
            signals=bullets(clause["signals"]),
            fix=clause["fix"],
            span_note=clause["span_note"],
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
                "source_url": clause["source"],
                "fetched_at": FETCHED,
                "derivation": clause.get("derivation", DERIVATION),
                "kind": "house_rule" if clause in HOUSE_RULES else "policy_restatement",
            }
        )

    # The corpus hash goes into every certificate. It is what makes a report
    # reproducible against a known snapshot of the rules.
    corpus_hash = hashlib.sha256(
        "".join(entry["sha256"] for entry in manifest).encode("utf-8")
    ).hexdigest()[:32]

    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "version": VERSION,
                "corpus_hash": corpus_hash,
                "clause_count": len(manifest),
                "fetched_at": FETCHED,
                "sources": sorted({c["source"] for c in every_clause}),
                "policy_clauses": len(CLAUSES),
                "house_rules": len(HOUSE_RULES),
                "note": (
                    "Two kinds of clause, distinguished by `kind`. "
                    "`policy_restatement` entries are structured restatements in "
                    "our own words of publicly published guidance — not "
                    "authoritative, not verbatim, not affiliated with YouTube. "
                    "`house_rule` entries are PREFLIGHT's own production "
                    "thresholds — loudness targets, caption availability, tag "
                    "hygiene — and are NOT platform policy. A finding cites the "
                    "clause it was judged under, and the citation is worthless "
                    "if a reader cannot tell whose rule it is."
                ),
                "clauses": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"wrote {len(manifest)} clauses + manifest to {OUT}")
    print(f"corpus_hash {corpus_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
