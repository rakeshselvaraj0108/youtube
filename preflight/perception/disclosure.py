"""On-screen secrets and personal data.

OCR already reads every word the video puts on screen. Nothing looked at
what those words *were* — so a run could report a clean video while the
creator's API key sat legible in a terminal window for nine seconds.

That is the failure this catches, and it is the one with real consequences
outside YouTube's policy: a leaked key is charged to the creator's account
within hours of upload, and a phone number on screen is a permanent
harassment vector. Both are silent — the creator does not know, and neither
does the platform.

Precision is the whole game here. A detector that flags every sixteen-digit
number as a credit card, or every `sk` as a secret, gets muted after the
second false alarm and then never catches the real one. So:

  * Card numbers are Luhn-checked. Sixteen digits that fail the checksum are
    an order number, not a card.
  * Keys are matched on published vendor prefixes with a length floor, not
    on entropy — entropy flags base64 thumbnails and minified JavaScript.
  * Phone numbers require a separator or a country code. A bare run of ten
    digits is a timestamp, a serial, or a score.

Everything found is redacted the moment it is captured. A tool that reports
"your API key is visible at 04:12" by printing the API key has not helped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Vendor prefixes are published and stable, which is what makes them safe to
# match on. The length floor is what stops "sk-" in ordinary prose matching.
_KEY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("OpenAI", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    ("NVIDIA", re.compile(r"\bnvapi-[A-Za-z0-9_-]{20,}")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("Stripe key", re.compile(r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{20,}")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("JSON Web Token", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.")),
]

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Requires a separator or a leading +country. A bare 10-digit run is far more
# often a timestamp, an order id or a score than a telephone number.
_PHONE = re.compile(
    r"(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?|\d{2,4}[\s.-])\d{3,4}[\s.-]?\d{3,4}\b"
)

_URL = re.compile(r"\bhttps?://[^\s<>\"']{4,}", re.IGNORECASE)

# 13-19 digits, optionally separated. Luhn decides whether it is real.
_CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")

# Matched near a value to catch `password: hunter2` in a terminal or a slide.
#
# The leading alternation is the important part. A plain `\b` before the
# keyword cannot match inside `DB_PASSWORD`, because `_` is a word character
# and there is therefore no boundary between `DB_` and `PASSWORD`. That made
# this miss every SCREAMING_SNAKE_CASE environment variable — which is the
# single most common way a real secret appears on screen: `.env` files,
# `export` lines and docker-compose blocks name things `DB_PASSWORD`,
# `AWS_SECRET_ACCESS_KEY`, `GITHUB_TOKEN`, never bare `password`. The
# lookbehind admits a separator-prefixed name while still refusing to match
# mid-word inside ordinary prose.
_CREDENTIAL_LABEL = re.compile(
    r"(?:\b|(?<=[_.\-]))"
    r"(?:password|passwd|pwd|secret|api[_ .\-]?key|access[_ .\-]?key|token|auth)"
    r"\b\s*[:=]\s*\S{4,}",
    re.IGNORECASE,
)

# `Authorization: Bearer <token>` — a space-separated scheme, so the
# `[:=]` form above never sees it. Common in dev tools, curl output and
# Postman screenshots.
_BEARER = re.compile(
    r"\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE
)

# A connection string carrying inline credentials, e.g.
# `postgres://admin:hunter2@10.0.0.5:5432/prod`. The password sits in the
# userinfo segment, so no keyword appears anywhere and every pattern above
# misses it.
_CONNECTION_STRING = re.compile(
    r"\b[a-z][a-z0-9+.-]{1,15}://[^\s:/@]{1,64}:[^\s:/@]{1,64}@[^\s/]{1,128}",
    re.IGNORECASE,
)


def luhn_valid(digits: str) -> bool:
    """The checksum every real card number satisfies.

    This is what separates a card detector from a sixteen-digit-number
    detector. Roughly nine in ten random digit runs of that length fail it,
    so it removes almost all of the false positives that would otherwise
    train a creator to ignore this finding.
    """
    numbers = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(numbers) <= 19:
        return False
    total = 0
    for position, digit in enumerate(reversed(numbers)):
        if position % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def redact(value: str, keep: int = 4) -> str:
    """Enough to recognise, never enough to use.

    Reporting "your key is on screen" by printing the key would put the
    secret into report.json, the HTML, the terminal scrollback and any CI
    log that captured the run — more copies than the video ever made.
    """
    stripped = value.strip()
    if len(stripped) <= keep:
        return "*" * len(stripped)
    return f"{stripped[:keep]}{'*' * min(len(stripped) - keep, 12)}"


@dataclass(frozen=True)
class Disclosure:
    """One piece of sensitive text, and where it was legible."""

    kind: str          # credential | email | phone | card | url
    label: str         # human name of what matched
    redacted: str
    start_ms: int
    end_ms: int
    severity: str

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "redacted": self.redacted,
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "severity": self.severity,
        }


def scan_text(text: str) -> list[tuple[str, str, str, str]]:
    """Find sensitive substrings. Returns (kind, label, raw, severity)."""
    hits: list[tuple[str, str, str, str]] = []

    for label, pattern in _KEY_PATTERNS:
        for match in pattern.finditer(text):
            hits.append(("credential", label, match.group(0), "CRITICAL"))

    for match in _CREDENTIAL_LABEL.finditer(text):
        hits.append(("credential", "labelled secret", match.group(0), "CRITICAL"))

    for match in _BEARER.finditer(text):
        hits.append(("credential", "authorization header", match.group(0), "CRITICAL"))

    for match in _CONNECTION_STRING.finditer(text):
        hits.append(
            ("credential", "connection string", match.group(0), "CRITICAL")
        )

    for match in _CARD.finditer(text):
        raw = match.group(0)
        if luhn_valid(raw):
            hits.append(("card", "payment card", raw, "CRITICAL"))

    for match in _EMAIL.finditer(text):
        hits.append(("email", "email address", match.group(0), "MEDIUM"))

    for match in _PHONE.finditer(text):
        # An email's digits, or a URL's, are not a phone number.
        if any(match.group(0).strip() in other for _, _, other, _ in hits):
            continue
        hits.append(("phone", "phone number", match.group(0), "MEDIUM"))

    for match in _URL.finditer(text):
        hits.append(("url", "link", match.group(0), "LOW"))

    return hits


def analyse(ocr_items: list) -> list[Disclosure]:
    """Scan everything OCR read, deduplicated by what and when.

    `ocr_items` is duck-typed on `start_ms`, `end_ms` and `text` — the same
    shape chunking already accepts, so this does not import the OCR module
    and OCR does not import this one.
    """
    # Collect first, merge second. Bucketing by a fixed time window looked
    # like deduplication but split on the window boundary instead: a key
    # legible from 4s to 12s landed in two buckets and was reported twice,
    # as though it had been shown, hidden, and shown again.
    raw_hits: list[Disclosure] = []
    for item in ocr_items or []:
        text = str(getattr(item, "text", "") or "")
        if not text.strip():
            continue
        start = int(getattr(item, "start_ms", 0))
        end = int(getattr(item, "end_ms", start))

        for kind, label, value, severity in scan_text(text):
            raw_hits.append(
                Disclosure(
                    kind=kind,
                    label=label,
                    # Redacted at capture. The raw value never survives into
                    # a structure that gets logged, keyed on, or serialised.
                    redacted=redact(value),
                    start_ms=start,
                    end_ms=max(end, start + 1),
                    severity=severity,
                )
            )

    return _merge(raw_hits)


# OCR reads the same overlay from consecutive keyframes, so one continuously
# visible string arrives as a run of near-touching spans. Anything closer
# than this is the same sighting rather than a second one.
MERGE_GAP_MS = 2000


def _merge(hits: list[Disclosure]) -> list[Disclosure]:
    """One continuously visible string is one disclosure."""
    merged: list[Disclosure] = []
    for hit in sorted(hits, key=lambda d: (d.kind, d.redacted, d.start_ms)):
        previous = merged[-1] if merged else None
        contiguous = (
            previous is not None
            and previous.kind == hit.kind
            and previous.redacted == hit.redacted
            and hit.start_ms - previous.end_ms <= MERGE_GAP_MS
        )
        if contiguous and previous is not None:
            merged[-1] = Disclosure(
                kind=previous.kind,
                label=previous.label,
                redacted=previous.redacted,
                start_ms=min(previous.start_ms, hit.start_ms),
                end_ms=max(previous.end_ms, hit.end_ms),
                severity=previous.severity,
            )
        else:
            merged.append(hit)

    return sorted(merged, key=lambda d: (d.start_ms, d.kind))


# Credentials and personal data are judged under different house rules, and
# a reader has to be able to tell which without reading the description.
_CLAUSE = {
    "credential": ("DISC-01", "Credential visible on screen"),
    "card": ("DISC-02", "Payment card visible on screen"),
    "email": ("DISC-02", "Email address visible on screen"),
    "phone": ("DISC-02", "Phone number visible on screen"),
    "url": ("DISC-02", "Link visible on screen"),
}

# A link on screen is ordinary — creators put them there deliberately. It is
# recorded as evidence but never raised as a finding on its own.
_REPORTABLE = {"credential", "card", "email", "phone"}


def to_findings(disclosures: list[Disclosure], corpus: Any = None) -> list:
    """Turn disclosures into findings, citing the house rule each falls under."""
    from preflight.models import Adversarial, Evidence, Finding, PolicyRef

    findings = []
    for index, item in enumerate(disclosures):
        if item.kind not in _REPORTABLE:
            continue
        clause_id, title = _CLAUSE[item.kind]
        section = "PREFLIGHT disclosure ruleset § 5.1" if clause_id == "DISC-01" else (
            "PREFLIGHT disclosure ruleset § 5.2"
        )
        findings.append(
            Finding(
                id=f"disc{index + 1}",
                clauseId=clause_id,
                category="Metadata",
                title=title,
                description=(
                    f"{item.label} legible on screen ({item.redacted})."
                ),
                startMs=item.start_ms,
                endMs=item.end_ms,
                severity=item.severity,  # type: ignore[arg-type]
                confidence=0.95 if item.kind in {"credential", "card"} else 0.85,
                modalities={"ocr": 0.95},
                evidence=Evidence(transcript=item.redacted),
                policy=PolicyRef(
                    clauseId=clause_id,
                    title=title,
                    section=section,
                    text=(
                        "Text matched a published credential format or a "
                        "checksum-valid card number. PREFLIGHT engineering "
                        "ruleset — not platform policy."
                    ),
                ),
                adversarial=Adversarial(
                    charge=f"{item.label} readable in the picture",
                    rationale=(
                        "Pattern match on the text OCR read, not a judgement "
                        "call. Card numbers are Luhn-checked; keys are matched "
                        "on vendor prefixes with a length floor, so prose "
                        "about secrets does not trigger this."
                    ),
                    confidence=0.95,
                ),
                suggestedFix="BLUR_REGION",
            )
        )
    return findings
