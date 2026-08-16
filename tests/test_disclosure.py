"""On-screen secrets and personal data.

The interesting tests here are the negatives. A detector that fires on
every sixteen-digit number and every "sk" gets muted after the second false
alarm, and then it is not there for the real one — so most of this file is
about what must *not* be flagged.
"""

from __future__ import annotations

import pytest

from preflight.perception.disclosure import (
    Disclosure,
    analyse,
    luhn_valid,
    redact,
    scan_text,
)


class Item:
    def __init__(self, text: str, start_ms: int = 0, end_ms: int = 1000) -> None:
        self.text, self.start_ms, self.end_ms = text, start_ms, end_ms


def kinds(text: str) -> set[str]:
    return {kind for kind, _, _, _ in scan_text(text)}


class TestLuhn:
    """The checksum is what makes a card detector a card detector."""

    @pytest.mark.parametrize(
        "number",
        [
            "4539578763621486",   # Visa test
            "5500005555555559",   # Mastercard test
            "371449635398431",    # Amex test, 15 digits
            "4539 5787 6362 1486",
        ],
    )
    def test_real_card_numbers_validate(self, number):
        assert luhn_valid(number)

    @pytest.mark.parametrize(
        "number",
        [
            "1234567812345678",   # sequential, fails checksum
            "0000000000000000",   # passes arithmetic but see below
            "4539578763621487",   # one digit off a valid card
            "12345",              # too short
            "12345678901234567890123",  # too long
        ],
    )
    def test_non_cards_are_rejected(self, number):
        if number == "0000000000000000":
            pytest.skip("all-zero satisfies Luhn; length/context handles it")
        assert not luhn_valid(number)

    def test_an_order_number_is_not_a_card(self):
        """The false positive that matters: a sixteen-digit order id on a
        confirmation screen must not be reported as a payment card."""
        assert not luhn_valid("8837261940572819")


class TestRedaction:
    def test_the_secret_never_survives_intact(self):
        secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
        hidden = redact(secret)
        assert secret not in hidden
        assert hidden.startswith("sk-a")

    def test_a_short_value_is_fully_masked(self):
        assert set(redact("abc")) == {"*"}

    def test_a_finding_carries_no_raw_secret(self):
        """The whole report is downstream of this. A key that reaches
        `Disclosure` reaches report.json, the HTML and any CI log."""
        secret = "nvapi-abcdefghijklmnopqrstuvwxyz0123456789"  # pragma: allowlist secret
        results = analyse([Item(f"export KEY={secret}")])
        assert results
        for finding in results:
            assert secret not in finding.redacted


class TestCredentials:
    @pytest.mark.parametrize(
        "text",
        [
            "sk-abcdefghijklmnopqrstuvwxyz123456",
            "nvapi-abcdefghijklmnopqrstuvwxyz0123456789",  # pragma: allowlist secret
            "AKIAIOSFODNN7EXAMPLE",
            "ghp_abcdefghijklmnopqrstuvwxyz1234567890",
            "AIzaSyD-abcdefghijklmnopqrstuvwxyz12345",  # pragma: allowlist secret
            "-----BEGIN RSA PRIVATE KEY-----",
            "password: hunter2000",
            "API_KEY = s3cr3tvalue",
        ],
    )
    def test_real_credentials_are_caught(self, text):
        assert "credential" in kinds(text), text

    @pytest.mark.parametrize(
        "text",
        [
            "the sk- prefix identifies an OpenAI key",
            "we discussed tokens and passwords in this video",
            "my password manager keeps everything safe",
            "skateboarding tutorial part 3",
        ],
    )
    def test_prose_about_secrets_is_not_a_secret(self, text):
        """Talking about credentials is not leaking one. This is the false
        positive most likely to appear in an educational video, which is
        exactly the audience this tool serves."""
        assert "credential" not in kinds(text), text

    @pytest.mark.parametrize(
        "text",
        [
            "DB_PASSWORD=hunter2supersecret",
            "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY",
            "GITHUB_TOKEN=abcdefghijklmnop123456",
            "export STRIPE_SECRET=sk_live_abcdefgh",
            "REACT_APP_API_KEY=abcd1234efgh5678",
        ],
    )
    def test_screaming_snake_case_env_vars_are_caught(self, text):
        """The commonest real shape of an on-screen leak, and it was missed.

        A plain `\\b` before the keyword cannot match inside `DB_PASSWORD`:
        `_` is a word character, so there is no boundary between `DB_` and
        `PASSWORD`. Every `.env` file, `export` line and docker-compose block
        names secrets exactly this way and none of them matched — the
        detector only fired on a bare `password:`, which is the one form
        almost nobody's screen actually shows.
        """
        assert "credential" in kinds(text), text

    @pytest.mark.parametrize(
        "text",
        [
            "Authorization: Bearer abc123def456ghi789",
            "curl -H 'Authorization: Basic YWRtaW46aHVudGVyMg=='",
        ],
    )
    def test_authorization_headers_are_caught(self, text):
        """`Bearer <token>` is space-separated, so the `key: value` form
        never sees it — and it is what every dev-tools screenshot shows."""
        assert "credential" in kinds(text), text

    @pytest.mark.parametrize(
        "text",
        [
            "DATABASE_URL=postgres://user:pw123@10.0.0.5:5432/prod",
            "redis://default:p4ssw0rd@redis-prod.internal:6379",
            "mongodb+srv://admin:s3cret@cluster0.mongodb.net/db",
        ],
    )
    def test_connection_strings_with_inline_credentials_are_caught(self, text):
        """The password sits in the userinfo segment with no keyword
        anywhere near it, so every keyword-based pattern misses it."""
        assert "credential" in kinds(text), text

    @pytest.mark.parametrize(
        "text",
        [
            "Bearer with me while I explain this",
            "https://example.com/docs/api-key-guide",
            "postgres://localhost:5432/dev",
            "the secret sauce recipe is simple",
        ],
    )
    def test_the_widened_patterns_do_not_over_match(self, text):
        """Widening the credential patterns must not start flagging prose,
        documentation links or a credential-free connection string — a
        detector that cries wolf gets muted, and then misses the real one."""
        assert "credential" not in kinds(text), text


class TestPersonalData:
    def test_an_email_is_found(self):
        assert "email" in kinds("contact me at creator@example.com")

    def test_a_phone_number_is_found(self):
        assert "phone" in kinds("call 555-123-4567 for details")
        assert "phone" in kinds("+44 20 7946 0958")

    def test_a_bare_digit_run_is_not_a_phone_number(self):
        """Timestamps, scores and serials are not phone numbers."""
        assert "phone" not in kinds("the score was 1234567890 points")

    def test_a_url_is_found(self):
        assert "url" in kinds("visit https://example.com/thing")

    def test_ordinary_speech_produces_nothing(self):
        assert scan_text("welcome back to the channel, today we are cooking") == []


class TestSpansAndDeduplication:
    def test_the_same_key_across_a_scene_is_one_finding(self):
        """A key legible for nine seconds is one disclosure, not forty —
        one per frame OCR happened to read it in."""
        secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
        items = [Item(secret, start_ms=t, end_ms=t + 400) for t in range(0, 4000, 400)]
        results = analyse(items)
        assert len([r for r in results if r.kind == "credential"]) == 1

    def test_a_continuous_sighting_does_not_split_on_a_window_boundary(self):
        """The first version bucketed by a fixed time window, which looked
        like deduplication but split on the boundary: a key legible from 4s
        to 12s was reported twice, as though it had been shown, hidden, and
        shown again."""
        secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
        items = [Item(secret, 4000, 9000), Item(secret, 9000, 12000)]
        found = [d for d in analyse(items) if d.kind == "credential"]
        assert len(found) == 1
        assert found[0].start_ms == 4000
        assert found[0].end_ms == 12000

    def test_a_genuine_second_sighting_stays_separate(self):
        """The other half: shown, hidden for a minute, shown again is two
        disclosures, and merging them would misreport when it was visible."""
        secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
        items = [Item(secret, 0, 1000), Item(secret, 90_000, 91_000)]
        found = [d for d in analyse(items) if d.kind == "credential"]
        assert len(found) == 2

    def test_a_credential_outranks_a_link(self):
        results = analyse([Item("https://x.co and sk-abcdefghijklmnopqrstuvwxyz1234")])
        by_kind = {r.kind: r.severity for r in results}
        assert by_kind["credential"] == "CRITICAL"
        assert by_kind["url"] == "LOW"

    def test_no_ocr_items_finds_nothing(self):
        assert analyse([]) == []
        assert analyse(None) == []

    def test_results_are_in_timeline_order(self):
        items = [
            Item("later@example.com", 5000, 5500),
            Item("earlier@example.com", 1000, 1500),
        ]
        found = analyse(items)
        assert [f.start_ms for f in found] == sorted(f.start_ms for f in found)

    def test_findings_serialise(self):
        import json

        found = analyse([Item("me@example.com")])
        json.dumps([f.to_json() for f in found])
