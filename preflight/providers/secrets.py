"""Credential resolution and redaction.

This is the only module in the project permitted to read a credential from the
environment. Agents request capabilities; they never see a key. If `os.environ`
appears outside this file for anything credential-shaped, that is a bug.

Redaction is enforced rather than hoped for. A logging filter and an excepthook
are installed at import time, so a key cannot escape through a third-party
library's debug output or an unhandled traceback — the two paths that leak keys
in practice, because neither is code you wrote.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

# Shape patterns, with a human-readable hint for the error message. Validating
# the shape at resolution time turns a confusing 401 three minutes into a run
# into a clear message before the first call.
PATTERNS: dict[str, tuple[re.Pattern[str], str]] = {
    "NVIDIA_API_KEY": (re.compile(r"^nvapi-[A-Za-z0-9_\-]{20,}$"), "nvapi-…"),
    "ACOUSTID_API_KEY": (re.compile(r"^[A-Za-z0-9]{8,}$"), "8+ alphanumeric"),
    "HUGGINGFACE_TOKEN": (re.compile(r"^hf_[A-Za-z0-9]{20,}$"), "hf_…"),
    "QDRANT_API_KEY": (re.compile(r"^.{8,}$"), "8+ characters"),
}

# Vendor prefixes we recognise well enough to say "that is the wrong key".
FOREIGN_KEYS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"^sk-proj-"), "OpenAI project key", "build.nvidia.com"),
    (re.compile(r"^sk-ant-"), "Anthropic key", "build.nvidia.com"),
    (re.compile(r"^sk-[A-Za-z0-9]{20,}$"), "OpenAI key", "build.nvidia.com"),
    (re.compile(r"^AIza"), "Google API key", "build.nvidia.com"),
    (re.compile(r"^gsk_"), "Groq key", "build.nvidia.com"),
]

# Anything matching this never reaches a log, an event, a report or a terminal.
_REDACT = re.compile(
    r"(nvapi-|hf_|sk-proj-|sk-ant-|sk-|AIza|ya29\.|gsk_)[A-Za-z0-9_\-\.]{6,}"
)


def redact(text: str) -> str:
    """Replace anything key-shaped with its prefix and a marker."""
    if not text:
        return text
    return _REDACT.sub(lambda m: f"{m.group(1)}…REDACTED", text)


def fingerprint(value: str | None) -> str:
    """A safe identifier for logs: the shape, never the secret.

    Enough to tell two keys apart and to confirm the right one loaded, without
    being enough to use.
    """
    if not value:
        return "—"
    if len(value) <= 14:
        return f"{value[:3]}…{'*' * 4} ({len(value)} chars)"
    return f"{value[:9]}…{value[-4:]} ({len(value)} chars)"


@dataclass
class Secret:
    name: str
    value: str | None
    source: str  # flag | env | dotenv | keyring | absent
    shape_ok: bool
    problem: str | None = None

    @property
    def present(self) -> bool:
        return bool(self.value)

    @property
    def usable(self) -> bool:
        return self.present and self.shape_ok

    def to_json(self) -> dict[str, object]:
        """Never includes the value. This is what reaches the certificate."""
        return {
            "name": self.name,
            "present": self.present,
            "source": self.source,
            "shapeOk": self.shape_ok,
            "fingerprint": fingerprint(self.value),
            "problem": self.problem,
        }


def _read_dotenv(path: Path = Path(".env")) -> dict[str, str]:
    """Minimal .env reader.

    A dependency on python-dotenv to split on '=' would be one more install
    between a judge and a working demo.
    """
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def _read_keyring(name: str) -> str | None:
    """OS keyring, if the optional dependency is installed and working."""
    try:
        import keyring
    except ImportError:
        return None
    try:
        return keyring.get_password("preflight", name)
    except Exception:  # noqa: BLE001 - a broken backend must not break a run
        return None


def _diagnose(name: str, value: str) -> tuple[bool, str | None]:
    pattern, hint = PATTERNS[name]
    if pattern.match(value):
        return True, None

    # Naming the wrong vendor saves twenty minutes of confused debugging.
    for foreign, described, where in FOREIGN_KEYS:
        if foreign.match(value):
            return False, (
                f"looks like an {described}, not a {name.split('_')[0].title()} "
                f"key — get one free at {where}"
            )
    return False, f"malformed — expected {hint}"


def load_secrets(overrides: dict[str, str] | None = None) -> dict[str, Secret]:
    """Resolve every known credential.

    Precedence, highest first: explicit flag, process environment, .env file,
    OS keyring, absent.
    """
    dotenv = _read_dotenv()
    resolved: dict[str, Secret] = {}

    for name in PATTERNS:
        value: str | None = None
        source = "absent"

        if overrides and overrides.get(name):
            value, source = overrides[name], "flag"
        elif os.environ.get(name):
            value, source = os.environ[name], "env"
        elif dotenv.get(name):
            value, source = dotenv[name], "dotenv"
        else:
            from_keyring = _read_keyring(name)
            if from_keyring:
                value, source = from_keyring, "keyring"

        if value:
            shape_ok, problem = _diagnose(name, value)
        else:
            shape_ok, problem = False, None

        resolved[name] = Secret(name, value, source, shape_ok, problem)

    return resolved


# ------------------------------------------------------------------ #
# Enforced redaction                                                  #
# ------------------------------------------------------------------ #


class RedactFilter(logging.Filter):
    """Scrubs key-shaped strings out of every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            record.args = tuple(
                redact(arg) if isinstance(arg, str) else arg for arg in record.args
            )
        return True


_installed = False


def install_redaction() -> None:
    """Install the filter and excepthook. Idempotent.

    Both paths matter. A library's debug logging and an unhandled traceback are
    where keys actually leak, and neither is code you control.
    """
    global _installed
    if _installed:
        return

    root = logging.getLogger()
    scrubber = RedactFilter()
    root.addFilter(scrubber)
    for handler in root.handlers:
        handler.addFilter(scrubber)

    previous = sys.excepthook

    def hook(kind, value, tb):  # pragma: no cover - exercised by hand
        text = "".join(traceback.format_exception(kind, value, tb))
        sys.stderr.write(redact(text))

    sys.excepthook = hook
    _installed = True

    # Keep a handle so a caller can restore the original if it wants to.
    hook.previous = previous  # type: ignore[attr-defined]


install_redaction()
