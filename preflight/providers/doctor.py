"""`preflight doctor` — diagnose everything, and say how to fix it.

Worth its build time twice over. A judge with a setup problem self-diagnoses
instead of giving up, and the zero-key capability becomes *visible* rather than
being a claim in a README.

Every failure line carries an actionable fix. A diagnosis without a remedy just
tells someone they have a problem they already knew about.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from preflight.providers.governor import GOVERNORS
from preflight.providers.registry import Registry
from preflight.providers.secrets import fingerprint, load_secrets

OK, WARN, FAIL, NOTE = "ok", "warn", "fail", "note"


@dataclass
class Check:
    status: str
    name: str
    detail: str = ""
    fix: str = ""

    @property
    def glyph(self) -> str:
        return {OK: "[+]", WARN: "[!]", FAIL: "[x]", NOTE: "[ ]"}[self.status]

    def to_json(self) -> dict[str, Any]:
        payload = {"status": self.status, "name": self.name, "detail": self.detail}
        if self.fix:
            payload["fix"] = self.fix
        return payload


@dataclass
class Report:
    sections: dict[str, list[Check]] = field(default_factory=dict)

    def add(self, section: str, check: Check) -> None:
        self.sections.setdefault(section, []).append(check)

    @property
    def failures(self) -> list[Check]:
        return [c for checks in self.sections.values() for c in checks if c.status == FAIL]

    @property
    def warnings(self) -> list[Check]:
        return [c for checks in self.sections.values() for c in checks if c.status == WARN]

    def to_json(self) -> dict[str, Any]:
        return {
            "sections": {
                name: [c.to_json() for c in checks]
                for name, checks in self.sections.items()
            },
            "failures": len(self.failures),
            "warnings": len(self.warnings),
        }


def _binary(name: str, *, required: bool, fix: str) -> Check:
    path = shutil.which(name)
    if not path:
        return Check(FAIL if required else NOTE, name, "not on PATH", fix)
    try:
        out = subprocess.run(
            [path, "-version" if name.startswith("ff") else "--version"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        version = out.splitlines()[0] if out else ""
        version = version.replace(f"{name} version ", "").split(" Copyright")[0]
    except Exception:  # noqa: BLE001
        version = "present"
    return Check(OK, name, version.strip()[:48])


def _hf_cached(fragment: str) -> Path | None:
    root = Path.home() / ".cache" / "huggingface" / "hub"
    if not root.is_dir():
        return None
    return next((e for e in root.iterdir() if fragment in e.name), None)


def _directory_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024 / 1024


def run_doctor(*, offline: bool = False, deep: bool = False) -> Report:
    report = Report()

    # ---- system -----------------------------------------------------
    report.add("SYSTEM", _binary(
        "ffmpeg", required=True,
        fix="winget install Gyan.FFmpeg  |  brew install ffmpeg  |  apt install ffmpeg",
    ))
    report.add("SYSTEM", _binary(
        "ffprobe", required=True,
        fix="ships with ffmpeg — install ffmpeg",
    ))
    report.add("SYSTEM", _binary(
        "tesseract", required=False,
        fix="optional — enables OCR. apt install tesseract-ocr",
    ))
    report.add("SYSTEM", _binary(
        "fpcalc", required=False,
        fix="optional — enables acoustic fingerprints. apt install libchromaprint-tools",
    ))
    report.add("SYSTEM", Check(
        OK, "python", f"{sys.version_info.major}.{sys.version_info.minor}."
                      f"{sys.version_info.micro}",
    ))

    # ---- local models -----------------------------------------------
    whisper = _hf_cached("faster-whisper-base.en")
    if whisper:
        report.add("LOCAL MODELS", Check(
            OK, "faster-whisper", f"base.en int8 cached, {_directory_size_mb(whisper):.0f} MB"
        ))
    else:
        report.add("LOCAL MODELS", Check(
            WARN, "faster-whisper", "not cached",
            "preflight models pull   (downloads once; never mid-run)",
        ))

    minilm = _hf_cached("all-MiniLM-L6-v2")
    report.add("LOCAL MODELS", Check(
        OK if minilm else NOTE,
        "all-MiniLM-L6-v2",
        f"cached, {_directory_size_mb(minilm):.0f} MB" if minilm else "not cached (optional)",
        "" if minilm else "optional — only needed for offline dense retrieval",
    ))

    # ---- credentials ------------------------------------------------
    secrets = load_secrets()
    for name, secret in secrets.items():
        if not secret.present:
            report.add("CREDENTIALS", Check(
                NOTE, name, "not configured (optional)",
                "get a free NVIDIA key at build.nvidia.com, then: "
                "echo 'NVIDIA_API_KEY=nvapi-...' >> .env"
                if name == "NVIDIA_API_KEY" else "",
            ))
        elif not secret.shape_ok:
            report.add("CREDENTIALS", Check(
                FAIL, name,
                f"{secret.problem}  [{secret.source}] {fingerprint(secret.value)}",
                "fix the value in .env, or unset it to run fully local",
            ))
        else:
            report.add("CREDENTIALS", Check(
                OK, name, f"{fingerprint(secret.value)}  [{secret.source}]"
            ))

    # ---- capability plan --------------------------------------------
    registry = Registry(secrets, offline=offline)
    for capability, resolution in registry.plan.items():
        if resolution.tier_label == "null":
            report.add("CAPABILITY PLAN", Check(
                WARN, capability, f"unavailable — {resolution.reason}",
            ))
        elif resolution.degraded:
            report.add("CAPABILITY PLAN", Check(
                OK, capability,
                f"{resolution.provider.id:<8} {resolution.label:<38} fallback",
            ))
        else:
            report.add("CAPABILITY PLAN", Check(
                OK, capability,
                f"{resolution.provider.id:<8} {resolution.label:<38} preferred",
            ))

    # ---- live round trips -------------------------------------------
    if deep:
        for capability, resolution in registry.plan.items():
            if resolution.tier_label != "hosted":
                continue
            ok, detail, ms = resolution.provider.healthcheck()
            report.add("HEALTH", Check(
                OK if ok else FAIL, capability, f"{detail} · {ms} ms",
                "" if ok else "check the key, or run with --offline",
            ))

    # ---- budgets ----------------------------------------------------
    for vendor, gov in GOVERNORS.items():
        report.add("BUDGET", Check(
            NOTE, vendor,
            f"{gov.rpm} rpm · budget {gov.ledger.budget or 'unlimited'} · "
            f"{gov.ledger.calls} used · circuit {gov.breaker.state}",
        ))

    # ---- what can actually run --------------------------------------
    preferred, fallback, unavailable = registry.summary()
    total = len(registry.plan)
    report.add("CAPABILITY", Check(
        OK if unavailable == 0 else WARN,
        "pipeline",
        f"{total - unavailable}/{total} capabilities served "
        f"({preferred} preferred, {fallback} fallback, {unavailable} unavailable)",
    ))

    offline_registry = Registry(secrets, offline=True)
    _, _, offline_unavailable = offline_registry.summary()
    report.add("CAPABILITY", Check(
        OK, "offline pipeline",
        f"{total - offline_unavailable}/{total} capabilities served with no network",
    ))

    return report
