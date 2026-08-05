"""Single-file HTML report.

`report.html` is one file. A judge double-clicks it and it works — offline, no
server, no network. That means the JS bundle, the CSS, the poster and every
evidence frame all travel inside it.

The React bundle is built once (`npm run build`); this inlines its assets and
injects the report as `window.__PREFLIGHT_REPORT__` before the bundle runs.
`src/lib/reportSource.ts` reads that and falls back to the fixture, so the same
build serves both the dev server and the CLI.

Inlining is done here rather than with a Vite plugin so the Python side owns
the whole emission path and there is one fewer npm dependency to install before
a report can be produced.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

SCRIPT_TAG = re.compile(
    r'<script[^>]*\ssrc="(?P<src>[^"]+)"[^>]*></script>', re.IGNORECASE
)
STYLE_TAG = re.compile(
    r'<link[^>]*\srel="stylesheet"[^>]*\shref="(?P<href>[^"]+)"[^>]*/?>', re.IGNORECASE
)
FONT_LINK = re.compile(r'<link[^>]*fonts\.(googleapis|gstatic)\.com[^>]*/?>', re.IGNORECASE)
PRECONNECT = re.compile(r'<link[^>]*rel="preconnect"[^>]*/?>', re.IGNORECASE)


class BundleMissing(FileNotFoundError):
    """The React bundle has not been built."""


def _asset(dist: Path, url: str) -> Path:
    return dist / url.lstrip("./").lstrip("/")


def _inline_css_assets(css: str, dist: Path) -> str:
    """Embed url(...) references a stylesheet points at."""

    def replace(match: re.Match[str]) -> str:
        raw = match.group(1).strip("'\"")
        if raw.startswith(("data:", "http:", "https:")):
            return match.group(0)
        target = _asset(dist, raw.split("?")[0])
        if not target.is_file():
            return match.group(0)
        suffix = target.suffix.lower().lstrip(".")
        mime = {
            "woff2": "font/woff2",
            "woff": "font/woff",
            "ttf": "font/ttf",
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "svg": "image/svg+xml",
        }.get(suffix, "application/octet-stream")
        payload = base64.b64encode(target.read_bytes()).decode("ascii")
        return f"url(data:{mime};base64,{payload})"

    return re.sub(r"url\(([^)]+)\)", replace, css)


def emit_html(
    report: dict[str, Any],
    dist: Path,
    destination: Path,
    *,
    after_report: dict[str, Any] | None = None,
) -> Path:
    index = Path(dist) / "index.html"
    if not index.is_file():
        raise BundleMissing(
            f"no built UI at {index}. Run `npm run build` first, or use "
            "--format json to emit JSON without the HTML report."
        )

    html = index.read_text(encoding="utf-8")
    dist = Path(dist)

    # Inline the stylesheet.
    def swap_style(match: re.Match[str]) -> str:
        target = _asset(dist, match.group("href"))
        if not target.is_file():
            return match.group(0)
        css = _inline_css_assets(target.read_text(encoding="utf-8"), dist)
        return f"<style>{css}</style>"

    html = STYLE_TAG.sub(swap_style, html)

    # Inline the script.
    def swap_script(match: re.Match[str]) -> str:
        target = _asset(dist, match.group("src"))
        if not target.is_file():
            return match.group(0)
        return f'<script type="module">{target.read_text(encoding="utf-8")}</script>'

    html = SCRIPT_TAG.sub(swap_script, html)

    # Remote fonts would make the page reach for the network on open. Strip the
    # links; the CSS already falls back to a monospace and a sans stack.
    html = FONT_LINK.sub("", html)
    html = PRECONNECT.sub("", html)

    payload = json.dumps(report, separators=(",", ":"), ensure_ascii=False)
    injection = [
        "<script>",
        f"window.__PREFLIGHT_REPORT__={payload};",
    ]
    if after_report is not None:
        after = json.dumps(after_report, separators=(",", ":"), ensure_ascii=False)
        injection.append(f"window.__PREFLIGHT_REPORT_AFTER__={after};")
    injection.append("</script>")
    script = "".join(injection)

    # Inject before </head> so the data exists before the bundle evaluates.
    if "</head>" in html:
        html = html.replace("</head>", script + "</head>", 1)
    else:  # pragma: no cover - defensive
        html = script + html

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    return destination


def emit_fixture(report: dict[str, Any], destination: Path) -> Path:
    """Write the UI's demo fixture as real engine output.

    This is what stops the demo showing numbers the engine cannot produce: the
    fixture stops being authored and becomes a recording of an actual run.
    """
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    body = f'''/**
 * GENERATED FILE — do not edit by hand.
 *
 * Emitted by `preflight check --emit-fixture`. This is a real analysis of a
 * real file, not authored data, which is what keeps the demo honest: the page
 * cannot show a number the engine is incapable of producing.
 */

import type {{ AnalysisReport }} from '@/types/analysis';

export const beforeReport: AnalysisReport = {payload} as AnalysisReport;

export const afterReport: AnalysisReport = beforeReport;

export const DEMO_DURATION_MS = beforeReport.video.durationMs;
'''
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(body, encoding="utf-8")
    return destination
