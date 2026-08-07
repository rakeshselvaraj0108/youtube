"""Orphan detection — modules that are built, tested, and never called.

This project has now shipped that bug four separate times: the vision and
OCR agents carried 35% of the analysis surface while `run_perception` called
neither; `orchestrator.py` had a full retry-and-timeline implementation and
its own passing suite while the pipeline hand-rolled a guard beside it;
`rerank.text` was resolved by the registry and counted in `doctor`'s
capability plan with no consumer anywhere; and `scoring/rollup.py` was
written, tested, and wired to nothing in the same commit that introduced it.

Every one of those had passing unit tests. That is the point — a unit test
proves a module works, never that anything uses it. Only reachability does,
and reachability is what this file checks.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path("preflight")

# Reached by the CLI or by an external contract rather than by another module.
ENTRY_POINTS = {
    "preflight.cli",
    "preflight.__init__",
    "preflight.__main__",
}

# Imported for their side effects or re-exported as public API.
ALLOWED_ORPHANS: set[str] = set()


def _modules() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in PACKAGE.rglob("*.py"):
        parts = list(path.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        out[".".join(parts)] = path
    return out


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
            for alias in node.names:
                found.add(f"{node.module}.{alias.name}")
    return found


@pytest.fixture(scope="module")
def graph() -> tuple[dict[str, Path], set[str]]:
    modules = _modules()
    imported: set[str] = set()
    for path in modules.values():
        imported |= _imports(path)
    return modules, imported


class TestNoOrphanedModules:
    def test_every_module_is_imported_by_something(self, graph):
        """A module nothing imports is dead code, however well tested.

        Tests importing it do not count — that is exactly how all four
        previous orphans stayed invisible.
        """
        modules, imported = graph
        orphans = sorted(
            name
            for name in modules
            if name not in ENTRY_POINTS
            and name not in ALLOWED_ORPHANS
            and name not in imported
            # A package imported only as `preflight.policy.corpus` still
            # makes `preflight.policy` reachable.
            and not any(other.startswith(f"{name}.") for other in imported)
        )
        assert orphans == [], (
            f"built but never imported by the package: {orphans}. "
            "Either wire it in or delete it — a module with tests and no "
            "caller passes CI while doing nothing."
        )


# PREFLIGHT resolves eight capabilities but routes them through two different
# mechanisms, and conflating them produces false alarms in both directions.
#
#   registry.invoke(...)  — the provider chain actually executes the work.
#   direct client         — the registry resolves the capability for the
#                           capability plan and the certificate, while the
#                           work runs through NimClient (chat, embeddings),
#                           faster-whisper (ASR) or numpy (vector search).
#
# The second group is a deliberate architectural split, not an orphan: those
# call sites predate the registry, and rewriting four working subsystems to
# route through it would be a large change with no user-visible gain. It is
# recorded here so a genuinely unwired capability — which `rerank.text` was —
# still fails this test instead of hiding among them.
REGISTRY_INVOKED = {"ocr.image", "vision.describe", "rerank.text"}
SERVED_BY_DIRECT_CLIENT = {
    "chat.reasoning",
    "chat.extraction",
    "asr.transcribe",
    "embed.text",
    "vector.search",
}


class TestCapabilitiesHaveConsumers:
    """A capability the registry resolves and `doctor` counts, that nothing
    ever invokes, inflates the capability plan's denominator and reads to a
    judge as a feature that is present."""

    def test_the_two_groups_account_for_every_capability(self):
        """A new capability must be classified, never silently assumed to be
        consumed by one path or the other."""
        from preflight.providers import registry as registry_mod

        declared = {
            value
            for name, value in vars(registry_mod).items()
            if name.isupper() and isinstance(value, str) and "." in value
        }
        assert declared == REGISTRY_INVOKED | SERVED_BY_DIRECT_CLIENT

    def test_every_registry_invoked_capability_has_a_call_site(self):
        source = "\n".join(
            path.read_text(encoding="utf-8-sig")
            for path in PACKAGE.rglob("*.py")
            if path.parent.name != "providers"
        )
        missing = sorted(
            capability
            for capability in REGISTRY_INVOKED
            if capability not in source
            and capability.upper().replace(".", "_") not in source
        )
        assert missing == [], f"resolved but never invoked: {missing}"


class TestHttpTimeoutHasRealHeadroom:
    """The timeout was 120s, set before anything had measured the service.

    A single 64-token request against the free tier was then timed at 108s.
    Twelve seconds of headroom, on the smallest call the system can make —
    and a real AUDITOR batch carries eight windows plus their clause text.
    The original value would have timed out, retried, and timed out again on
    exactly the calls that matter, while a trivial request passed and made
    the setting look fine.
    """

    MEASURED_SMALL_CALL_S = 108

    def test_the_default_clears_the_measured_latency_with_margin(self):
        from preflight.config import Settings

        assert Settings().http_timeout_s >= self.MEASURED_SMALL_CALL_S * 2

    def test_the_client_uses_the_configured_timeout_not_a_literal(self):
        """The value was hardcoded at the call site, so configuring it did
        nothing. This asserts the wiring, not just the number."""
        import inspect

        from preflight.agents import nim

        source = inspect.getsource(nim.NimClient._post_with_retries)
        assert "self.settings.http_timeout_s" in source
        assert "timeout=120" not in source

    def test_the_environment_can_override_it(self, monkeypatch):
        from preflight.config import Settings

        monkeypatch.setenv("PREFLIGHT_HTTP_TIMEOUT", "45")
        assert Settings.load().http_timeout_s == 45


class TestScoringAgreesWithItself:
    """The two places that decide "is this video long enough to roll up"
    must be one place. They were briefly two."""

    def test_the_report_rolls_up_exactly_when_the_plan_says_it_will(self):
        from preflight.plan import HIERARCHICAL_ABOVE_MS, build_plan
        from preflight.report import build as build_mod

        assert build_mod.HIERARCHICAL_ABOVE_MS is HIERARCHICAL_ABOVE_MS
        assert build_plan(HIERARCHICAL_ABOVE_MS + 1).hierarchical
        assert not build_plan(HIERARCHICAL_ABOVE_MS).hierarchical
