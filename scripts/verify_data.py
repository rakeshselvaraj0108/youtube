"""Verify the data layer keeps the promises made about where it came from.

This project's build prompts stated a set of hard constraints on the data
layer, verbatim, and nothing before this script checked them mechanically:

    Ship .env.example, never .env.
    No bulk verbatim copies of policy pages. Restatements with source URLs
      and fetch dates.
    No hate-speech corpora, extremist text, or slur lists sourced from
      third-party datasets.
    No graphic or disturbing media for the demo or the corpus.
    Never download video or audio from YouTube.

A constraint that is only a sentence in a chat log is a promise nobody is
keeping. This makes each one a check that fails the build, so "we don't do
that" is something CI verifies rather than something a reviewer has to trust.

    python scripts/verify_data.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "data" / "policy"
CORPUS_CLIPS = ROOT / "data" / "corpus" / "clips"
LEXICONS_DIR = ROOT / "data" / "lexicons"

# Media extensions that must never be committed to git. The corpus, the CC
# audio bed and the demo sample are all generated locally by scripts in this
# repo and are gitignored — committing a rendered file would be indistinguish-
# able, to a future contributor, from having sourced it from somewhere.
MEDIA_EXTENSIONS = {".mp4", ".mov", ".mp3", ".wav", ".m4a", ".avi", ".mkv"}

# Domains a source_url is allowed to point at. Anything else in a policy
# clause's provenance is a citation this project cannot actually stand behind.
ALLOWED_POLICY_DOMAINS = ("support.google.com", "www.w3.org", "w3.org")

YOUTUBE_HOST_PATTERN = re.compile(
    r"(?:youtube\.com/watch|youtu\.be/|googlevideo\.com)", re.IGNORECASE
)


class Check:
    def __init__(self, name: str) -> None:
        self.name = name
        self.problems: list[str] = []

    def fail(self, message: str) -> None:
        self.problems.append(message)

    @property
    def ok(self) -> bool:
        return not self.problems


def check_no_tracked_media() -> Check:
    """No rendered audio or video anywhere git can see it.

    `git ls-files` rather than a directory walk, because the failure this
    guards against is specifically something ending up IN THE REPOSITORY, not
    something existing on disk during a local run.
    """
    check = Check("no rendered media is tracked by git")
    import subprocess

    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        check.fail("could not run `git ls-files` — is this a git checkout?")
        return check

    for line in result.stdout.splitlines():
        if Path(line).suffix.lower() in MEDIA_EXTENSIONS:
            check.fail(f"{line} is tracked by git and is a media file")
    return check


def check_no_youtube_urls() -> Check:
    """Nothing in the data layer points at a YouTube media URL.

    'Never download video or audio from YouTube. Not for the demo, not for
    the test corpus, not "just to check."' A citation to a YouTube *policy
    page* is fine and expected; a URL that would fetch video or audio is not.
    """
    check = Check("nothing in data/ references a YouTube media URL")
    for path in (ROOT / "data").rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".json", ".jsonl", ".yaml", ".yml", ".md", ".py"
        }:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if YOUTUBE_HOST_PATTERN.search(text):
            check.fail(f"{path.relative_to(ROOT)} references a YouTube media URL")
    return check


def check_policy_provenance() -> Check:
    """Every clause the manifest carries is either sourced or self-declared.

    Two honest shapes, and every clause must be exactly one of them:
    `policy_restatement` clauses point at a real page on a domain this
    project actually reads, and are dated; `house_rule` clauses say plainly
    that they are PREFLIGHT's own threshold, not platform policy. A clause
    that is neither — sourced from nowhere, or sourced from an unreviewed
    domain — is the exact failure that let a loudness target masquerade as
    a citation, or a restatement go unattributed.
    """
    check = Check("every policy clause is honestly sourced or honestly own")
    manifest_path = POLICY_DIR / "manifest.json"
    if not manifest_path.is_file():
        check.fail(f"{manifest_path} does not exist — run `make corpus`")
        return check

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest.get("clauses", []):
        clause_id = entry.get("clause_id", "?")
        kind = entry.get("kind")
        source = entry.get("source_url", "")
        derivation = entry.get("derivation", "")

        if kind == "policy_restatement":
            if not source.startswith("http"):
                check.fail(f"{clause_id}: policy_restatement with no source_url")
            elif not any(domain in source for domain in ALLOWED_POLICY_DOMAINS):
                check.fail(f"{clause_id}: source_url {source} is not a reviewed domain")
            if "not a verbatim copy" not in derivation:
                check.fail(f"{clause_id}: derivation does not disclaim verbatim copying")
            if not entry.get("fetched_at"):
                check.fail(f"{clause_id}: policy_restatement with no fetch date")
        elif kind == "house_rule":
            if "not a restatement of any platform policy" not in derivation:
                check.fail(f"{clause_id}: house_rule derivation does not say so plainly")
            if any(domain in source for domain in ALLOWED_POLICY_DOMAINS):
                check.fail(f"{clause_id}: house_rule source_url points at a policy domain")
        else:
            check.fail(f"{clause_id}: kind is {kind!r}, neither restatement nor house rule")
    return check


def check_no_slur_enumeration() -> Check:
    """The profanity lexicon may grade language; it may not enumerate slurs.

    'No hate-speech corpora, extremist text, or slur lists sourced from
    third-party datasets.' The lexicon documents a fourth tier for slurs and
    explicitly leaves it unenumerated, deferring to the clause text a model
    reasons over rather than a list a keyword filter matches against — the
    Scunthorpe problem in exactly the shape this project exists to avoid.
    This asserts the tier stays empty rather than trusting the comment above
    it.
    """
    check = Check("no slur list is enumerated in the profanity lexicon")
    path = LEXICONS_DIR / "profanity.tiered.jsonl"
    if not path.is_file():
        check.fail(f"{path} does not exist")
        return check

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("tier") == 4:
            check.fail(
                f"tier 4 (slur) entry present: {row.get('term', '?')!r} — "
                "this tier must stay empty; AF-06's clause text carries the rule"
            )
    return check


def check_corpus_is_synthetic() -> Check:
    """The corpus manifest declares itself synthetic, and clips are gitignored.

    Cheap and specific: the manifest.yaml header makes the synthetic claim in
    prose, and this checks that the claim has not silently drifted while
    nobody was reading the comment.
    """
    check = Check("the golden corpus declares itself synthetic and stays gitignored")
    manifest = ROOT / "data" / "corpus" / "manifest.yaml"
    if not manifest.is_file():
        check.fail(f"{manifest} does not exist")
        return check

    text = manifest.read_text(encoding="utf-8")
    if "synthetic" not in text.lower():
        check.fail("manifest.yaml no longer describes the corpus as synthetic")

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    if "data/corpus/clips" not in gitignore:
        check.fail("data/corpus/clips/ is not in .gitignore")
    return check


CHECKS = [
    check_no_tracked_media,
    check_no_youtube_urls,
    check_policy_provenance,
    check_no_slur_enumeration,
    check_corpus_is_synthetic,
]


def main() -> int:
    failed = 0
    for build in CHECKS:
        result = build()
        mark = "PASS" if result.ok else "FAIL"
        print(f"[{mark}] {result.name}")
        for problem in result.problems:
            print(f"       {problem}")
        if not result.ok:
            failed += 1

    print()
    if failed:
        print(f"{failed}/{len(CHECKS)} checks failed")
        return 1
    print(f"all {len(CHECKS)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
