"""Content-addressed artifact store.

Every artifact is keyed by the BLAKE3 hash of its inputs. Two consequences that
matter more than they sound:

1. Re-running `preflight check` on an unchanged video costs **zero API calls**
   and produces a byte-identical report. That is what makes a live demo safe on
   a rate-limited free tier, and it is what makes `report_hash` a
   reproducibility proof rather than a decoration.

2. Changing the policy corpus invalidates only retrieval and adjudication.
   Ingest and speech survive, because their keys do not include the corpus.

Namespaces mirror the pipeline stages:

    v/  video bytes            -> probe, audio, frames, poster
    t/  audio + asr model id   -> transcript
    f/  fingerprint audio      -> acoustic fingerprints
    p/  policy corpus          -> FAISS index
    r/  everything above       -> report
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

try:  # pragma: no cover - exercised by whichever branch is installed
    from blake3 import blake3 as _blake3

    def _hash_bytes(data: bytes) -> str:
        return _blake3(data).hexdigest()

    def _hash_file(path: Path, chunk: int = 1 << 20) -> str:
        hasher = _blake3()
        with path.open("rb") as handle:
            while block := handle.read(chunk):
                hasher.update(block)
        return hasher.hexdigest()

    HASH_NAME = "blake3"
except ImportError:  # pragma: no cover
    # Degrade, never fail. blake2b is in the stdlib and gives the same
    # properties we actually rely on; the algorithm name travels in the
    # attestation so a reader always knows which one produced the digest.
    import hashlib

    def _hash_bytes(data: bytes) -> str:
        return hashlib.blake2b(data, digest_size=32).hexdigest()

    def _hash_file(path: Path, chunk: int = 1 << 20) -> str:
        hasher = hashlib.blake2b(digest_size=32)
        with path.open("rb") as handle:
            while block := handle.read(chunk):
                hasher.update(block)
        return hasher.hexdigest()

    HASH_NAME = "blake2b"


Namespace = Literal["v", "t", "f", "p", "r"]


def hash_bytes(data: bytes) -> str:
    return _hash_bytes(data)


def hash_file(path: Path) -> str:
    return _hash_file(Path(path))


def hash_text(text: str) -> str:
    return _hash_bytes(text.encode("utf-8"))


def hash_json(value: Any) -> str:
    """Hash a structure by its canonical serialisation.

    Sorted keys and no incidental whitespace, so a dict that differs only in
    insertion order hashes the same. Without this, cache keys would depend on
    Python dict ordering and a re-run would miss for no reason.
    """
    return hash_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


def hash_many(parts: Iterable[str]) -> str:
    """Combine component hashes into one key, order-sensitively."""
    return hash_text(" ".join(parts))


def prefixed(digest: str) -> str:
    """Display form carried in reports and attestations, e.g. `b3:9f2c81…`."""
    tag = "b3" if HASH_NAME == "blake3" else "b2"
    return f"{tag}:{digest}"


@dataclass(frozen=True)
class Entry:
    """A directory in the store. Present only if `ready` was written."""

    root: Path

    @property
    def marker(self) -> Path:
        return self.root / ".ready"

    @property
    def exists(self) -> bool:
        # A directory without the marker is a partial write from a crashed or
        # interrupted run. Treating it as a miss is what keeps a half-extracted
        # frame set from being served as a cache hit.
        return self.marker.is_file()

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def commit(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.marker.write_text("", encoding="utf-8")

    def discard(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

    def read_json(self, name: str) -> Any:
        return json.loads(self.path(name).read_text(encoding="utf-8"))

    def write_json(self, name: str, value: Any) -> Path:
        target = self.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        return target


class Store:
    """Read-through content-addressed store rooted at `.preflight/cache`."""

    def __init__(self, root: Path | str = ".preflight/cache") -> None:
        self.root = Path(root)

    def entry(self, namespace: Namespace, key: str) -> Entry:
        return Entry(self.root / namespace / key)

    def stats(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for namespace in ("v", "t", "f", "p", "r"):
            directory = self.root / namespace
            counts[namespace] = (
                sum(1 for child in directory.iterdir() if (child / ".ready").is_file())
                if directory.is_dir()
                else 0
            )
        return counts

    def clear(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)
