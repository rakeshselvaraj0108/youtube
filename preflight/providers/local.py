"""Local providers. No network, no key, no vendor.

These are the reason PREFLIGHT runs on a stranger's machine. `available()`
checks a binary or a cached model and never touches the network, so resolution
stays instant and offline.

Nothing here downloads mid-run. A model that is not cached reports unavailable
and points at `preflight models pull` — discovering a 140MB download halfway
through a demo is a worse outcome than a documented gap.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from preflight.providers.base import BaseProvider, Result, Served, Unavailable


def _hf_cached(fragment: str) -> bool:
    """Is a Hugging Face model already in the local cache?"""
    root = Path.home() / ".cache" / "huggingface" / "hub"
    if not root.is_dir():
        return False
    return any(fragment in entry.name for entry in root.iterdir())


class LocalWhisper(BaseProvider):
    id = "local"
    tier_label = "local"
    model = "faster-whisper base.en (int8)"

    def __init__(self, capability: str, model_id: str = "base.en") -> None:
        super().__init__(capability)
        self.model_id = model_id
        self.model = f"faster-whisper {model_id} (int8)"

    def available(self) -> tuple[bool, str]:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False, "faster-whisper not installed (pip install 'preflight[asr]')"
        if not _hf_cached(f"faster-whisper-{self.model_id}"):
            return False, f"{self.model_id} not cached — run: preflight models pull"
        return True, f"{self.model_id} int8 cached"

    def invoke(self, **kwargs: Any) -> Result:
        # Transcription is driven by perception/asr.py, which owns the cache
        # key and the artifact store. This provider exists so the capability
        # plan can report which backend served speech.
        return Served(
            value={"backend": "faster-whisper", "model_id": self.model_id},
            provider=f"local:{self.model}",
            tier=1,
            calls=0,
        )


class LocalMiniLM(BaseProvider):
    id = "local"
    tier_label = "local"
    model = "all-MiniLM-L6-v2"

    def available(self) -> tuple[bool, str]:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return False, "sentence-transformers not installed"
        if not _hf_cached("all-MiniLM-L6-v2"):
            return False, "all-MiniLM-L6-v2 not cached — run: preflight models pull"
        return True, "all-MiniLM-L6-v2 cached"

    def invoke(self, *, texts: list[str], **kwargs: Any) -> Result:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return Unavailable("sentence-transformers not installed", "local")

        started = time.monotonic()
        model = SentenceTransformer("all-MiniLM-L6-v2")
        vectors = model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        return Served(
            value=np.asarray(vectors, dtype=np.float32),
            provider=f"local:{self.model}",
            tier=1,
            calls=0,
            latency_ms=int((time.monotonic() - started) * 1000),
        )


# Where the standard installers put tesseract when they do not touch PATH.
# The Windows installer (UB-Mannheim) and Homebrew both routinely leave the
# binary unreachable from a shell that was already open, and every one of
# those machines reports "tesseract not on PATH" for a tool that is sitting
# right there. OCR carries the entire on-screen-secret surface — a leaked API
# key is the highest-consequence finding this engine produces — so silently
# skipping it because of a PATH quirk is the most expensive possible way to
# be technically correct.
_TESSERACT_FALLBACKS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    "/opt/homebrew/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/usr/bin/tesseract",
)


def find_tesseract() -> str | None:
    """PATH first, then the places the installers actually use."""
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in _TESSERACT_FALLBACKS:
        if Path(candidate).is_file():
            return candidate
    return None


class LocalTesseract(BaseProvider):
    id = "local"
    tier_label = "local"
    model = "tesseract"

    def available(self) -> tuple[bool, str]:
        binary = find_tesseract()
        if not binary:
            return False, "tesseract not installed"
        try:
            out = subprocess.run(
                [binary, "--version"], capture_output=True, text=True, timeout=10
            ).stdout
            version = out.splitlines()[0].split()[-1] if out else "?"
        except Exception:  # noqa: BLE001
            version = "?"
        self.model = f"tesseract {version}"
        return True, f"tesseract {version}"

    def invoke(self, *, image: Path, **kwargs: Any) -> Result:
        """Words with boxes and confidences, not a blob of text.

        `image_to_string` throws away everything A05 needs. A caption's role
        is decided by where it sits and how long it persists, so position is
        not a nicety here — without a box there is no way to tell a corner
        watermark from meme text, and both would land in the same bucket.
        """
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            return Unavailable("pytesseract/Pillow not installed", "local")

        # pytesseract does its own PATH lookup and will raise
        # TesseractNotFoundError even though `available()` just located the
        # binary — point it at the one actually found.
        binary = find_tesseract()
        if binary:
            pytesseract.pytesseract.tesseract_cmd = binary

        started = time.monotonic()
        with Image.open(image) as handle:
            width, height = handle.size
            data = pytesseract.image_to_data(
                handle, output_type=pytesseract.Output.DICT
            )

        words: list[dict[str, Any]] = []
        for index, text in enumerate(data.get("text", [])):
            text = (text or "").strip()
            if not text:
                continue
            try:
                confidence = float(data["conf"][index])
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            if confidence < 0:  # tesseract's "no estimate" sentinel
                continue
            words.append(
                {
                    "text": text,
                    # Normalised so downstream rules are resolution-independent.
                    "box": (
                        data["left"][index] / max(width, 1),
                        data["top"][index] / max(height, 1),
                        data["width"][index] / max(width, 1),
                        data["height"][index] / max(height, 1),
                    ),
                    "conf": confidence / 100.0,
                    "line": (data["block_num"][index], data["line_num"][index]),
                }
            )

        return Served(
            value={"words": words, "text": " ".join(w["text"] for w in words)},
            provider=f"local:{self.model}",
            tier=1,
            calls=0,
            latency_ms=int((time.monotonic() - started) * 1000),
        )


class NumpyVectorStore(BaseProvider):
    """Exact cosine search over an in-memory matrix.

    The corpus is roughly a hundred chunks. Exact search over a 119x1024 matrix
    is a single dot product — microseconds, no service, no index to corrupt, no
    approximate-recall failure mode to debug. A vector database earns its place
    at a million vectors; here it would be a dependency with nothing to do.

    Kept as a provider rather than hardcoded so a real vector store can serve
    the same capability when one is configured.
    """

    id = "local"
    tier_label = "local"
    model = "numpy exact search"

    def available(self) -> tuple[bool, str]:
        return True, "numpy exact cosine search"

    def invoke(
        self,
        *,
        matrix: np.ndarray | None = None,
        query: np.ndarray | None = None,
        top_k: int = 12,
        **kwargs: Any,
    ) -> Result:
        if matrix is None or query is None:
            return Unavailable("no vectors supplied", "local")

        started = time.monotonic()
        # Rows are L2-normalised upstream, so the dot product is cosine.
        similarity = matrix @ query
        order = np.argsort(-similarity)[:top_k]
        return Served(
            value=[(int(i), float(similarity[i])) for i in order],
            provider="local:numpy",
            tier=1,
            calls=0,
            latency_ms=int((time.monotonic() - started) * 1000),
        )


class NoLocalReranker(BaseProvider):
    """There is no honest local reranker, so this says so.

    Faking one — returning the input order and calling it reranked — would make
    the ablation table meaningless. Retrieval continues on raw RRF order and
    the report records that no reranking happened.
    """

    id = "local"
    tier_label = "local"
    model = "none"

    def available(self) -> tuple[bool, str]:
        return False, "no local reranker — retrieval uses raw RRF order"

    def invoke(self, **kwargs: Any) -> Result:
        return Unavailable("no local reranker; using RRF order", "local")
