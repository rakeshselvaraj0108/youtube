"""Qdrant as a vector.search provider.

Qdrant serves the same capability as the numpy store and is preferred when
configured. Worth being straight about why it is a *tier* rather than the
implementation:

The policy corpus is roughly 120 chunks. Exact cosine search over a 120x1024
matrix is one dot product — microseconds, no service to run, no index to
rebuild, and no approximate-recall failure mode. At that size a vector database
is a dependency with nothing to do, and making it mandatory would break the
property that this tool runs on a stranger's machine with nothing installed.

Where it does earn its place, and why this exists:

- **The archive.** The Drift Watcher re-lints a back catalogue. A creator with
  two hundred videos has hundreds of thousands of window embeddings, and that
  is past the point where an in-memory matrix is sensible.
- **Persistence.** Embeddings survive between runs and across machines instead
  of being rebuilt from cache.
- **Filtered search.** `clause_id`, `severity`, `video_hash` as payload filters
  is real functionality the numpy path fakes with a list comprehension.

So: configured, it serves. Absent, numpy serves, and the report says which.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import numpy as np

from preflight.providers.base import BaseProvider, Result, Served, Unavailable
from preflight.providers.governor import governor
from preflight.providers.secrets import Secret, redact

VENDOR = "qdrant"
DEFAULT_URL = "http://localhost:6333"
POLICY_COLLECTION = "preflight_policy"
WINDOW_COLLECTION = "preflight_windows"


class QdrantVectorStore(BaseProvider):
    id = VENDOR
    tier_label = "hosted"
    model = "qdrant"

    def __init__(
        self,
        capability: str,
        *,
        url: str = DEFAULT_URL,
        secret: Secret | None = None,
        collection: str = POLICY_COLLECTION,
    ) -> None:
        super().__init__(capability)
        self.url = url.rstrip("/")
        self.secret = secret
        self.collection = collection
        self.model = f"qdrant {collection}"
        self.gov = governor(VENDOR, rpm=600, call_budget=0)
        self._client = None

    # ---------------------------------------------------------------- #

    def _connect(self):
        if self._client is not None:
            return self._client
        from qdrant_client import QdrantClient

        api_key = self.secret.value if self.secret and self.secret.present else None
        self._client = QdrantClient(url=self.url, api_key=api_key, timeout=10)
        return self._client

    def available(self) -> tuple[bool, str]:
        """Offline check only — the import and the configured URL.

        Deliberately does NOT ping the server. `available()` runs for every
        provider in every chain at startup, and a TCP connect to a host that is
        not there costs a timeout each time. Reachability is `doctor`'s job.
        """
        try:
            import qdrant_client  # noqa: F401
        except ImportError:
            return False, "qdrant-client not installed (pip install 'preflight[qdrant]')"
        if not self.url:
            return False, "QDRANT_URL not configured"
        return True, f"configured at {self.url}"

    def healthcheck(self) -> tuple[bool, str, int]:
        ok, reason = self.available()
        if not ok:
            return False, reason, 0
        started = time.monotonic()
        try:
            collections = self._connect().get_collections()
            names = [c.name for c in collections.collections]
            elapsed = int((time.monotonic() - started) * 1000)
            present = self.collection in names
            return True, (
                f"{len(names)} collection(s); "
                f"{self.collection} {'present' if present else 'not yet created'}"
            ), elapsed
        except Exception as exc:  # noqa: BLE001
            return False, redact(str(exc))[:160], int((time.monotonic() - started) * 1000)

    # ---------------------------------------------------------------- #

    def ensure_collection(self, dim: int) -> Result:
        try:
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            return Unavailable("qdrant-client not installed", VENDOR)

        try:
            client = self._connect()
            existing = {c.name for c in client.get_collections().collections}
            if self.collection not in existing:
                client.create_collection(
                    collection_name=self.collection,
                    # Cosine, because vectors are L2-normalised everywhere else
                    # in this codebase and switching metric here would silently
                    # change what "similarity" means between tiers.
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
            return Served(value=self.collection, provider=f"{VENDOR}:{self.collection}")
        except Exception as exc:  # noqa: BLE001
            return Unavailable(f"qdrant unreachable: {redact(str(exc))}", VENDOR, True)

    def upsert(self, vectors: np.ndarray, payloads: list[dict[str, Any]]) -> Result:
        try:
            from qdrant_client.models import PointStruct
        except ImportError:
            return Unavailable("qdrant-client not installed", VENDOR)

        if vectors.shape[0] != len(payloads):
            return Unavailable(
                f"{vectors.shape[0]} vectors for {len(payloads)} payloads", VENDOR
            )

        created = self.ensure_collection(int(vectors.shape[1]))
        if not created:
            return created

        started = time.monotonic()
        try:
            points = [
                PointStruct(
                    # Deterministic id from the payload, so re-indexing the same
                    # corpus updates in place rather than duplicating.
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, str(payload.get("id", index)))),
                    vector=[float(v) for v in vectors[index]],
                    payload=payload,
                )
                for index, payload in enumerate(payloads)
            ]
            self._connect().upsert(collection_name=self.collection, points=points)
        except Exception as exc:  # noqa: BLE001
            return Unavailable(f"upsert failed: {redact(str(exc))}", VENDOR, True)

        return Served(
            value=len(payloads),
            provider=f"{VENDOR}:{self.collection}",
            calls=1,
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    def invoke(
        self,
        *,
        query: np.ndarray | None = None,
        top_k: int = 12,
        where: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Result:
        if query is None:
            return Unavailable("no query vector supplied", VENDOR)

        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
        except ImportError:
            return Unavailable("qdrant-client not installed", VENDOR)

        query_filter = None
        if where:
            query_filter = Filter(
                must=[
                    FieldCondition(key=key, match=MatchValue(value=value))
                    for key, value in where.items()
                ]
            )

        started = time.monotonic()
        try:
            hits = self._connect().search(
                collection_name=self.collection,
                query_vector=[float(v) for v in query],
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
        except Exception as exc:  # noqa: BLE001
            # Unreachable is a demotion, not a crash: the numpy tier below can
            # serve the same capability.
            return Unavailable(f"search failed: {redact(str(exc))}", VENDOR, True)

        return Served(
            value=[
                (int(hit.payload.get("index", 0)), float(hit.score))
                for hit in hits
            ],
            provider=f"{VENDOR}:{self.collection}",
            tier=0,
            calls=1,
            latency_ms=int((time.monotonic() - started) * 1000),
            meta={"payloads": [hit.payload for hit in hits]},
        )
