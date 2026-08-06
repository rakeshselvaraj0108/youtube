"""Building the retrieval index.

Dense vectors are cached against the corpus digest and the embedding model id,
so changing a single clause re-embeds the corpus once and every subsequent run
is free. That is also what makes the Drift Watcher cheap: a clause edit
invalidates retrieval without touching ingest or speech.

Embedding backends, in preference order:

1. **NIM** when a key is present — `nv-embedqa-e5-v5`, asymmetric (passages and
   queries are embedded with different `input_type` values, which is what that
   model is trained for).
2. **sentence-transformers** when installed — offline, no key.
3. **None.** BM25 alone still works, and retrieval reports itself as sparse-only
   rather than pretending it had vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from preflight import cas
from preflight.agents.nim import NimClient, NimUnavailable
from preflight.config import Settings
from preflight.policy.corpus import Corpus
from preflight.policy.retrieval import Retriever, normalize

EMBED_BATCH = 32


@dataclass
class IndexBuild:
    retriever: Retriever
    backend: str
    dense: bool
    calls: int
    log: list[str]
    embed_query: Callable[[str], np.ndarray | None]
    scope: str = "policy"

    @property
    def size(self) -> int:
        return len(self.retriever.corpus.chunks)


@dataclass
class ScopedIndexes:
    """One index per retrieval scope.

    Each agent searches only the clauses that could possibly apply to it. The
    policy triad never retrieves the paid-promotion clause; the metadata agent
    never retrieves a firearms clause. Beyond precision this is cheaper — a
    query embeds once and searches thirty vectors instead of a hundred and
    twenty.
    """

    indexes: dict[str, IndexBuild]
    log: list[str] = field(default_factory=list)
    calls: int = 0

    def __getitem__(self, scope: str) -> IndexBuild:
        return self.indexes[scope]

    def get(self, scope: str) -> IndexBuild | None:
        return self.indexes.get(scope)

    @property
    def backend(self) -> str:
        backends = {index.backend for index in self.indexes.values()}
        return next(iter(backends)) if len(backends) == 1 else "mixed"

    @property
    def dense(self) -> bool:
        return any(index.dense for index in self.indexes.values())


def _nim_embeddings(
    corpus: Corpus, client: NimClient, model: str
) -> tuple[np.ndarray | None, int, list[str]]:
    log: list[str] = []
    texts = [chunk.for_prompt() for chunk in corpus.chunks]
    vectors: list[np.ndarray] = []
    calls_before = client.usage.calls

    try:
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start : start + EMBED_BATCH]
            matrix = client.embed(batch, model=model, input_type="passage")
            if matrix is None:
                return None, 0, ["embedding endpoint returned no vectors"]
            vectors.append(matrix)
    except NimUnavailable as exc:
        return None, client.usage.calls - calls_before, [f"dense embeddings unavailable: {exc}"]

    stacked = np.vstack(vectors)
    calls = client.usage.calls - calls_before
    log.append(
        f"embedded {len(texts)} chunks with {model} "
        f"({calls} call{'s' if calls != 1 else ''}, dim {stacked.shape[1]})"
    )
    return stacked, calls, log


def _local_embeddings(corpus: Corpus) -> tuple[np.ndarray | None, list[str]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None, []

    model = SentenceTransformer("all-MiniLM-L6-v2")
    matrix = model.encode(
        [chunk.for_prompt() for chunk in corpus.chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return matrix.astype(np.float32), [
        f"embedded {len(corpus.chunks)} chunks locally with all-MiniLM-L6-v2"
    ]


def _reranker(registry) -> Callable | None:
    """A cross-encoder rerank callable, when something can actually serve one.

    `rerank.text` was resolved by the registry, reported by `doctor` and
    counted in the capability plan while nothing in the project ever called
    it — a capability that existed on paper only. There is deliberately no
    local fallback (see `NoLocalReranker`): faking a reranker by returning
    the input order would make the ablation table meaningless, so offline
    runs return None here and retrieval stays on raw RRF order.
    """
    if registry is None:
        return None
    from preflight.providers.registry import RERANK_TEXT

    if registry.is_degraded(RERANK_TEXT):
        return None

    def rerank(query: str, passages: list[str]):
        result = registry.invoke(RERANK_TEXT, query=query, passages=passages)
        return result.value if result else None

    return rerank


def build_index(
    corpus: Corpus,
    settings: Settings,
    store: cas.Store,
    client: NimClient | None = None,
    registry=None,
) -> IndexBuild:
    """Assemble a retriever, using the best embedding backend available."""
    log: list[str] = []
    embeddings: np.ndarray | None = None
    backend = "bm25"
    calls = 0

    cache_key = cas.hash_json(
        {"digest": corpus.digest, "model": settings.models.embed, "v": 2}
    )
    entry = store.entry("p", f"index-{cache_key}")

    if entry.exists:
        payload = entry.read_json("index.json")
        embeddings = np.array(payload["vectors"], dtype=np.float32)
        backend = payload["backend"]
        log.append(f"index cache hit · {backend} · {embeddings.shape[0]} chunks")
    else:
        if client is not None and client.online:
            embeddings, calls, nim_log = _nim_embeddings(
                corpus, client, settings.models.embed
            )
            log.extend(nim_log)
            if embeddings is not None:
                backend = f"nim:{settings.models.embed}"

        if embeddings is None:
            embeddings, local_log = _local_embeddings(corpus)
            log.extend(local_log)
            if embeddings is not None:
                backend = "local:all-MiniLM-L6-v2"

        if embeddings is not None:
            entry.discard()
            entry.root.mkdir(parents=True, exist_ok=True)
            entry.write_json(
                "index.json", {"backend": backend, "vectors": embeddings.tolist()}
            )
            entry.commit()

    if embeddings is None:
        log.append("sparse retrieval only — BM25 without dense fusion")

    rerank = _reranker(registry)
    retriever = Retriever(corpus, embeddings=embeddings, rerank=rerank)
    if rerank is not None:
        log.append("cross-encoder reranking the fused pool")

    def embed_query(text: str) -> np.ndarray | None:
        if embeddings is None:
            return None
        if backend.startswith("nim:") and client is not None and client.online:
            try:
                matrix = client.embed([text], model=settings.models.embed, input_type="query")
            except NimUnavailable:
                return None
            return None if matrix is None else matrix[0]
        if backend.startswith("local:"):
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                return None
            model = SentenceTransformer("all-MiniLM-L6-v2")
            vector = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
            return normalize(vector.astype(np.float32))[0]
        return None

    return IndexBuild(
        retriever=retriever,
        backend=backend,
        dense=embeddings is not None,
        calls=calls,
        log=log,
        embed_query=embed_query,
        scope=getattr(corpus, "scope_name", "policy"),
    )


def build_scoped_indexes(
    corpus: Corpus,
    settings: Settings,
    store: cas.Store,
    client: NimClient | None = None,
    scopes: list[str] | None = None,
    registry=None,
) -> ScopedIndexes:
    """Build one index per scope.

    Scopes are cached independently, keyed on the sub-corpus digest, so editing
    a metadata clause re-embeds thirty chunks rather than a hundred and twenty
    — which also means the Drift Watcher only pays for what actually moved.
    """
    wanted = scopes or corpus.scopes
    built: dict[str, IndexBuild] = {}
    log: list[str] = []
    calls = 0

    for scope in wanted:
        sub = corpus.scoped(scope)
        if not sub.chunks:
            continue
        index = build_index(sub, settings, store, client, registry)
        index.scope = scope
        built[scope] = index
        calls += index.calls
        log.append(f"{scope}: {len(sub.chunks)} chunks from {len(sub.clauses)} clauses")
        log.extend(f"  {line}" for line in index.log)

    return ScopedIndexes(indexes=built, log=log, calls=calls)
