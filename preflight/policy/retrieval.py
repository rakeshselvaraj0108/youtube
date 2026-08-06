"""Hybrid retrieval — BM25 fused with dense embeddings by RRF.

Two deliberate decisions:

**BM25 is implemented here rather than imported.** It is roughly thirty lines of
numpy and removes a dependency from the zero-key path. More importantly it is
the retriever that always works: dense embeddings need either a network call or
a 2GB torch install, and neither can be assumed on a stranger's machine.

**No FAISS.** The corpus is tens of chunks. Exact search over a small matrix is
instant and removes an entire class of approximate-index bugs. FAISS earns its
place at a million vectors, not at seventy.

Fusion is Reciprocal Rank Fusion:

    RRF(d) = Σ_r  1 / (k + rank_r(d))

Dense retrieval misses exact tokens — a specific slur, a named event — and BM25
catches them. BM25 misses paraphrase and dense catches that. RRF needs no score
calibration between the two, which is why it is used instead of a weighted sum.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np

from preflight.policy.corpus import Chunk, Corpus

RRF_K = 60

# Given (query, passages), returns (index_into_passages, score) best-first, or
# None when the reranker could not answer.
Reranker = Callable[[str, "list[str]"], "list[tuple[int, float]] | None"]

# Weighted RRF. Transcript language and policy language rarely share tokens
# ("this is fucked" vs "strong profanity"), so dense carries the semantic load
# and BM25 exists to catch the exact tokens dense misses — a specific slur, a
# named event. Giving them equal weight on a 70-chunk corpus lets BM25's
# incidental word matches outrank real semantic hits.
DENSE_WEIGHT = 1.0
SPARSE_WEIGHT = 0.5

# A BM25 hit below this fraction of the query's own top score is an incidental
# match on a common word, not a lexical signal worth fusing.
SPARSE_FLOOR = 0.35

TOKEN = re.compile(r"[a-z0-9']+")

# Words that carry no retrieval signal. Kept short deliberately: an aggressive
# stoplist removes tokens like "not" and "no" that genuinely change meaning in
# a policy context.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "that", "the", "there", "these", "this", "to",
    "was", "were", "will", "with",
}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    reranked: bool = False

    @property
    def provenance(self) -> str:
        parts = []
        if self.sparse_rank is not None:
            parts.append(f"bm25#{self.sparse_rank + 1}")
        if self.dense_rank is not None:
            parts.append(f"dense#{self.dense_rank + 1}")
        if self.reranked:
            parts.append("reranked")
        return "+".join(parts) or "none"


class BM25:
    """Okapi BM25 over the corpus chunks."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        self.docs = [tokenize(f"{c.clause_title} {c.section} {c.text}") for c in chunks]
        self.lengths = np.array([len(d) for d in self.docs], dtype=np.float64)
        self.avg_length = float(self.lengths.mean()) if len(self.docs) else 0.0

        self.term_frequencies = [Counter(d) for d in self.docs]
        document_frequency: Counter[str] = Counter()
        for doc in self.docs:
            document_frequency.update(set(doc))

        total = len(self.docs)
        self.idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    def scores(self, query: str) -> np.ndarray:
        terms = tokenize(query)
        out = np.zeros(len(self.docs), dtype=np.float64)
        if not terms or self.avg_length == 0:
            return out

        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avg_length)
        for term in terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            freqs = np.array(
                [tf.get(term, 0) for tf in self.term_frequencies], dtype=np.float64
            )
            out += idf * (freqs * (self.k1 + 1)) / (freqs + norm)
        return out


def _rank_order(scores: np.ndarray, limit: int, floor_ratio: float = 0.0) -> list[int]:
    """Indices of the top `limit` scores, best first.

    `floor_ratio` drops hits below that fraction of the best score, which is
    what keeps a weak incidental match out of the fusion pool entirely rather
    than merely ranking it low.
    """
    if scores.size == 0:
        return []
    best = float(scores.max())
    if best <= 0:
        return []
    threshold = best * floor_ratio
    order = np.argsort(-scores)
    return [int(i) for i in order[:limit] if scores[i] > threshold]


class Retriever:
    """Hybrid retriever. Dense vectors are optional; BM25 always runs."""

    def __init__(
        self,
        corpus: Corpus,
        *,
        embeddings: np.ndarray | None = None,
        rerank: Reranker | None = None,
    ) -> None:
        self.corpus = corpus
        self.bm25 = BM25(corpus.chunks)
        self.embeddings = embeddings
        self.rerank = rerank
        if embeddings is not None and embeddings.shape[0] != len(corpus.chunks):
            raise ValueError(
                f"embedding matrix has {embeddings.shape[0]} rows for "
                f"{len(corpus.chunks)} chunks"
            )

    @property
    def dense_available(self) -> bool:
        return self.embeddings is not None

    @property
    def rerank_available(self) -> bool:
        return self.rerank is not None

    def _reranked(self, query: str, ranked: list[tuple[int, float]]) -> list[int] | None:
        """Reorder the fused pool with a cross-encoder, or give up quietly.

        RRF fuses two rankings without ever reading a passage against the
        query; a cross-encoder does, which is the whole reason to spend a call
        here. Every failure path returns None and leaves RRF order untouched —
        a reranker that returns nonsense must not be able to reorder the
        adjudicator's evidence, and one that is simply unavailable is the
        normal offline case rather than an error.
        """
        assert self.rerank is not None
        passages = [self.corpus.chunks[index].text for index, _ in ranked]
        try:
            scored = self.rerank(query, passages)
        except Exception:  # noqa: BLE001 - retrieval must survive a bad reranker
            return None
        if not scored:
            return None

        out: list[int] = []
        taken: set[int] = set()
        for position, _score in scored:
            if not isinstance(position, int) or not 0 <= position < len(ranked):
                return None
            chunk_index = ranked[position][0]
            if chunk_index in taken:
                continue
            taken.add(chunk_index)
            out.append(chunk_index)
        # A partial ranking is not an error — anything the reranker declined to
        # score keeps its RRF position, behind everything it did score.
        out.extend(index for index, _ in ranked if index not in taken)
        return out

    def search(
        self,
        query: str,
        *,
        top_k: int = 6,
        pool: int = 12,
        query_vector: np.ndarray | None = None,
    ) -> list[Hit]:
        sparse_order = _rank_order(self.bm25.scores(query), pool, SPARSE_FLOOR)

        dense_order: list[int] = []
        if self.embeddings is not None and query_vector is not None:
            # Vectors are L2-normalised, so the dot product is cosine similarity.
            similarity = self.embeddings @ query_vector
            dense_order = [int(i) for i in np.argsort(-similarity)[:pool]]

        fused: dict[int, float] = {}
        sparse_rank: dict[int, int] = {}
        dense_rank: dict[int, int] = {}

        for rank, index in enumerate(sparse_order):
            fused[index] = fused.get(index, 0.0) + SPARSE_WEIGHT / (RRF_K + rank)
            sparse_rank[index] = rank
        for rank, index in enumerate(dense_order):
            fused[index] = fused.get(index, 0.0) + DENSE_WEIGHT / (RRF_K + rank)
            dense_rank[index] = rank

        ranked = sorted(fused.items(), key=lambda item: -item[1])

        # Rerank the whole fused pool, not the truncated head: reordering the
        # top_k the fusion already chose cannot rescue a clause RRF ranked
        # 11th, which is the case a cross-encoder exists to catch.
        reordered: list[int] | None = None
        if self.rerank is not None and len(ranked) > 1:
            reordered = self._reranked(query, ranked)

        if reordered is None:
            ordered = ranked[:top_k]
            was_reranked = False
        else:
            scores = dict(ranked)
            ordered = [(index, scores[index]) for index in reordered[:top_k]]
            was_reranked = True

        return [
            Hit(
                chunk=self.corpus.chunks[index],
                score=round(score, 6),
                sparse_rank=sparse_rank.get(index),
                dense_rank=dense_rank.get(index),
                reranked=was_reranked,
            )
            for index, score in ordered
        ]

    def scope_score(self, query_vector: np.ndarray | None) -> float:
        """Best cosine similarity between the query and any clause chunk.

        DIAGNOSTIC ONLY — do not gate on this.

        The obvious quota optimisation is to skip windows whose best similarity
        falls below a floor. Measured on this corpus (scripts/eval_retrieval.py)
        that does not work: in-scope queries span 0.212-0.329 while clean
        controls span 0.212-0.356. The populations overlap completely, and the
        single highest-scoring query in the set is a control about colour
        grading. Any threshold that silences the controls also silences real
        violations.

        Batching is the quota lever that actually holds — eight windows per
        AUDITOR call — plus skipping windows that contain no speech at all.
        """
        if self.embeddings is None or query_vector is None:
            return 1.0
        return float((self.embeddings @ query_vector).max())

    def clauses_for(self, query: str, *, top_k: int = 3, **kwargs) -> list[Hit]:
        """Top clauses, deduplicated so three hits are three different clauses.

        Without this, a query can retrieve Scope, Yellow and Red of one clause
        and the adjudicator never sees a competing rule.
        """
        hits = self.search(query, top_k=top_k * 4, **kwargs)
        seen: set[str] = set()
        out: list[Hit] = []
        for hit in hits:
            if hit.chunk.clause_id in seen:
                continue
            seen.add(hit.chunk.clause_id)
            out.append(hit)
            if len(out) >= top_k:
                break
        return out


def normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows so dot products are cosine similarities."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)
