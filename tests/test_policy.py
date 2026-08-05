"""Policy corpus, chunking and hybrid retrieval."""

from __future__ import annotations

import numpy as np
import pytest

from preflight.chunking import build_windows, iou
from preflight.perception.asr import Segment, Transcript, Word
from preflight.policy.corpus import load_corpus
from preflight.policy.retrieval import BM25, Retriever, normalize, tokenize


@pytest.fixture(scope="module")
def corpus():
    return load_corpus("data/policy")


class TestCorpus:
    def test_loads_every_clause(self, corpus):
        assert len(corpus.clauses) == 14
        assert len({c.clause_id for c in corpus.clauses}) == 14

    def test_parses_frontmatter(self, corpus):
        clause = corpus.clause("AF-01")
        assert clause is not None
        assert clause.title == "Inappropriate language"
        assert clause.severity_default in {"LIMITING", "DEMONETIZING"}
        assert clause.source_url.startswith("http")
        assert clause.fetched_at

    def test_chunks_at_heading_level(self, corpus):
        sections = {c.section for c in corpus.chunks if c.clause_id == "AF-01"}
        assert {"Scope", "Green", "Yellow", "Red"} <= sections

    def test_every_chunk_carries_a_citation(self, corpus):
        for chunk in corpus.chunks:
            assert chunk.clause_id in chunk.citation
            assert chunk.text.strip()

    def test_digest_is_stable_across_loads(self, corpus):
        assert load_corpus("data/policy").digest == corpus.digest

    def test_prompt_form_names_the_clause(self, corpus):
        rendered = corpus.chunks[0].for_prompt()
        assert corpus.chunks[0].clause_id in rendered


class TestTokenizer:
    def test_lowercases_and_strips_punctuation(self):
        assert tokenize("Blood, and INJURY!") == ["blood", "injury"]

    def test_keeps_negations(self):
        """An aggressive stoplist would drop 'not' and 'no', which change
        meaning entirely in a policy context."""
        assert "not" in tokenize("content that is not graphic")
        assert "no" in tokenize("no rope no anchor")


class TestBM25:
    def test_scores_documents_containing_the_term(self, corpus):
        bm25 = BM25(corpus.chunks)
        scores = bm25.scores("firearms")
        assert scores.max() > 0
        best = corpus.chunks[int(np.argmax(scores))]
        assert best.clause_id == "AF-08"

    def test_unknown_terms_score_zero(self, corpus):
        assert BM25(corpus.chunks).scores("zzzzqqqq").max() == 0.0

    def test_empty_query_scores_zero(self, corpus):
        assert BM25(corpus.chunks).scores("").max() == 0.0


class TestRetriever:
    def test_works_without_dense_vectors(self, corpus):
        hits = Retriever(corpus).clauses_for("firearms manufacturing", top_k=3)
        assert hits
        assert all(h.dense_rank is None for h in hits)

    def test_clauses_for_returns_distinct_clauses(self, corpus):
        """Three hits must be three different clauses, or the adjudicator never
        sees a competing rule."""
        hits = Retriever(corpus).clauses_for("violence blood injury", top_k=3)
        assert len({h.chunk.clause_id for h in hits}) == len(hits)

    def test_rejects_a_mismatched_embedding_matrix(self, corpus):
        with pytest.raises(ValueError):
            Retriever(corpus, embeddings=np.zeros((3, 8), dtype=np.float32))

    def test_dense_hits_are_fused_and_attributed(self, corpus):
        rng = np.random.default_rng(0)
        embeddings = normalize(
            rng.normal(size=(len(corpus.chunks), 16)).astype(np.float32)
        )
        retriever = Retriever(corpus, embeddings=embeddings)
        query = embeddings[5].copy()

        hits = retriever.search("firearms", top_k=5, query_vector=query)
        assert any(h.dense_rank is not None for h in hits)
        assert any("dense" in h.provenance for h in hits)

    def test_scope_score_is_the_best_similarity(self, corpus):
        embeddings = normalize(
            np.eye(len(corpus.chunks), 16, dtype=np.float32)
        )
        retriever = Retriever(corpus, embeddings=embeddings)
        assert retriever.scope_score(embeddings[0]) == pytest.approx(1.0, abs=1e-5)

    def test_scope_score_without_vectors_never_gates(self, corpus):
        assert Retriever(corpus).scope_score(None) == 1.0


class TestNormalize:
    def test_rows_become_unit_length(self):
        matrix = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
        assert np.allclose(np.linalg.norm(normalize(matrix), axis=1), 1.0)

    def test_zero_rows_do_not_divide_by_zero(self):
        assert np.isfinite(normalize(np.zeros((2, 3), dtype=np.float32))).all()


class TestChunking:
    def _transcript(self) -> Transcript:
        words = [
            Word(w=f"w{i}", start_ms=i * 1000, end_ms=i * 1000 + 900, conf=0.9)
            for i in range(120)
        ]
        return Transcript(
            language="en",
            duration_ms=120_000,
            words=words,
            segments=[Segment(start_ms=0, end_ms=120_000, text=" ".join(w.w for w in words))],
        )

    def test_windows_overlap_by_the_configured_amount(self):
        windows = build_windows(
            self._transcript(), 120_000, [], chunk_ms=30_000, overlap_ms=5_000
        )
        assert windows[0].start_ms == 0
        assert windows[1].start_ms == 25_000
        assert windows[0].end_ms == 30_000

    def test_a_sentence_on_a_boundary_appears_whole_somewhere(self):
        """The reason overlap exists at all."""
        windows = build_windows(
            self._transcript(), 120_000, [], chunk_ms=30_000, overlap_ms=5_000
        )
        # Words spanning 29s-33s must sit entirely inside at least one window.
        assert any(w.start_ms <= 29_000 and w.end_ms >= 33_000 for w in windows)

    def test_windows_never_exceed_the_runtime(self):
        for window in build_windows(self._transcript(), 120_000, []):
            assert window.end_ms <= 120_000

    def test_zero_duration_yields_no_windows(self):
        assert build_windows(None, 0, []) == []

    def test_silent_windows_are_marked_contentless(self):
        windows = build_windows(None, 60_000, [])
        assert windows
        assert all(not w.has_content for w in windows)

    def test_query_merges_transcript_and_ocr(self):
        windows = build_windows(self._transcript(), 60_000, [])
        windows[0].ocr = ["WARNING"]
        assert "WARNING" in windows[0].query()


class TestIoU:
    def test_identical_spans_are_one(self):
        assert iou((0, 100), (0, 100)) == 1.0

    def test_disjoint_spans_are_zero(self):
        assert iou((0, 100), (200, 300)) == 0.0

    def test_half_overlap(self):
        assert iou((0, 100), (50, 150)) == pytest.approx(1 / 3)

    def test_degenerate_spans_do_not_divide_by_zero(self):
        assert iou((0, 0), (0, 0)) == 0.0
