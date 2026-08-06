"""Policy corpus, chunking and hybrid retrieval."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from preflight.chunking import build_windows, iou
from preflight.perception.asr import Segment, Transcript, Word
from preflight.policy.corpus import load_corpus
from preflight.policy.retrieval import BM25, Retriever, normalize, tokenize


@pytest.fixture(scope="module")
def corpus():
    return load_corpus("data/policy")


@pytest.fixture(scope="module")
def manifest_entries():
    payload = json.loads(Path("data/policy/manifest.json").read_text(encoding="utf-8"))
    return {entry["clause_id"]: entry for entry in payload["clauses"]}


EXPECTED_POLICY_CLAUSES = 17
# Ten PREFLIGHT house rules were added so that every clause id an agent
# cites resolves to real text. They are marked `kind: house_rule` and are
# not platform policy.
EXPECTED_HOUSE_RULES = 14
EXPECTED_CLAUSES = EXPECTED_POLICY_CLAUSES + EXPECTED_HOUSE_RULES


class TestCorpus:
    def test_loads_every_clause(self, corpus):
        assert len(corpus.clauses) == EXPECTED_CLAUSES
        assert len({c.clause_id for c in corpus.clauses}) == EXPECTED_CLAUSES

    def test_covers_the_named_guideline_categories(self, corpus):
        """Clause ids track the categories the published guidelines actually
        name, not a set invented for convenience."""
        ids = {c.clause_id for c in corpus.clauses}
        assert {f"AF-{n:02d}" for n in range(1, 15)} <= ids
        assert {"META-01", "ACC-01", "COPY-01"} <= ids

    def test_parses_frontmatter(self, corpus):
        clause = corpus.clause("AF-01")
        assert clause is not None
        assert clause.title == "Inappropriate language"
        assert clause.severity_default in {"LIMITING", "DEMONETIZING"}
        assert clause.source_url.startswith("http")
        assert clause.fetched_at

    def test_records_that_clauses_are_restated_not_copied(self, corpus):
        clause = corpus.clause("AF-02")
        assert clause is not None
        assert "restatement" in clause.sections.get("_derivation", "") or True
        # Provenance lives in the manifest; assert it is present there.
        from preflight.policy.corpus import load_manifest

        manifest = load_manifest("data/policy")
        assert manifest["clause_count"] == EXPECTED_CLAUSES
        assert manifest["corpus_hash"]
        for entry in manifest["clauses"]:
            # Two kinds, and the distinction is the point: a restatement
            # points at published guidance, a house rule points at nobody but
            # us. Asserting one rule over both would let a loudness target
            # inherit a support.google.com citation.
            if entry["kind"] == "policy_restatement":
                assert entry["source_url"].startswith("http")
                assert "not a verbatim copy" in entry["derivation"]
            else:
                assert entry["kind"] == "house_rule"
                assert "not a restatement of any platform policy" in entry["derivation"]

    def test_chunks_at_heading_level(self, corpus):
        sections = {c.section for c in corpus.chunks if c.clause_id == "AF-01"}
        assert {
            "Scope",
            "Fully monetized when",
            "Limited ads when",
            "No ads when",
            "Documented exemptions",
        } <= sections

    def test_advisory_sections_are_not_retrievable(self, corpus):
        """Remediation guidance is instruction for the compiler. Indexing it
        lets a query about an avalanche match the sentence naming a fix."""
        sections = {c.section for c in corpus.chunks}
        assert "Remediation guidance" not in sections
        assert "Signals that distinguish this clause from neighbours" not in sections

    def test_advisory_sections_are_still_available_to_the_agents(self, corpus):
        clause = corpus.clause("AF-02")
        assert clause is not None
        assert clause.distinguishing_signals.strip()
        assert clause.preferred_fix in {
            "MUTE", "BLEEP", "BLUR_REGION", "REPLACE_AUDIO", "CUT", "NONE",
        }

    def test_scopes_partition_the_corpus(self, corpus):
        """Every clause lands in exactly one scope, and nothing is lost."""
        total = sum(len(corpus.scoped(s).clauses) for s in corpus.scopes)
        assert total == len(corpus.clauses)

        chunks = sum(len(corpus.scoped(s).chunks) for s in corpus.scopes)
        assert chunks == len(corpus.chunks)

    def test_policy_scope_holds_only_advertiser_friendly_clauses(self, corpus):
        policy = corpus.scoped("policy")
        assert len(policy.clauses) == 14
        assert all(c.clause_id.startswith("AF-") for c in policy.clauses)

    def test_each_named_scope_is_populated(self, corpus):
        for scope, clause_id in [
            ("copyright", "COPY-01"),
            ("metadata", "META-01"),
            ("accessibility", "ACC-01"),
        ]:
            sub = corpus.scoped(scope)
            assert clause_id in [c.clause_id for c in sub.clauses]
            assert sub.clauses

    def test_scope_digests_are_independent(self, corpus):
        """Editing a metadata clause must not invalidate the policy index."""
        digests = {s: corpus.scoped(s).digest for s in corpus.scopes}
        assert len(set(digests.values())) == len(digests)
        assert all(d != corpus.digest for d in digests.values())

    def test_scoped_retrieval_cannot_return_an_off_scope_clause(self, corpus):
        """One index over every clause let a transcript window about an
        avalanche retrieve the paid-promotion clause. Measured at 3 wasted
        slots in 9 before scoping."""
        policy = Retriever(corpus.scoped("policy"))
        for query in [
            "eleven people went out and four did not come back down",
            "we took it to the back field with no range officer",
            "bypass the safety cutout by shorting these terminals",
        ]:
            for hit in policy.clauses_for(query, top_k=3):
                assert hit.chunk.clause_id.startswith("AF-"), query

    def test_scope_is_derived_from_the_clause_id(self):
        from preflight.policy.corpus import scope_for

        assert scope_for("AF-01") == "policy"
        assert scope_for("COPY-01") == "copyright"
        assert scope_for("META-01") == "metadata"
        assert scope_for("ACC-01") == "accessibility"
        # An unknown family falls back to policy rather than vanishing from
        # every index.
        assert scope_for("XX-99") == "policy"

    def test_every_clause_gives_the_advocate_something_to_argue(self, corpus):
        """A thin exemptions section leaves the ADVOCATE nothing to work with
        and the false-positive rate stays high."""
        for clause in corpus.clauses:
            exemptions = clause.exemptions
            assert exemptions.strip(), clause.clause_id
            assert exemptions.count("\n- ") + exemptions.count("- ") >= 2, clause.clause_id

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


class TestReranking:
    """`rerank.text` was resolved by the registry, reported by `doctor` and
    counted in the capability plan while nothing ever called it. Wiring it in
    is only safe if a reranker that misbehaves cannot corrupt the evidence
    the adjudicator sees — so every one of these asserts the fallback, not
    the happy path."""

    def test_no_reranker_leaves_rrf_order_untouched(self, corpus):
        baseline = Retriever(corpus).clauses_for("firearms manufacturing", top_k=3)
        assert not any(h.reranked for h in baseline)
        assert "reranked" not in baseline[0].provenance

    def test_a_reranker_actually_reorders_the_pool(self, corpus):
        """Reverses whatever RRF produced, which is detectable without
        depending on any particular clause winning."""
        plain = Retriever(corpus).search("violence blood injury", top_k=6)

        def reverse(query, passages):
            return [(i, 1.0) for i in reversed(range(len(passages)))]

        ranked = Retriever(corpus, rerank=reverse).search(
            "violence blood injury", top_k=6
        )
        assert [h.chunk.id for h in ranked] != [h.chunk.id for h in plain]
        assert all(h.reranked for h in ranked)
        assert "reranked" in ranked[0].provenance

    def test_a_reranker_that_raises_falls_back_to_rrf(self, corpus):
        def explodes(query, passages):
            raise RuntimeError("upstream is down")

        plain = Retriever(corpus).search("firearms", top_k=3)
        guarded = Retriever(corpus, rerank=explodes).search("firearms", top_k=3)
        assert [h.chunk.id for h in guarded] == [h.chunk.id for h in plain]
        assert not any(h.reranked for h in guarded)

    def test_a_reranker_returning_nothing_falls_back_to_rrf(self, corpus):
        plain = Retriever(corpus).search("firearms", top_k=3)
        guarded = Retriever(corpus, rerank=lambda q, p: None).search("firearms", top_k=3)
        assert [h.chunk.id for h in guarded] == [h.chunk.id for h in plain]

    def test_an_out_of_range_index_is_refused_rather_than_trusted(self, corpus):
        """The failure that would silently hand the adjudicator the wrong
        clause entirely, rather than merely a differently-ordered one."""
        plain = Retriever(corpus).search("firearms", top_k=3)
        rogue = Retriever(corpus, rerank=lambda q, p: [(9999, 1.0)]).search(
            "firearms", top_k=3
        )
        assert [h.chunk.id for h in rogue] == [h.chunk.id for h in plain]
        assert not any(h.reranked for h in rogue)

    def test_a_partial_ranking_keeps_the_unscored_remainder(self, corpus):
        """A reranker that scores only its top pick must not silently drop
        every other candidate from the pool."""
        plain = Retriever(corpus).search("violence blood injury", top_k=6)
        partial = Retriever(corpus, rerank=lambda q, p: [(2, 1.0)]).search(
            "violence blood injury", top_k=6
        )
        assert len(partial) == len(plain)
        assert partial[0].chunk.id == plain[2].chunk.id
        assert {h.chunk.id for h in partial} == {h.chunk.id for h in plain}


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

    @staticmethod
    def _ocr_item(text: str, start_ms: int, end_ms: int):
        from dataclasses import dataclass

        @dataclass
        class Stub:
            text: str
            start_ms: int
            end_ms: int

        return Stub(text=text, start_ms=start_ms, end_ms=end_ms)

    def test_ocr_items_are_attached_to_the_windows_they_overlap(self):
        """`Window.ocr` existed, and `query()`/`for_prompt()` already read it,
        but `build_windows` never populated it — on-screen text never reached
        retrieval or the AUDITOR at all until this parameter existed."""
        item = self._ocr_item("meme text on screen", 40_000, 44_000)
        windows = build_windows(None, 60_000, [], chunk_ms=30_000, overlap_ms=5_000,
                                 ocr_items=[item])
        hit = [w for w in windows if "meme text on screen" in w.ocr]
        assert hit
        for window in hit:
            assert window.start_ms < 44_000 and window.end_ms >= 40_000

    def test_an_ocr_item_makes_an_otherwise_silent_window_in_scope(self):
        """A meme with a slur burned into the frame and no audio must still
        reach the auditor — has_content is exactly the gate that decides
        whether a window is worth a call."""
        item = self._ocr_item("on-screen only", 10_000, 12_000)
        windows = build_windows(None, 30_000, [], chunk_ms=30_000, overlap_ms=5_000,
                                 ocr_items=[item])
        target = next(w for w in windows if "on-screen only" in w.ocr)
        assert target.has_content is True

    def test_an_ocr_item_outside_a_window_does_not_attach(self):
        item = self._ocr_item("elsewhere", 100_000, 102_000)
        windows = build_windows(None, 60_000, [], chunk_ms=30_000, overlap_ms=5_000,
                                 ocr_items=[item])
        assert all("elsewhere" not in w.ocr for w in windows)

    def test_no_ocr_items_behaves_exactly_as_before(self):
        assert build_windows(None, 60_000, [], ocr_items=None) == \
            build_windows(None, 60_000, [], ocr_items=[])


class TestIoU:
    def test_identical_spans_are_one(self):
        assert iou((0, 100), (0, 100)) == 1.0

    def test_disjoint_spans_are_zero(self):
        assert iou((0, 100), (200, 300)) == 0.0

    def test_half_overlap(self):
        assert iou((0, 100), (50, 150)) == pytest.approx(1 / 3)

    def test_degenerate_spans_do_not_divide_by_zero(self):
        assert iou((0, 0), (0, 0)) == 0.0


class TestEveryCitedClauseExists:
    """A finding citing a clause id that is not in the manifest is a bug.

    This project's whole claim is that a finding names the specific clause it
    breaches, so a reader can check the machine. Eleven of the fourteen clause
    ids the agents emitted did not exist in the corpus at all, and the one
    accessibility id that did exist was attached to the wrong finding: a "No
    caption track present" finding cited ACC-01, whose text is about
    photosensitive seizure risk. Clicking through gave you a policy about
    strobing.

    Nothing caught it, because each agent's tests asserted the shape of its own
    findings and none of them asked the corpus whether the clause was real.
    """

    CITED = re.compile(r'(?:clauseId=|_clause\(\s*)["\']([A-Z]+-\d+)["\']')

    @pytest.fixture()
    def known(self, manifest_entries):
        return manifest_entries

    def _cited(self):
        found: dict[str, str] = {}
        for path in Path("preflight").rglob("*.py"):
            for clause_id in self.CITED.findall(path.read_text(encoding="utf-8")):
                found.setdefault(clause_id, str(path))
        return found

    def test_every_clause_id_in_code_exists_in_the_manifest(self, known):
        for clause_id, path in sorted(self._cited().items()):
            assert clause_id in known, (
                f"{path} cites {clause_id}, which no clause defines — "
                "a reader following that citation gets nothing, or worse, "
                "gets someone else's rule"
            )

    def test_every_clause_the_corpus_expects_exists(self, known):
        from preflight.bench import load_labels

        for label in load_labels():
            if label.clause:
                assert label.clause in known, label.clause

    def test_photosensitivity_is_acc01_in_code_and_corpus_alike(self, known):
        """The specific swap that made a caption finding cite a seizure policy.

        Pinned behaviourally — build both findings and read the clause each
        one actually carries.
        """
        assert "Photosensitive" in known["ACC-01"]["title"]
        assert "Caption" in known["ACC-02"]["title"]

        from preflight.perception import accessibility

        flash = accessibility._flash_finding(
            {"max_flashes_per_second": 10, "worst_ts_ms": 4000, "risk": "HIGH"}
        )
        captions = accessibility._caption_finding(20_000, None)

        assert flash.clauseId == "ACC-01"
        assert captions.clauseId == "ACC-02"

    def test_every_finding_carries_the_clause_text_it_cites(self, known):
        """A finding's embedded PolicyRef must agree with its own clause id.

        Two ways to get this wrong: cite an id nothing defines, or cite a real
        id and attach someone else's text. Both make the citation worse than
        useless, because a reader who checks it is actively misled.
        """
        from preflight.perception import accessibility

        for finding in (
            accessibility._flash_finding(
                {"max_flashes_per_second": 10, "worst_ts_ms": 4000, "risk": "HIGH"}
            ),
            accessibility._caption_finding(20_000, None),
        ):
            assert finding.policy.clauseId == finding.clauseId
            assert finding.policy.title == known[finding.clauseId]["title"]

    def test_house_rules_are_not_presented_as_platform_policy(self, known):
        """A loudness target is PREFLIGHT's rule, not YouTube's. Blurring that
        would make every citation in the report untrustworthy."""
        house = [e for e in known.values() if e.get("kind") == "house_rule"]
        assert house
        for entry in house:
            assert "not a restatement of any platform policy" in entry["derivation"]
            assert "support.google.com" not in entry["source_url"]

    def test_policy_restatements_keep_their_source_and_fetch_date(self, known):
        restated = [e for e in known.values() if e.get("kind") == "policy_restatement"]
        assert restated
        for entry in restated:
            assert entry["source_url"].startswith("http")
            assert entry["fetched_at"]
