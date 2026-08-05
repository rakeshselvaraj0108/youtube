"""Retrieval sanity check.

Not the golden corpus — this is a fast smoke test that the right clause reaches
the adjudicator for representative transcript language. Recall@3 is the metric
that matters: the triad can dismiss a clause that does not apply, but it cannot
rule on a clause it never saw.

    python scripts/eval_retrieval.py
"""

from __future__ import annotations

from preflight import cas
from preflight.agents.nim import NimClient
from preflight.config import Settings
from preflight.policy.corpus import load_corpus
from preflight.policy.index import build_index

CASES: list[tuple[str | None, str]] = [
    ("AF-01", "this is fucked we need to be off this face in ten minutes"),
    ("AF-10", "eleven people went out that morning and four of them didn't come back down"),
    ("AF-05", "we skipped the fixed line entirely, no rope no anchor, downclimbing wet rock"),
    ("AF-12", "we cracked the whisky we'd been hauling since base camp"),
    ("AF-02", "look at the blood that's a bad one, get the kit open, get pressure on it"),
    ("AF-09", "the ministry hands permits to operators who never set foot on the mountain"),
    ("AF-06", "he used a slur about their nationality and the whole room went quiet"),
    ("AF-07", "they were passing a joint around the base camp tent"),
    # Clean control: nothing here should be in scope.
    (None, "today we're going to talk about how to colour grade footage in post"),
    (None, "the weather forecast showed a clear window on tuesday morning"),
]


def main() -> int:
    settings = Settings.load()
    store = cas.Store(settings.cache_dir)
    client = NimClient(settings, store)
    corpus = load_corpus(settings.policy_dir)
    index = build_index(corpus, settings, store, client)

    print(f"backend      {index.backend}")
    print(f"corpus       {len(corpus.clauses)} clauses · {len(corpus.chunks)} chunks")
    print()

    hits = 0
    wanted = 0
    in_scope: list[float] = []
    controls: list[float] = []

    for expected, query in CASES:
        vector = index.embed_query(query)
        results = index.retriever.clauses_for(query, top_k=3, query_vector=vector)
        ids = [r.chunk.clause_id for r in results]
        scope = index.retriever.scope_score(vector)

        if expected is None:
            label = "ctrl"
            controls.append(scope)
            verdict = f"got {ids}"
        else:
            wanted += 1
            in_scope.append(scope)
            found = expected in ids
            hits += found
            label = "PASS" if found else "MISS"
            verdict = f"want {expected} · got {ids}"

        print(f"  [{label}] scope={scope:.3f}  {query[:48]:50} {verdict}")

    print()
    print(f"recall@3     {hits}/{wanted}")
    if in_scope and controls:
        # The gate lives between these two populations. If they overlap, no
        # single threshold can separate in-scope windows from clean ones and
        # gating would cost recall for nothing.
        print(f"in-scope     min {min(in_scope):.3f}  max {max(in_scope):.3f}")
        print(f"controls     min {min(controls):.3f}  max {max(controls):.3f}")
        margin = min(in_scope) - max(controls)
        print(f"separation   {margin:+.3f}", "(separable)" if margin > 0 else "(overlapping)")
    print(f"LLM calls    {client.usage.calls} ({client.usage.cached} cached)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
