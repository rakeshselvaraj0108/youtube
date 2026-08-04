"""Content-addressed store."""

from __future__ import annotations

import json

import pytest

from preflight import cas


def test_hash_is_stable_and_input_sensitive():
    assert cas.hash_text("abc") == cas.hash_text("abc")
    assert cas.hash_text("abc") != cas.hash_text("abd")


def test_hash_json_ignores_key_order():
    """Cache keys must not depend on Python dict insertion order."""
    a = {"model": "llama", "temp": 0.3, "chunks": [1, 2]}
    b = {"chunks": [1, 2], "temp": 0.3, "model": "llama"}
    assert cas.hash_json(a) == cas.hash_json(b)


def test_hash_json_is_value_sensitive():
    assert cas.hash_json({"temp": 0.3}) != cas.hash_json({"temp": 0.4})


def test_hash_many_is_order_sensitive():
    assert cas.hash_many(["a", "b"]) != cas.hash_many(["b", "a"])


def test_hash_file_matches_hash_bytes(tmp_path):
    payload = b"preflight" * 4096
    target = tmp_path / "blob.bin"
    target.write_bytes(payload)
    assert cas.hash_file(target) == cas.hash_bytes(payload)


def test_prefixed_names_the_algorithm():
    digest = cas.hash_text("x")
    assert cas.prefixed(digest).split(":")[0] in {"b3", "b2"}
    assert cas.prefixed(digest).endswith(digest)


class TestEntry:
    def test_entry_is_a_miss_until_committed(self, tmp_path):
        store = cas.Store(tmp_path)
        entry = store.entry("v", "deadbeef")
        assert not entry.exists

        entry.root.mkdir(parents=True)
        entry.write_json("meta.json", {"a": 1})
        # Written but not committed: still a miss. This is what stops a run
        # that crashed mid-extraction from serving a half-built frame set.
        assert not entry.exists

        entry.commit()
        assert entry.exists

    def test_round_trips_json(self, tmp_path):
        entry = cas.Store(tmp_path).entry("r", "k")
        entry.root.mkdir(parents=True)
        entry.write_json("report.json", {"overall": 34, "verdict": "DO_NOT_PUBLISH"})
        assert entry.read_json("report.json")["overall"] == 34

    def test_written_json_is_canonical(self, tmp_path):
        """Sorted keys, so a report file diffs cleanly between runs."""
        entry = cas.Store(tmp_path).entry("r", "k")
        entry.root.mkdir(parents=True)
        path = entry.write_json("x.json", {"b": 1, "a": 2})
        assert list(json.loads(path.read_text()).keys()) == ["a", "b"]
        assert path.read_text().index('"a"') < path.read_text().index('"b"')

    def test_discard_removes_the_entry(self, tmp_path):
        entry = cas.Store(tmp_path).entry("v", "k")
        entry.root.mkdir(parents=True)
        entry.commit()
        assert entry.exists
        entry.discard()
        assert not entry.exists


class TestStore:
    def test_namespaces_are_isolated(self, tmp_path):
        store = cas.Store(tmp_path)
        assert store.entry("v", "k").root != store.entry("t", "k").root

    def test_stats_counts_only_committed_entries(self, tmp_path):
        store = cas.Store(tmp_path)
        store.entry("v", "a").root.mkdir(parents=True)  # partial
        committed = store.entry("v", "b")
        committed.root.mkdir(parents=True)
        committed.commit()

        assert store.stats()["v"] == 1
        assert store.stats()["r"] == 0

    def test_clear_empties_the_store(self, tmp_path):
        store = cas.Store(tmp_path)
        entry = store.entry("v", "k")
        entry.root.mkdir(parents=True)
        entry.commit()
        store.clear()
        assert store.stats()["v"] == 0


@pytest.mark.parametrize("value", ["", "a", "x" * 10_000])
def test_hash_handles_any_length(value):
    assert len(cas.hash_text(value)) == 64
