"""Metadata filtering, and the reason it has to happen before ranking.

The invariant under test throughout: a filtered search over a mixed corpus must
behave like an unfiltered search over a corpus containing only the matching
documents. Anything that ranks first and filters afterwards violates it, and
every test here is written to fail in that case.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from poma_memory import api
from poma_memory.metadata import MetadataNotIndexed
from poma_memory.search import HybridSearch
from poma_memory.server import _IndexCache, _handle
from poma_memory.store import Store

GATE = 0.35


def _write_rules(root: Path, rules: list[dict]) -> None:
    (root / ".poma-metadata.json").write_text(json.dumps({"rules": rules}))


def _kind_rules(root: Path) -> None:
    _write_rules(root, [
        {"glob": "in/**/*.md", "metadata": {"kind": "in"}},
        {"glob": "out/**/*.md", "metadata": {"kind": "out"}},
    ])


class _StubEmbedder:
    """Cosines fixed per chunkset, masked exactly as the real embedder masks.

    Mirrors `_EmbedderBase.search`: mask, then take the top-k, so the top-1 the
    empty gate reads is the top-1 of the narrowed corpus.
    """

    empty_gate = GATE
    min_score = 0.10

    def __init__(self, store: Store, cosines: dict[str, float]):
        rows = store.get_all_chunksets()
        self._rows = rows
        self._scores = np.array(
            [cosines.get(Path(r["file_path"]).name, 0.0) for r in rows],
            dtype=np.float32,
        )

    def search(self, query, top_k=10, allowed_ids=None):
        scores = self._scores
        if allowed_ids is not None:
            keep = np.fromiter(
                (r["chunkset_id"] in allowed_ids for r in self._rows),
                dtype=bool, count=len(self._rows),
            )
            scores = np.where(keep, scores, -np.inf)
        order = np.argsort(scores)[-top_k:][::-1]
        hits = []
        for idx in order:
            score = float(scores[idx])
            if score < self.min_score:
                continue
            r = self._rows[idx]
            hits.append({
                "chunkset_id": r["chunkset_id"],
                "file_path": r["file_path"],
                "chunk_ids": json.loads(r["chunk_ids"]),
                "contents": r["contents"],
                "score": score,
            })
        return hits


def _build(root: Path, files: dict[str, str]) -> Store:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    _kind_rules(root)
    api.index(path=root)
    return Store(root / ".poma-memory.db")


# --- the crux ---

def test_gate_reads_the_narrowed_corpus_not_the_whole_one():
    """An out-of-scope document must not vouch for in-scope ones.

    Filtering after ranking lets the gate see 0.60 from an excluded document,
    pass, and then hand back the weak in-scope hit the gate exists to suppress.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        store = _build(root, {
            "out/strong.md": "# Strong\n\nSemantic search with cosine similarity.\n",
            "in/weak.md": "# Weak\n\nUnrelated administrative trivia.\n",
        })
        hybrid = HybridSearch(store, enable_semantic=False)
        hybrid._semantic = _StubEmbedder(store, {"strong.md": 0.60, "weak.md": 0.30})

        assert hybrid._semantic.search("q")[0]["score"] == pytest.approx(0.60)
        assert hybrid.search("q", top_k=3, where={"kind": "in"}) == []
        store.close()


def test_in_scope_hits_are_not_crowded_out_of_the_candidate_window():
    """The candidate list is top_k*3 deep; out-of-scope documents can fill it.

    Ranking first and filtering after leaves nothing, because every candidate
    was excluded before the filter ever looked at an in-scope document.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        files = {f"out/o{i}.md": f"# Out {i}\n\nCosine similarity ranking notes.\n"
                 for i in range(8)}
        files["in/target.md"] = "# Target\n\nCosine similarity ranking notes.\n"
        store = _build(root, files)

        cosines = {f"o{i}.md": 0.90 for i in range(8)}
        cosines["target.md"] = 0.50
        hybrid = HybridSearch(store, enable_semantic=False)
        hybrid._semantic = _StubEmbedder(store, cosines)

        results = hybrid.search("cosine similarity", top_k=1, where={"kind": "in"})
        assert [Path(r["file_path"]).name for r in results] == ["target.md"]
        store.close()


def test_an_in_scope_document_below_the_gate_is_still_suppressed():
    """Narrowing must not disable the gate, only re-aim it."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        store = _build(root, {
            "in/a.md": "# A\n\nAdministrative trivia.\n",
            "out/b.md": "# B\n\nMore trivia.\n",
        })
        hybrid = HybridSearch(store, enable_semantic=False)
        hybrid._semantic = _StubEmbedder(store, {"a.md": 0.20, "b.md": 0.25})
        assert hybrid.search("q", top_k=3, where={"kind": "in"}) == []
        store.close()


# --- equivalence against a separately built corpus ---

IN_DOCS = {
    "in/vectors.md": "# Vectors\n\nCosine similarity over embedded chunksets.\n",
    "in/storage.md": "# Storage\n\nSQLite with WAL journalling and FTS5.\n",
}
OUT_DOCS = {
    "out/pastry.md": "# Pastry\n\nLaminated dough, butter temperature, proving.\n",
    "out/birds.md": "# Birds\n\nMigration routes of the arctic tern.\n",
}


def _names(results):
    return {Path(r["file_path"]).name for r in results}


@pytest.mark.parametrize("query,expect_hits", [
    ("cosine similarity embeddings", True),
    ("laminated butter dough proving", False),
])
def test_filtered_mixed_corpus_matches_a_corpus_built_from_the_subset(
        query, expect_hits):
    """The whole invariant, on the real embedder.

    Membership rather than order: BM25 IDF is computed over whatever corpus is
    present, so the two runs can rank in-scope documents differently even when
    they retrieve the same ones. The gate decision and the result set are the
    parts that must not depend on documents the caller excluded.
    """
    with tempfile.TemporaryDirectory() as td:
        mixed = Path(td) / "mixed"
        subset = Path(td) / "subset"
        mixed.mkdir()
        subset.mkdir()
        for rel, text in {**IN_DOCS, **OUT_DOCS}.items():
            p = mixed / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        for rel, text in IN_DOCS.items():
            p = subset / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        _kind_rules(mixed)
        _kind_rules(subset)
        api.index(path=mixed)
        api.index(path=subset)

        filtered = api.search(query, path=mixed, top_k=5, where={"kind": "in"})
        reference = api.search(query, path=subset, top_k=5)

        assert _names(filtered) == _names(reference)
        assert bool(filtered) is bool(reference)
        if not expect_hits:
            assert filtered == []


def test_semantic_order_is_preserved_because_cosines_are_absolute():
    """Unlike BM25 IDF, a cosine does not depend on what else is indexed."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        store = _build(root, {
            "in/a.md": "# A\n\nCosine similarity.\n",
            "in/b.md": "# B\n\nCosine similarity again.\n",
            "out/c.md": "# C\n\nCosine similarity elsewhere.\n",
        })
        hybrid = HybridSearch(store, enable_semantic=False)
        hybrid._semantic = _StubEmbedder(
            store, {"a.md": 0.50, "b.md": 0.70, "c.md": 0.95})
        hits = hybrid._semantic.search("q", top_k=10)
        assert [Path(h["file_path"]).name for h in hits] == ["c.md", "b.md", "a.md"]

        allowed = hybrid._allowed_ids({"kind": "in"})
        hits = hybrid._semantic.search("q", top_k=10, allowed_ids=allowed)
        assert [Path(h["file_path"]).name for h in hits] == ["b.md", "a.md"]
        store.close()


# --- refusing rather than guessing ---

def test_filtering_an_unscanned_index_raises_instead_of_returning_empty():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        store = _build(root, {"in/a.md": "# A\n\nSQLite storage notes.\n"})
        store._conn.execute("UPDATE files SET metadata = ''")
        store._conn.commit()
        store.close()

        with pytest.raises(MetadataNotIndexed) as e:
            api.search("storage", path=root, where={"kind": "in"})
        assert "poma-memory index" in str(e.value)
        assert e.value.count == 1


def test_an_unfiltered_search_on_the_same_index_still_works():
    """The refusal is about the predicate, not about the index being usable."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        store = _build(root, {"in/a.md": "# A\n\nSQLite storage notes.\n"})
        store._conn.execute("UPDATE files SET metadata = ''")
        store._conn.commit()
        store.close()
        api.search("storage", path=root)  # must not raise


def test_backfill_clears_the_refusal():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        store = _build(root, {"in/a.md": "# A\n\nSQLite storage notes.\n"})
        store._conn.execute("UPDATE files SET metadata = ''")
        store._conn.commit()
        store.close()
        api.index(path=root)
        api.search("storage", path=root, where={"kind": "in"})  # must not raise


def test_a_predicate_nothing_matches_returns_empty_rather_than_raising():
    """Scanned and says no is a different answer from never scanned."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        store = _build(root, {"in/a.md": "# A\n\nSQLite storage notes.\n"})
        store.close()
        assert api.search("storage", path=root, where={"kind": "nope"}) == []


# --- daemon ---

@pytest.fixture()
def mixed_dir():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for rel, text in {**IN_DOCS, **OUT_DOCS}.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        _kind_rules(root)
        api.index(path=root)
        yield root


def test_where_round_trips_through_the_daemon(mixed_dir):
    cache = _IndexCache()
    req = {"op": "search", "query": "cosine similarity embeddings",
           "path": str(mixed_dir), "where": {"kind": "in"}}
    resp = _handle(req, cache)
    assert resp["ok"]
    assert _names(resp["results"]) <= {"vectors.md", "storage.md"}


def test_a_different_predicate_does_not_rebuild_the_cached_index(mixed_dir):
    """Masking is per query. If a predicate rebuilt the index, the daemon's
    whole reason to exist would be gone while it still looked healthy."""
    cache = _IndexCache()
    base = {"op": "search", "query": "cosine similarity", "path": str(mixed_dir)}
    _handle({**base, "where": {"kind": "in"}}, cache)
    builds = cache.builds
    _handle({**base, "where": {"kind": "out"}}, cache)
    _handle(base, cache)
    assert cache.builds == builds


def test_the_daemon_reports_an_unscanned_index_with_a_code(mixed_dir):
    store = Store(mixed_dir / ".poma-memory.db")
    store._conn.execute("UPDATE files SET metadata = ''")
    store._conn.commit()
    store.close()

    resp = _handle({"op": "search", "query": "cosine", "path": str(mixed_dir),
                    "where": {"kind": "in"}}, _IndexCache())
    assert resp["ok"] is False
    assert resp["code"] == "metadata_not_indexed"
    assert resp["files_without_metadata"] == 4


def test_the_daemon_rejects_a_malformed_predicate(mixed_dir):
    resp = _handle({"op": "search", "query": "cosine", "path": str(mixed_dir),
                    "where": {"kind": 7}}, _IndexCache())
    assert resp["ok"] is False and resp["code"] == "bad_where"


# --- BM25 masking detail ---

def test_an_in_scope_document_scoring_zero_is_kept():
    """Masked rows come back at 0.0 when fewer than k score positive, and an
    in-scope document can legitimately score 0.0 too — so membership decides."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        store = _build(root, {
            "in/a.md": "# A\n\nzebra xylophone\n",
            "out/b.md": "# B\n\nSQLite storage notes\n",
            "out/c.md": "# C\n\nSQLite storage notes again\n",
        })
        hybrid = HybridSearch(store, enable_semantic=False)
        allowed = hybrid._allowed_ids({"kind": "in"})
        hits = hybrid._bm25.search("sqlite storage", top_k=10, allowed_ids=allowed)
        assert all(Path(h["file_path"]).name == "a.md" for h in hits)
        store.close()


# --- CLI encoding ---

def test_cli_where_encoding():
    from poma_memory.cli import _parse_where
    assert _parse_where(None) is None
    assert _parse_where(["kind=event"]) == {"kind": "event"}
    assert _parse_where(["kind=a", "kind=b", "status=x"]) == {
        "kind": ["a", "b"], "status": "x"}
    with pytest.raises(SystemExit):
        _parse_where(["novalue"])
