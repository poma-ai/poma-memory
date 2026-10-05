"""MemoryIndex: a warm, locked index for a process that serves one database.

What these pin is the cost model, not the ranking (test_store_and_search
covers that): after an ingest a search must not write, and an ingest must
embed only what is new.
"""

import os
import threading

import numpy as np
import pytest

pytest.importorskip("model2vec")

import poma_memory  # noqa: E402
from poma_memory import MemoryIndex, index, index_file, search  # noqa: E402
from poma_memory.store import Store  # noqa: E402


def _write(d, name, text):
    p = d / "live" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _entry(i):
    return (f"---\nrepo: poma-ai/r{i % 2}\nkind: decision\n---\n"
            f"# Decision {i}\nWe chose option {i} because of constraint {i}.\n")


@pytest.fixture
def live(tmp_path):
    for i in range(6):
        _write(tmp_path, f"d{i}.md", _entry(i))
    root = tmp_path / "live"
    index(path=str(root))
    return root


class _Counter:
    """Counts rows written by embedding batches and single-row updates."""

    def __init__(self, monkeypatch):
        self.batches, self.singles, self.threads = [], 0, set()
        orig_batch = Store.update_chunkset_embeddings
        orig_one = Store.update_chunkset_embedding

        def batch(store, pairs):
            self.batches.append(len(pairs))
            self.threads.add(threading.current_thread().name)
            return orig_batch(store, pairs)

        def one(store, cs_id, emb):
            self.singles += 1
            return orig_one(store, cs_id, emb)

        monkeypatch.setattr(Store, "update_chunkset_embeddings", batch)
        monkeypatch.setattr(Store, "update_chunkset_embedding", one)

    @property
    def rows(self):
        return sum(self.batches) + self.singles


def test_search_matches_the_function_api(live):
    h = MemoryIndex(live / ".poma-memory.db")
    want = search("option constraint", path=str(live), top_k=5)
    got = h.search("option constraint", top_k=5)
    assert [r["file_path"] for r in got] == [r["file_path"] for r in want]
    h.close()


def test_where_filter_is_passed_through(live):
    h = MemoryIndex(live / ".poma-memory.db")
    got = h.search("option constraint", top_k=10, where={"repo": "poma-ai/r1"})
    assert got and all("d1.md" in r["file_path"] or "d3.md" in r["file_path"]
                       or "d5.md" in r["file_path"] for r in got)
    h.close()


def test_a_missing_database_is_an_empty_result_and_creates_nothing(tmp_path):
    h = MemoryIndex(tmp_path / "gone" / ".poma-memory.db")
    assert h.search("anything") == []
    h.ensure_embeddings()
    assert not (tmp_path / "gone").exists()


def test_a_search_after_ensure_embeddings_writes_nothing(live, monkeypatch):
    h = MemoryIndex(live / ".poma-memory.db")
    h.ensure_embeddings()
    c = _Counter(monkeypatch)
    for _ in range(3):
        h.search("option constraint")
    assert c.rows == 0 and h.builds == 1
    h.close()


def test_an_ingest_embeds_only_the_new_chunksets(live, monkeypatch):
    h = MemoryIndex(live / ".poma-memory.db")
    h.ensure_embeddings()
    store = Store(live / ".poma-memory.db")
    before = len(store.get_all_chunksets())
    store.close()

    new = _write(live.parent, "d99.md", _entry(99))
    with h.lock:
        index_file(str(new), path=str(live))
        c = _Counter(monkeypatch)
        h.ensure_embeddings()

    store = Store(live / ".poma-memory.db")
    added = len(store.get_all_chunksets()) - before
    store.close()
    assert added >= 1
    assert c.rows == added, "embedded rows that already had an embedding"
    assert len(c.batches) == 1 and c.singles == 0, "one transaction, not per row"
    c2 = _Counter(monkeypatch)
    assert any("d99.md" in r["file_path"] for r in h.search("option 99"))
    assert c2.rows == 0
    h.close()


def test_the_model_is_not_reloaded_on_a_rebuild(live):
    h = MemoryIndex(live / ".poma-memory.db")
    h.ensure_embeddings()
    first = h._cache.get(h.db_path)._semantic._model
    with h.lock:
        index_file(str(_write(live.parent, "d98.md", _entry(98))), path=str(live))
        h.ensure_embeddings()
    assert h.builds == 2
    assert h._cache.get(h.db_path)._semantic._model is first
    h.close()


def test_a_dimension_change_re_embeds_everything(live, monkeypatch):
    h = MemoryIndex(live / ".poma-memory.db")
    h.ensure_embeddings()
    h.close()
    store = Store(live / ".poma-memory.db")
    rows = store.get_all_chunkset_embeddings()
    store.update_chunkset_embeddings(
        [(cs_id, np.zeros(7, dtype=np.float32).tobytes()) for cs_id, _ in rows]
    )
    store.close()
    c = _Counter(monkeypatch)
    h2 = MemoryIndex(live / ".poma-memory.db")
    h2.ensure_embeddings()
    assert c.rows == len(rows) and len(c.batches) == 1
    assert h2.search("option constraint")
    h2.close()


def test_concurrent_searches_and_ingests_do_not_fail(live, monkeypatch):
    h = MemoryIndex(live / ".poma-memory.db")
    h.ensure_embeddings()
    errors = []

    def reader():
        try:
            for _ in range(15):
                h.search("option constraint")
        except Exception as e:      # noqa: BLE001
            errors.append(e)

    def writer():
        try:
            for i in range(200, 205):
                with h.lock:
                    index_file(str(_write(live.parent, f"w{i}.md", _entry(i))),
                               path=str(live))
                    h.ensure_embeddings()
        except Exception as e:      # noqa: BLE001
            errors.append(e)

    c = _Counter(monkeypatch)
    threads = [threading.Thread(target=reader, name=f"reader{i}") for i in range(6)]
    threads.append(threading.Thread(target=writer, name="writer"))
    [t.start() for t in threads]
    [t.join(120) for t in threads]
    assert not any(t.is_alive() for t in threads), "deadlock"
    assert not errors, errors
    assert c.threads == {"writer"}, f"a reader wrote embeddings: {c.threads}"
    assert any("w204.md" in r["file_path"] for r in h.search("option 204"))
    h.close()


def test_exported_at_the_package_root():
    assert poma_memory.MemoryIndex is MemoryIndex


def test_a_recreated_database_is_not_served_from_the_old_file(live):
    db = live / ".poma-memory.db"
    h = MemoryIndex(db)
    assert h.search("option constraint")
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(str(db) + suffix)
        except FileNotFoundError:
            pass
    for f in live.glob("*"):
        if f.is_file() and f.suffix == ".md":
            f.unlink()
    _write(live.parent, "z.md", "# Zebras\nStriped animals graze on the savanna.\n")
    index(path=str(live))
    got = h.search("striped animals savanna")
    assert got and all("z.md" in r["file_path"] for r in got)
    assert not h.search("option constraint")
    h.close()


def test_a_mixed_size_corpus_repairs_itself(live):
    db = live / ".poma-memory.db"
    MemoryIndex(db).ensure_embeddings()
    store = Store(db)
    ids = [i for i, _ in store.get_all_chunkset_embeddings()]
    store.update_chunkset_embeddings([
        (ids[0], np.zeros(7, dtype=np.float32).tobytes()),
        (ids[1], None),
    ])
    store.close()
    h = MemoryIndex(db)
    h.ensure_embeddings()
    assert h._cache.get(db)._semantic is not None
    store = Store(db)
    assert all(e is not None and len(e) // 4 == 512
               for _, e in store.get_all_chunkset_embeddings())
    store.close()
    h.close()


def test_a_failed_embed_raises_and_is_retried(live, monkeypatch):
    from poma_memory.semantic_search import Model2VecSearch
    h = MemoryIndex(live / ".poma-memory.db")
    h.ensure_embeddings()
    with h.lock:
        index_file(str(_write(live.parent, "d97.md", _entry(97))), path=str(live))
        real = Model2VecSearch._embed_texts
        monkeypatch.setattr(Model2VecSearch, "_embed_texts",
                            lambda self, texts: (_ for _ in ()).throw(OSError("quota")))
        with pytest.raises(RuntimeError):
            h.ensure_embeddings()
        monkeypatch.setattr(Model2VecSearch, "_embed_texts", real)
        h.ensure_embeddings()
    c = _Counter(monkeypatch)
    assert any("d97.md" in r["file_path"] for r in h.search("option 97 constraint"))
    assert c.rows == 0
    assert h._cache.get(h.db_path)._semantic is not None
    h.close()


def test_a_short_embedder_response_is_an_error_not_a_misalignment(live, monkeypatch):
    from poma_memory.semantic_search import Model2VecSearch
    store = Store(live / ".poma-memory.db")
    store.update_chunkset_embeddings(
        [(i, None) for i, _ in store.get_all_chunkset_embeddings()]
    )
    monkeypatch.setattr(Model2VecSearch, "_embed_texts",
                        lambda self, texts: np.zeros((len(texts) - 1, 512),
                                                     dtype=np.float32))
    with pytest.raises(ValueError, match="vectors for"):
        Model2VecSearch(store)
    assert all(e is None for _, e in store.get_all_chunkset_embeddings())
    store.close()


def test_a_closed_handle_stays_closed(live):
    h = MemoryIndex(live / ".poma-memory.db")
    h.search("option constraint")
    h.close()
    with pytest.raises(RuntimeError):
        h.search("option constraint")
    with pytest.raises(RuntimeError):
        h.ensure_embeddings()


def test_a_base_install_without_model2vec_is_bm25_by_design_and_does_not_raise(
        live, monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "model2vec", None)
    monkeypatch.setattr("poma_memory.semantic_search._MODELS", {})
    h = MemoryIndex(live / ".poma-memory.db")
    h.ensure_embeddings()
    h.ensure_embeddings()
    assert h.builds == 1, "an absent optional dependency must not drop the cache"
    assert h.search("option constraint")
    h.close()


def test_an_openai_outage_raises_and_keeps_the_stored_vectors(live, monkeypatch):
    class _Down:
        class embeddings:                                   # noqa: N801
            @staticmethod
            def create(**_kw):
                raise OSError("quota")

    monkeypatch.setenv("POMA_EMBEDDER", "openai")
    monkeypatch.setattr("poma_memory.semantic_search._get_openai_client",
                        lambda: _Down())
    db = live / ".poma-memory.db"
    store = Store(db)
    ids = [i for i, _ in store.get_all_chunkset_embeddings()]
    foreign = np.zeros(3072, dtype=np.float32).tobytes()
    store.update_chunkset_embeddings([(i, foreign) for i in ids[:-1]] + [(ids[-1], None)])
    store.close()
    h = MemoryIndex(db)
    with pytest.raises(RuntimeError, match="OSError"):
        h.ensure_embeddings()
    store = Store(db)
    kept = [e for _, e in store.get_all_chunkset_embeddings() if e is not None]
    store.close()
    assert len(kept) == len(ids) - 1 and all(e == foreign for e in kept)
    h.close()


def test_an_unstable_inode_does_not_rebuild_on_every_call(live, monkeypatch):
    real = os.stat
    counter = iter(range(10**9))

    def flaky(path, *a, **kw):
        st = real(path, *a, **kw)
        if str(path).endswith(".poma-memory.db"):
            return os.stat_result((st.st_mode, next(counter), st.st_dev, st.st_nlink,
                                   st.st_uid, st.st_gid, st.st_size,
                                   st.st_atime, st.st_mtime, st.st_ctime))
        return st

    monkeypatch.setattr(os, "stat", flaky)
    h = MemoryIndex(live / ".poma-memory.db")
    for _ in range(5):
        h.search("option constraint")
    assert h.builds == 1
    h.close()
