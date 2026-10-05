"""A long-lived, thread-safe handle on one index, for a process that serves it.

`api.search` opens the database, builds the BM25 corpus, loads the embedding
matrix and closes it all again, on every call. That is right for a CLI and for
a hook; it is wrong for a service, where it pays the build per request and,
because a build writes any missing embeddings, turns a read into a write.

`MemoryIndex` keeps one warm `HybridSearch` per process and rebuilds it only
when another connection has committed to the database (`PRAGMA data_version`)
or the file at `db_path` has been replaced (see `server._IndexCache`, which
does the work). A filesystem that reports an unstable inode rebuilds on every
call: correct, but it costs what a rebuild costs. It adds the two things a caller
needs and the daemon does not expose:

* `lock`, a re-entrant lock every operation here takes. A caller that mutates
  the index through `index_file`, `index` or `forget` holds it across the
  mutation and `ensure_embeddings`, so no search can observe the half-way state
  and no second thread can write embeddings while the first is mid-batch.
* `ensure_embeddings`, which brings the warm index up to date NOW and embeds
  only the rows that have no embedding yet, in one transaction. Called under
  the lock right after an ingest, it moves that cost out of whichever search
  arrives next.

Writes still go through `api.index_file`, `api.index` and `api.forget`; they
open their own connection. Hold `handle.lock` around them.
"""

from __future__ import annotations

import threading
from pathlib import Path

from poma_memory.server import _IndexCache


class MemoryIndex:
    """One database, one warm search index, one lock."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.lock = threading.RLock()
        self._cache = _IndexCache(limit=1)
        self._closed = False

    @property
    def builds(self) -> int:
        """How many times the warm index was (re)built. For tests and metrics."""
        return self._cache.builds

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("MemoryIndex is closed")

    def ensure_embeddings(self) -> None:
        """Make the warm index current, embedding only rows that lack one.

        A no-op when the database does not exist or has not changed since the
        last build. After it returns, a search issues no embedding writes until
        the next commit from another connection.

        Raises `RuntimeError` when semantic search is installed but the build
        could not produce it (the embedder failed: a quota, a network error, a
        model that would not load). With `POMA_EMBEDDER=openai` an outage also
        raises, and leaves the stored OpenAI vectors alone instead of replacing
        them with local ones. `HybridSearch` swallows that and degrades to
        BM25, which is right for a one-off search and wrong here: the caller
        asked for the index to be made current, and a BM25-only index cached
        under the current stamp would serve every later search until the next
        commit. The entry is dropped so the next call retries.
        """
        with self.lock:
            self._check_open()
            if not self.db_path.exists():
                return
            hybrid = self._cache.get(self.db_path)
            err = hybrid.semantic_error
            # An ImportError is the optional extra not being installed: a
            # BM25-only index is what this install is meant to have, so it is
            # not a failure and the entry stays cached.
            if err is not None and not isinstance(err, ImportError):
                self._cache.drop(self.db_path)
                raise RuntimeError(
                    f"semantic index could not be built ({type(err).__name__}: "
                    f"{err}); search would be BM25-only"
                ) from err

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        empty_gate: float | None = None,
        where: dict | None = None,
    ) -> list[dict]:
        """Same arguments, result and exceptions as `poma_memory.search`."""
        with self.lock:
            self._check_open()
            if not self.db_path.exists():
                return []
            return self._cache.get(self.db_path).search(
                query, top_k=top_k, min_score=min_score,
                empty_gate=empty_gate, where=where,
            )

    def close(self) -> None:
        """Shutdown only, once no thread is inside `search`. Final: every later
        call raises `RuntimeError`."""
        with self.lock:
            self._closed = True
            self._cache.close_all()
