"""BM25s-based keyword search on chunksets. Always available."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import bm25s
import numpy as np

if TYPE_CHECKING:
    from poma_memory.store import Store


class BM25Search:
    """BM25s search over chunkset contents."""

    def __init__(self, store: Store):
        self._store = store
        self._chunksets: list[dict] = []
        self._retriever: bm25s.BM25 | None = None
        self._build_index()

    def _build_index(self) -> None:
        self._chunksets = self._store.get_all_chunksets()
        if not self._chunksets:
            return

        corpus = [cs["contents"] for cs in self._chunksets]
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en", show_progress=False)

        # An index whose every chunkset tokenizes to nothing -- a corpus of
        # pure stopwords -- gives bm25s an empty vocabulary, and `BM25.index`
        # raises `ValueError: max() iterable argument is empty`. That happens
        # in `HybridSearch.__init__`, so it took out `search` and the daemon
        # alike, filtered or not, on a corpus that is odd but perfectly legal.
        # Leaving `_retriever` as None is what `search` already does for an
        # empty index: BM25 contributes nothing, and the semantic side, which
        # has no vocabulary, still answers.
        if not getattr(corpus_tokens, "vocab", None):
            return

        self._retriever = bm25s.BM25()
        self._retriever.index(corpus_tokens, show_progress=False)

    def search(self, query: str, top_k: int = 10,
               allowed_ids: set[int] | None = None) -> list[dict]:
        """Search chunksets by keyword.

        `allowed_ids` narrows the corpus BEFORE ranking: the top-k is taken
        over the allowed documents only, so an out-of-scope document cannot
        occupy the candidate window. It does NOT make the ranking identical to
        an index built from those documents alone — BM25 scores against
        corpus-wide IDF and average document length, which the excluded
        documents still contribute to. What holds is that every in-scope
        document is rankable and no other document is returned; an earlier
        comment here claimed the stronger property and was wrong (see
        `tests/test_filtering.py`). Narrowing here is about recall — the empty
        gate reads a cosine, not a BM25 score, so the semantic side is what
        makes the gate correct (see `semantic_search`).

        Returns:
            List of dicts: [{chunkset_id, file_path, chunk_ids, contents, score}]
        """
        if not self._retriever or not self._chunksets:
            return []

        query_tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)

        if allowed_ids is None:
            results, scores = self._retriever.retrieve(
                query_tokens, k=min(top_k, len(self._chunksets)),
                show_progress=False,
            )
            ranked = list(zip(results[0], scores[0]))
        else:
            # `retrieve(k=...)` with a `weight_mask` takes the top k over the
            # WHOLE corpus and only then hands them back, so out-of-scope
            # documents still occupy the window. They sit at 0.0 after masking,
            # and an in-scope document can legitimately score 0.0 too — BM25
            # gives every document that shares no term with the query the same
            # zero — so masked documents win the tie on index order and the
            # in-scope one never reaches the membership drop below. Reproduced:
            # 200 out-of-scope documents plus one in-scope document sharing no
            # term with the query, BM25-only, top_k=1 -> []; the same document
            # in an index built from the matching files alone -> found. That is
            # the invariant this module's docstring claims, and it is also the
            # DEFAULT install, since the semantic side (whose -inf mask has no
            # such tie) needs the [semantic] extra.
            #
            # So rank inside the allowed set instead of masking and hoping.
            # `get_scores` is bm25s's own per-document scoring for one query;
            # the ordering among allowed documents is identical to what a
            # retrieve over that subset would give, and out-of-scope documents
            # cannot occupy the window because they are never in it.
            allowed_idx = np.fromiter(
                (i for i, cs in enumerate(self._chunksets)
                 if cs["chunkset_id"] in allowed_ids),
                dtype=np.int64,
            )
            if allowed_idx.size == 0:
                return []
            # TOKEN STRINGS, never `query_tokens.ids[0]`. `bm25s.tokenize`
            # builds a fresh vocabulary for whatever it is given, so a query's
            # ids are positions in the QUERY's vocab while `get_scores` reads
            # them as positions in the INDEX's. The two coincide on small
            # corpora -- they did on the five-document check that first passed
            # -- and diverge silently on real ones: scores came back ~100x too
            # small, every one of 292 of 300 random comparisons ranking the
            # wrong documents, because the terms being scored were whichever
            # ones happened to sit at those indices. `retrieve` does this
            # mapping for you; `get_scores` does not.
            str_tokens = bm25s.tokenize([query], stopwords="en",
                                        show_progress=False, return_ids=False)
            if not str_tokens[0]:
                # A query that tokenizes to nothing -- all stopwords, or all
                # punctuation. `get_scores` indexes `[0]` unconditionally and
                # raises IndexError; `retrieve` handles it, so the unfiltered
                # path never saw this and neither did any test until the
                # filtered path stopped going through `retrieve`. Every
                # document scores zero on an empty query, which is what the
                # unfiltered path returns too.
                all_scores = np.zeros(len(self._chunksets), dtype=np.float32)
            else:
                all_scores = self._retriever.get_scores(str_tokens[0])
            subset = all_scores[allowed_idx]
            k = min(top_k, allowed_idx.size)
            # Ties break toward the lower corpus index. That is NOT the order
            # `retrieve` picks for equal scores -- measured over 80
            # query/corpus pairs with every document allowed, 13 differed and
            # every one was an exact tie, 3 of them at the top-k boundary, so
            # the returned SET can differ by a tied document. Non-tie
            # divergence: 0. Deterministic and correct; simply not identical.
            top = np.argsort(-subset, kind="stable")[:k]
            ranked = [(int(allowed_idx[t]), float(subset[t])) for t in top]

        hits = []
        for idx, score in ranked:
            cs = self._chunksets[idx]
            # Drop by membership, never by score. Belt and braces now that the
            # ranking is built from the allowed set: an in-scope document can
            # legitimately score 0.0, so the score could never tell an excluded
            # document from an included one.
            if allowed_ids is not None and cs["chunkset_id"] not in allowed_ids:
                continue
            hits.append({
                "chunkset_id": cs["chunkset_id"],
                "file_path": cs["file_path"],
                "chunk_ids": json.loads(cs["chunk_ids"]) if isinstance(cs["chunk_ids"], str) else cs["chunk_ids"],
                "contents": cs["contents"],
                "score": float(score),
            })

        return hits
