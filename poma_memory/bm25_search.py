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

        self._retriever = bm25s.BM25()
        self._retriever.index(corpus_tokens, show_progress=False)

    def search(self, query: str, top_k: int = 10,
               allowed_ids: set[int] | None = None) -> list[dict]:
        """Search chunksets by keyword.

        `allowed_ids` narrows the corpus BEFORE ranking, via bm25s's own
        `weight_mask`: the mask is multiplied into the score vector ahead of
        top-k, so ranks among in-scope documents are exactly what they would be
        in an index built from those documents alone. Masking here is about
        recall — the empty gate reads a cosine, not a BM25 score, so the
        semantic side is what makes the gate correct (see `semantic_search`).

        Returns:
            List of dicts: [{chunkset_id, file_path, chunk_ids, contents, score}]
        """
        if not self._retriever or not self._chunksets:
            return []

        weight_mask = None
        if allowed_ids is not None:
            weight_mask = np.fromiter(
                (1.0 if cs["chunkset_id"] in allowed_ids else 0.0
                 for cs in self._chunksets),
                dtype=np.float32, count=len(self._chunksets),
            )

        # Caveat if the retriever is ever constructed with a different method:
        # bm25s applies `scores *= weight_mask` BEFORE adding its
        # `nonoccurrence_array`, so under "bm25l"/"bm25+" a masked document
        # would come back with a non-zero score. `bm25s.BM25()` defaults to
        # lucene, where that array is None. The membership drop below does not
        # depend on the score, so it holds either way.
        query_tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        results, scores = self._retriever.retrieve(
            query_tokens, k=min(top_k, len(self._chunksets)),
            show_progress=False, weight_mask=weight_mask,
        )

        hits = []
        for idx, score in zip(results[0], scores[0]):
            cs = self._chunksets[idx]
            # Drop by membership, never by score. A masked document still comes
            # back at 0.0 when fewer than k documents score positive, and an
            # in-scope document can legitimately score 0.0 too — so the score
            # cannot tell the two apart and the id set can.
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
