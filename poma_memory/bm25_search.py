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

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search chunksets by keyword.

        Returns:
            List of dicts: [{chunkset_id, file_path, chunk_ids, contents, score}]
        """
        if not self._retriever or not self._chunksets:
            return []

        query_tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        results, scores = self._retriever.retrieve(
            query_tokens, k=min(top_k, len(self._chunksets)),
            show_progress=False,
        )

        hits = []
        for idx, score in zip(results[0], scores[0]):
            cs = self._chunksets[idx]
            hits.append({
                "chunkset_id": cs["chunkset_id"],
                "file_path": cs["file_path"],
                "chunk_ids": json.loads(cs["chunk_ids"]) if isinstance(cs["chunk_ids"], str) else cs["chunk_ids"],
                "contents": cs["contents"],
                "score": float(score),
            })

        return hits
