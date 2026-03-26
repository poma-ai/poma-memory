"""Semantic search via model2vec (default) or OpenAI embeddings.

Install with: pip install poma-memory[semantic]
For OpenAI: pip install poma-memory[openai]
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from poma_memory.store import Store

# model2vec is optional — imported at class init time
_M2V_MODEL = "minishlab/potion-retrieval-32M"
_M2V_CUTOFF = 0.10
_M2V_DIMS = 512

_OAI_MODEL = "text-embedding-3-large"
_OAI_CUTOFF = 0.25
_OAI_DIMS = 3072


def _get_openai_client():
    """Return OpenAI client if API key is set and SDK available, else None."""
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
        return OpenAI()
    except ImportError:
        return None


class _EmbedderBase:
    """Common logic for cosine search over stored embeddings."""

    min_score: float = 0.0
    expected_dims: int = 0

    def __init__(self, store: Store):
        self._store = store
        self._chunkset_ids: list[int] = []
        self._chunkset_map: dict[int, dict] = {}
        self._embeddings: np.ndarray | None = None
        self._build_index()

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def _embed_query(self, query: str) -> np.ndarray:
        raise NotImplementedError

    def _build_index(self) -> None:
        chunksets = self._store.get_all_chunksets()
        if not chunksets:
            return
        self._chunkset_map = {cs["chunkset_id"]: cs for cs in chunksets}
        stored = self._store.get_all_chunkset_embeddings()

        # Detect dimension mismatch (model switch) — wipe stale embeddings
        dim_ok = True
        for _, emb_bytes in stored:
            if emb_bytes is not None:
                dim = len(emb_bytes) // 4  # float32 = 4 bytes
                if dim != self.expected_dims:
                    dim_ok = False
                break

        needs_reembed = not dim_ok or not stored or any(
            emb is None for _, emb in stored
        )

        if needs_reembed:
            if not dim_ok:
                # Wipe all embeddings so they get recomputed
                for cs_id, _ in stored:
                    self._store.update_chunkset_embedding(cs_id, None)
            # Use to_embed field (normalized text) when available, fall back to contents
            texts = [cs.get("to_embed") or cs["contents"] for cs in chunksets]
            embeddings = self._embed_texts(texts)
            for cs, emb in zip(chunksets, embeddings):
                emb_bytes = emb.astype(np.float32).tobytes()
                self._store.update_chunkset_embedding(cs["chunkset_id"], emb_bytes)
            self._chunkset_ids = [cs["chunkset_id"] for cs in chunksets]
            self._embeddings = embeddings.astype(np.float32)
        else:
            ids, embs = [], []
            for cs_id, emb_bytes in stored:
                if emb_bytes is not None:
                    arr = np.frombuffer(emb_bytes, dtype=np.float32)
                    ids.append(cs_id)
                    embs.append(arr)
            if embs:
                self._chunkset_ids = ids
                self._embeddings = np.stack(embs)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if self._embeddings is None or len(self._chunkset_ids) == 0:
            return []
        query_vec = self._embed_query(query)
        norms = np.linalg.norm(self._embeddings, axis=1)
        query_norm = np.linalg.norm(query_vec)
        denom = norms * query_norm
        denom = np.where(denom > 0, denom, 1.0)
        scores = np.dot(self._embeddings, query_vec) / denom
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        hits = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < self.min_score:
                continue
            cs_id = self._chunkset_ids[idx]
            cs = self._chunkset_map.get(cs_id)
            if cs is None:
                continue
            hits.append({
                "chunkset_id": cs_id,
                "file_path": cs["file_path"],
                "chunk_ids": json.loads(cs["chunk_ids"]) if isinstance(cs["chunk_ids"], str) else cs["chunk_ids"],
                "contents": cs["contents"],
                "score": score,
            })
        return hits


class Model2VecSearch(_EmbedderBase):
    """Local vector search using model2vec (30MB, no API key)."""

    min_score = _M2V_CUTOFF
    expected_dims = _M2V_DIMS

    def __init__(self, store: Store):
        from model2vec import StaticModel
        self._model = StaticModel.from_pretrained(_M2V_MODEL)
        super().__init__(store)

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts).astype(np.float32)

    def _embed_query(self, query: str) -> np.ndarray:
        return self._model.encode([query])[0].astype(np.float32)


class OpenAISearch(_EmbedderBase):
    """Vector search using OpenAI text-embedding-3-large."""

    min_score = _OAI_CUTOFF
    expected_dims = _OAI_DIMS

    def __init__(self, store: Store, client):
        self._client = client
        super().__init__(store)

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        all_embs = []
        for i in range(0, len(texts), 2048):
            batch = texts[i:i + 2048]
            resp = self._client.embeddings.create(model=_OAI_MODEL, input=batch)
            all_embs.extend([e.embedding for e in resp.data])
        return np.array(all_embs, dtype=np.float32)

    def _embed_query(self, query: str) -> np.ndarray:
        resp = self._client.embeddings.create(model=_OAI_MODEL, input=[query])
        return np.array(resp.data[0].embedding, dtype=np.float32)


# Kept as alias for backwards compatibility with existing HybridSearch import
SemanticSearch = Model2VecSearch


def create_search(store: Store) -> _EmbedderBase:
    """Auto-select: OpenAI if OPENAI_API_KEY set, else model2vec."""
    client = _get_openai_client()
    if client is not None:
        try:
            return OpenAISearch(store, client)
        except Exception:
            pass
    return Model2VecSearch(store)
