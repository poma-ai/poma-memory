"""Semantic search via model2vec.

Install with: pip install poma-memory[semantic]
"""

from __future__ import annotations

import json
import os
import sys
import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from poma_memory.store import Store

# model2vec is optional — imported at class init time
_M2V_MODEL = "minishlab/potion-retrieval-32M"
_M2V_CUTOFF = 0.10
_M2V_DIMS = 512
# Empty gate: if even the BEST hit's cosine is below this, the corpus has no
# answer for the query and the whole result set should be suppressed (RRF
# fused scores are rank-based and saturate, so they cannot express "nothing
# here is relevant" — only an absolute similarity can). Calibrated 2026-07-02
# on a real .agent/ corpus (poma-core, 12 answerable + 10 irrelevant queries):
# irrelevant top-1 cosines 0.14–0.25, answerable 0.44–0.71 — 0.35 sits mid-gap,
# rejecting 10/10 irrelevant while keeping 12/12 answerable.
_M2V_EMPTY_GATE = 0.35


class _EmbedderBase:
    """Common logic for cosine search over stored embeddings."""

    min_score: float = 0.0
    empty_gate: float = 0.0
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

        # Any row whose embedding is the wrong size (a model switch, or two
        # processes with different POMA_EMBEDDER settings writing one database)
        # is treated as missing and re-embedded with the rest. Checking only the
        # first stored row left a mixed corpus unrepairable: np.stack failed on
        # every rebuild and semantic search stayed off for good.
        by_id: dict[int, bytes | None] = {
            cs_id: (
                emb if emb is not None
                and len(emb) // 4 == self.expected_dims else None
            )
            for cs_id, emb in stored
        }

        # Embed ONLY the rows that have no embedding. This used to re-embed the
        # whole corpus whenever any one row was NULL, so every ingest into a
        # warm index cost a full pass of the model and one commit per row
        # (measured: 602 UPDATEs for one file added to a 300-file corpus). The
        # rows that do have an embedding are trusted: the dimension check above
        # is the only invalidation, as before.
        missing = [
            cs for cs in chunksets if by_id.get(cs["chunkset_id"]) is None
        ]
        if missing:
            # Use to_embed field (normalized text) when available, fall back to contents
            texts = [cs.get("to_embed") or cs["contents"] for cs in missing]
            embeddings = self._embed_texts(texts)
            if len(embeddings) != len(texts):
                # zip() would truncate and, worse, store the vectors after a
                # dropped one against the wrong rows, where they would stay.
                raise ValueError(
                    f"embedder returned {len(embeddings)} vectors for "
                    f"{len(texts)} texts"
                )
            fresh = [
                (cs["chunkset_id"], emb.astype(np.float32).tobytes())
                for cs, emb in zip(missing, embeddings)
            ]
            self._store.update_chunkset_embeddings(fresh)
            by_id.update(fresh)

        ids, embs = [], []
        for cs_id in sorted(by_id):
            emb_bytes = by_id[cs_id]
            if emb_bytes is not None and cs_id in self._chunkset_map:
                ids.append(cs_id)
                embs.append(np.frombuffer(emb_bytes, dtype=np.float32))
        if embs:
            self._chunkset_ids = ids
            self._embeddings = np.stack(embs)

    def search(self, query: str, top_k: int = 10,
               allowed_ids: set[int] | None = None) -> list[dict]:
        """Perform cosine similarity search on stored chunkset embeddings.

        `allowed_ids` narrows the corpus BEFORE the ranking, and this is the
        step that makes metadata filtering correct rather than merely
        available. `HybridSearch` derives its empty gate from the top-1 cosine
        of whatever this returns; if out-of-scope documents were still in the
        ranking at that point, the gate would answer for a corpus the caller
        did not ask about — suppressing a real in-scope hit when out-of-scope
        bulk drags top-1 down, or returning [] indistinguishable from "no
        answer" when an out-of-scope document cleared the gate and was filtered
        away afterwards.
        """
        if self._embeddings is None or len(self._chunkset_ids) == 0:
            return []
        query_vec = self._embed_query(query)
        norms = np.linalg.norm(self._embeddings, axis=1)
        query_norm = np.linalg.norm(query_vec)
        denom = norms * query_norm
        denom = np.where(denom > 0, denom, 1.0)
        scores = np.dot(self._embeddings, query_vec) / denom
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        if allowed_ids is not None:
            keep = np.fromiter(
                (cs_id in allowed_ids for cs_id in self._chunkset_ids),
                dtype=bool, count=len(self._chunkset_ids),
            )
            scores = np.where(keep, scores, -np.inf)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        hits = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < self.min_score:
                continue
            cs_id = self._chunkset_ids[idx]
            if allowed_ids is not None and cs_id not in allowed_ids:
                continue
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


_MODELS: dict = {}
_MODEL_LOCK = threading.Lock()


def _load_static_model(name: str):
    """Load the model2vec model without re-resolving it against the hub.

    `StaticModel.from_pretrained` defaults to `force_download=True`, so every
    call re-fetches a model that is already on disk. Measured: 7.85s with the
    default against 0.11s with `force_download=False` on a warm cache — and a
    search hook runs this on every Grep.

    An earlier version of this fix set HF_HUB_OFFLINE around the call instead.
    That worked, but it mutates process-global state in a library other people
    embed, and it is unsafe once two loads can overlap on different threads (the
    search daemon). model2vec's own argument needs neither.

    Falls back to the default behaviour if this model2vec build predates the
    argument, so an older install degrades to slow rather than broken.
    """
    from model2vec import StaticModel

    # One loaded model per process and name. A resident index rebuilds its
    # embedding matrix every time another connection commits, and each rebuild
    # used to load the model again (~0.1s and tens of MB) for nothing: the model
    # does not change when the corpus does. The lock keeps two threads that
    # arrive together from loading it twice.
    with _MODEL_LOCK:
        model = _MODELS.get(name)
        if model is None:
            try:
                model = StaticModel.from_pretrained(name, force_download=False)
            except TypeError:
                model = StaticModel.from_pretrained(name)
            _MODELS[name] = model
        return model


class Model2VecSearch(_EmbedderBase):
    """Local vector search using model2vec (30MB, no API key)."""

    min_score = _M2V_CUTOFF
    empty_gate = _M2V_EMPTY_GATE
    expected_dims = _M2V_DIMS

    def __init__(self, store: Store):
        self._model = _load_static_model(_M2V_MODEL)
        super().__init__(store)

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts).astype(np.float32)

    def _embed_query(self, query: str) -> np.ndarray:
        return self._model.encode([query])[0].astype(np.float32)


# Kept as alias for backwards compatibility with existing HybridSearch import
SemanticSearch = Model2VecSearch


def create_search(store: Store) -> _EmbedderBase:
    """Return the embedder: local model2vec, which is private, free, has no
    network use and is deterministic across processes.

    POMA_EMBEDDER is no longer a selector. A non-empty value other than
    "model2vec" is reported on stderr and ignored. Stored vectors of another
    size are treated as missing and re-embedded locally on the next build.
    """
    choice = os.environ.get("POMA_EMBEDDER", "").strip().lower()
    if choice not in ("", "model2vec"):
        print(
            f"poma-memory: POMA_EMBEDDER={choice[:40]!r} is not supported; "
            "using local model2vec (stored vectors of another size are "
            "re-embedded)",
            file=sys.stderr,
        )
    return Model2VecSearch(store)
