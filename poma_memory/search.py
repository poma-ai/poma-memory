"""Hybrid search: BM25s (always) + semantic (optional) + RRF fusion."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from poma_memory.bm25_search import BM25Search
from poma_memory.metadata import MetadataNotIndexed, matches, normalize_where
from poma_primecut_nano import expand_chunk_ids, assemble_context

if TYPE_CHECKING:
    from poma_memory.store import Store

# Attempt optional semantic search
try:
    from poma_memory.semantic_search import create_search as _create_semantic
    HAS_SEMANTIC = True
except ImportError:
    HAS_SEMANTIC = False


class HybridSearch:
    """BM25 + optional semantic search with Reciprocal Rank Fusion."""

    def __init__(self, store: Store, enable_semantic: bool = True):
        self._store = store
        self._bm25 = BM25Search(store)
        self._semantic = None

        # Metadata lives here, not on the Store, so that it is rebuilt with the
        # rest of this object. The search daemon caches one HybridSearch per
        # database and discards it when PRAGMA data_version moves; a map hung
        # off the long-lived Store would survive a reindex and answer for the
        # previous corpus.
        self._file_meta = store.get_file_metadata_map()
        self._chunkset_files = store.get_chunkset_files()
        self._files_without_metadata = store.count_files_without_metadata()

        if enable_semantic and HAS_SEMANTIC:
            try:
                self._semantic = _create_semantic(store)
            except Exception as e:
                # Fall back to BM25 only, but surface WHY on stderr — a silent
                # `pass` here hid a perpetual "Semantic: no" for a long time.
                # (Callers that must stay quiet, e.g. hooks, redirect stderr.)
                import sys
                print(
                    f"poma-memory: semantic search unavailable, using BM25 only "
                    f"({type(e).__name__}: {e})",
                    file=sys.stderr,
                )

    def search(
        self,
        query: str,
        top_k: int = 5,
        max_per_file: int = 3,
        min_score: float = 0.0,
        empty_gate: float | None = None,
        where: dict | None = None,
    ) -> list[dict]:
        """Search with hybrid BM25 + semantic fusion.

        Args:
            query: Search query
            top_k: Number of results to return
            max_per_file: Max hits from a single file (prevents domination)
            min_score: Drop results below this RRF-fused score (0.0 = no floor)
            empty_gate: Suppress ALL results when the best semantic hit's
                cosine is below this. None = the embedder's calibrated
                default; 0.0 disables. Env override: POMA_MEMORY_EMPTY_GATE.
            where: Metadata predicate, e.g. {"kind": ["decision", "lesson"]}.
                AND across keys, OR within a list. The corpus is narrowed
                before anything is ranked — see `_allowed_ids`.

        Returns:
            List of dicts: [{file_path, score, context, chunk_ids}]

        Raises:
            MetadataNotIndexed: `where` was given but this index has files with
                no metadata recorded, so the answer could not be honest.
        """
        allowed_ids = self._allowed_ids(where)
        if allowed_ids is not None and not allowed_ids:
            # Nothing matches the predicate. Returning early also avoids handing
            # the searchers an all-zero mask, which is not a meaningful ranking.
            return []

        # BM25 always runs
        bm25_hits = self._bm25.search(query, top_k=top_k * 3,
                                      allowed_ids=allowed_ids)

        if self._semantic:
            vec_hits = self._semantic.search(query, top_k=top_k * 3,
                                             allowed_ids=allowed_ids)
            # Empty gate on ABSOLUTE similarity of the single best semantic
            # hit. RRF fused scores are rank-based: something always tops the
            # list, and a top-of-both-lists hit scores ~0.033 whether it is a
            # genuine answer or merely the best of a bad lot — so no fused-
            # score floor can express "the corpus has no answer". The top-1
            # cosine can: below the (per-embedder calibrated) gate, inject
            # nothing at all. BM25-only mode has no cosine, hence no gate.
            gate = self._resolve_empty_gate(empty_gate)
            top1 = vec_hits[0]["score"] if vec_hits else 0.0
            if top1 < gate:
                return []
            merged = _reciprocal_rank_fusion(bm25_hits, vec_hits, k=top_k * 3)
        else:
            merged = bm25_hits

        # Per-file hit limiting: group by file, cap per file, then flatten
        file_hits: dict[str, list] = {}
        file_scores: dict[str, float] = {}
        for hit in merged:
            fp = hit["file_path"]
            file_hits.setdefault(fp, [])
            if len(file_hits[fp]) < max_per_file:
                file_hits[fp].append(hit)
            file_scores[fp] = max(file_scores.get(fp, 0), hit["score"])

        sorted_files = sorted(file_scores, key=lambda f: file_scores[f], reverse=True)

        # Assemble cheatsheets: merge hits per file into one context block
        results = []
        for file_path in sorted_files[:top_k]:
            fhits = file_hits[file_path]
            all_chunk_ids: list[int] = []
            for h in fhits:
                all_chunk_ids.extend(h["chunk_ids"])

            # Deduplicate while preserving order
            seen: set[int] = set()
            unique_ids: list[int] = []
            for cid in all_chunk_ids:
                if cid not in seen:
                    seen.add(cid)
                    unique_ids.append(cid)

            file_chunks = self._store.get_chunks_for_file(file_path)

            if file_chunks:
                chunk_dicts = [
                    {
                        "chunk_index": c["local_index"],
                        "content": c["content"],
                        "depth": c["depth"],
                        "parent_chunk_index": c["parent_chunk_id"],
                    }
                    for c in file_chunks
                ]
                expanded_ids = expand_chunk_ids(chunk_dicts, unique_ids)
                context = assemble_context(chunk_dicts, expanded_ids)
            else:
                expanded_ids = unique_ids
                context = "\n".join(h["contents"] for h in fhits)

            # Age of this result = newest matched chunkset in the file (0.0 = unknown,
            # e.g. legacy rows indexed before upserted_at existed).
            cs_ids = [h["chunkset_id"] for h in fhits if "chunkset_id" in h]
            results.append({
                "file_path": file_path,
                "score": file_scores[file_path],
                "context": context,
                "chunk_ids": expanded_ids,
                "upserted_at": self._store.max_chunkset_upserted(cs_ids),
            })

        # Relevance floor: drop weak hits. RRF scores cluster ~0.027+ when both
        # BM25 and semantic corroborate a result, vs ~0.016 for single-signal
        # noise — a floor in that gap keeps corroborated context and injects
        # nothing when nothing is genuinely relevant.
        if min_score > 0.0:
            results = [r for r in results if r["score"] >= min_score]
        return results

    def _allowed_ids(self, where: dict | None) -> set[int] | None:
        """Chunkset ids the predicate admits, or None when there is no predicate.

        Resolved before ranking, which is the entire point: the empty gate is
        taken from the top-1 cosine of the ranked list, so a corpus narrowed
        afterwards would leave the gate answering for documents the caller
        excluded.
        """
        where = normalize_where(where)
        if where is None:
            return None
        if self._files_without_metadata:
            # An empty result here would be indistinguishable from an honest
            # "nothing matches", so refuse instead of guessing.
            raise MetadataNotIndexed(self._files_without_metadata,
                                     self._store.db_path)
        keep_files = {
            path for path, meta in self._file_meta.items()
            if matches(meta, where)
        }
        return {
            cs_id for cs_id, file_path in self._chunkset_files
            if file_path in keep_files
        }

    def _resolve_empty_gate(self, empty_gate: float | None) -> float:
        """Precedence: explicit param > POMA_MEMORY_EMPTY_GATE env >
        embedder's calibrated default. 0.0 (any layer) disables."""
        if empty_gate is not None:
            return empty_gate
        env = os.environ.get("POMA_MEMORY_EMPTY_GATE")
        if env is not None:
            try:
                return float(env)
            except ValueError:
                pass
        return getattr(self._semantic, "empty_gate", 0.0)


def _reciprocal_rank_fusion(
    bm25_hits: list[dict],
    vec_hits: list[dict],
    k: int = 10,
    rrf_k: int = 60,
) -> list[dict]:
    """Merge two ranked lists using Reciprocal Rank Fusion.

    RRF score = sum(1 / (rrf_k + rank)) across all lists.
    Score-scale-agnostic — no weight tuning needed.
    """
    scores: dict[int, float] = {}
    hit_map: dict[int, dict] = {}

    for rank, hit in enumerate(bm25_hits):
        cs_id = hit["chunkset_id"]
        scores[cs_id] = scores.get(cs_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        hit_map[cs_id] = hit

    for rank, hit in enumerate(vec_hits):
        cs_id = hit["chunkset_id"]
        scores[cs_id] = scores.get(cs_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        if cs_id not in hit_map:
            hit_map[cs_id] = hit

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

    results = []
    for cs_id in sorted_ids[:k]:
        hit = hit_map[cs_id].copy()
        hit["score"] = scores[cs_id]
        results.append(hit)

    return results
