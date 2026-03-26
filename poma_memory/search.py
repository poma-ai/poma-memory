"""Hybrid search: BM25s (always) + semantic (optional) + RRF fusion."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from poma_memory.bm25_search import BM25Search
from poma_memory.retrieval import expand_chunk_ids, assemble_context

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

        if enable_semantic and HAS_SEMANTIC:
            try:
                self._semantic = _create_semantic(store)
            except Exception:
                pass  # model not available, fall back to BM25 only

    def search(
        self,
        query: str,
        top_k: int = 5,
        max_per_file: int = 3,
    ) -> list[dict]:
        """Search with hybrid BM25 + semantic fusion.

        Args:
            query: Search query
            top_k: Number of results to return
            max_per_file: Max hits from a single file (prevents domination)

        Returns:
            List of dicts: [{file_path, score, context, chunk_ids}]
        """
        # BM25 always runs
        bm25_hits = self._bm25.search(query, top_k=top_k * 3)

        if self._semantic:
            vec_hits = self._semantic.search(query, top_k=top_k * 3)
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

            results.append({
                "file_path": file_path,
                "score": file_scores[file_path],
                "context": context,
                "chunk_ids": expanded_ids,
            })

        return results


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
