"""Tree-walk expansion for search results.

Given a set of matched chunk IDs, expands to include ancestors and
siblings to produce complete hierarchical context.
"""

from __future__ import annotations

from poma_memory._constants import ELLIPSIS_MARKER


def expand_chunk_ids(
    chunks: list[dict],
    hit_chunk_ids: list[int],
) -> list[int]:
    """Expand hit chunk IDs to include ancestors for complete context.

    Args:
        chunks: All chunks for the file
        hit_chunk_ids: Chunk IDs from search hits

    Returns:
        Expanded list of chunk IDs (deduplicated, sorted by index)
    """
    chunk_by_idx = {c["chunk_index"]: c for c in chunks}
    expanded: set[int] = set(hit_chunk_ids)

    # Walk up parent chain for each hit
    for cid in hit_chunk_ids:
        current = cid
        while current is not None:
            expanded.add(current)
            parent = chunk_by_idx.get(current, {}).get("parent_chunk_index")
            current = parent

    return sorted(expanded)


def expand_chunk_ids_deep(
    chunks: list[dict],
    hit_chunk_ids: list[int],
) -> list[int]:
    """Expand hit chunk IDs using relatively-deepest filtering + subtree expansion.

    Smarter than expand_chunk_ids(): instead of just walking up the parent
    chain, it first identifies the deepest unique hits (removing ancestors
    of other hits), expands their full subtrees (children), then adds
    parents for context. Ported from poma-core/retrieval.py.

    Args:
        chunks: All chunks for the file (need chunk_index, depth)
        hit_chunk_ids: Chunk IDs from search hits

    Returns:
        Expanded list of chunk IDs (deduplicated, sorted by index)
    """
    if not hit_chunk_ids or not chunks:
        return sorted(set(hit_chunk_ids))

    sorted_chunks = sorted(chunks, key=lambda c: c["chunk_index"])
    index_to_depth = {c["chunk_index"]: c["depth"] for c in sorted_chunks}

    candidates = set(hit_chunk_ids)

    def is_ancestor(idx1: int, idx2: int) -> bool:
        """True if idx1 is an ancestor of idx2 in document tree."""
        if idx1 >= idx2:
            return False
        d1 = index_to_depth.get(idx1, 0)
        d2 = index_to_depth.get(idx2, 0)
        if d1 >= d2:
            return False
        # Scan between idx1 and idx2: all must be deeper than d1
        for c in sorted_chunks:
            ci = c["chunk_index"]
            if ci <= idx1:
                continue
            if ci >= idx2:
                break
            if c["depth"] <= d1:
                return False
        return True

    # Find relatively deepest: remove any that is ancestor of another
    relatively_deepest = set(candidates)
    for idx1 in candidates:
        for idx2 in candidates:
            if idx1 != idx2 and is_ancestor(idx1, idx2):
                relatively_deepest.discard(idx1)
                break

    def get_children(chunk_index: int) -> list[int]:
        """Get all descendants (subtree below this chunk)."""
        base_depth = index_to_depth.get(chunk_index, 0)
        children = []
        found = False
        for c in sorted_chunks:
            if c["chunk_index"] == chunk_index:
                found = True
                continue
            if found:
                if c["depth"] <= base_depth:
                    break
                children.append(c["chunk_index"])
        return children

    def get_parents(chunk_index: int) -> list[int]:
        """Get all ancestors by walking up via depth."""
        parents = []
        current_depth = index_to_depth.get(chunk_index, 0)
        for c in reversed(sorted_chunks):
            if c["chunk_index"] >= chunk_index:
                continue
            if c["depth"] < current_depth:
                parents.append(c["chunk_index"])
                current_depth = c["depth"]
        return parents

    # Collect: all original hits + children of deepest + parents of everything
    result = set(hit_chunk_ids)
    for idx in relatively_deepest:
        result.update(get_children(idx))
    for idx in list(result):
        result.update(get_parents(idx))

    return sorted(result)


def assemble_context(
    chunks: list[dict],
    chunk_ids: list[int],
) -> str:
    """Assemble readable context from a set of chunk IDs.

    Inserts ellipsis markers between non-contiguous chunks.

    Args:
        chunks: All chunks for the file
        chunk_ids: Sorted chunk IDs to include

    Returns:
        Formatted text with depth-based indentation and ellipsis gaps.
    """
    chunk_by_idx = {c["chunk_index"]: c for c in chunks}
    lines: list[str] = []
    prev_idx: int | None = None

    for cid in chunk_ids:
        chunk = chunk_by_idx.get(cid)
        if chunk is None:
            continue

        # Insert ellipsis for non-contiguous chunks
        if prev_idx is not None and cid - prev_idx > 1:
            lines.append(ELLIPSIS_MARKER)

        depth = chunk["depth"]
        indent = "  " * depth
        lines.append(f"{indent}{chunk['content']}")
        prev_idx = cid

    return "\n".join(lines)
