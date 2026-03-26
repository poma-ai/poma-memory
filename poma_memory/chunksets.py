"""Root-to-leaf path grouping for retrieval.

Groups chunks into chunksets — complete hierarchical paths from document
root to leaf nodes. Each chunkset is a self-contained retrieval unit.
"""

from __future__ import annotations

from poma_memory.normalize import normalize_for_embedding


def build_ancestor_maps(
    chunks: list[dict],
) -> tuple[dict[int, int | None], dict[int, tuple[int, ...]]]:
    """Build parent and ancestor maps from chunks with parent_chunk_index.

    Returns:
        parent_by_index: {chunk_idx: parent_idx or None}
        ancestors_by_index: {chunk_idx: (ancestor0, ..., parent)}
    """
    parent_by_index: dict[int, int | None] = {
        c["chunk_index"]: c.get("parent_chunk_index")
        for c in chunks
    }

    ancestors_by_index: dict[int, tuple[int, ...]] = {}
    for chunk in chunks:
        idx = chunk["chunk_index"]
        ancestors: list[int] = []
        current = parent_by_index.get(idx)
        while current is not None:
            ancestors.append(current)
            current = parent_by_index.get(current)
        ancestors_by_index[idx] = tuple(reversed(ancestors))

    return parent_by_index, ancestors_by_index


def chunks_to_chunksets(
    chunks: list[dict],
    target_tokens: int = 512,
) -> list[dict]:
    """Build chunksets from chunks using ancestor paths.

    Simple strategy: each leaf chunk (no children) becomes a chunkset
    containing itself and all its ancestors. Adjacent leaves under the
    same parent are collapsed into a single chunkset.

    Args:
        chunks: List with chunk_index, content, depth, parent_chunk_index
        target_tokens: Target token budget per chunkset

    Returns:
        List of dicts: [{chunkset_index, chunk_ids: list[int], contents: str}]
    """
    if not chunks:
        return []

    parent_map, ancestor_map = build_ancestor_maps(chunks)

    # Find children for each chunk
    children: dict[int, list[int]] = {}
    for c in chunks:
        children.setdefault(c["chunk_index"], [])
        parent = c.get("parent_chunk_index")
        if parent is not None:
            children.setdefault(parent, []).append(c["chunk_index"])

    # Leaf chunks = those with no children
    leaves = [c["chunk_index"] for c in chunks if not children.get(c["chunk_index"])]

    # Build chunksets: group consecutive leaves under same parent
    chunk_by_idx = {c["chunk_index"]: c for c in chunks}
    chunksets: list[dict] = []
    current_group: list[int] = []
    current_parent: int | None = None

    def flush_group() -> None:
        nonlocal current_group, current_parent
        if not current_group:
            return

        # Collect ancestor chain from the first leaf
        first_leaf = current_group[0]
        ancestors = list(ancestor_map.get(first_leaf, ()))

        # Chunkset = ancestors + all leaves in group
        chunk_ids = ancestors + current_group

        # Deduplicate while preserving order
        seen: set[int] = set()
        unique_ids: list[int] = []
        for cid in chunk_ids:
            if cid not in seen:
                seen.add(cid)
                unique_ids.append(cid)

        contents = "\n".join(
            chunk_by_idx[cid]["content"] for cid in unique_ids
            if cid in chunk_by_idx
        )

        chunksets.append({
            "chunkset_index": len(chunksets),
            "chunk_ids": unique_ids,
            "contents": contents,
            "to_embed": normalize_for_embedding(contents),
        })

        current_group = []
        current_parent = None

    for leaf_idx in leaves:
        leaf_parent = parent_map.get(leaf_idx)

        if current_parent is not None and leaf_parent != current_parent:
            flush_group()

        current_group.append(leaf_idx)
        current_parent = leaf_parent

    flush_group()

    return chunksets
