"""Root-to-leaf path grouping for retrieval.

Groups chunks into chunksets — complete hierarchical paths from document
root to leaf nodes. Each chunkset is a self-contained retrieval unit.

Two variants are available:
- chunks_to_chunksets(): Simple leaf-grouping strategy (fast, predictable).
- chunks_to_chunksets_optimized(): Collapse/merge/sibling-fill algorithm
  ported from poma-core (fewer, more distinct chunksets).
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


def chunks_to_chunksets_optimized(
    chunks: list[dict],
    target_tokens: int = 512,
    max_tokens: int = 7000,
    max_siblings: int = 3,
    sibling_budget_ratio: float = 0.4,
) -> list[dict]:
    """Build chunksets using collapse/merge/sibling-fill algorithm.

    More sophisticated than chunks_to_chunksets(): produces fewer, more
    distinct chunksets with minimal repeated ancestors. Ported from poma-core.

    Algorithm:
    1. Collapse contiguous same-depth chunks into blocks under token budget
    2. Merge adjacent blocks upward when possible
    3. Build chunksets around merged blocks with full ancestor chains
    4. Fill preceding siblings within budget (closest-first, capped)
    5. Deduplicate by superset removal

    Args:
        chunks: List with chunk_index, content, depth, parent_chunk_index
        target_tokens: Soft target tokens per chunkset
        max_tokens: Hard maximum tokens per chunkset
        max_siblings: Max sibling blocks to add per chunkset
        sibling_budget_ratio: Max fraction of budget for siblings (0.4 = 40%)
    """
    if not chunks:
        return []

    from poma_memory._constants import ELLIPSIS_MARKER

    # Estimate token count (~1.3 tokens per word)
    def _est_tokens(text: str) -> int:
        return max(1, int(len(text.split()) * 1.3))

    chunk_by_idx = {c["chunk_index"]: c for c in chunks}
    token_by_idx = {c["chunk_index"]: _est_tokens(c["content"]) for c in chunks}

    max_chunk_tokens = max(token_by_idx.values())
    token_limit = max(target_tokens, round(max_chunk_tokens * 1.1))
    token_limit = min(token_limit, max_tokens)

    parent_by_index, ancestors_by_index = build_ancestor_maps(chunks)

    ancestor_tokens = {
        idx: sum(token_by_idx.get(a, 0) for a in ancs)
        for idx, ancs in ancestors_by_index.items()
    }

    # Sort deepest-first for grouping
    chunks_sorted = sorted(chunks, key=lambda c: (c["depth"], c["chunk_index"]), reverse=True)

    def collapse(chs, limit):
        """Collapse contiguous same-depth chunks into blocks."""
        groups = []
        cur, cur_depth, tot = [], None, 0
        for ch in chs:
            d = ch["depth"]
            if cur_depth is None:
                cur_depth = d
            contiguous = (
                d == cur_depth
                and (not cur or ch["chunk_index"] == cur[-1]["chunk_index"] - 1)
            )
            if contiguous and tot + token_by_idx[ch["chunk_index"]] <= limit:
                cur.append(ch)
                tot += token_by_idx[ch["chunk_index"]]
            else:
                if cur:
                    groups.append((cur, tot))
                cur, tot, cur_depth = [ch], token_by_idx[ch["chunk_index"]], d
        if cur:
            groups.append((cur, tot))

        return [
            {
                "start_chunk": min(c["chunk_index"] for c in g),
                "end_chunk": max(c["chunk_index"] for c in g),
                "depth": g[0]["depth"],
                "total_tokens": t,
            }
            for g, t in groups
        ]

    def merge(groups, limit):
        """Merge adjacent blocks upward when possible."""
        out, i = [], 0
        while i < len(groups):
            cur = groups[i]
            if i + 1 < len(groups):
                nxt = groups[i + 1]
                if (
                    cur["depth"] > nxt["depth"]
                    and cur["end_chunk"] == nxt["start_chunk"] - 1
                    and cur["total_tokens"] + nxt["total_tokens"] <= limit
                ):
                    out.append({
                        "start_chunk": cur["start_chunk"],
                        "end_chunk": nxt["end_chunk"],
                        "depth": nxt["depth"],
                        "total_tokens": cur["total_tokens"] + nxt["total_tokens"],
                    })
                    i += 2
                    continue
            out.append(cur)
            i += 1
        return out

    # Iterative adjustment so any leaf->root chain fits
    merged = []
    tl = token_limit
    while tl > 0:
        collapsed = collapse(chunks_sorted, tl)
        merged = merge(collapsed, tl)
        max_path = max(
            (blk["total_tokens"] + ancestor_tokens.get(blk["end_chunk"], 0)
             for blk in merged),
            default=0,
        )
        if max_path <= target_tokens:
            break
        tl -= 10

    def flat_indices(blk):
        return list(range(blk["start_chunk"], blk["end_chunk"] + 1))

    # Group blocks by parent for sibling discovery
    parent_to_blocks: dict[int | None, list[dict]] = {}
    for blk in sorted(merged, key=lambda b: b["start_chunk"]):
        seed = blk["end_chunk"]
        parent = parent_by_index.get(seed)
        parent_to_blocks.setdefault(parent, []).append(blk)

    # Build chunksets, deepest-first, with global coverage tracking
    flat_lists = []
    global_covered: set[int] = set()

    ordered_blocks = sorted(
        merged, key=lambda b: (b["depth"], b["end_chunk"]), reverse=True
    )

    for blk in ordered_blocks:
        core_idxs = set(flat_indices(blk))
        if core_idxs.issubset(global_covered):
            continue

        idxs_set = set(core_idxs)
        tokens_used = sum(token_by_idx.get(i, 0) for i in core_idxs)

        # Prepend all ancestors
        seed = blk["end_chunk"]
        for a in ancestors_by_index.get(seed, ()):
            if a not in idxs_set:
                idxs_set.add(a)
                tokens_used += token_by_idx.get(a, 0)

        # Fill preceding siblings within budget
        budget = target_tokens
        if tokens_used < budget:
            core_parent = parent_by_index.get(seed)
            same_parent = parent_to_blocks.get(core_parent, [])
            preceding = sorted(
                [b for b in same_parent if b["end_chunk"] < blk["start_chunk"]],
                key=lambda b: b["end_chunk"],
                reverse=True,
            )
            siblings_added = 0
            sib_tokens = 0
            cap = int(min(budget * sibling_budget_ratio, tokens_used))
            for sib in preceding:
                sib_range = range(sib["start_chunk"], sib["end_chunk"] + 1)
                if all(i in global_covered for i in sib_range):
                    continue
                new_idxs = [i for i in sib_range if i not in idxs_set]
                if not new_idxs:
                    continue
                delta = sum(token_by_idx.get(i, 0) for i in new_idxs)
                if tokens_used + delta <= budget and sib_tokens + delta <= cap:
                    idxs_set.update(new_idxs)
                    tokens_used += delta
                    sib_tokens += delta
                    siblings_added += 1
                else:
                    break
                if siblings_added >= max_siblings:
                    break

        idxs = sorted(idxs_set)
        if idxs:
            flat_lists.append(idxs)
            global_covered.update(idxs)

    # Dedup subsets
    unique_lists = []
    for lst in sorted(flat_lists, key=len, reverse=True):
        if not any(set(u).issuperset(lst) for u in unique_lists):
            unique_lists.append(lst)
    unique_lists.sort(key=lambda x: x[0])

    # Build final chunksets
    chunksets = []
    for i, lst in enumerate(unique_lists):
        parts, prev = [], -999
        for idx in lst:
            if prev != -999 and idx > prev + 1:
                parts.append(ELLIPSIS_MARKER)
            if idx in chunk_by_idx:
                parts.append(chunk_by_idx[idx]["content"])
            prev = idx
        contents = "\n".join(parts)
        chunksets.append({
            "chunkset_index": i,
            "chunk_ids": lst,
            "contents": contents,
            "to_embed": normalize_for_embedding(contents),
        })

    return chunksets
