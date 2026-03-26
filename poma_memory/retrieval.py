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
