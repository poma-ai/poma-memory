"""Chunk creation, depth normalization, and parent linkage.

Converts arrow-prefixed output from indent_light() into structured
chunk dicts with depth and parent_chunk_index fields.
"""

from __future__ import annotations


def parse_indented_text(arrow_text: str) -> list[dict]:
    """Parse arrow-prefixed text into chunk dicts.

    Args:
        arrow_text: Output from indent_light(), lines prefixed with arrows.

    Returns:
        List of dicts: [{chunk_index, content, depth}]
    """
    chunks = []
    for i, line in enumerate(arrow_text.split("\n")):
        if not line.strip():
            continue
        depth = 0
        while depth < len(line) and line[depth] == "\u2192":  # →
            depth += 1
        content = line[depth:]
        if content.strip():
            chunks.append({
                "chunk_index": len(chunks),
                "content": content,
                "depth": depth,
            })
    return chunks


def normalize_depths(chunks: list[dict]) -> list[dict]:
    """Assign parent_chunk_index to each chunk based on depth hierarchy.

    Walks the chunk list and for each chunk finds the nearest preceding
    chunk with depth = current_depth - 1.

    Args:
        chunks: List from parse_indented_text()

    Returns:
        Same list with parent_chunk_index added to each dict.
    """
    # Stack of (depth, chunk_index) for tracking parent chain
    stack: list[tuple[int, int]] = []

    for chunk in chunks:
        depth = chunk["depth"]

        # Pop stack entries that are at same or deeper depth
        while stack and stack[-1][0] >= depth:
            stack.pop()

        # Parent is top of stack (if any)
        chunk["parent_chunk_index"] = stack[-1][1] if stack else None

        stack.append((depth, chunk["chunk_index"]))

    return chunks
