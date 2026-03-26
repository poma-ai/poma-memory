"""MCP server for poma-memory. Exposes index, search, and status tools.

Install with: pip install poma-memory[mcp]
Run with: poma-memory-mcp
Register with: claude mcp add --transport stdio --scope user poma-memory -- poma-memory-mcp
"""

from __future__ import annotations

import json
import sys

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("poma-memory")


@mcp.tool()
def poma_search(query: str, path: str = ".agent/", top_k: int = 5) -> str:
    """Search indexed .agent/ content with structure-preserving hierarchical context.

    Returns ranked results with ancestor headings and [...] gap markers
    for non-contiguous sections. Uses hybrid BM25 + semantic search when
    model2vec is installed.

    Args:
        query: Search query (keywords or natural language)
        path: Directory that was indexed (default: .agent/)
        top_k: Number of results to return (default: 5)
    """
    from poma_memory.api import search

    results = search(query=query, path=path, top_k=top_k)

    if not results:
        return "No results found."

    output = []
    for i, r in enumerate(results, 1):
        output.append(
            f"--- Result {i} (score: {r['score']:.4f}) ---\n"
            f"File: {r['file_path']}\n"
            f"{r['context']}"
        )

    return "\n\n".join(output)


@mcp.tool()
def poma_index(path: str = ".agent/", file: str | None = None, glob: str = "**/*.md") -> str:
    """Index or re-index markdown files for semantic search.

    Supports incremental updates: only processes new content appended
    to existing files. Full reindex if file content was modified (not just appended).

    Args:
        path: Directory to index (default: .agent/)
        file: Optional single file to index (for incremental updates)
        glob: File pattern to match (default: **/*.md)
    """
    from pathlib import Path

    from poma_memory.store import Store
    from poma_memory.incremental import update_file

    if file:
        p = Path(path)
        db_path = str(p / ".poma-memory.db")
        store = Store(db_path)
        result = update_file(store, file)
        store.close()
        return (
            f"{file}: {result['status']}"
            f" ({result.get('new_chunks', 0)} chunks,"
            f" {result.get('new_chunksets', 0)} chunksets)"
        )

    from poma_memory.api import index as api_index

    result = api_index(path=path, glob=glob)
    return (
        f"Indexed {result['files_indexed']} files:"
        f" {result['chunks_created']} chunks,"
        f" {result['chunksets_created']} chunksets"
    )


@mcp.tool()
def poma_status(path: str = ".agent/") -> str:
    """Show poma-memory index status for a directory.

    Args:
        path: Directory that was indexed (default: .agent/)
    """
    from poma_memory.api import status

    info = status(path=path)

    if not info["files"]:
        return "No indexed files. Use poma_index to index .agent/ first."

    lines = [
        f"Files:     {len(info['files'])}",
        f"Chunks:    {info['total_chunks']}",
        f"Chunksets: {info['total_chunksets']}",
        f"Semantic:  {'yes' if info['has_embeddings'] else 'no'}",
    ]
    for f in info["files"]:
        lines.append(f"  - {f}")

    return "\n".join(lines)


def main():
    """Entry point for poma-memory-mcp command."""
    print("poma-memory MCP server starting", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
