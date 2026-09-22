"""MCP server for poma-memory. Exposes index, search, and status tools.

Install with: pip install poma-memory[mcp]
Run with: poma-memory-mcp
Register with: claude mcp add --transport stdio --scope user poma-memory -- poma-memory-mcp
"""

from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore", message="A NumPy version", category=UserWarning)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("poma-memory")


@mcp.tool()
def poma_search(
    query: str,
    path: str = ".agent/",
    top_k: int = 5,
    min_score: float = 0.0,
    empty_gate: float | None = None,
    where: dict | None = None,
) -> str:
    """Search indexed .agent/ content with structure-preserving hierarchical context.

    Returns ranked results with ancestor headings and [...] gap markers
    for non-contiguous sections. Uses hybrid BM25 + semantic search when
    model2vec is installed.

    Args:
        query: Search query (keywords or natural language)
        path: Directory that was indexed (default: .agent/)
        top_k: Number of results to return (default: 5)
        min_score: Drop results below this fused score (0.0 = no floor)
        empty_gate: Suppress ALL results when the best semantic hit's cosine
            is below this (default: embedder-calibrated; 0 disables)
        where: Metadata predicate over the indexed files, e.g.
            {"kind": ["decision", "lesson"]}. AND across keys, OR within a
            list, case-sensitive equality. Requires the index to have been
            built with metadata (see `.poma-metadata.json`).
    """
    from poma_memory.api import search
    from poma_memory.metadata import MetadataIncomplete

    try:
        results = search(
            query=query, path=path, top_k=top_k, min_score=min_score,
            empty_gate=empty_gate, where=where,
        )
    except (MetadataIncomplete, ValueError) as e:
        return f"Search failed: {e}"

    if not results:
        return "No results found."

    from poma_memory.api import format_updated
    output = []
    for i, r in enumerate(results, 1):
        updated = format_updated(r.get("upserted_at"))
        updated_line = f"Updated: {updated}\n" if updated else ""
        output.append(
            f"--- Result {i} (score: {r['score']:.4f}) ---\n"
            f"File: {r['file_path']}\n"
            f"{updated_line}"
            f"{r['context']}"
        )

    return "\n\n".join(output)


@mcp.tool()
def poma_index(path: str = ".agent/", file: str | None = None,
               glob: str = "**/*.md", prune: bool | None = None) -> str:
    """Index or re-index markdown files for semantic search.

    Supports incremental updates: only processes new content appended
    to existing files. Full reindex if file content was modified (not just appended).

    Args:
        path: Directory to index (default: .agent/)
        file: Optional single file to index (for incremental updates)
        glob: File pattern to match (default: **/*.md)
        prune: Remove indexed files that are gone from disk. None (default)
            removes them unless that looks like a directory that failed to
            mount rather than a deletion; True always removes; False never.
    """
    if file:
        from poma_memory.api import index_file
        from poma_memory.metadata import MetadataRulesError

        try:
            result = index_file(file, path=path)
        except (ValueError, MetadataRulesError) as e:
            return f"Index failed: {e}"
        return (
            f"{file}: {result['status']}"
            f" ({result.get('new_chunks', 0)} chunks,"
            f" {result.get('new_chunksets', 0)} chunksets)"
        )

    from poma_memory.api import index as api_index

    result = api_index(path=path, glob=glob, prune=prune)
    summary = (
        f"Indexed {result['files_indexed']} files:"
        f" {result['chunks_created']} chunks,"
        f" {result['chunksets_created']} chunksets"
    )
    # The agent calling this tool sees only what is returned; `index` writes the
    # prune lines to stderr, which on a stdio MCP server reaches the client's
    # log and not the model. Removing documents from the index is not something
    # a caller should have to read a logfile to discover.
    if result.get("pruned"):
        summary += (f" ({len(result['pruned'])} removed: no longer on disk)")
    if result.get("prune_held_back"):
        # Deliberately loud: the index is missing most of its files and nothing
        # was deleted, which the agent has to know to interpret later searches.
        summary += (f" ({len(result['prune_held_back'])} indexed files are"
                    " missing and were NOT removed — it looks like a directory"
                    " that failed to mount rather than a deletion. Call again"
                    " with prune=True if they really are gone.)")
    return summary


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
    # An agent told "N files have no metadata" by poma_search needs a surface
    # that confirms and quantifies it; without these it has none.
    missing = info.get("files_without_metadata", 0)
    stale = info.get("stale_rules", [])
    if info.get("rules_error"):
        lines.append(f"Metadata:  rules file unusable - {info['rules_error']}")
    elif missing:
        lines.append(f"Metadata:  {missing} file(s) unscanned - run poma_index")
    elif stale:
        lines.append(f"Metadata:  {len(stale)} file(s) on an earlier rule set "
                     "- run poma_index")
    else:
        lines.append("Metadata:  complete")
    for f in stale[:3]:
        lines.append(f"  ! earlier rule set: {f}")
    for f in info.get("unparsed_frontmatter", []):
        lines.append(f"  ! unparsed front-matter: {f}")
    for f in info["files"]:
        lines.append(f"  - {f}")

    return "\n".join(lines)


def main():
    """Entry point for poma-memory-mcp command."""
    print("poma-memory MCP server starting", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
