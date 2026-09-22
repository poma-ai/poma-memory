"""MCP server for poma-memory. Exposes index, search, forget and status tools.

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
            mount rather than a deletion; True removes them anyway; False
            never. Neither touches rows outside `path`, and neither removes
            anything when `path` itself is absent -- use `poma_forget` for a
            directory that is gone for good.
    """
    if file:
        from poma_memory.api import index_file
        from poma_memory.metadata import MetadataRulesError

        try:
            result = index_file(file, path=path)
        except (OSError, ValueError, MetadataRulesError) as e:
            # OSError too: a missing or unreadable file raised straight out of
            # the tool, and the agent got a transport-level error instead of a
            # sentence it could act on.
            return f"Index failed: {file}: {getattr(e, 'strerror', None) or e}"
        return (
            f"{file}: {result['status']}"
            f" ({result.get('new_chunks', 0)} chunks,"
            f" {result.get('new_chunksets', 0)} chunksets)"
        )

    from poma_memory.api import index as api_index
    from poma_memory.metadata import MetadataRulesError

    try:
        result = api_index(path=path, glob=glob, prune=prune)
    except (OSError, MetadataRulesError) as e:
        # The `file` branch above has always caught this; the directory branch
        # did not, so a `.poma-metadata.json` with a typo in it raised out of
        # the tool as a traceback rather than naming the file and the typo.
        return f"Index failed: {e}"
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
def poma_forget(path: str, db_path: str | None = None) -> str:
    """Remove every indexed row under a directory, whether or not it still exists.

    Use this when a search refuses with "still hold metadata resolved against
    an earlier rule set" and names files under a directory that has been
    deleted or renamed. `poma_index` cannot clear those: it only removes
    documents under a directory it can still see, deliberately, so that a run
    over one directory can never delete another's rows.

    Args:
        path: Directory whose rows to remove (it need not still exist)
        db_path: Database holding them. Required once `path` itself is gone,
            because the default database lives inside it -- the refusal
            message names the database to pass here.
    """
    import sqlite3

    from poma_memory.api import forget

    try:
        result = forget(path, db_path=db_path)
    except (OSError, sqlite3.DatabaseError) as e:
        # `sqlite3.DatabaseError` too: a `db_path` that is not a database
        # reached the agent as a transport error rather than a sentence, which
        # is the defect fixed one function above for `poma_index`.
        return f"Forget failed: {db_path or path}: {e}"
    n = len(result["forgotten"])
    if not n:
        return f"Nothing indexed under {result['root']} in {result['db_path']}."
    return f"Forgot {n} file(s) under {result['root']}."


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
