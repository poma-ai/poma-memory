"""Public API: index(), search(), status()."""

from __future__ import annotations

from pathlib import Path

from poma_memory.store import Store
from poma_memory.incremental import update_file
from poma_memory.search import HybridSearch


def index(
    path: str | Path = ".agent/",
    db_path: str | Path | None = None,
    glob: str = "**/*.md",
) -> dict:
    """Index all markdown files in a directory.

    Args:
        path: Directory to index (default: .agent/)
        db_path: SQLite database path (default: {path}/.poma-memory.db)
        glob: File pattern to match (default: **/*.md)

    Returns:
        dict with keys: files_indexed, chunks_created, chunksets_created
    """
    path = Path(path)
    if db_path is None:
        db_path = path / ".poma-memory.db"

    store = Store(db_path)
    total_chunks = 0
    total_chunksets = 0
    files_indexed = 0

    for md_file in sorted(path.glob(glob)):
        if md_file.name.startswith("."):
            continue
        result = update_file(store, str(md_file))
        if result["status"] in ("updated", "reindexed"):
            files_indexed += 1
            total_chunks += result.get("new_chunks", 0)
            total_chunksets += result.get("new_chunksets", 0)

    store.close()
    return {
        "files_indexed": files_indexed,
        "chunks_created": total_chunks,
        "chunksets_created": total_chunksets,
    }


def search(
    query: str,
    path: str | Path = ".agent/",
    db_path: str | Path | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Search indexed content.

    Args:
        query: Search query
        path: Directory that was indexed (for default db_path)
        db_path: SQLite database path
        top_k: Number of results to return

    Returns:
        List of dicts with keys: file_path, score, context, chunk_ids
    """
    path = Path(path)
    if db_path is None:
        db_path = path / ".poma-memory.db"

    store = Store(db_path)
    hybrid = HybridSearch(store)
    results = hybrid.search(query, top_k=top_k)
    store.close()
    return results


def status(
    path: str | Path = ".agent/",
    db_path: str | Path | None = None,
) -> dict:
    """Show index status.

    Returns:
        dict with keys: files, total_chunks, total_chunksets, has_embeddings
    """
    path = Path(path)
    if db_path is None:
        db_path = path / ".poma-memory.db"

    if not Path(db_path).exists():
        return {"files": [], "total_chunks": 0, "total_chunksets": 0, "has_embeddings": False}

    store = Store(db_path)
    info = store.status()
    store.close()
    return info
