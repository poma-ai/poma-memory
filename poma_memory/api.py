"""Public API: index(), search(), status()."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

from poma_memory.store import Store
from poma_memory.incremental import update_file
from poma_memory.metadata import (
    current_rules_hash, load_rules, resolve_paths, rules_hash,
)
from poma_memory.search import HybridSearch

def _disk_state(path: str) -> str:
    """"gone" | "present" | "unknown".

    `os.path.exists` collapses the last two: it returns False for a permission
    error on a parent directory, an unmounted volume and a dead network mount,
    none of which mean the file was deleted. Treating those as deletions
    records a live document as scanned-and-empty, permanently and silently.
    """
    try:
        os.lstat(path)
        return "present"
    except FileNotFoundError:
        return "gone"
    except OSError:
        return "unknown"


def format_updated(upserted_at: float | None) -> str | None:
    """Render a chunkset's upsert time as a YYYY-MM-DD date for result output.

    Returns None for 0/missing (legacy rows indexed before age tracking) so
    callers can omit the line rather than print a misleading epoch-0 date.
    """
    if not upserted_at:
        return None
    try:
        return datetime.fromtimestamp(upserted_at).strftime("%Y-%m-%d")
    except (ValueError, OSError, OverflowError):
        return None


def index(
    path: str | Path = ".agent/",
    db_path: str | Path | None = None,
    glob: str = "**/*.md",
) -> dict:
    """Index all markdown files in a directory.

    Also resolves per-file metadata from `.poma-metadata.json` path rules and
    from each document's own front-matter. This is the only backfill mechanism
    there is: a legacy row with no metadata, and every row after the rule file
    changes, are refreshed here with an in-place UPDATE — no re-chunking and no
    re-embedding.

    Args:
        path: Directory to index (default: .agent/)
        db_path: SQLite database path (default: {path}/.poma-memory.db)
        glob: File pattern to match (default: **/*.md)

    Returns:
        dict with keys: files_indexed, chunks_created, chunksets_created,
        metadata_refreshed
    """
    path = Path(path)
    if db_path is None:
        db_path = path / ".poma-memory.db"

    store = Store(db_path)
    rules, raw = load_rules(path)
    new_hash = rules_hash(raw)
    path_meta = resolve_paths(path, rules)

    total_chunks = 0
    total_chunksets = 0
    files_indexed = 0
    refreshed = False
    seen: set[str] = set()
    unreadable: list[str] = []

    # Taken before the loop, so the dot-prefix rule below asks whether a row
    # already existed rather than whether this run just made one.
    tracked_set = set(store.all_file_paths())

    for md_file in sorted(path.glob(glob)):
        key = os.path.realpath(md_file)
        if md_file.name.startswith(".") and key not in tracked_set:
            # Dot-prefixed names are not indexed by this command. But once a row
            # exists -- `index --file` puts one there -- refusing to revisit it
            # leaves metadata that NO glob can ever backfill, so a filtered
            # search refuses permanently and the remediation it prints cannot
            # work. Skipping only unindexed ones keeps the default behaviour and
            # removes the trap.
            continue
        seen.add(key)
        try:
            result = update_file(
                store, str(md_file),
                path_metadata=path_meta.get(key),
                rules_hash=new_hash,
            )
        except (OSError, UnicodeDecodeError) as e:
            # A dangling symlink or an unreadable file used to abort the whole
            # run with an uncaught exception, skipping store.close(). One bad
            # file should cost that file, not the index.
            #
            # UnicodeDecodeError is a ValueError, not an OSError, so catching
            # OSError alone still let one Latin-1 byte in one document kill the
            # run and lose every file already indexed. It also has no `strerror`.
            unreadable.append(key)
            seen.discard(key)
            print(f"poma-memory: {md_file}: "
                  f"{getattr(e, 'strerror', None) or e}; skipped",
                  file=sys.stderr)
            continue
        if result["status"] == "unreadable":
            unreadable.append(key)
            seen.discard(key)
            print(f"poma-memory: {md_file}: could not be read; metadata left "
                  "unresolved", file=sys.stderr)
            continue
        refreshed = refreshed or result.get("metadata_refreshed", False)
        if result["status"] in ("updated", "reindexed"):
            files_indexed += 1
            total_chunks += result.get("new_chunks", 0)
            total_chunksets += result.get("new_chunksets", 0)

    # Rows whose file is gone from disk. Their indexed content demonstrably has
    # no metadata to find, and leaving them at '' would make the index
    # permanently incomplete — every filtered search would refuse forever.
    #
    # `fp not in seen` is NOT enough to call a row orphaned: this run may have
    # been given a narrower `glob` than the one that indexed it, and a file that
    # still exists has simply not been scanned yet. Recording it as scanned-and-
    # empty would be a lie that no later run corrects, because '{}' looks done.
    # Leaving it at '' is honest — a filtered search refuses until a run whose
    # glob covers it fills it in.
    scanned = store.get_file_metadata_map()
    # Re-read: the loop above may have added rows.
    states = {fp: _disk_state(fp) for fp in store.all_file_paths()
              if fp not in seen}
    orphaned = [fp for fp, st in states.items()
                if st == "gone" and fp not in scanned]
    for fp in orphaned:
        store.set_file_metadata(fp, "{}", False, new_hash)
        print(f"poma-memory: {fp} is indexed but no longer on disk; recorded "
              "as having no metadata", file=sys.stderr)

    # Rows this run did not reach — a narrower glob, a name `index()` skips, a
    # file it could not read. Their metadata is whatever an earlier rule set
    # produced. Per-file hashes mean the next run that reaches them fixes it,
    # but nothing would otherwise say the rule set is only partly applied.
    stale = store.files_with_other_rules(new_hash)
    if stale:
        shown = ", ".join(stale[:3]) + (", ..." if len(stale) > 3 else "")
        print(f"poma-memory: {len(stale)} file(s) still hold metadata from an "
              f"earlier rule set and were not reached by this run ({shown}). "
              "Re-run with a glob that matches them.", file=sys.stderr)

    store.close()
    return {
        "files_indexed": files_indexed,
        "chunks_created": total_chunks,
        "chunksets_created": total_chunksets,
        "metadata_refreshed": refreshed,
        "unreadable": unreadable,
        "stale_rules": stale,
    }


def index_file(
    file: str | Path,
    path: str | Path = ".agent/",
    db_path: str | Path | None = None,
) -> dict:
    """Index one file, resolving it against the directory's path rules.

    Single-file mode still has to see the rules: resolving this file without
    them would store `{}` where a rule says `kind: event`, and the row would
    then look scanned-and-empty rather than unscanned — wrong, and invisible.

    Returns:
        the `update_file` result dict (status and counts)
    """
    path = Path(path)
    if db_path is None:
        db_path = path / ".poma-memory.db"

    store = Store(db_path)
    rules, raw = load_rules(path)
    path_meta = resolve_paths(path, rules)
    if Path(file).name.startswith("."):
        # `index()` does not pick dot-prefixed names up on its own, so this
        # command is what puts the row there. Once it exists `index` does
        # revisit it, which is what keeps a rule edit from stranding it.
        print(f"poma-memory: {file} starts with '.'; `poma-memory index` "
              "will not discover it on its own, but will keep this row "
              "up to date now that it exists", file=sys.stderr)
    result = update_file(
        store, str(file),
        path_metadata=path_meta.get(os.path.realpath(file)),
        rules_hash=rules_hash(raw),
    )
    store.close()
    return result


def search(
    query: str,
    path: str | Path = ".agent/",
    db_path: str | Path | None = None,
    top_k: int = 5,
    min_score: float = 0.0,
    empty_gate: float | None = None,
    where: dict | None = None,
) -> list[dict]:
    """Search indexed content.

    Args:
        query: Search query
        path: Directory that was indexed (for default db_path)
        db_path: SQLite database path
        top_k: Number of results to return
        min_score: Drop results below this fused score (0.0 = no floor)
        empty_gate: Suppress ALL results when the best semantic hit's cosine
            is below this (None = embedder's calibrated default, 0.0 =
            disable; env override POMA_MEMORY_EMPTY_GATE)
        where: Metadata predicate, e.g. {"kind": ["decision", "lesson"]}.
            AND across keys, OR within a list, case-sensitive equality.

    Returns:
        List of dicts with keys: file_path, score, context, chunk_ids

    Raises:
        MetadataNotIndexed: `where` was given against an index that has files
            with no metadata recorded. Run `index()` to backfill.
        MetadataStale: `where` was given against an index whose rows were
            resolved against a rule set that is no longer current. Run
            `index()` with a glob that covers them.
    """
    path = Path(path)
    if db_path is None:
        db_path = path / ".poma-memory.db"

    store = Store(db_path)
    hybrid = HybridSearch(store)
    try:
        results = hybrid.search(
            query, top_k=top_k, min_score=min_score, empty_gate=empty_gate,
            where=where,
            # Only when there is a predicate: locating the rules file costs a
            # read and can raise on a malformed one, and an unfiltered search
            # neither uses metadata nor should start failing because of it.
            rules_hash=current_rules_hash(path, db_path) if where else None,
        )
    finally:
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
        return {"files": [], "total_chunks": 0, "total_chunksets": 0,
                "has_embeddings": False, "files_without_metadata": 0,
                "unparsed_frontmatter": []}

    store = Store(db_path)
    info = store.status()
    store.close()
    return info
