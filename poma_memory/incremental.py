"""Incremental update logic for append-only markdown files."""

from __future__ import annotations

import hashlib
import os

from poma_memory.chunker import indent_light
from poma_memory.tree import parse_indented_text, normalize_depths
from poma_memory.chunksets import chunks_to_chunksets
from poma_memory.store import Store


def update_file(store: Store, file_path: str) -> dict:
    """Incrementally update index for a single file.

    For append-only files, only processes new content from the last
    known byte offset. Falls back to full reindex if the existing
    prefix was modified.

    Returns:
        dict with status ("unchanged", "updated", "reindexed") and counts.
    """
    # Normalize to absolute path to prevent duplicates from relative vs absolute indexing
    file_path = os.path.realpath(file_path)
    stat = os.stat(file_path)
    record = store.get_file_record(file_path)

    # Check if file is unchanged
    if record and record["mtime"] == stat.st_mtime:
        return {"status": "unchanged"}

    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # Try incremental (append-only fast path)
    if record and record["byte_offset"] > 0:
        prefix = full_text[: record["byte_offset"]]
        prefix_hash = _hash(prefix)

        if prefix_hash == record["content_hash"]:
            # Prefix unchanged — append-only case
            new_text = full_text[record["byte_offset"] :]
            if not new_text.strip():
                store.upsert_file_record(
                    file_path, len(full_text),
                    prefix_hash, stat.st_mtime,
                )
                return {"status": "unchanged"}

            return _incremental_update(
                store, file_path, full_text, new_text, stat.st_mtime
            )

    # Full reindex (first time or prefix was modified)
    return _full_reindex(store, file_path, full_text, stat.st_mtime)


def _incremental_update(
    store: Store,
    file_path: str,
    full_text: str,
    new_text: str,
    mtime: float,
) -> dict:
    """Process only the appended portion of a file."""
    # Get heading context from existing chunks for proper depth assignment
    last_heading = store.get_last_heading_chunk(file_path)

    # Inject synthetic heading context so indent_light knows the current depth
    if last_heading:
        depth = last_heading["depth"]
        prefix = "#" * max(1, depth + 1) + " " + last_heading["content"] + "\n"
        chunker_input = prefix + new_text
        skip_first = True
    else:
        chunker_input = new_text
        skip_first = False

    # Chunk the new text
    arrow_text = indent_light(chunker_input, extract_title=not skip_first)
    new_chunks = parse_indented_text(arrow_text)
    new_chunks = normalize_depths(new_chunks)

    if skip_first and new_chunks:
        new_chunks = new_chunks[1:]

    if not new_chunks:
        store.upsert_file_record(
            file_path, len(full_text),
            _hash(full_text), mtime,
        )
        return {"status": "updated", "new_chunks": 0, "new_chunksets": 0}

    # Re-index local_index continuing from existing max
    max_idx = store.get_max_local_index(file_path)
    for i, chunk in enumerate(new_chunks):
        chunk["chunk_index"] = max_idx + 1 + i

    # Re-normalize parent pointers for the new chunk batch
    new_chunks = normalize_depths(new_chunks)

    # Store new chunks
    store.insert_chunks(file_path, new_chunks)

    # Build chunksets for new chunks only
    new_chunksets = chunks_to_chunksets(new_chunks)
    # Offset chunkset indices
    existing_chunksets = len(store.get_all_chunksets())
    for cs in new_chunksets:
        cs["chunkset_index"] = existing_chunksets + cs["chunkset_index"]
    store.insert_chunksets(file_path, new_chunksets)

    store.upsert_file_record(
        file_path, len(full_text),
        _hash(full_text), mtime,
    )

    return {
        "status": "updated",
        "new_chunks": len(new_chunks),
        "new_chunksets": len(new_chunksets),
    }


def _full_reindex(
    store: Store, file_path: str, full_text: str, mtime: float,
) -> dict:
    """Full reindex: delete existing data and re-chunk entire file."""
    store.delete_file_data(file_path)

    arrow_text = indent_light(full_text)
    chunks = parse_indented_text(arrow_text)
    chunks = normalize_depths(chunks)

    if not chunks:
        store.upsert_file_record(file_path, len(full_text),
                                  _hash(full_text), mtime)
        return {"status": "reindexed", "new_chunks": 0, "new_chunksets": 0}

    store.insert_chunks(file_path, chunks)

    chunksets = chunks_to_chunksets(chunks)
    store.insert_chunksets(file_path, chunksets)

    store.upsert_file_record(
        file_path, len(full_text),
        _hash(full_text), mtime,
    )

    return {
        "status": "reindexed",
        "new_chunks": len(chunks),
        "new_chunksets": len(chunksets),
    }


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
