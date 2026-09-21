"""Incremental update logic for append-only markdown files."""

from __future__ import annotations

import hashlib
import json
import os

from poma_primecut_nano.chunker import indent_light
from poma_primecut_nano.tree import parse_indented_text, normalize_depths
from poma_primecut_nano import chunks_to_chunksets
from poma_memory import frontmatter, metadata as meta_mod
from poma_memory.store import Store


def _resolve_metadata(
    file_path: str, path_metadata: dict | None, text: str | None = None,
) -> tuple[str, bool]:
    """Combine path-rule metadata with the file's own front-matter.

    `text` is the already-read document when there is one. Without it only the
    head of the file is read: metadata backfill must not cost a full read of
    every document, and front-matter cannot legally live past the head anyway.
    """
    if text is None:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read(frontmatter.MAX_BYTES)
        except OSError:
            text = ""
    fm, ok = frontmatter.parse(text)
    return json.dumps(meta_mod.merge(path_metadata, fm), sort_keys=True), not ok


def update_file(
    store: Store,
    file_path: str,
    path_metadata: dict | None = None,
    refresh_metadata: bool = False,
) -> dict:
    """Incrementally update index for a single file.

    For append-only files, only processes new content from the last
    known byte offset. Falls back to full reindex if the existing
    prefix was modified.

    `path_metadata` is what the caller's path rules say about this file;
    `refresh_metadata` forces metadata to be re-resolved even when the content
    is untouched, which is how a rule-file edit reaches rows whose mtime has
    not moved.

    Returns:
        dict with status ("unchanged", "updated", "reindexed") and counts.
    """
    # Normalize to absolute path to prevent duplicates from relative vs absolute indexing
    file_path = os.path.realpath(file_path)
    stat = os.stat(file_path)
    record = store.get_file_record(file_path)

    # Check if file is unchanged
    if record and record["mtime"] == stat.st_mtime:
        # Content is untouched, but metadata may not be: a legacy row has never
        # had any, and a rule edit changes what this file resolves to without
        # touching the file. Both are an in-place UPDATE — no re-chunk, no
        # re-embed.
        if refresh_metadata or not record.get("metadata"):
            meta_json, fm_unparsed = _resolve_metadata(file_path, path_metadata)
            store.set_file_metadata(file_path, meta_json, fm_unparsed)
        return {"status": "unchanged"}

    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    meta_json, fm_unparsed = _resolve_metadata(file_path, path_metadata, full_text)

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
                    meta_json, fm_unparsed,
                )
                return {"status": "unchanged"}

            return _incremental_update(
                store, file_path, full_text, new_text, stat.st_mtime,
                meta_json, fm_unparsed,
            )

    # Full reindex (first time or prefix was modified)
    return _full_reindex(store, file_path, full_text, stat.st_mtime,
                         meta_json, fm_unparsed)


def _incremental_update(
    store: Store,
    file_path: str,
    full_text: str,
    new_text: str,
    mtime: float,
    meta_json: str = "{}",
    fm_unparsed: bool = False,
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
            _hash(full_text), mtime, meta_json, fm_unparsed,
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
        _hash(full_text), mtime, meta_json, fm_unparsed,
    )

    return {
        "status": "updated",
        "new_chunks": len(new_chunks),
        "new_chunksets": len(new_chunksets),
    }


def _full_reindex(
    store: Store, file_path: str, full_text: str, mtime: float,
    meta_json: str = "{}", fm_unparsed: bool = False,
) -> dict:
    """Full reindex: delete existing data and re-chunk entire file."""
    store.delete_file_data(file_path)

    arrow_text = indent_light(full_text)
    chunks = parse_indented_text(arrow_text)
    chunks = normalize_depths(chunks)

    if not chunks:
        store.upsert_file_record(file_path, len(full_text),
                                  _hash(full_text), mtime,
                                  meta_json, fm_unparsed)
        return {"status": "reindexed", "new_chunks": 0, "new_chunksets": 0}

    store.insert_chunks(file_path, chunks)

    chunksets = chunks_to_chunksets(chunks)
    store.insert_chunksets(file_path, chunksets)

    store.upsert_file_record(
        file_path, len(full_text),
        _hash(full_text), mtime, meta_json, fm_unparsed,
    )

    return {
        "status": "reindexed",
        "new_chunks": len(chunks),
        "new_chunksets": len(chunksets),
    }


def _hash(text: str) -> str:
    """Hash a string using SHA-256."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
