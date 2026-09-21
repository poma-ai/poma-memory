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
) -> tuple[str, bool] | None:
    """Combine path-rule metadata with the file's own front-matter.

    `text` is the already-read document when there is one. Without it only the
    head of the file is read: metadata backfill must not cost a full read of
    every document, and front-matter cannot legally live past the head anyway.

    Returns None when the file could not be read. That is NOT the same as "it
    has no front-matter": swallowing the error and parsing "" stamps a live
    document with path-rule metadata alone, flags it parsed, and counts it
    scanned — permanently and silently wrong, and indistinguishable afterwards
    from a document that really carries nothing.
    """
    if text is None:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read(frontmatter.MAX_BYTES)
                # A cut-short head disagrees with the full read in BOTH
                # directions: an unterminated block looks unparsed, and a
                # `---` line straddling the boundary looks terminated when it
                # is not. Whether a file's metadata parsed must not depend on
                # which code path last touched it, so once a truncated head
                # turns out to carry a fence at all, pay for the rest. Files
                # with no front-matter — the overwhelming majority — still
                # cost one bounded read.
                if (len(text) == frontmatter.MAX_BYTES
                        and text.lstrip("\ufeff").startswith(frontmatter.FENCE)):
                    # Bounded by the same limit the parser applies to a block,
                    # so the two paths still agree and a huge file that merely
                    # opens with `---` cannot be pulled into memory whole.
                    text += f.read(frontmatter.MAX_BLOCK_BYTES)
        except OSError:
            return None
    fm, ok = frontmatter.parse(text)
    return json.dumps(meta_mod.merge(path_metadata, fm), sort_keys=True), not ok


def update_file(
    store: Store,
    file_path: str,
    path_metadata: dict | None = None,
    rules_hash: str = "",
) -> dict:
    """Incrementally update index for a single file.

    For append-only files, only processes new content from the last
    known byte offset. Falls back to full reindex if the existing
    prefix was modified.

    `path_metadata` is what the caller's path rules say about this file, and
    `rules_hash` identifies the rule set it came from. The hash is recorded on
    the row, so a file re-resolves exactly when the rules that produced it have
    changed — a rule edit moves no mtime, and a global "rules changed" flag
    cannot express "this row was covered but that one was not".

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
        stale = (not record.get("metadata")
                 or record.get("rules_hash") != rules_hash)
        resolved = _resolve_metadata(file_path, path_metadata) if stale else None
        if stale and resolved is not None:
            meta_json, fm_unparsed = resolved
            store.set_file_metadata(file_path, meta_json, fm_unparsed, rules_hash)
        elif stale:
            # Unreadable right now. Leave the row exactly as it was so the
            # next run that can read it resolves it properly.
            return {"status": "unreadable", "metadata_refreshed": False}
        return {"status": "unchanged", "metadata_refreshed": stale}

    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    resolved = _resolve_metadata(file_path, path_metadata, full_text)
    # The content was read above, so this cannot be a read failure; the guard
    # is here so a future change to _resolve_metadata cannot silently reach
    # the upserts below with nothing.
    meta_json, fm_unparsed = resolved if resolved is not None else ("", False)

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
                    meta_json, fm_unparsed, rules_hash,
                )
                return {"status": "unchanged"}

            return _incremental_update(
                store, file_path, full_text, new_text, stat.st_mtime,
                meta_json, fm_unparsed, rules_hash,
            )

    # Full reindex (first time or prefix was modified)
    return _full_reindex(store, file_path, full_text, stat.st_mtime,
                         meta_json, fm_unparsed, rules_hash)


def _incremental_update(
    store: Store,
    file_path: str,
    full_text: str,
    new_text: str,
    mtime: float,
    meta_json: str = "{}",
    fm_unparsed: bool = False,
    rules_hash: str = "",
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
            _hash(full_text), mtime, meta_json, fm_unparsed, rules_hash,
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
        _hash(full_text), mtime, meta_json, fm_unparsed, rules_hash,
    )

    return {
        "status": "updated",
        "new_chunks": len(new_chunks),
        "new_chunksets": len(new_chunksets),
    }


def _full_reindex(
    store: Store, file_path: str, full_text: str, mtime: float,
    meta_json: str = "{}", fm_unparsed: bool = False, rules_hash: str = "",
) -> dict:
    """Full reindex: delete existing data and re-chunk entire file."""
    store.delete_file_data(file_path)

    arrow_text = indent_light(full_text)
    chunks = parse_indented_text(arrow_text)
    chunks = normalize_depths(chunks)

    if not chunks:
        store.upsert_file_record(file_path, len(full_text),
                                  _hash(full_text), mtime,
                                  meta_json, fm_unparsed, rules_hash)
        return {"status": "reindexed", "new_chunks": 0, "new_chunksets": 0}

    store.insert_chunks(file_path, chunks)

    chunksets = chunks_to_chunksets(chunks)
    store.insert_chunksets(file_path, chunksets)

    store.upsert_file_record(
        file_path, len(full_text),
        _hash(full_text), mtime, meta_json, fm_unparsed, rules_hash,
    )

    return {
        "status": "reindexed",
        "new_chunks": len(chunks),
        "new_chunksets": len(chunksets),
    }


def _hash(text: str) -> str:
    """Hash a string using SHA-256."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
