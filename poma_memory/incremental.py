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


def _stat_agrees(record: dict, stat: os.stat_result) -> bool:
    """Whether the recorded stat signature still matches the file on disk.

    Size AND ctime, because mtime is settable and the other two are not in the
    same way. `os.utime` restores mtime, which is what archive and sync tools
    do; size catches any edit that changes the file's length; and ctime catches
    the rest, because it moves on ANY write and `utime` itself bumps it. An
    equal-length in-place substitution with the timestamp restored -- `sed -i`
    on a status token, then a restore -- is invisible to mtime and size
    together, and is the case ctime is here for.

    Windows caveat: `st_ctime` there is CREATION time and does not move on
    write, so on Windows this degrades to the mtime-and-size pair. That is why
    ctime is an additional signal and not a replacement for size.

    A row written before these columns existed records 0, which will not match
    a real file, so it reads as changed and is re-read once. Deliberate: the
    alternative -- trusting mtime alone when the signature is unknown -- leaves
    every legacy row permanently unprotected, because the short-circuit it then
    takes is also the path that never records one. One extra read per file,
    once, buys the check for good, and costs no re-chunking or re-embedding
    because an unchanged file still matches its content hash on the append path
    and comes back "unchanged". A false positive from `chmod` or a rename costs
    exactly the same single read.
    """
    return (record.get("size_bytes") == stat.st_size
            and record.get("ctime") == stat.st_ctime)


def update_file(
    store: Store,
    file_path: str,
    path_metadata: dict | None = None,
    rules_hash: str = "",
    rules_root: str = "",
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

    # Check if file is unchanged. mtime alone is not enough: archive and sync
    # tools that preserve timestamps carry a CHANGED file across with its old
    # mtime, and the row then keeps metadata the document no longer says while
    # `status` reports the index complete. See `_stat_agrees` for what the two
    # extra fields each catch and what still gets through on Windows. The stat
    # already happened, so the check costs nothing.
    if record and record["mtime"] == stat.st_mtime and _stat_agrees(record, stat):
        # Content is untouched, but metadata may not be: a legacy row has never
        # had any, and a rule edit changes what this file resolves to without
        # touching the file. Both are an in-place UPDATE — no re-chunk, no
        # re-embed.
        # `rules_root` belongs in this test as much as the hash does. Without
        # it, a row carrying metadata and a matching hash but no recorded root
        # is refused by search (an unverifiable row is not trusted) and skipped
        # by this refresh, so `index` could never heal it -- refusing forever,
        # which is the failure this whole mechanism exists to prevent.
        stale = (not record.get("metadata")
                 or record.get("rules_hash") != rules_hash
                 or record.get("rules_root") != rules_root)
        resolved = _resolve_metadata(file_path, path_metadata) if stale else None
        if stale and resolved is not None:
            meta_json, fm_unparsed = resolved
            store.set_file_metadata(file_path, meta_json, fm_unparsed,
                                    rules_hash, rules_root)
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
                # `_hash(full_text)`, not `prefix_hash`: byte_offset advances to
                # the whole file, and content_hash must describe the same span.
                # Recording the old prefix's hash against the new offset broke
                # that invariant, so the NEXT real append failed its prefix
                # check and fell into a full re-chunk and re-embed of the file.
                store.upsert_file_record(
                    file_path, len(full_text),
                    _hash(full_text), stat.st_mtime,
                    meta_json, fm_unparsed, rules_hash, rules_root,
                    stat.st_size, stat.st_ctime,
                )
                return {"status": "unchanged"}

            return _incremental_update(
                store, file_path, full_text, new_text, stat.st_mtime,
                meta_json, fm_unparsed, rules_hash, rules_root,
                stat.st_size, stat.st_ctime,
            )

    # Full reindex (first time or prefix was modified)
    return _full_reindex(store, file_path, full_text, stat.st_mtime,
                         meta_json, fm_unparsed, rules_hash, rules_root,
                         stat.st_size, stat.st_ctime)


def _incremental_update(
    store: Store,
    file_path: str,
    full_text: str,
    new_text: str,
    mtime: float,
    meta_json: str = "{}",
    fm_unparsed: bool = False,
    rules_hash: str = "",
    rules_root: str = "",
    size_bytes: int = 0,
    ctime: float = 0.0,
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
            rules_root, size_bytes, ctime,
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

    # Build chunksets for new chunks only, continuing THIS file's sequence.
    #
    # It used to offset by the global chunkset count, against a column that is
    # `UNIQUE(file_path, local_index)` — per file. While the corpus only grew
    # that produced gaps and out-of-order indices but no error; once anything
    # removes chunksets the count goes down, later appends reuse indices this
    # file already holds, and the insert dies with an IntegrityError mid-run.
    # Deleting a file now prunes its chunksets, so the count really does fall.
    # Reproduced: a.md at [0, 1, 5] after a delete, then appends landing on
    # 3, 4, and finally 5 again.
    #
    # `max + 1` is what the chunk path above already does, and it is correct
    # over gapped sequences too, so existing databases heal rather than needing
    # a migration.
    new_chunksets = chunks_to_chunksets(new_chunks)
    max_cs = store.get_max_chunkset_local_index(file_path)
    for cs in new_chunksets:
        cs["chunkset_index"] = max_cs + 1 + cs["chunkset_index"]
    store.insert_chunksets(file_path, new_chunksets)

    store.upsert_file_record(
        file_path, len(full_text),
        _hash(full_text), mtime, meta_json, fm_unparsed, rules_hash,
        rules_root, size_bytes, ctime,
    )

    return {
        "status": "updated",
        "new_chunks": len(new_chunks),
        "new_chunksets": len(new_chunksets),
    }


def _full_reindex(
    store: Store, file_path: str, full_text: str, mtime: float,
    meta_json: str = "{}", fm_unparsed: bool = False, rules_hash: str = "",
    rules_root: str = "",
    size_bytes: int = 0,
    ctime: float = 0.0,
) -> dict:
    """Full reindex: delete existing data and re-chunk entire file."""
    store.delete_file_data(file_path)

    arrow_text = indent_light(full_text)
    chunks = parse_indented_text(arrow_text)
    chunks = normalize_depths(chunks)

    if not chunks:
        store.upsert_file_record(file_path, len(full_text),
                                  _hash(full_text), mtime,
                                  meta_json, fm_unparsed, rules_hash,
                                  rules_root, size_bytes, ctime)
        return {"status": "reindexed", "new_chunks": 0, "new_chunksets": 0}

    store.insert_chunks(file_path, chunks)

    chunksets = chunks_to_chunksets(chunks)
    store.insert_chunksets(file_path, chunksets)

    store.upsert_file_record(
        file_path, len(full_text),
        _hash(full_text), mtime, meta_json, fm_unparsed, rules_hash,
        rules_root, size_bytes, ctime,
    )

    return {
        "status": "reindexed",
        "new_chunks": len(chunks),
        "new_chunksets": len(chunksets),
    }


def _hash(text: str) -> str:
    """Hash a string using SHA-256."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
