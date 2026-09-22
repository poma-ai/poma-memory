"""Public API: index(), search(), status()."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

from poma_memory.store import Store
from poma_memory.incremental import update_file
from poma_memory.metadata import (
    RULES_FILENAME, MetadataIncomplete, MetadataRulesError, load_rules,
    resolve_paths, rules_hash,
    stale_files,
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


# Below this many missing files, a run prunes without asking however small the
# index is: three deletions out of four documents is a plausible afternoon, and
# refusing there would make the guard a nuisance rather than a safeguard.
_PRUNE_FLOOR = 5


def index(
    path: str | Path = ".agent/",
    db_path: str | Path | None = None,
    glob: str = "**/*.md",
    prune: bool | None = None,
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
        prune: Remove indexed files that are gone from disk. None (default)
            removes them unless a directory under `path` has vanished or the
            removals would be most of the index, both of which look more like
            something that failed to mount than a deletion. True removes them
            anyway; False never does. Neither reaches rows outside `path`, and
            neither removes anything at all when `path` itself is absent — for
            a directory that is gone for good, see `forget`.

    Returns:
        dict with keys: files_indexed, chunks_created, chunksets_created,
        metadata_refreshed, unreadable, stale_rules, pruned, prune_held_back
    """
    path = Path(path)
    if db_path is None:
        db_path = path / ".poma-memory.db"

    # Recorded on every row this run writes, so a later search re-reads THIS
    # directory's rules for these files however the database is addressed.
    root_key = os.path.realpath(path)

    # A RUN MUST NOT CREATE THE DIRECTORY IT WAS ASKED TO INDEX. `Store` makes
    # its database's parent, and the default database sits inside `path`, so
    # `poma-memory index R` with R gone recreated R and left an empty database
    # in it. That is not a cosmetic wart: R's absence is what the root-present
    # gate below reads, so the next run saw a present root, found three rows
    # whose files were gone, and — under the floor, with the parent directory
    # now existing so the missing-directory rule did not apply either — deleted
    # every one of them. Two ordinary commands, no flags, no prompt:
    #
    #     poma-memory index R                 # R and a stray db appear
    #     poma-memory index R --db shared.db  # rows 3 -> 0
    #
    # Reproduced end to end. `search` and `forget` already refuse to open a
    # database that is not there; this is the same rule for the third surface.
    # An EXISTING database outside the root is untouched by this and still runs
    # (it reports and skips pruning, below), because opening it creates nothing.
    if _disk_state(root_key) != "present" and not Path(db_path).exists():
        print(f"poma-memory: {root_key} is not present and there is no index "
              f"at {db_path}; nothing to do. Nothing was created.",
              file=sys.stderr)
        return {"files_indexed": 0, "chunks_created": 0, "chunksets_created": 0,
                "metadata_refreshed": False, "unreadable": [], "stale_rules": [],
                "pruned": [], "prune_held_back": []}

    store = Store(db_path)
    try:
        rules, _ = load_rules(path)
        new_hash = rules_hash(rules)
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
            try:
                inside = os.path.commonpath([key, root_key]) == root_key
            except ValueError:
                inside = False
            if not inside:
                # `index_file` refuses exactly this below, for the same reason:
                # `path` supplies the rules and is recorded as their source. A
                # symlink pointing out of the tree lands a row that gate 2 then
                # excludes from pruning forever, so it can never be removed or
                # refreshed -- and a later rules edit refuses every filtered
                # search with no way out.
                print(f"poma-memory: {md_file} resolves to {key}, outside "
                      f"{root_key}; skipped", file=sys.stderr)
                continue
            seen.add(key)
            try:
                result = update_file(
                    store, str(md_file),
                    path_metadata=path_meta.get(key),
                    rules_hash=new_hash,
                    rules_root=root_key,
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

        # Rows whose file is gone from disk are PRUNED — chunks, chunksets and the
        # row itself. Keeping their content meant a deleted or renamed document went
        # on answering searches, including filtered ones, where a caller narrowing to
        # `kind=decision` reasonably reads the result as the current decision set.
        #
        # This is the only destructive operation in the package, and losing a row
        # costs a re-chunk AND a re-embed of it — real money on a paid embedder, and
        # more than that, because a single NULL embedding makes the semantic index
        # re-embed every chunkset it holds. So it is gated three ways, each of which
        # was a reproduced way to delete live data:
        #
        # 1. THE ROOT MUST BE PRESENT. `_disk_state` maps FileNotFoundError to
        #    "gone", and an unmounted volume is not "some other OSError" — its
        #    contents are ENOENT, so every row under it read "gone" and the whole
        #    corpus was deleted by an `index` run against a drive that was not
        #    mounted. The earlier guard only ever covered EACCES, which was never
        #    the dangerous errno.
        # 2. ONLY ROWS UNDER THIS RUN'S ROOT. `all_file_paths()` is the whole
        #    database, and two roots may share one (`--db`, and the README
        #    advertises it). A run given root A has no rules for root B, cannot see
        #    it, and must not delete it — renaming B's directory made an `index` of
        #    A destroy B's index entirely.
        # 3. RE-CHECKED IMMEDIATELY BEFORE THE DELETE, because the state was
        #    sampled for every row before any of them were deleted, and an editor
        #    saving atomically inside that window would lose a file that exists.
        #
        # `fp not in seen` alone is NOT authority to delete: this run may have been
        # given a narrower `glob`, and a file that still exists has simply not been
        # scanned. `_disk_state` answers the real question.
        #
        # NONE of these three has an override, `--prune` included. `--prune` is
        # a statement about the threshold — "yes, remove that many" — and the
        # directory is present and inspectable when a user makes it. Letting it
        # also override gate 1 would put the destructive answer behind the flag
        # people type for ordinary deletions, at exactly the moment they cannot
        # tell "deleted" from "not mounted": measured, `index <root> --prune`
        # against an absent root took 12 rows and 12 chunksets to zero. The
        # route out of a root that really is gone is `forget`, which is a
        # different word for a different question.
        root_present = _disk_state(root_key) == "present"
        candidates = []
        if root_present:
            for fp in store.all_file_paths():
                if fp in seen:
                    continue
                try:
                    under_root = os.path.commonpath([fp, root_key]) == root_key
                except ValueError:
                    under_root = False      # different drives on Windows
                if under_root:
                    candidates.append(fp)
        elif store.all_file_paths():
            print(f"poma-memory: {root_key} is not present; skipping the check "
                  "for indexed files that have been deleted. If it is gone for "
                  f"good, `poma-memory forget {root_key} --db {db_path}` "
                  "removes its rows.", file=sys.stderr)

        # Sampled for every candidate first, so the set being deleted is decided
        # from one consistent view...
        gone = [fp for fp in candidates if _disk_state(fp) == "gone"]

        # A whole subtree vanishing is NOT the same event as a file being
        # deleted, and the filesystem cannot tell them apart: an unmounted
        # mountpoint nested inside a present root reports ENOENT for everything
        # under it, exactly as a deleted directory does. The root-present gate
        # above only catches the case where the ROOT is the mountpoint.
        #
        # So this does not try to divine intent. It refuses to remove most of an
        # index in one run without being asked, which is the outcome worth
        # blocking whatever produced it, and leaves ordinary single-file deletes
        # automatic. `prune=True` is the explicit yes; `prune=False` never
        # removes anything.
        held_back: list[str] = []
        if gone and prune is None:
            # PRE-EXISTING rows only. Counting `seen` whole let files this
            # run created vouch for the ones it was about to delete: eight new
            # documents elsewhere in the tree raised the denominator enough to
            # prune an entire vanished subtree silently, which is the exact
            # event this guard exists to stop.
            tracked_here = len(candidates) + len(seen & tracked_set)
            # A vanished DIRECTORY is held whatever the proportion. Otherwise
            # the guard erodes: a subtree held back today becomes a minority of
            # the index as the corpus grows and is pruned silently on some later
            # run, with no warning at all. `--prune` still clears it.
            lost_dirs = {os.path.dirname(fp) for fp in gone
                         if _disk_state(os.path.dirname(fp)) != "present"}
            if lost_dirs or len(gone) > max(_PRUNE_FLOOR, tracked_here // 2):
                held_back, gone = gone, []
                shown = ", ".join(held_back[:3]) + (
                    ", ..." if len(held_back) > 3 else "")
                why = ("their directory is missing too" if lost_dirs
                       else "that is most of this index")
                print(
                    f"poma-memory: {len(held_back)} of {tracked_here} indexed "
                    f"file(s) under {root_key} are missing ({shown}) and "
                    f"{why}, which looks more like a directory that did not "
                    "mount than a deletion, so nothing was removed. Re-run "
                    "with --prune to remove them.", file=sys.stderr)
        elif prune is False:
            held_back, gone = gone, []

        pruned: list[str] = []
        for fp in gone:
            # ...then re-checked here, because deleting the first row takes time and
            # the file may have come back before this one's turn.
            if _disk_state(fp) != "gone":
                continue
            store.delete_file_data(fp)
            pruned.append(fp)
            print(f"poma-memory: {fp} is no longer on disk; removed from the index",
                  file=sys.stderr)

        # Rows this run did not reach — a narrower glob, a name `index()` skips, a
        # file it could not read. Their metadata is whatever an earlier rule set
        # produced. Per-file hashes mean the next run that reaches them fixes it,
        # but nothing would otherwise say the rule set is only partly applied.
        #
        # Advisory, and it re-reads OTHER roots' rules files: rows in a shared
        # database point wherever they came from. A corrupt rules file over there
        # must not take down an indexing run over here, which has already done its
        # work and committed it — so this reports and continues rather than
        # raising past `store.close()`.
        stale: list[str] = []
        try:
            stale = stale_files(store.scanned_rows_rules())
        except MetadataIncomplete as e:
            print(f"poma-memory: could not check whether other indexed files are "
                  f"on the current rule set ({e})", file=sys.stderr)
        if stale:
            shown = ", ".join(stale[:3]) + (", ..." if len(stale) > 3 else "")
            # A row whose file is GONE cannot be reached by any glob, so telling
            # the user to widen one sends them nowhere. `--prune` is the only
            # thing that clears those.
            # ...and in a shared database they may not belong to this root at
            # all. A row whose FILE is gone can be reached by no glob and no
            # re-read, and `index --prune` will not touch a directory it cannot
            # see -- so the only command that clears it is `forget`, with the
            # database named, because the default one lives inside the
            # directory that is missing.
            vanished = [fp for fp in stale if _disk_state(fp) == "gone"]
            how = (f"Run `poma-memory forget <dir> --db {db_path}` for the "
                   "directory each one was indexed from."
                   if vanished else "Re-run with a glob that matches them.")
            print(f"poma-memory: {len(stale)} file(s) still hold metadata from "
                  f"an earlier rule set and were not reached by this run "
                  f"({shown}). " + how, file=sys.stderr)

        return {
            "files_indexed": files_indexed,
            "chunks_created": total_chunks,
            "chunksets_created": total_chunksets,
            "metadata_refreshed": refreshed,
            "unreadable": unreadable,
            "stale_rules": stale,
            "pruned": pruned,
            "prune_held_back": held_back,
        }
    finally:
        store.close()


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

    # `path` supplies the rules AND is recorded on the row as their source, so
    # a file outside it gets a root that cannot describe it: `path_meta` misses,
    # the row is stamped `{}` against that root's CURRENT hash, and it therefore
    # reads as up to date forever while being silently absent from every
    # filtered result and counted complete by `status`. The MCP tool reaches
    # this with its own defaults — `poma_index(file=X)` keeps `path=".agent/"`.
    file_real = os.path.realpath(file)
    root_real = os.path.realpath(path)
    try:
        inside = os.path.commonpath([file_real, root_real]) == root_real
    except ValueError:
        inside = False  # different drives on Windows
    if not inside:
        raise ValueError(
            f"{file_real} is not inside {root_real}, so that directory's "
            f"{RULES_FILENAME} cannot describe it. Pass `path` as the "
            "directory the file lives in."
        )

    store = Store(db_path)
    try:
        rules, _ = load_rules(path)
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
            path_metadata=path_meta.get(file_real),
            rules_hash=rules_hash(rules),
            rules_root=os.path.realpath(path),
        )
        return result
    finally:
        store.close()


def forget(path: str | Path, db_path: str | Path | None = None) -> dict:
    """Remove every indexed row under `path`, whether or not it still exists.

    The one thing `index --prune` deliberately cannot do, given its own name.
    Pruning asks "which of MY documents are gone?" and answers it only from a
    directory it can see; this asks "forget this directory", which is a
    statement about the directory rather than about its contents, and is the
    only honest answer to two states that otherwise have none:

    * **A deleted root in a shared database.** Two roots, one database
      (`--db`). Delete root B's directory and B's rows read stale forever --
      `load_rules` on a missing directory returns no rules, whose hash is not
      the one on the row -- so every filtered search over A refuses, including
      searches that have nothing to do with B.
    * **A renamed root.** `mv proj proj-renamed` is the same state reached by
      an ordinary command, and the database usually moves with the directory,
      so the rows point at a path that no longer exists while the index that
      holds them is the live one.

    Deliberately NOT part of `index`: a command that deletes rows for a
    directory nobody can inspect must be typed on purpose, not reached by the
    flag used for everyday deletions.

    Raises:
        FileNotFoundError: no database at `db_path` (or at the default inside
            `path`, which is the usual case once `path` itself is gone -- the
            database moved with the directory, and `--db` names where it is).
    """
    path = Path(path)
    if db_path is None:
        db_path = path / ".poma-memory.db"
    # Checked BEFORE `Store`, which creates the parent directory: the default
    # database sits inside `path`, so opening it for a directory the user
    # deleted recreated that directory, left an empty database in it, and made
    # the next `index` run see a present root. The remedy must not re-create
    # what the user removed.
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"no index at {db_path}. If the database is elsewhere -- which it "
            f"is whenever {path} moved or was deleted -- name it with `--db`."
        )

    root_key = os.path.realpath(path)
    store = Store(db_path)
    try:
        removed = []
        for fp in store.all_file_paths():
            try:
                under_root = os.path.commonpath([fp, root_key]) == root_key
            except ValueError:
                continue                    # different drives on Windows
            if under_root:
                store.delete_file_data(fp)
                removed.append(fp)
        return {"forgotten": sorted(removed), "root": root_key,
                "db_path": str(db_path)}
    finally:
        store.close()


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
        MetadataRulesError: a rules file a row points at cannot be read.
    """
    path = Path(path)
    if db_path is None:
        db_path = path / ".poma-memory.db"

    if not Path(db_path).exists():
        # `Store` creates the parent directory, so opening a database that is
        # not there recreates a root the user deleted and leaves an empty
        # database in it -- which then reads as a present root to the next
        # `index` run. Nothing to search is not a reason to write anything.
        # The daemon already answers this shape with an empty result.
        return []

    store = Store(db_path)
    hybrid = HybridSearch(store)
    try:
        results = hybrid.search(
            query, top_k=top_k, min_score=min_score, empty_gate=empty_gate,
            where=where,
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
                "unparsed_frontmatter": [], "stale_rules": []}

    store = Store(db_path)
    try:
        info = store.status()
        try:
            info["stale_rules"] = stale_files(store.scanned_rows_rules())
        except MetadataRulesError as e:
            # `status` reports; it does not fail. An unreadable rules file is
            # itself the thing worth showing.
            info["stale_rules"] = []
            info["rules_error"] = str(e)
        return info
    finally:
        store.close()


