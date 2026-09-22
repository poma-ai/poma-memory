"""Per-file metadata: caller-supplied path rules, front-matter, and the predicate.

Two sources populate one opaque JSON blob per file:

1. **Path rules** -- a ``.poma-metadata.json`` at the root of the indexed
   directory, mapping globs to metadata dicts. This is the source that needs no
   producer change: a corpus that already encodes type structurally (one
   directory per kind, one file per kind) gets filterable metadata by naming
   what is already true about its layout.
2. **Front-matter** -- see ``frontmatter.py``. Wins per key over a path rule,
   because a document stating something about itself beats a rule stating it
   from outside.

poma-memory interprets none of it. ``kind``, ``event``, ``decision`` are the
caller's words; this module stores and compares them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat as stat_module
from pathlib import Path, PurePosixPath

RULES_FILENAME = ".poma-metadata.json"


class MetadataIncomplete(RuntimeError):
    """Base: a predicate cannot be answered honestly from what is recorded.

    Every surface that refuses a filtered search catches this rather than the
    concrete classes, so a new way for the recorded metadata to be untrustworthy
    cannot quietly start returning results instead of refusing.
    """


class MetadataNotIndexed(MetadataIncomplete):
    """A metadata predicate was used against an index that has none recorded.

    Raised instead of returning ``[]``, because an empty result from a corpus
    that was never scanned for metadata is indistinguishable from an honest "no
    document matches" -- the exact failure this feature exists to remove from
    the search path.
    """

    def __init__(self, count: int, db_path: str | os.PathLike | None = None,
                 examples: list[str] | None = None):
        self.count = count
        self.db_path = str(db_path) if db_path is not None else None
        self.examples = list(examples or [])
        where = f" in {self.db_path}" if self.db_path else ""
        # Name a few of them: the usual way to get stuck here is a file that no
        # `index` run's glob covers, and a bare count does not tell you which.
        shown = ", ".join(self.examples[:3])
        if shown and count > len(self.examples[:3]):
            shown += ", ..."
        detail = f" (e.g. {shown})" if shown else ""
        # A dot-prefixed name is not reachable by ANY glob `index` is given:
        # it skips those by name. Telling that user to "pass the glob that
        # matches it" sends them round a loop with no exit, so name the one
        # command that can heal the row.
        hidden = [p for p in self.examples if os.path.basename(p).startswith(".")]
        route = (
            " A file whose name starts with '.' is skipped by `index` unless it "
            f"is already indexed; run `poma-memory index --file {hidden[0]}` for "
            "it." if hidden else
            " If a file is not covered by the default glob, pass the glob that "
            "matches it."
        )
        super().__init__(
            f"{count} indexed file(s){where} have no metadata recorded{detail}, "
            "so a metadata filter cannot be answered honestly. Run `poma-memory "
            "index` to backfill (it re-reads metadata only; no re-chunking or "
            "re-embedding)." + route
        )


def _stuck_remedy(examples: list[str], roots: list[str],
                  db_path: str | None) -> str:
    """The sentence telling someone how to unstick a row, by WHY it is stuck.

    Shared by every refusal about a row that was scanned and can no longer be
    trusted, because every one of them has the same three cases and getting any
    of them wrong sends the user somewhere that silently does nothing. Naming
    `index` is useless when the run that would fix the row is the run that just
    skipped it, and a deleted file can be reached by no glob and no re-read.

    `os.stat` and `os.access`, never `open`: this runs inside an exception
    constructor on the search path, and opening a FIFO blocks forever waiting
    for a writer. Measured: >6 s and counting. A dead network mount does the
    same. `os.access` can disagree with real openability under ACLs or as root,
    which is acceptable in a diagnostic string where a hang is not.
    """
    gone, unreadable = [], []
    for path in examples:
        try:
            st = os.stat(path)
        except FileNotFoundError:
            gone.append(path)
            continue
        except OSError:
            unreadable.append(path)
            continue
        if not stat_module.S_ISREG(st.st_mode) or not os.access(path, os.R_OK):
            unreadable.append(path)

    roots = [r for r in roots if r]
    db = f" --db {db_path}" if db_path else ""
    if gone:
        # Print the command, whole. Two earlier cuts printed one that did not
        # work, and the second was worse than the first: "run `poma-memory
        # index`" sends you to the run that just skipped the row, and "run
        # `index <dir> --prune`" names a command that will not prune a
        # directory it cannot see AND whose default database lives inside that
        # directory -- so following it RECREATED the directory the user had
        # deleted, left an empty database in it, printed "Indexed 0 files", and
        # changed nothing.
        if len(roots) == 1:
            each = f"`poma-memory forget {roots[0]}{db}`"
        elif roots:
            each = (f"`poma-memory forget <dir>{db}` for each of "
                    + ", ".join(roots[:3]))
        else:
            each = f"`poma-memory forget <dir>{db}`"
        return (f" {len(gone)} of them no longer exist (e.g. {gone[0]}); no "
                "glob and no re-read can reach a deleted file, and "
                "`index --prune` will not remove rows for a directory it "
                f"cannot see. Run {each} to drop them.")
    if unreadable:
        return (f" {len(unreadable)} of them cannot be read "
                f"(e.g. {unreadable[0]}); fix the permissions, then re-run "
                "`poma-memory index` — the run cannot resolve a file it "
                "cannot open.")
    where = f" (e.g. {roots[0]})" if roots else ""
    return (f" Run `poma-memory index` over the directory each one came "
            f"from{where}; it re-reads metadata only, with no re-chunking and "
            "no re-embedding.")


class MetadataUnreadable(MetadataIncomplete):
    """A row was scanned, but what it holds is not a metadata object.

    A hand edit or a third-party writer, and it passes the completeness check
    because the column is not `''`. Skipping such a row dropped its document
    out of every filtered result silently -- an answer the caller cannot tell
    from a correct one, which is what this whole mechanism exists to refuse.

    It carries the same remedy as `MetadataStale` rather than a fixed "re-run
    index", because the row can be stuck for the same three reasons: an
    unrelated `index` run over a corpus where six other files are missing
    reports "6 missing, kept", heals nothing, and the refusal repeats verbatim.
    """

    def __init__(self, count: int, db_path: str | os.PathLike | None = None,
                 examples: list[str] | None = None,
                 roots: list[str] | None = None):
        self.count = count
        self.db_path = str(db_path) if db_path is not None else None
        self.examples = list(examples or [])
        self.roots = [r for r in (roots or []) if r]
        where = f" in {self.db_path}" if self.db_path else ""
        shown = ", ".join(self.examples[:3])
        if shown and count > len(self.examples[:3]):
            shown += ", ..."
        detail = f" (e.g. {shown})" if shown else ""
        super().__init__(
            f"{count} indexed file(s){where} have metadata recorded that is "
            f"not a JSON object{detail}, so a metadata filter cannot be "
            "answered honestly."
            + _stuck_remedy(self.examples, self.roots, self.db_path)
        )


class MetadataStale(MetadataIncomplete):
    """Rows hold metadata resolved against a rule set that is not the current one.

    The row is not ``''`` -- it was scanned -- so nothing else on the search
    path notices. But a rule edit changes what a file *means* without touching
    the file, so answering from the old resolution is a confident wrong answer
    in both directions: the predicate that used to match still does, and the one
    that should now match does not. Refusing is the same discipline
    `MetadataNotIndexed` applies to a row that was never scanned at all.
    """

    def __init__(self, count: int, db_path: str | os.PathLike | None = None,
                 examples: list[str] | None = None,
                 roots: list[str] | None = None):
        self.count = count
        self.db_path = str(db_path) if db_path is not None else None
        self.examples = list(examples or [])
        # The directories the stale rows were indexed FROM. Without them the
        # remedy can only name a flag, and the user runs it in the directory
        # they are already in, where the scope gate makes it a no-op.
        self.roots = [r for r in (roots or []) if r]
        where = f" in {self.db_path}" if self.db_path else ""
        shown = ", ".join(self.examples[:3])
        if shown and count > len(self.examples[:3]):
            shown += ", ..."
        detail = f" (e.g. {shown})" if shown else ""
        super().__init__(
            f"{count} indexed file(s){where} still hold metadata resolved "
            f"against an earlier rule set{detail}, so a metadata filter would "
            "answer from rules that are no longer in effect."
            + _stuck_remedy(self.examples, self.roots, self.db_path)
        )


class MetadataRulesError(MetadataIncomplete, ValueError):
    """`.poma-metadata.json` exists but cannot be used.

    A `MetadataIncomplete` as well as a `ValueError`: if the rules cannot be
    read then a predicate cannot be answered honestly, which is the same
    contract the other two carry. It reaches the search path now that staleness
    is checked per query, and every surface already catches the base — leaving
    it outside meant a typo in the rules file surfaced as a raw traceback.
    """


def rules_path(root: str | Path) -> Path:
    return Path(root) / RULES_FILENAME


def load_rules(root: str | Path) -> tuple[list[dict], str]:
    """Load and validate the rules file. Returns (rules, raw_text).

    A missing file is not an error -- it means "no path rules", and every file
    falls back to front-matter alone. A malformed one IS an error naming the
    file: silently ignoring it would resolve every document to ``{}`` and make
    every filtered search return nothing, for a reason nobody can see.
    """
    p = rules_path(root)
    if not p.exists():
        return [], ""
    # utf-8-sig: an editor-written BOM is not a syntax error the user can see,
    # and `json.loads` rejects it. The front-matter parser already strips one.
    #
    # The READ needs the same guard the parse below has. Left bare it raised
    # PermissionError or UnicodeDecodeError -- neither a `MetadataIncomplete`,
    # so no surface caught it -- and staleness is checked per query now, so a
    # rules file saved as UTF-16 by a Windows shell tracebacked out of `status`,
    # `search` AND `index`, which is the remedy the message recommends.
    try:
        raw = p.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as e:
        raise MetadataRulesError(f"{p}: cannot be read ({e})") from e
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as e:
        raise MetadataRulesError(f"{p}: invalid JSON ({e})") from e
    if not isinstance(doc, dict) or not isinstance(doc.get("rules"), list):
        raise MetadataRulesError(f"{p}: expected an object with a 'rules' list")

    for i, rule in enumerate(doc["rules"]):
        if not isinstance(rule, dict):
            raise MetadataRulesError(f"{p}: rule {i} is not an object")
        if not isinstance(rule.get("glob"), str) or not rule["glob"]:
            raise MetadataRulesError(f"{p}: rule {i} needs a non-empty 'glob' string")
        # `Path.glob` raises NotImplementedError on an absolute pattern, which
        # names nothing; and '..' would attach metadata to files outside the
        # directory being indexed.
        # A backslash and a drive letter are how the posix-only checks below
        # get bypassed: PurePosixPath does not split on a backslash and does
        # not see `C:` as absolute, so `..\..\etc\*.md` reads as one harmless
        # component here and escapes the root on Windows. A bare `:` elsewhere
        # is a legal POSIX filename character and is left alone — a directory
        # really can be called `notes:2026`.
        if "\\" in rule["glob"]:
            raise MetadataRulesError(
                f"{p}: rule {i} glob {rule['glob']!r} must use '/' separators")
        if re.match(r"^[A-Za-z]:", rule["glob"]):
            raise MetadataRulesError(
                f"{p}: rule {i} glob {rule['glob']!r} must be relative to the "
                "indexed directory (no drive letter)")
        if PurePosixPath(rule["glob"]).is_absolute() or rule["glob"].startswith("/"):
            raise MetadataRulesError(
                f"{p}: rule {i} glob {rule['glob']!r} must be relative to the "
                "indexed directory")
        if ".." in PurePosixPath(rule["glob"]).parts:
            raise MetadataRulesError(
                f"{p}: rule {i} glob {rule['glob']!r} escapes the indexed "
                "directory")
        meta = rule.get("metadata")
        if not isinstance(meta, dict) or not meta:
            raise MetadataRulesError(f"{p}: rule {i} needs a non-empty 'metadata' object")
        for k, v in meta.items():
            ok = isinstance(v, str) or (
                isinstance(v, list) and all(isinstance(x, str) for x in v)
            )
            if not ok:
                raise MetadataRulesError(
                    f"{p}: rule {i} key {k!r} must be a string or a list of strings"
                )
    return doc["rules"], raw


def rules_hash(rules: list[dict]) -> str:
    """Identity of the rule set, stored so a rule edit is detectable.

    Editing the rules file touches no indexed document, and `update_file`
    short-circuits on mtime equality -- so without this every row would keep
    metadata resolved from the superseded rules, silently.

    Over the PARSED rules in canonical form, not the raw bytes. Hashing bytes
    made a reformat with no change of meaning -- a trailing newline, CRLF from a
    checkout, a formatter's re-indent -- refuse every filtered search until the
    corpus was re-indexed. Detection is unchanged: any rule that differs differs
    here too.
    """
    return hashlib.sha256(
        json.dumps(rules, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def stale_by_root(rows: list[tuple[str, str, str]]) -> dict[str, list[str]]:
    """`stale_files`, grouped by the directory whose rules produced each row.

    The remedy for a stale row depends on WHERE it came from, so the caller
    needs the root and not only the path: a row from a directory that no longer
    exists is cleared by `forget <that directory> --db <this database>`, and
    naming either half without the other sends the user somewhere that does
    nothing. A row with no recorded root groups under ''.
    """
    by_root: dict[str, list[tuple[str, str]]] = {}
    for file_path, root, stored in rows:
        by_root.setdefault(root, []).append((file_path, stored))

    out: dict[str, list[str]] = {}
    for root, items in by_root.items():
        if not root:
            out.setdefault("", []).extend(fp for fp, _ in items)
            continue
        current = rules_hash(load_rules(root)[0])
        bad = sorted(fp for fp, stored in items if stored != current)
        if bad:
            out[root] = bad
    return {k: v for k, v in out.items() if v}


def stale_files(rows: list[tuple[str, str, str]]) -> list[str]:
    """Which scanned rows carry a rule set that is no longer current.

    `rows` is (file_path, rules_root, rules_hash) from `store.scanned_rows_rules`.
    Each row names the directory whose rules produced it, so the current rules
    are re-read from THAT directory and compared. Nothing is inferred from where
    the database happens to live.

    That inference is what the previous cut got wrong twice at once. It skipped
    the check whenever the database was not beside the rules file, which left
    `--db` callers with no check at all -- and the row-versus-row fallback it
    used instead is blind exactly when every row is stale together, which is the
    normal shape of a rule edit. It also refused forever when two roots shared
    one database, since neither run could restamp the other's rows.

    A row with no recorded root predates this and cannot be verified, so it
    counts as stale: one re-index settles it, where trusting it would be a guess.
    """
    by_root: dict[str, list[tuple[str, str]]] = {}
    for file_path, root, stored in rows:
        by_root.setdefault(root, []).append((file_path, stored))

    stale: list[str] = []
    for root, items in by_root.items():
        if not root:
            stale.extend(fp for fp, _ in items)
            continue
        current = rules_hash(load_rules(root)[0])
        stale.extend(fp for fp, stored in items if stored != current)
    return sorted(stale)


def resolve_paths(root: str | Path, rules: list[dict]) -> dict[str, dict]:
    """Map each rule-matched file to its metadata. First matching rule wins.

    Globs are expanded with ``Path.glob`` -- the same engine `index()` uses for
    its own ``--glob`` -- so ``**`` means here exactly what it means there. The
    alternative, matching each path against a translated pattern, is a second
    glob implementation that can disagree with the first.
    """
    root = Path(root)
    resolved: dict[str, dict] = {}
    for rule in rules:
        for match in root.glob(rule["glob"]):
            if not match.is_file():
                continue
            key = os.path.realpath(match)
            if key not in resolved:  # first rule wins
                resolved[key] = dict(rule["metadata"])
    return resolved


def merge(path_meta: dict | None, fm_meta: dict | None) -> dict:
    """Combine the two sources. Front-matter wins per key."""
    out = dict(path_meta or {})
    out.update(fm_meta or {})
    return out


def matches(meta: dict, where: dict) -> bool:
    """AND across keys, OR within a list value, case-sensitive string equality.

    A list-valued field matches on non-empty intersection. A key the document
    does not carry never matches -- the predicate cannot express absence, by
    design; producers emit the key explicitly and callers filter positively.
    """
    for key, want in where.items():
        allowed = {want} if isinstance(want, str) else set(want)
        have = meta.get(key)
        if have is None:
            return False
        have_set = {have} if isinstance(have, str) else set(have)
        if not (have_set & allowed):
            return False
    return True


def normalize_where(where: dict | None) -> dict | None:
    """Validate a predicate and reject shapes the grammar does not cover."""
    if where is None:
        return None
    if not isinstance(where, dict):
        # Not reachable from `--where`, which always builds a dict, but very
        # reachable from the Python API and from any other socket client.
        # Left as AttributeError it escapes the daemon's `code:` contract and
        # the CLI's except clause, and the user gets a traceback.
        #
        # The type check has to come BEFORE the emptiness one. Testing
        # falsiness first meant `[]`, `""` and `0` were read as "no predicate"
        # and answered with the ENTIRE corpus, ok:true -- a whole-corpus answer
        # the caller cannot tell apart from a correct filtered one, which is the
        # failure this feature exists to remove. `[1, 2]` raised; `[]` did not.
        raise ValueError(f"where: expected a dict, got {type(where).__name__}")
    if not where:
        # An empty dict is a real predicate that constrains nothing, and asking
        # for no constraint is not an error.
        return None
    out = {}
    for key, want in where.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"where: key must be a non-empty string, got {key!r}")
        if isinstance(want, str):
            out[key] = want
        elif isinstance(want, (list, tuple)) and want and all(
            isinstance(x, str) for x in want
        ):
            out[key] = list(want)
        else:
            raise ValueError(
                f"where[{key!r}]: expected a string or a non-empty list of "
                f"strings, got {want!r}"
            )
    return out
