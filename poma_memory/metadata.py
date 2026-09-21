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
from pathlib import Path, PurePosixPath

RULES_FILENAME = ".poma-metadata.json"


class MetadataNotIndexed(RuntimeError):
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
        super().__init__(
            f"{count} indexed file(s){where} have no metadata recorded{detail}, "
            "so a metadata filter cannot be answered honestly. Run `poma-memory "
            "index` to backfill (it re-reads metadata only; no re-chunking or "
            "re-embedding); if a file is not covered by the default glob, pass "
            "the glob that matches it."
        )


class MetadataRulesError(ValueError):
    """`.poma-metadata.json` exists but cannot be used."""


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
    raw = p.read_text(encoding="utf-8")
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


def rules_hash(raw: str) -> str:
    """Identity of the rule set, stored so a rule edit is detectable.

    Editing the rules file touches no indexed document, and `update_file`
    short-circuits on mtime equality -- so without this every row would keep
    metadata resolved from the superseded rules, silently.
    """
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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
    if not where:
        return None
    if not isinstance(where, dict):
        # Not reachable from `--where`, which always builds a dict, but very
        # reachable from the Python API and from any other socket client.
        # Left as AttributeError it escapes the daemon's `code:` contract and
        # the CLI's except clause, and the user gets a traceback.
        raise ValueError(f"where: expected a dict, got {type(where).__name__}")
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
