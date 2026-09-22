"""CLI entry point: poma-memory index|search|status|mcp."""

from __future__ import annotations

import argparse
import json
import sys
import warnings

warnings.filterwarnings("ignore", message="A NumPy version", category=UserWarning)


def main(argv: list[str] | None = None) -> None:
    """Entry point for poma-memory command."""
    parser = argparse.ArgumentParser(
        prog="poma-memory",
        description="Structure-preserving memory for AI agents.",
    )
    sub = parser.add_subparsers(dest="command")

    # index
    p_index = sub.add_parser("index", help="Index markdown files")
    p_index.add_argument("path", nargs="?", default=".agent/",
                         help="Directory to index (default: .agent/)")
    p_index.add_argument("--file", help="Index a single file")
    p_index.add_argument("--db", help="Database path (default: {path}/.poma-memory.db)")
    p_index.add_argument("--glob", default="**/*.md", help="File pattern")
    p_prune = p_index.add_mutually_exclusive_group()
    p_prune.add_argument("--prune", dest="prune", action="store_true", default=None,
                         help="Remove indexed files that are gone from disk, "
                              "even when that is most of the index. Never rows "
                              "outside this directory, and nothing at all when "
                              "the directory itself is absent (see `forget`)")
    p_prune.add_argument("--no-prune", dest="prune", action="store_false",
                         help="Never remove indexed files that are gone")

    # search
    p_search = sub.add_parser("search", help="Search indexed content")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--path", default=".agent/",
                          help="Directory that was indexed")
    p_search.add_argument("--db", help="Database path")
    p_search.add_argument("--top", "--top-k", type=int, default=5, help="Number of results")
    p_search.add_argument("--min-score", type=float, default=0.0, dest="min_score",
                          help="Drop results below this fused score (0.0 = no floor)")
    p_search.add_argument("--empty-gate", type=float, default=None, dest="empty_gate",
                          help="Suppress ALL results when the best semantic hit's cosine "
                               "is below this (default: embedder-calibrated; 0 disables)")
    p_search.add_argument("--where", action="append", default=None, metavar="KEY=VALUE",
                          help="Metadata filter, repeatable. Repeats of one key are "
                               "OR'd, different keys are AND'd. Requires an index "
                               "built with metadata (.poma-metadata.json).")
    p_search.add_argument("--socket", default="auto",
                          help="Daemon socket: 'auto' (default), a path, or "
                               "'off' to force in-process search")
    p_search.add_argument("--json", action="store_true", dest="as_json",
                          help="Output as JSON")

    # forget
    p_forget = sub.add_parser(
        "forget",
        help="Remove every indexed row under a directory, gone or not")
    p_forget.add_argument("path", help="Directory whose rows to remove")
    p_forget.add_argument("--db", help="Database path. Required once the "
                                       "directory itself is gone, since the "
                                       "default one lives inside it")

    # status
    p_status = sub.add_parser("status", help="Show index status")
    p_status.add_argument("--path", default=".agent/")
    p_status.add_argument("--db", help="Database path")

    # mcp
    sub.add_parser("mcp", help="Start MCP server (requires: pip install poma-memory[mcp])")

    # serve
    p_serve = sub.add_parser(
        "serve", help="Run a resident search daemon on a unix socket")
    p_serve.add_argument("--socket", help="Socket path (default: per-user)")
    p_serve.add_argument("--idle-timeout", type=float, default=1800.0,
                         help="Exit after this many idle seconds (0 = never)")
    p_serve.add_argument("--quiet", action="store_true")

    parser.add_argument("--version", action="store_true",
                        help="Print the installed version and exit")

    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        from poma_memory import __version__
        print(__version__)
        return

    if args.command == "index":
        _cmd_index(args)
    elif args.command == "search":
        _cmd_search(args)
    elif args.command == "forget":
        _cmd_forget(args)
    elif args.command == "status":
        _cmd_status(args)
    elif args.command == "mcp":
        _cmd_mcp()
    elif args.command == "serve":
        _cmd_serve(args)


def _cmd_index(args: argparse.Namespace) -> None:
    """Index command: index all markdown files in a directory."""
    from poma_memory.api import index, index_file

    from poma_memory.metadata import MetadataRulesError

    if args.file:
        # Single file mode. Goes through index_file so the directory's path
        # rules still apply — resolving one file without them would record
        # "scanned, no metadata" where a rule says otherwise.
        try:
            result = index_file(args.file, path=args.path, db_path=args.db)
        except (OSError, ValueError, MetadataRulesError) as e:
            # OSError as well as ValueError. `index()` already treats an
            # unreadable or missing document as costing that file and not the
            # run; here the same file tracebacked out of `main` instead --
            # `poma-memory index --file nope.md` printed a FileNotFoundError
            # stack, while the same file with one Latin-1 byte printed a clean
            # message, because UnicodeDecodeError happens to be a ValueError.
            print(f"poma-memory: {args.file}: "
                  f"{getattr(e, 'strerror', None) or e}", file=sys.stderr)
            raise SystemExit(2)
        print(f"{args.file}: {result['status']}"
              f" ({result.get('new_chunks', 0)} chunks,"
              f" {result.get('new_chunksets', 0)} chunksets)")
    else:
        try:
            result = index(path=args.path, db_path=args.db, glob=args.glob,
                           prune=getattr(args, "prune", None))
        except MetadataRulesError as e:
            print(f"poma-memory: {e}", file=sys.stderr)
            raise SystemExit(2)
        summary = (f"Indexed {result['files_indexed']} files:"
                   f" {result['chunks_created']} chunks,"
                   f" {result['chunksets_created']} chunksets")
        # Pruning is the one thing this command does that destroys data. On
        # stderr alone it is invisible to anything reading stdout.
        if result.get("pruned"):
            summary += (f" ({len(result['pruned'])} removed:"
                        " no longer on disk)")
        if result.get("prune_held_back"):
            summary += (f" ({len(result['prune_held_back'])} missing, kept"
                        " — see above)")
        print(summary)


def _parse_where(pairs: list[str] | None) -> dict | None:
    """Turn repeated `--where key=value` into the predicate dict.

    Repeats of one key become a list (OR); distinct keys stay separate (AND).
    A flat encoding on purpose: anything richer on the command line would be a
    filter DSL, and the predicate deliberately is not one.
    """
    if not pairs:
        return None
    out: dict[str, list[str]] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        key = key.strip()
        if not sep or not key:
            raise SystemExit(f"--where expects KEY=VALUE, got {pair!r}")
        out.setdefault(key, []).append(value)
    # One value stays a scalar so the wire form matches what the API documents.
    return {k: (v[0] if len(v) == 1 else v) for k, v in out.items()}


# Daemon codes that are a real answer about the index, not a daemon fault.
# Falling through to the in-process path on one of these reaches the same
# refusal a model load later — and `bad_rules` used to be missing here, so the
# fallback ran and the user got a raw traceback instead of the daemon's clean
# message.
_REFUSAL_CODES = frozenset({"bad_rules"})


def _cmd_search(args: argparse.Namespace) -> None:
    """Search command: search indexed content."""
    import os

    from poma_memory.metadata import MetadataIncomplete, normalize_where

    where = _parse_where(getattr(args, "where", None))
    try:
        normalize_where(where)
    except ValueError as e:
        print(f"poma-memory: {e}", file=sys.stderr)
        raise SystemExit(2)

    # The env overrides documented on `search` (POMA_MEMORY_EMPTY_GATE selects the
    # relevance gate, POMA_EMBEDDER selects the embedder) are read inside the
    # process that runs the search. Under the daemon that is a long-lived tmux
    # process whose environment was frozen days ago — so the same command would
    # silently mean something different depending on whether a daemon happened to
    # be up. Resolve the gate here and send it explicitly; refuse the daemon
    # entirely when the embedder is overridden, since that changes which vectors
    # a query is compared against and cannot be passed in a request.
    empty_gate = args.empty_gate
    if empty_gate is None:
        env_gate = os.environ.get("POMA_MEMORY_EMPTY_GATE", "").strip()
        if env_gate:
            try:
                empty_gate = float(env_gate)
            except ValueError:
                empty_gate = None

    embedder_override = bool(os.environ.get("POMA_EMBEDDER", "").strip())

    results = None
    if getattr(args, "socket", None) != "off" and not embedder_override:
        # Try the resident daemon first: it holds the model and the embedding
        # matrix, which is ~0.37s of the ~0.52s a cold search costs. Any failure
        # (no daemon, stale socket, daemon mid-restart) falls through to the
        # in-process path, so behaviour never depends on the daemon being up.
        try:
            from poma_memory.server import request

            sock = None if args.socket in (None, "auto") else args.socket
            # Resolve here, not in the daemon: its cwd is not ours.
            from pathlib import Path as _P
            _path = str(_P(args.path).expanduser().resolve()) if args.path else None
            _db = str(_P(args.db).expanduser().resolve()) if args.db else None
            resp = request({
                "op": "search",
                "query": args.query,
                "path": _path,
                "db_path": _db,
                "top_k": args.top,
                "min_score": args.min_score,
                "empty_gate": empty_gate,
                "where": where,
            }, sock)
            if resp.get("ok"):
                results = resp.get("results", [])
            elif str(resp.get("code", "")) in _REFUSAL_CODES or str(
                    resp.get("code", "")).startswith("metadata_"):
                # A real answer, not a daemon problem. Falling through to the
                # in-process path would reach the same refusal ~0.5s and one
                # model load later.
                print(f"poma-memory: {resp.get('error')}", file=sys.stderr)
                raise SystemExit(2)
        except SystemExit:
            raise
        except Exception:
            results = None

    if results is None:
        from poma_memory.api import search

        try:
            results = search(
                query=args.query,
                path=args.path,
                db_path=args.db,
                top_k=args.top,
                min_score=args.min_score,
                empty_gate=empty_gate,
                where=where,
            )
        except MetadataIncomplete as e:
            # Only this one. A bare `ValueError` here would swallow, say, a
            # corrupt chunk_ids blob and report it as a metadata problem.
            print(f"poma-memory: {e}", file=sys.stderr)
            raise SystemExit(2)

    if args.as_json:
        print(json.dumps(results, indent=2))
        return

    if not results:
        print("No results found.")
        return

    from poma_memory.api import format_updated
    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {r['score']:.4f}) ---")
        print(f"File: {r['file_path']}")
        updated = format_updated(r.get("upserted_at"))
        if updated:
            print(f"Updated: {updated}")
        print(r["context"])


def _cmd_forget(args: argparse.Namespace) -> None:
    """Forget command: drop every row under a directory."""
    from poma_memory.api import forget

    try:
        result = forget(args.path, db_path=args.db)
    except OSError as e:
        print(f"poma-memory: {e}", file=sys.stderr)
        raise SystemExit(2)
    n = len(result["forgotten"])
    if not n:
        print(f"Nothing indexed under {result['root']} in {result['db_path']}.")
        return
    print(f"Forgot {n} file(s) under {result['root']}.")


def _cmd_status(args: argparse.Namespace) -> None:
    """Status command: show index status."""
    from poma_memory.api import status

    info = status(path=args.path, db_path=args.db)

    if not info["files"]:
        print("No indexed files. Run: poma-memory index")
        return

    print(f"Files:     {len(info['files'])}")
    print(f"Chunks:    {info['total_chunks']}")
    print(f"Chunksets: {info['total_chunksets']}")
    print(f"Semantic:  {'yes' if info['has_embeddings'] else 'no'}")
    # Both ways the index can be behind, because this is the surface a user
    # checks when a filtered search refuses. Reporting only the first said
    # "complete" while every filtered search exited 2.
    missing = info.get("files_without_metadata", 0)
    stale = info.get("stale_rules", [])
    if info.get("rules_error"):
        print(f"Metadata:  rules file unusable - {info['rules_error']}")
    elif missing:
        print(f"Metadata:  {missing} file(s) unscanned - run `poma-memory index`")
    elif stale:
        print(f"Metadata:  {len(stale)} file(s) on an earlier rule set - "
              "run `poma-memory index`")
    else:
        print("Metadata:  complete")
    for f in stale[:3]:
        print(f"  ! earlier rule set: {f}")
    for f in info.get("unparsed_frontmatter", []):
        print(f"  ! unparsed front-matter: {f}")
    for f in info["files"]:
        print(f"  - {f}")


def _cmd_serve(args: argparse.Namespace) -> None:
    """Serve command: run the resident search daemon."""
    import sys

    from poma_memory.server import serve

    sock = args.socket
    if sock:
        from pathlib import Path as _P
        sock = str(_P(sock).expanduser().resolve())
    sys.exit(serve(socket_path=sock, idle_timeout=args.idle_timeout,
                   quiet=args.quiet))


def _cmd_mcp() -> None:
    """Start the MCP server (stdio transport)."""
    try:
        from poma_memory.mcp_server import main as mcp_main
    except ImportError:
        print(
            "MCP dependencies not installed.\n"
            "Run: pip install poma-memory[mcp]",
            file=sys.stderr,
        )
        sys.exit(1)
    mcp_main()


if __name__ == "__main__":
    main()
