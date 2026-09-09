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
    p_search.add_argument("--socket", default="auto",
                          help="Daemon socket: 'auto' (default), a path, or "
                               "'off' to force in-process search")
    p_search.add_argument("--json", action="store_true", dest="as_json",
                          help="Output as JSON")

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
    elif args.command == "status":
        _cmd_status(args)
    elif args.command == "mcp":
        _cmd_mcp()
    elif args.command == "serve":
        _cmd_serve(args)


def _cmd_index(args: argparse.Namespace) -> None:
    """Index command: index all markdown files in a directory."""
    from poma_memory.api import index
    from poma_memory.store import Store
    from poma_memory.incremental import update_file
    from pathlib import Path

    if args.file:
        # Single file mode
        path = Path(args.path)
        db_path = args.db or str(path / ".poma-memory.db")
        store = Store(db_path)
        result = update_file(store, args.file)
        store.close()
        print(f"{args.file}: {result['status']}"
              f" ({result.get('new_chunks', 0)} chunks,"
              f" {result.get('new_chunksets', 0)} chunksets)")
    else:
        result = index(path=args.path, db_path=args.db, glob=args.glob)
        print(f"Indexed {result['files_indexed']} files:"
              f" {result['chunks_created']} chunks,"
              f" {result['chunksets_created']} chunksets")


def _cmd_search(args: argparse.Namespace) -> None:
    """Search command: search indexed content."""
    import os

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
            }, sock)
            if resp.get("ok"):
                results = resp.get("results", [])
        except Exception:
            results = None

    if results is None:
        from poma_memory.api import search

        results = search(
            query=args.query,
            path=args.path,
            db_path=args.db,
            top_k=args.top,
            min_score=args.min_score,
            empty_gate=empty_gate,
        )

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
