"""CLI entry point: poma-memory index|search|status."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> None:
    """Entry point for poma-memory command."""
    parser = argparse.ArgumentParser(
        prog="poma-memory",
        description="Structure-preserving memory for AI agents.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

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
    p_search.add_argument("--json", action="store_true", dest="as_json",
                          help="Output as JSON")

    # status
    p_status = sub.add_parser("status", help="Show index status")
    p_status.add_argument("--path", default=".agent/")
    p_status.add_argument("--db", help="Database path")

    args = parser.parse_args(argv)

    if args.command == "index":
        _cmd_index(args)
    elif args.command == "search":
        _cmd_search(args)
    elif args.command == "status":
        _cmd_status(args)


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
    from poma_memory.api import search

    results = search(
        query=args.query,
        path=args.path,
        db_path=args.db,
        top_k=args.top,
    )

    if args.as_json:
        print(json.dumps(results, indent=2))
        return

    if not results:
        print("No results found.")
        return

    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {r['score']:.4f}) ---")
        print(f"File: {r['file_path']}")
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


if __name__ == "__main__":
    main()
