"""Resident search daemon over a unix socket.

Why this exists
---------------
Every `poma-memory search` invocation pays for work that has nothing to do with
the query. Measured on a warm cache, one search costs ~0.52s end to end:

    python startup + imports (bm25s -> scipy.sparse)   ~0.14s
    model2vec load + reading every stored embedding    ~0.37s
    the actual hybrid search                           <0.01s

A search hook that fires on every Grep pays that per call, and an umbrella
session searching several `.agent` roots pays it once per root, sequentially.
Keeping the model and the embedding matrix in memory turns a 0.52s call into a
few milliseconds of IPC, and the saving scales with the number of roots.

Shape
-----
One daemon serves every index on the machine: the db path travels in the
request, so an umbrella session hitting four repos still talks to one warm
process. Requests are one JSON object per line, replies likewise:

    {"op": "search", "query": "...", "path": "/abs/repo/.agent", "top_k": 8}
    -> {"ok": true, "results": [...]}

Paths must be absolute — see `_resolve_db`.

Connections are handled on daemon threads; the SQLite work for a given index is
serialised behind that index's own lock. One client that connects and never
sends must not stall every search hook on the machine, and a cold index load on
one root (~0.4s) must not queue a warm search on another. A warm search measures
13-30ms, so per-index serialisation costs nothing worth reclaiming.

Freshness: `reindex-agent.sh` rewrites the db behind our back, so each cached
index is stamped with `PRAGMA data_version` and rebuilt when another connection
commits. File mtime cannot do this job — the store runs in WAL mode, so a commit
lands in the -wal file and the main db's stat never moves. A warm daemon serving
a stale vector index would be worse than a slow one.

The socket is user-only (0600) in a user-only directory. That matters more than
disclosure: a reply's text is injected into the model's context by the megavibe
search hook, so a socket another local user could hijack is a prompt-injection
channel, not just a read of your notes.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

from poma_memory.metadata import (
    MetadataIncomplete, MetadataNotIndexed, MetadataRulesError, MetadataStale,
    normalize_where,
)
from poma_memory.search import HybridSearch
from poma_memory.store import Store

DEFAULT_IDLE_TIMEOUT = 1800.0
MAX_CACHED_INDEXES = 8
MAX_REQUEST_BYTES = 1 << 20
CONN_TIMEOUT = 10.0
MAX_INFLIGHT = 64


def default_socket_path() -> Path:
    """Per-user socket path. XDG_RUNTIME_DIR when the platform provides one."""
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        return Path(runtime) / "poma-memory.sock"
    return Path.home() / ".poma-memory" / "serve.sock"


class _RelativePath(ValueError):
    """A client sent a relative path. The daemon must not guess whose cwd it is."""


def _resolve_db(path: str | None, db_path: str | None) -> Path:
    """Resolve the db to open. Paths must be absolute.

    The daemon's cwd is its own — wherever it happened to be started — and has
    nothing to do with the caller's. Resolving a relative path here silently
    answers for the wrong index, which is worse than any error: the caller gets
    a confident, empty, wrong result. Clients resolve before sending; the daemon
    refuses what it cannot resolve honestly.
    """
    raw = db_path or path or ""
    candidate = Path(raw).expanduser()
    if not raw or not candidate.is_absolute():
        raise _RelativePath(
            f"path must be absolute, got {raw!r} — the daemon cannot resolve it "
            "against the caller's working directory"
        )
    if db_path:
        return candidate
    return candidate / ".poma-memory.db"


def _lock_key(req: dict) -> str:
    """Which index this request serialises on: the RESOLVED database path.

    The same string `_IndexCache` keys its entries on, and it has to be. Keyed
    on the raw request instead, two spellings of one index took two different
    locks -- `{"path": "/r/.agent"}` and
    `{"db_path": "/r/.agent/.poma-memory.db"}` are the same database, and the
    CLI sends the second whenever `--db` is passed. Two threads then entered
    `_IndexCache.get` for one entry, and one closed the Store the other was
    reading. Measured on 320 mixed requests before the fix: 13 failures
    ("Cannot operate on a closed database", "bad parameter or other API
    misuse", "tuple index out of range") and 4 cache builds instead of 1.

    Anything with no resolvable database -- `ping`, `stats`, a relative path
    `_handle` will refuse anyway -- shares the empty key and touches no index.
    """
    try:
        return str(_resolve_db(req.get("path"), req.get("db_path")))
    except Exception:
        return ""


class _IndexCache:
    """db path -> warm HybridSearch, invalidated when the db file changes."""

    def __init__(self, limit: int = MAX_CACHED_INDEXES):
        self._limit = limit
        self._entries: dict[str, dict] = {}
        # `_entries` is touched by every worker thread. Held only around the
        # dict itself, never across a build: a cold build is ~0.4s, and holding
        # this through one would re-serialise every index behind the slowest,
        # which is what the per-index `_LockTable` exists to avoid.
        self._guard = threading.Lock()
        # Every rebuild is ~0.4s of model load and embedding reads. A daemon that
        # silently rebuilds on every request has lost its entire reason to exist
        # while still looking healthy, so the count is observable via `stats`.
        self.builds = 0

    @staticmethod
    def _stamp(store: Store | None, db: Path) -> tuple:
        """A value that changes whenever another process writes this database.

        `PRAGMA data_version` is the authority: SQLite bumps it on our connection
        when a *different* connection commits, which is exactly the reindex case.
        File mtime/size cannot do this job alone — the store runs in WAL mode, so
        a commit lands in the -wal file and the main db's stat does not move at
        all. A cache keyed on stat would have served a stale vector index after
        every reindex, silently. (Caught by test_cache_rebuilds_when_the_db_changes.)

        The -wal stat is carried too, as a belt for the moment before a
        checkpoint, and as the whole answer if the pragma is ever unavailable.
        """
        version = None
        if store is not None:
            try:
                # rollback() first: in autocommit this is a no-op, but if any read
                # transaction is still open the connection is pinned to an older
                # WAL snapshot and data_version would report the view we already
                # have instead of the one on disk. Cheap insurance against
                # serving a stale index.
                store._conn.rollback()
                version = store._conn.execute("PRAGMA data_version").fetchone()[0]
            except Exception:
                version = None

        if version is not None:
            # Authoritative and exact: SQLite bumps this for our connection when
            # ANOTHER connection commits, which is precisely the reindex case.
            # Nothing else belongs in the stamp — the -wal file's mtime and size
            # move on checkpoint and on our own writes with no content change,
            # so including it made the cache rebuild on almost every request.
            return ("data_version", version)

        def _stat(p: Path) -> tuple | None:
            try:
                st = p.stat()
                return (st.st_mtime_ns, st.st_size)
            except OSError:
                return None

        # No pragma (an unusable connection): fall back to file state, -wal
        # included, since in WAL mode the main db alone can sit still across a
        # commit. Coarser and prone to spurious rebuilds, but never stale.
        return ("stat", _stat(db), _stat(db.with_name(db.name + "-wal")))

    def get(self, db: Path) -> HybridSearch:
        key = str(db)
        with self._guard:
            entry = self._entries.get(key)

        if entry is not None:
            if self._stamp(entry["store"], db) == entry["stamp"]:
                entry["used"] = time.time()
                return entry["search"]
            # Another process wrote the index — drop the warm copy and rebuild.
            # Closing IS safe here, unlike in `_evict`: the caller holds this
            # database's lock, so no other thread can be inside a search on it.
            with self._guard:
                if self._entries.get(key) is entry:
                    del self._entries[key]
            try:
                entry["store"].close()
            except Exception:
                pass

        # Stamp before AND after building. An indexer that commits while
        # HybridSearch is reading would otherwise be recorded as "already
        # included" — the entry would carry the new data_version alongside an
        # index built from the old snapshot, and stay stale indefinitely.
        for _ in range(3):
            store = Store(db, check_same_thread=False)
            try:
                before = self._stamp(store, db)
                search = HybridSearch(store)
                after = self._stamp(store, db)
            except Exception:
                store.close()        # a corrupt db must not leak the handle
                raise
            if before == after:
                break
            store.close()
        else:
            # Three collisions means something is writing continuously; serve
            # this one uncached rather than spin.
            store = Store(db, check_same_thread=False)
            search = HybridSearch(store)
            after = self._stamp(store, db)

        with self._guard:
            self.builds += 1
            self._entries[key] = {
                "store": store,
                "search": search,
                "stamp": after,
                "used": time.time(),
            }
            self._evict()
        return search

    def _evict(self) -> None:
        """Drop the least recently used entries. Caller holds `_guard`.

        It DROPS the reference and does not close the `Store`, which is the
        whole point. Eviction crosses databases by construction — building an
        entry for db A is what evicts db B — so the per-index lock cannot make
        it safe, and closing here closed a connection another thread was
        running a query on. With `MAX_CACHED_INDEXES = 8`, nine databases
        queried concurrently was enough: "Cannot operate on a closed database",
        and on the next run a **SIGSEGV** (exit 139). Ten databases against a
        limit of twenty is clean, which is the tell.

        Dropping is sufficient because nothing else needs to happen. The thread
        mid-search still holds the `HybridSearch`, which holds the `Store`, so
        the connection stays alive exactly as long as it is in use and CPython
        closes it on the last reference — there is no cycle here, so that is
        immediate rather than at some later collection. The cost of an eviction
        that was still warm is one rebuild, which is what eviction means.
        """
        while len(self._entries) > self._limit:
            oldest = min(self._entries, key=lambda k: self._entries[k]["used"])
            self._entries.pop(oldest, None)

    def close_all(self) -> None:
        """Shutdown only, once no worker is still running — see `serve`.

        The join before this is bounded (a 5 s budget across ALL workers), so
        "the threads have been joined" was a guarantee the code did not
        provide: one in-flight search slower than the budget met a closed
        connection, which is the same close-under-a-live-reader class `_evict`
        was fixed for, and with a query in flight it is the same segfault. The
        caller now checks, and skips this when anything is still alive."""
        with self._guard:
            entries = list(self._entries.values())
            self._entries.clear()
        for entry in entries:
            try:
                entry["store"].close()
            except Exception:
                pass

    def stats(self) -> list[dict]:
        with self._guard:
            items = list(self._entries.items())
        return [
            {"db": k, "age_s": round(time.time() - v["used"], 1)}
            for k, v in items
        ]


def _handle(req: dict, cache: _IndexCache) -> dict:
    op = req.get("op", "search")

    if op == "ping":
        # The version lets a supervisor notice that an upgraded poma-memory is
        # still being served by a daemon started from the old build, and restart
        # it. Nothing in the request/response shape can detect that: the stale
        # daemon answers ok:true with old behaviour.
        from poma_memory import __version__
        return {"ok": True, "pid": os.getpid(), "version": __version__}

    if op == "stats":
        return {"ok": True, "pid": os.getpid(), "indexes": cache.stats(),
                "builds": cache.builds}

    if op == "shutdown":
        return {"ok": True, "shutdown": True}

    if op != "search":
        return {"ok": False, "error": f"unknown op: {op!r}"}

    query = req.get("query") or ""
    if not query.strip():
        return {"ok": False, "error": "empty query"}

    try:
        db = _resolve_db(req.get("path"), req.get("db_path"))
    except _RelativePath as e:
        return {"ok": False, "error": str(e)}
    if not db.exists():
        # Not an error: a root without an index is simply skipped by callers.
        return {"ok": True, "results": [], "db": str(db), "indexed": False}

    # `x or default` silently rewrites a valid 0: `--top 0` would come back as 5
    # and the daemon would disagree with the in-process path. Only None means
    # "not supplied".
    top_k = req.get("top_k")
    min_score = req.get("min_score")
    # Validate the predicate on its own, before any search work. Wrapping the
    # whole call in `except ValueError` labelled unrelated failures — a corrupt
    # `chunk_ids` blob raises json.JSONDecodeError, which IS a ValueError — as
    # a bad predicate, and the CLI then treats that as a real answer and stops
    # instead of falling through to the in-process path.
    try:
        where = normalize_where(req.get("where"))
    except ValueError as e:
        return {"ok": False, "code": "bad_where", "error": str(e)}
    search = cache.get(db)
    try:
        results = search.search(
            query,
            top_k=5 if top_k is None else int(top_k),
            min_score=0.0 if min_score is None else float(min_score),
            empty_gate=req.get("empty_gate"),
            where=where,
        )
    except MetadataIncomplete as e:
        # A machine-readable code, not just prose. The client has to tell this
        # apart from every other failure: falling back to the in-process path
        # would raise the same thing half a second and one model load later,
        # and matching on the message text is not something a client should be
        # asked to do. Additive — every other failure keeps the old shape.
        code = ("metadata_not_indexed" if isinstance(e, MetadataNotIndexed)
                else "metadata_stale" if isinstance(e, MetadataStale)
                else "bad_rules" if isinstance(e, MetadataRulesError)
                else "metadata_incomplete")
        resp = {"ok": False, "code": code, "error": str(e)}
        if isinstance(e, MetadataNotIndexed):
            # Kept for clients written against the original shape.
            resp["files_without_metadata"] = e.count
        return resp
    return {"ok": True, "results": results, "db": str(db), "indexed": True}


def request(payload: dict, socket_path: str | Path | None = None,
            timeout: float = 10.0) -> dict:
    """Send one request to a running daemon. Raises OSError if none is there."""
    sock_path = Path(socket_path) if socket_path else default_socket_path()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(str(sock_path))
        s.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        try:
            # Half-close so the server sees EOF. On a local socket it can read,
            # answer and close first, in which case this raises ENOTCONN — the
            # request was still delivered, and the reply is waiting in the
            # buffer, so this is not a failure.
            s.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        buf = b""
        while len(buf) < MAX_REQUEST_BYTES:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf.decode("utf-8") or "{}")


def _is_live(sock_path: Path) -> bool:
    try:
        request({"op": "ping"}, sock_path, timeout=2.0)
        return True
    except Exception:
        return False


class _LockTable:
    """A lock per database path, created on demand."""

    def __init__(self):
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def for_key(self, key: str) -> threading.Lock:
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = self._locks[key] = threading.Lock()
            return lock

    def all(self) -> list:
        with self._guard:
            return list(self._locks.values())


def _serve_conn(conn: socket.socket, cache: _IndexCache, lock: "_LockTable",
                stop: threading.Event, inflight: tuple) -> None:
    """Handle one connection: read outside the lock, work inside it.

    Threads exist to stop ONE slow client from stalling the machine. Reading a
    request can block for the whole connection timeout — a client that connects
    and never sends is enough — and doing that on the accept loop meant every
    search hook in every session waited behind it. The SQLite work itself stays
    serialised under `lock`, because connections are not thread-safe and a warm
    search is sub-millisecond anyway.
    """
    try:
        with conn:
            conn.settimeout(CONN_TIMEOUT)
            try:
                buf = b""
                oversize = False
                while b"\n" not in buf:
                    if len(buf) >= MAX_REQUEST_BYTES:
                        oversize = True
                        break
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                if oversize:
                    resp = {"ok": False, "error": "request too large"}
                else:
                    req = json.loads(buf.decode("utf-8").strip() or "{}")
                    # Cheap ops need no index lock at all; a search takes only
                    # the lock for the index it touches.
                    with lock.for_key(_lock_key(req)):
                        resp = _handle(req, cache)
            except Exception as e:
                # One bad request must never take the daemon down.
                resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            try:
                conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
            except Exception:
                pass
            if resp.get("shutdown"):
                stop.set()
    finally:
        with inflight[1]:
            inflight[0].discard(threading.current_thread())


def serve(socket_path: str | Path | None = None,
          idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
          quiet: bool = False) -> int:
    """Run the daemon. Returns a process exit code."""
    sock_path = Path(socket_path) if socket_path else default_socket_path()
    sock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # mkdir's mode is ignored when the directory already exists, so tighten it.
    # A group/other-writable parent lets another local user unlink our socket and
    # bind their own, and replies land in the model's context.
    try:
        os.chmod(sock_path.parent, 0o700)
    except OSError:
        pass

    if sock_path.exists():
        if _is_live(sock_path):
            if not quiet:
                print(f"poma-memory: already serving on {sock_path}", file=sys.stderr)
            return 3
        # Stale socket from a killed daemon — nothing is listening.
        try:
            sock_path.unlink()
        except OSError:
            pass

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old_umask = os.umask(0o177)          # socket readable by this user only
    try:
        srv.bind(str(sock_path))
    except OSError as e:
        os.umask(old_umask)
        print(f"poma-memory: cannot bind {sock_path}: {e}", file=sys.stderr)
        return 1
    os.umask(old_umask)
    srv.listen(64)
    # Poll rather than block forever on accept: a `shutdown` request is handled
    # on a worker thread, so the loop has to come up for air to notice it. The
    # idle timeout is tracked as a deadline instead of an accept timeout.
    srv.settimeout(1.0)

    cache = _IndexCache()
    # One lock PER INDEX, not one for the process: a cold load takes ~0.4s and
    # under a global lock every other session's warm search queued behind it.
    lock = _LockTable()
    stop = threading.Event()
    # (threads, lock): touched by the accept loop and by every worker as it
    # finishes, so it needs its own guard.
    inflight: tuple = (set(), threading.Lock())
    try:
        bound_ino = sock_path.stat().st_ino
    except OSError:
        bound_ino = None
    if not quiet:
        print(f"poma-memory: serving on {sock_path} (pid {os.getpid()})",
              file=sys.stderr)

    deadline = time.time() + idle_timeout
    try:
        while True:
            if stop.is_set():
                break
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                if idle_timeout > 0 and time.time() >= deadline:
                    if not quiet:
                        print("poma-memory: idle timeout, exiting", file=sys.stderr)
                    break
                continue
            except InterruptedError:
                continue
            except OSError as e:
                # EMFILE and friends: log and keep serving rather than die.
                if not quiet:
                    print(f"poma-memory: accept failed: {e}", file=sys.stderr)
                time.sleep(0.1)
                continue
            deadline = time.time() + idle_timeout

            with inflight[1]:
                busy = len(inflight[0]) >= MAX_INFLIGHT
            if busy:
                # Refuse rather than pile up threads. Clients fall back in-process.
                try:
                    conn.sendall(b'{"ok": false, "error": "server busy"}\n')
                    conn.close()
                except Exception:
                    pass
                continue

            t = threading.Thread(target=_serve_conn,
                                 args=(conn, cache, lock, stop, inflight),
                                 daemon=True)
            with inflight[1]:
                inflight[0].add(t)
            t.start()
            if stop.is_set():
                break
    except KeyboardInterrupt:
        pass
    finally:
        # Order matters. Stop accepting first, then let in-flight handlers finish
        # before closing the databases underneath them, then remove the socket
        # file — and only if it is still the one this process bound. A
        # replacement daemon may already have unlinked ours and bound its own at
        # the same path; unlinking blindly would delete the live socket and leave
        # the machine with a daemon nobody can reach.
        stop.set()
        srv.close()
        join_until = time.time() + 5.0
        with inflight[1]:
            workers = list(inflight[0])
        for t in workers:
            t.join(timeout=max(0.0, join_until - time.time()))
        for one in lock.all():
            one.acquire(timeout=2.0)
        # Only when nothing is still running. The join above is best-effort, and
        # closing a database out from under a live query is worse than leaving
        # the handles to the exiting process: the OS reclaims them either way.
        if all(not t.is_alive() for t in workers):
            cache.close_all()
        try:
            if bound_ino is None or sock_path.stat().st_ino == bound_ino:
                sock_path.unlink()
        except OSError:
            pass
    return 0
