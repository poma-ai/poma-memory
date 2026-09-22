"""Tests for the resident search daemon (poma_memory.server)."""

import json
import os
import socket
import tempfile
import threading
import time
from pathlib import Path

import pytest

from poma_memory.incremental import update_file
from poma_memory.server import _IndexCache, _resolve_db, _RelativePath, request, serve
from poma_memory.store import Store

SAMPLE_MD = """\
# Deploy Notes

## Rollback

**Decision:** roll back with the previous image tag, never by reverting main.

- Keep the last three tags on the registry.
- A rollback that needs a rebuild is not a rollback.
"""


@pytest.fixture
def indexed_dir():
    """A directory with one indexed markdown file and its .poma-memory.db."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        md = d / "notes.md"
        md.write_text(SAMPLE_MD, encoding="utf-8")
        store = Store(d / ".poma-memory.db")
        update_file(store, str(md))
        store.close()
        yield d


@pytest.fixture
def running_daemon(indexed_dir):
    """Serve on a temp socket in a background thread; shut it down after."""
    sock = Path(tempfile.mkdtemp()) / "t.sock"
    t = threading.Thread(
        target=serve,
        kwargs={"socket_path": sock, "idle_timeout": 30.0, "quiet": True},
        daemon=True,
    )
    t.start()
    # Wait for a successful ping, not for the socket file: the path appears at
    # bind() and connections are only accepted after listen(), so a stat-based
    # wait can connect into that gap and get ECONNREFUSED. (In production a
    # client that loses this race just falls back to in-process search.)
    for _ in range(200):
        try:
            if request({"op": "ping"}, sock, timeout=2.0)["ok"]:
                break
        except (OSError, ValueError):
            time.sleep(0.05)
    else:
        pytest.fail("daemon never came up")
    yield sock
    try:
        request({"op": "shutdown"}, sock)
    except OSError:
        pass
    t.join(timeout=5)


def test_ping(running_daemon):
    resp = request({"op": "ping"}, running_daemon)
    assert resp["ok"] is True
    assert resp["pid"] == os.getpid()


def test_search_returns_results(running_daemon, indexed_dir):
    resp = request(
        {"op": "search", "query": "rollback", "path": str(indexed_dir), "top_k": 5},
        running_daemon,
    )
    assert resp["ok"] is True
    assert resp["indexed"] is True
    assert resp["results"], "expected at least one hit for an indexed term"
    assert "file_path" in resp["results"][0]


def test_relative_path_is_refused_not_guessed(running_daemon):
    """The daemon's cwd is not the caller's; guessing would answer wrongly."""
    resp = request({"op": "search", "query": "rollback", "path": ".agent/"},
                   running_daemon)
    assert resp["ok"] is False
    assert "absolute" in resp["error"]


def test_missing_index_is_empty_not_an_error(running_daemon):
    with tempfile.TemporaryDirectory() as empty:
        resp = request({"op": "search", "query": "rollback", "path": empty},
                       running_daemon)
    assert resp["ok"] is True
    assert resp["indexed"] is False
    assert resp["results"] == []


def test_empty_query_is_an_error(running_daemon, indexed_dir):
    resp = request({"op": "search", "query": "   ", "path": str(indexed_dir)},
                   running_daemon)
    assert resp["ok"] is False


def test_unknown_op_does_not_kill_the_daemon(running_daemon):
    resp = request({"op": "nonsense"}, running_daemon)
    assert resp["ok"] is False
    assert request({"op": "ping"}, running_daemon)["ok"] is True


def test_malformed_request_does_not_kill_the_daemon(running_daemon):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(10)
        s.connect(str(running_daemon))
        s.sendall(b"{not json\n")
        s.shutdown(socket.SHUT_WR)
        body = s.recv(65536)
    assert json.loads(body)["ok"] is False
    assert request({"op": "ping"}, running_daemon)["ok"] is True


def test_socket_is_user_only(running_daemon):
    assert (running_daemon.stat().st_mode & 0o077) == 0, \
        "socket must not be readable by other users"


def test_cache_rebuilds_when_the_db_changes(indexed_dir):
    """A reindex writes the db; a warm daemon must not serve the stale index."""
    db = indexed_dir / ".poma-memory.db"
    cache = _IndexCache()
    first = cache.get(db)
    assert cache.get(db) is first, "unchanged db should reuse the warm index"

    time.sleep(0.01)
    store = Store(db)
    extra = indexed_dir / "more.md"
    extra.write_text("# Extra\n\nA second note about deploys.\n", encoding="utf-8")
    update_file(store, str(extra))
    store.close()

    assert cache.get(db) is not first, "changed db must invalidate the warm index"
    cache.close_all()


def test_cache_evicts_beyond_its_limit(indexed_dir):
    cache = _IndexCache(limit=1)
    cache.get(indexed_dir / ".poma-memory.db")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "n.md").write_text(SAMPLE_MD, encoding="utf-8")
        store = Store(d / ".poma-memory.db")
        update_file(store, str(d / "n.md"))
        store.close()
        cache.get(d / ".poma-memory.db")
        assert len(cache.stats()) == 1
    cache.close_all()


def test_stale_socket_is_replaced(indexed_dir):
    """A killed daemon leaves the socket file behind; the next one must bind."""
    sock = Path(tempfile.mkdtemp()) / "stale.sock"
    sock.write_text("")                        # not a socket, nothing listening
    t = threading.Thread(
        target=serve,
        kwargs={"socket_path": sock, "idle_timeout": 30.0, "quiet": True},
        daemon=True,
    )
    t.start()
    for _ in range(100):
        try:
            if request({"op": "ping"}, sock)["ok"]:
                break
        except (OSError, ValueError):
            time.sleep(0.05)
    else:
        pytest.fail("daemon did not take over the stale socket path")
    request({"op": "shutdown"}, sock)
    t.join(timeout=5)


def test_resolve_db_rules():
    assert _resolve_db("/abs/dir", None) == Path("/abs/dir/.poma-memory.db")
    assert _resolve_db(None, "/abs/custom.db") == Path("/abs/custom.db")
    with pytest.raises(_RelativePath):
        _resolve_db("rel/dir", None)
    with pytest.raises(_RelativePath):
        _resolve_db(None, None)


def test_a_stalled_client_does_not_block_another(running_daemon, indexed_dir):
    """One client that connects and never sends must not stall the machine.

    This is the whole reason connections are handled off the accept loop: the
    search hook fires on every Grep in every session, and they share one daemon.
    """
    stalled = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stalled.connect(str(running_daemon))          # connect, send nothing
    try:
        start = time.time()
        resp = request({"op": "ping"}, running_daemon, timeout=5.0)
        elapsed = time.time() - start
        assert resp["ok"] is True
        assert elapsed < 2.0, f"second client waited {elapsed:.1f}s behind a stalled one"
    finally:
        stalled.close()


def test_reindex_is_visible_through_the_socket(running_daemon, indexed_dir):
    """End-to-end freshness: a warm daemon must not serve a pre-reindex view."""
    first = request(
        {"op": "search", "query": "kubernetes", "path": str(indexed_dir), "top_k": 5},
        running_daemon,
    )
    assert first["ok"] is True
    assert not first["results"], "term should be absent before the reindex"

    new_file = indexed_dir / "k8s.md"
    new_file.write_text(
        "# Cluster\n\n## Kubernetes\n\n**Decision:** kubernetes rollouts are "
        "gated on the readiness probe, never on a fixed sleep.\n",
        encoding="utf-8",
    )
    store = Store(indexed_dir / ".poma-memory.db")
    update_file(store, str(new_file))
    store.close()

    second = request(
        {"op": "search", "query": "kubernetes", "path": str(indexed_dir), "top_k": 5},
        running_daemon,
    )
    assert second["ok"] is True
    assert second["results"], "warm daemon served a stale index after a reindex"


def test_oversized_request_is_refused_cleanly(running_daemon):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(15)
        s.connect(str(running_daemon))
        try:
            s.sendall(b"x" * (2 << 20))           # no newline, over the cap
            s.shutdown(socket.SHUT_WR)
        except OSError:
            pass                                   # server may close first
        body = s.recv(65536)
    assert json.loads(body)["ok"] is False
    assert request({"op": "ping"}, running_daemon)["ok"] is True


@pytest.mark.parametrize("top_k", [0, 1, 3, 500])
def test_daemon_and_in_process_agree(running_daemon, indexed_dir, top_k):
    """The daemon may be faster. It may never be different.

    `--top 0` caught a real divergence: the server coerced with `or 5`, so a
    valid zero became the default and the daemon returned five hits where the
    in-process path returned none.
    """
    from poma_memory.api import search as in_process

    direct = in_process(query="rollback", path=indexed_dir, top_k=top_k)
    resp = request(
        {"op": "search", "query": "rollback", "path": str(indexed_dir),
         "top_k": top_k},
        running_daemon,
    )
    assert resp["ok"] is True
    assert [r["file_path"] for r in resp["results"]] == \
           [r["file_path"] for r in direct]
    assert len(resp["results"]) == len(direct)


def test_concurrent_searches_share_one_connection_safely(running_daemon, indexed_dir):
    """Two requests in flight at once must not trip SQLite's thread check.

    Sequential requests hide this: each worker thread exits and Python hands the
    next one the same thread identifier, so the check passes by luck.
    """
    errors = []

    def hit():
        try:
            r = request({"op": "search", "query": "rollback",
                         "path": str(indexed_dir), "top_k": 5}, running_daemon,
                        timeout=20)
            if not r.get("ok"):
                errors.append(r.get("error"))
        except Exception as e:                     # pragma: no cover
            errors.append(f"{type(e).__name__}: {e}")

    request({"op": "search", "query": "rollback", "path": str(indexed_dir)},
            running_daemon)                       # warm the cache first
    warm = request({"op": "stats"}, running_daemon)["builds"]

    threads = [threading.Thread(target=hit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, f"concurrent searches failed: {errors[:3]}"

    # The real symptom of a cross-thread connection is not an error the caller
    # sees: `_stamp` catches it, reports "unknown version", and the cache rebuilds
    # from scratch every single request — 0.4s each, silently, forever. Assert the
    # warm index actually survived concurrent use.
    after = request({"op": "stats"}, running_daemon)["builds"]
    assert after == warm, (
        f"cache rebuilt {after - warm} times under concurrent load — the warm "
        "index is not being reused"
    )


def test_ping_reports_the_version(running_daemon):
    """A supervisor needs this to notice it is serving an upgraded install."""
    from poma_memory import __version__
    assert request({"op": "ping"}, running_daemon)["version"] == __version__


def test_env_override_is_not_taken_from_the_daemons_environment(running_daemon,
                                                                indexed_dir):
    """POMA_MEMORY_EMPTY_GATE must mean the same with and without a daemon.

    The daemon is a long-lived process whose environment was frozen whenever it
    started. If the gate were resolved there, the same command would return
    different results depending on whether a daemon happened to be running.
    """
    import subprocess
    import sys

    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    env["POMA_MEMORY_EMPTY_GATE"] = "0.99"        # gate everything out
    env.pop("POMA_EMBEDDER", None)

    def run(socket_arg):
        return subprocess.run(
            [sys.executable, "-m", "poma_memory.cli", "search", "rollback",
             "--path", str(indexed_dir), "--top-k", "3", "--socket", socket_arg],
            capture_output=True, text=True, env=env, timeout=120,
        ).stdout

    assert run(str(running_daemon)) == run("off"), \
        "daemon and in-process disagreed under an environment override"


def test_one_index_addressed_two_ways_takes_one_lock(running_daemon, indexed_dir):
    """The lock key must be the RESOLVED database, not the raw request.

    `{"path": "/r/.agent"}` and `{"db_path": "/r/.agent/.poma-memory.db"}` name
    one index, and the CLI sends the second whenever `--db` is passed. Keyed on
    the raw request they took two different locks, so two threads entered
    `_IndexCache.get` for one entry and one closed the Store the other was
    reading. Measured before the fix on this shape: 13 failures in 320 requests
    ("Cannot operate on a closed database", "bad parameter or other API
    misuse", "tuple index out of range") and 4 cache builds instead of 1.
    """
    db = str(indexed_dir / ".poma-memory.db")
    # Warm it first, so `builds` counts only rebuilds caused by the two
    # spellings fighting over one entry.
    request({"op": "search", "query": "rollback", "path": str(indexed_dir)},
            running_daemon, timeout=60)
    warm = request({"op": "stats"}, running_daemon)["builds"]
    errors = []

    def hit(use_db):
        for _ in range(30):
            try:
                resp = request({
                    "op": "search", "query": "rollback",
                    "path": str(indexed_dir),
                    "db_path": db if use_db else None, "top_k": 3,
                }, running_daemon, timeout=30)
                if not resp.get("ok"):
                    errors.append(resp)
            except Exception as e:                      # noqa: BLE001
                errors.append(f"{type(e).__name__}: {e}")

    threads = [threading.Thread(target=hit, args=(i % 2 == 0,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)

    assert errors == [], f"{len(errors)} concurrent failures, e.g. {errors[0]}"
    assert request({"op": "stats"}, running_daemon)["builds"] == warm, \
        "the two spellings rebuilt the cache instead of sharing one entry"


def test_the_lock_key_is_the_resolved_database():
    """The unit of the above: both spellings resolve to one key."""
    from poma_memory.server import _lock_key
    a = {"op": "search", "path": "/r/.agent"}
    b = {"op": "search", "path": "/r/.agent",
         "db_path": "/r/.agent/.poma-memory.db"}
    assert _lock_key(a) == _lock_key(b) == "/r/.agent/.poma-memory.db"
    # Nothing resolvable (ping, stats, a relative path) shares the empty key.
    assert _lock_key({"op": "ping"}) == ""
    assert _lock_key({"op": "search", "path": "relative/.agent"}) == ""
