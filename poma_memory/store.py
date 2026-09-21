"""SQLite + FTS5 storage layer for chunks and chunksets."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    file_path    TEXT PRIMARY KEY,
    byte_offset  INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT NOT NULL DEFAULT '',
    mtime        REAL NOT NULL DEFAULT 0,
    metadata     TEXT NOT NULL DEFAULT '',
    fm_unparsed  INTEGER NOT NULL DEFAULT 0
);

-- Index-wide state that is not about any one file. Created with IF NOT EXISTS
-- rather than added by ALTER, so it appears on databases written by older
-- versions without a migration step.
CREATE TABLE IF NOT EXISTS index_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path        TEXT NOT NULL,
    local_index      INTEGER NOT NULL,
    content          TEXT NOT NULL,
    depth            INTEGER NOT NULL,
    parent_chunk_id  INTEGER,
    embedding        BLOB,
    UNIQUE(file_path, local_index)
);

CREATE TABLE IF NOT EXISTS chunksets (
    chunkset_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path    TEXT NOT NULL,
    local_index  INTEGER NOT NULL,
    chunk_ids    TEXT NOT NULL,
    contents     TEXT NOT NULL,
    to_embed     TEXT NOT NULL DEFAULT '',
    embedding    BLOB,
    upserted_at  REAL NOT NULL DEFAULT 0,
    UNIQUE(file_path, local_index)
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    content='chunks',
    content_rowid='chunk_id',
    tokenize='porter unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunksets_fts USING fts5(
    contents,
    content='chunksets',
    content_rowid='chunkset_id',
    tokenize='porter unicode61'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content) VALUES (new.chunk_id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content)
        VALUES('delete', old.chunk_id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS chunksets_ai AFTER INSERT ON chunksets BEGIN
    INSERT INTO chunksets_fts(rowid, contents)
        VALUES (new.chunkset_id, new.contents);
END;

CREATE TRIGGER IF NOT EXISTS chunksets_ad AFTER DELETE ON chunksets BEGIN
    INSERT INTO chunksets_fts(chunksets_fts, rowid, contents)
        VALUES('delete', old.chunkset_id, old.contents);
END;
"""


class Store:
    """SQLite + FTS5 storage for poma-memory."""

    def __init__(self, db_path: str | Path, check_same_thread: bool = True):
        """Open the store.

        `check_same_thread=False` is for the search daemon only, which hands one
        cached connection to whichever worker thread serves the next request and
        serialises every use of it behind a single lock. SQLite's thread check
        guards against unsynchronised sharing; the daemon provides that
        synchronisation itself. Do NOT pass False without an equivalent lock —
        and note that the thread check passes by luck in short-lived threads,
        because Python reuses thread identifiers once a thread exits.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path),
                                     check_same_thread=check_same_thread)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Run schema migrations for databases created by older versions."""
        # v0.2.0: added to_embed column to chunksets
        try:
            self._conn.execute(
                "ALTER TABLE chunksets ADD COLUMN to_embed TEXT NOT NULL DEFAULT ''"
            )
            self._conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists

        # v0.3.8: added upserted_at to chunksets (per-chunkset age signal). Existing
        # rows default to 0 = "unknown age"; true timestamps accrue as content is
        # (re)indexed. The append fast path preserves old rows, so chunks appended
        # over time keep their original insert time — real age for append-only logs.
        try:
            self._conn.execute(
                "ALTER TABLE chunksets ADD COLUMN upserted_at REAL NOT NULL DEFAULT 0"
            )
            self._conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists

        # v0.6.0: per-file metadata (path rules + front-matter) as opaque JSON.
        # Three states matter and '' is one of them: '' means this row predates
        # metadata indexing, '{}' means it was scanned and has none. Collapsing
        # them would make a filtered search on a legacy index return [] that is
        # indistinguishable from "no document matches".
        for column, ddl in (
            ("metadata", "ALTER TABLE files ADD COLUMN metadata TEXT NOT NULL DEFAULT ''"),
            ("fm_unparsed", "ALTER TABLE files ADD COLUMN fm_unparsed INTEGER NOT NULL DEFAULT 0"),
        ):
            try:
                self._conn.execute(ddl)
                self._conn.commit()
            except sqlite3.OperationalError:
                pass  # column already exists

    # --- Index-wide state ---

    def get_index_meta(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM index_meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_index_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            """INSERT INTO index_meta (key, value) VALUES (?, ?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
            (key, value),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # --- File tracking ---

    def get_file_record(self, file_path: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM files WHERE file_path = ?", (file_path,)
        ).fetchone()
        return dict(row) if row else None

    def upsert_file_record(
        self, file_path: str, byte_offset: int, content_hash: str, mtime: float,
        metadata: str | None = None, fm_unparsed: bool | None = None,
    ) -> None:
        """Write a file row. `metadata=None` leaves any existing value alone.

        Callers that are not metadata-aware (and older callers) must not blank
        the column just by touching the row, so None means "don't change it"
        rather than "set it to empty".
        """
        flag = None if fm_unparsed is None else int(fm_unparsed)
        self._conn.execute(
            """INSERT INTO files (file_path, byte_offset, content_hash, mtime,
                                  metadata, fm_unparsed)
               VALUES (?, ?, ?, ?, COALESCE(?, ''), COALESCE(?, 0))
               ON CONFLICT(file_path) DO UPDATE SET
                   byte_offset=excluded.byte_offset,
                   content_hash=excluded.content_hash,
                   mtime=excluded.mtime,
                   metadata=COALESCE(?, files.metadata),
                   fm_unparsed=COALESCE(?, files.fm_unparsed)""",
            (file_path, byte_offset, content_hash, mtime, metadata, flag,
             metadata, flag),
        )
        self._conn.commit()

    def set_file_metadata(self, file_path: str, metadata: str,
                          fm_unparsed: bool = False) -> None:
        """Update metadata in place, without re-chunking or re-embedding.

        This is the whole backfill mechanism. `_full_reindex` would delete,
        re-chunk and re-embed the file, which on the OpenAI embedder is real
        money for a column that can be filled from the file head.
        """
        self._conn.execute(
            "UPDATE files SET metadata = ?, fm_unparsed = ? WHERE file_path = ?",
            (metadata, int(fm_unparsed), file_path),
        )
        self._conn.commit()

    def all_file_paths(self) -> list[str]:
        rows = self._conn.execute("SELECT file_path FROM files").fetchall()
        return [r["file_path"] for r in rows]

    def count_files_without_metadata(self) -> int:
        """Files never scanned for metadata. Non-zero blocks a filtered search."""
        return self._conn.execute(
            "SELECT COUNT(*) AS c FROM files WHERE metadata = ''"
        ).fetchone()["c"]

    def files_without_metadata(self, limit: int = 3) -> list[str]:
        """A few paths that have never been scanned, to name in the refusal.

        A bare count does not tell you which file to cover, and the usual way
        to get stuck is a file no `index` run's glob reaches.
        """
        rows = self._conn.execute(
            "SELECT file_path FROM files WHERE metadata = '' "
            "ORDER BY file_path LIMIT ?", (limit,),
        ).fetchall()
        return [r["file_path"] for r in rows]

    def get_file_metadata_map(self) -> dict[str, dict]:
        """file_path -> parsed metadata, skipping rows that have none recorded."""
        rows = self._conn.execute(
            "SELECT file_path, metadata FROM files WHERE metadata != ''"
        ).fetchall()
        out: dict[str, dict] = {}
        for r in rows:
            try:
                value = json.loads(r["metadata"])
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict):
                out[r["file_path"]] = value
        return out

    def unparsed_frontmatter_files(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT file_path FROM files WHERE fm_unparsed = 1 ORDER BY file_path"
        ).fetchall()
        return [r["file_path"] for r in rows]

    # --- Chunks ---

    def insert_chunks(self, file_path: str, chunks: list[dict]) -> list[int]:
        """Insert chunks and return their chunk_ids."""
        ids = []
        for c in chunks:
            cur = self._conn.execute(
                """INSERT INTO chunks (file_path, local_index, content, depth, parent_chunk_id)
                   VALUES (?, ?, ?, ?, ?)""",
                (file_path, c["chunk_index"], c["content"], c["depth"],
                 c.get("parent_chunk_index")),
            )
            ids.append(cur.lastrowid)
        self._conn.commit()
        return ids

    def get_chunks_for_file(self, file_path: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE file_path = ? ORDER BY local_index",
            (file_path,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_max_local_index(self, file_path: str) -> int:
        row = self._conn.execute(
            "SELECT MAX(local_index) as m FROM chunks WHERE file_path = ?",
            (file_path,),
        ).fetchone()
        return row["m"] if row and row["m"] is not None else -1

    def get_last_heading_chunk(self, file_path: str) -> dict | None:
        """Get the last chunk that looks like a heading (depth <= 1)."""
        row = self._conn.execute(
            """SELECT * FROM chunks WHERE file_path = ? AND depth <= 1
               ORDER BY local_index DESC LIMIT 1""",
            (file_path,),
        ).fetchone()
        return dict(row) if row else None

    def delete_file_data(self, file_path: str) -> None:
        """Delete all chunks and chunksets for a file."""
        self._conn.execute("DELETE FROM chunksets WHERE file_path = ?", (file_path,))
        self._conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
        self._conn.execute("DELETE FROM files WHERE file_path = ?", (file_path,))
        self._conn.commit()

    # --- Chunksets ---

    def insert_chunksets(self, file_path: str, chunksets: list[dict]) -> None:
        # Stamp insert time so search can surface content age. Appends insert only
        # NEW chunksets (old ones are untouched), so each chunkset's timestamp
        # reflects when its content first entered the index.
        now = time.time()
        for cs in chunksets:
            self._conn.execute(
                """INSERT INTO chunksets (file_path, local_index, chunk_ids, contents, to_embed, upserted_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (file_path, cs["chunkset_index"],
                 json.dumps(cs["chunk_ids"]), cs["contents"],
                 cs.get("to_embed", ""), now),
            )
        self._conn.commit()

    def max_chunkset_upserted(self, chunkset_ids: list[int]) -> float:
        """Newest upsert time among the given chunksets (0.0 if none/unknown).

        Used to surface the age of a search result: the freshest matched chunkset
        in a file answers "how recent is this context?".
        """
        if not chunkset_ids:
            return 0.0
        placeholders = ",".join("?" for _ in chunkset_ids)
        row = self._conn.execute(
            f"SELECT MAX(upserted_at) AS m FROM chunksets WHERE chunkset_id IN ({placeholders})",
            list(chunkset_ids),
        ).fetchone()
        return float(row["m"]) if row and row["m"] is not None else 0.0

    def get_chunkset_files(self) -> list[tuple[int, str]]:
        """(chunkset_id, file_path) for every chunkset.

        Two columns rather than `get_all_chunksets`, which carries every
        chunkset's full text: resolving a metadata predicate needs only the
        mapping, and the search path already loads the contents once.
        """
        rows = self._conn.execute(
            "SELECT chunkset_id, file_path FROM chunksets"
        ).fetchall()
        return [(r["chunkset_id"], r["file_path"]) for r in rows]

    def get_all_chunksets(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM chunksets ORDER BY file_path, local_index"
        ).fetchall()
        return [dict(r) for r in rows]

    # --- FTS search ---

    def fts_search_chunksets(self, query: str, limit: int = 20) -> list[dict]:
        """BM25-ranked FTS5 search on chunksets."""
        rows = self._conn.execute(
            """SELECT cs.*, rank
               FROM chunksets_fts
               JOIN chunksets cs ON chunksets_fts.rowid = cs.chunkset_id
               WHERE chunksets_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (query, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def fts_search_chunks(self, query: str, limit: int = 50) -> list[dict]:
        """BM25-ranked FTS5 search on individual chunks."""
        rows = self._conn.execute(
            """SELECT c.*, rank
               FROM chunks_fts
               JOIN chunks c ON chunks_fts.rowid = c.chunk_id
               WHERE chunks_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (query, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Embeddings ---

    def update_chunk_embedding(self, chunk_id: int, embedding: bytes) -> None:
        """Persist the embedding blob for a chunk row."""
        self._conn.execute(
            "UPDATE chunks SET embedding = ? WHERE chunk_id = ?",
            (embedding, chunk_id),
        )
        self._conn.commit()

    def update_chunkset_embedding(self, chunkset_id: int, embedding: bytes | None) -> None:
        """Set or clear the embedding blob for a chunkset row."""
        self._conn.execute(
            "UPDATE chunksets SET embedding = ? WHERE chunkset_id = ?",
            (embedding, chunkset_id),
        )
        self._conn.commit()

    def get_all_chunkset_embeddings(self) -> list[tuple[int, bytes | None]]:
        """Return every chunkset id with its embedding (or None if missing), ordered by id."""
        rows = self._conn.execute(
            "SELECT chunkset_id, embedding FROM chunksets ORDER BY chunkset_id"
        ).fetchall()
        return [(r["chunkset_id"], r["embedding"]) for r in rows]

    # --- Status ---

    def status(self) -> dict:
        """Summarize indexed files, chunk/chunkset counts, and whether any embedding exists.

        Embeddings live on the ``chunksets`` table (one vector per chunkset), not on
        ``chunks``. Checking only ``chunks.embedding`` reports ``has_embeddings=False``
        even when every chunkset is fully embedded.
        """
        files = self._conn.execute("SELECT file_path FROM files").fetchall()
        chunk_count = self._conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()["c"]
        chunkset_count = self._conn.execute("SELECT COUNT(*) as c FROM chunksets").fetchone()["c"]
        has_emb = self._conn.execute(
            "SELECT COUNT(*) as c FROM chunksets WHERE embedding IS NOT NULL"
        ).fetchone()["c"] > 0

        return {
            "files": [r["file_path"] for r in files],
            "total_chunks": chunk_count,
            "total_chunksets": chunkset_count,
            "has_embeddings": has_emb,
            "files_without_metadata": self.count_files_without_metadata(),
            "unparsed_frontmatter": self.unparsed_frontmatter_files(),
        }
