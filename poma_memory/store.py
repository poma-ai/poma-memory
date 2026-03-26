"""SQLite + FTS5 storage layer for chunks and chunksets."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    file_path    TEXT PRIMARY KEY,
    byte_offset  INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT NOT NULL DEFAULT '',
    mtime        REAL NOT NULL DEFAULT 0
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

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
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

    def close(self) -> None:
        self._conn.close()

    # --- File tracking ---

    def get_file_record(self, file_path: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM files WHERE file_path = ?", (file_path,)
        ).fetchone()
        return dict(row) if row else None

    def upsert_file_record(
        self, file_path: str, byte_offset: int, content_hash: str, mtime: float
    ) -> None:
        self._conn.execute(
            """INSERT INTO files (file_path, byte_offset, content_hash, mtime)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(file_path) DO UPDATE SET
                   byte_offset=excluded.byte_offset,
                   content_hash=excluded.content_hash,
                   mtime=excluded.mtime""",
            (file_path, byte_offset, content_hash, mtime),
        )
        self._conn.commit()

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
        for cs in chunksets:
            self._conn.execute(
                """INSERT INTO chunksets (file_path, local_index, chunk_ids, contents, to_embed)
                   VALUES (?, ?, ?, ?, ?)""",
                (file_path, cs["chunkset_index"],
                 json.dumps(cs["chunk_ids"]), cs["contents"],
                 cs.get("to_embed", "")),
            )
        self._conn.commit()

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
        self._conn.execute(
            "UPDATE chunks SET embedding = ? WHERE chunk_id = ?",
            (embedding, chunk_id),
        )
        self._conn.commit()

    def update_chunkset_embedding(self, chunkset_id: int, embedding: bytes | None) -> None:
        self._conn.execute(
            "UPDATE chunksets SET embedding = ? WHERE chunkset_id = ?",
            (embedding, chunkset_id),
        )
        self._conn.commit()

    def get_all_chunkset_embeddings(self) -> list[tuple[int, bytes | None]]:
        rows = self._conn.execute(
            "SELECT chunkset_id, embedding FROM chunksets ORDER BY chunkset_id"
        ).fetchall()
        return [(r["chunkset_id"], r["embedding"]) for r in rows]

    # --- Status ---

    def status(self) -> dict:
        files = self._conn.execute("SELECT file_path FROM files").fetchall()
        chunk_count = self._conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()["c"]
        chunkset_count = self._conn.execute("SELECT COUNT(*) as c FROM chunksets").fetchone()["c"]
        has_emb = self._conn.execute(
            "SELECT COUNT(*) as c FROM chunks WHERE embedding IS NOT NULL"
        ).fetchone()["c"] > 0

        return {
            "files": [r["file_path"] for r in files],
            "total_chunks": chunk_count,
            "total_chunksets": chunkset_count,
            "has_embeddings": has_emb,
        }
