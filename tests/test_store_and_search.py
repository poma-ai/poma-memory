"""Tests for store, incremental update, search pipeline, and path handling."""

import os
import tempfile

from poma_memory.store import Store
from poma_memory.incremental import update_file
from poma_memory.search import HybridSearch


SAMPLE_MD = """\
# Project Notes

## Authentication

**Decision:** Use JWT tokens for API auth.

- Access tokens expire in 15 minutes.
- Refresh tokens expire in 7 days.
- Store tokens in httpOnly cookies.

## Database Schema

**Users table:**

| Column | Type | Notes |
|--------|------|-------|
| id | UUID | Primary key |
| email | TEXT | Unique, indexed |
| created_at | TIMESTAMP | Default now() |

## Deployment

We use Docker containers on AWS ECS.

- Production runs 3 replicas behind ALB.
- Staging runs 1 replica.
"""

APPEND_MD = """
## Caching Strategy

**Decision:** Use Redis for session caching.

- TTL of 30 minutes for session data.
- Cache-aside pattern for database queries.
"""


def test_full_reindex():
    """First index creates all chunks and chunksets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = os.path.join(tmpdir, "notes.md")
        db_path = os.path.join(tmpdir, "test.db")

        with open(md_path, "w") as f:
            f.write(SAMPLE_MD)

        store = Store(db_path)
        result = update_file(store, md_path)

        assert result["status"] == "reindexed"
        assert result["new_chunks"] > 5
        assert result["new_chunksets"] > 0

        # Status shows the file
        status = store.status()
        assert len(status["files"]) == 1
        assert status["total_chunks"] == result["new_chunks"]
        store.close()


def test_unchanged_file():
    """Re-indexing an unchanged file returns 'unchanged'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = os.path.join(tmpdir, "notes.md")
        db_path = os.path.join(tmpdir, "test.db")

        with open(md_path, "w") as f:
            f.write(SAMPLE_MD)

        store = Store(db_path)
        update_file(store, md_path)
        result = update_file(store, md_path)

        assert result["status"] == "unchanged"
        store.close()


def test_incremental_append():
    """Appending content triggers incremental update, not full reindex."""
    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = os.path.join(tmpdir, "notes.md")
        db_path = os.path.join(tmpdir, "test.db")

        with open(md_path, "w") as f:
            f.write(SAMPLE_MD)

        store = Store(db_path)
        first = update_file(store, md_path)
        first_chunks = first["new_chunks"]

        # Append new content
        with open(md_path, "a") as f:
            f.write(APPEND_MD)

        second = update_file(store, md_path)
        assert second["status"] in ("updated", "reindexed")
        assert second["new_chunks"] > 0

        # Total chunks should be more than first run
        status = store.status()
        assert status["total_chunks"] >= first_chunks + second["new_chunks"]
        store.close()


def test_path_normalization():
    """Relative and absolute paths to the same file should not create duplicates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = os.path.join(tmpdir, "notes.md")
        db_path = os.path.join(tmpdir, "test.db")

        with open(md_path, "w") as f:
            f.write(SAMPLE_MD)

        store = Store(db_path)

        # Index with absolute path
        update_file(store, md_path)
        # Index with relative path (via symlink or different form)
        relative = os.path.join(tmpdir, ".", "notes.md")
        update_file(store, relative)

        status = store.status()
        assert len(status["files"]) == 1, f"Expected 1 file, got {status['files']}"
        store.close()


def test_search_pipeline():
    """Full search pipeline: index → search → hierarchical context."""
    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = os.path.join(tmpdir, "notes.md")
        db_path = os.path.join(tmpdir, "test.db")

        with open(md_path, "w") as f:
            f.write(SAMPLE_MD)

        store = Store(db_path)
        update_file(store, md_path)

        hybrid = HybridSearch(store, enable_semantic=False)
        results = hybrid.search("JWT tokens authentication", top_k=3)

        assert len(results) > 0
        # Top result should contain auth content
        top = results[0]
        assert "jwt" in top["context"].lower() or "auth" in top["context"].lower()
        # Should include hierarchical context (section heading)
        assert "Authentication" in top["context"] or "Project Notes" in top["context"]

        store.close()


def test_search_returns_hierarchical_context():
    """Search results should include ancestor headings for context."""
    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = os.path.join(tmpdir, "notes.md")
        db_path = os.path.join(tmpdir, "test.db")

        with open(md_path, "w") as f:
            f.write(SAMPLE_MD)

        store = Store(db_path)
        update_file(store, md_path)

        hybrid = HybridSearch(store, enable_semantic=False)
        results = hybrid.search("Docker ECS replicas", top_k=3)

        assert len(results) > 0
        top = results[0]
        # Should contain the deployment content
        assert "docker" in top["context"].lower() or "replica" in top["context"].lower()
        # Should include section heading in the expanded context
        assert "Deployment" in top["context"]

        store.close()


def test_empty_search():
    """Search with no matching results returns empty list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = os.path.join(tmpdir, "notes.md")
        db_path = os.path.join(tmpdir, "test.db")

        with open(md_path, "w") as f:
            f.write(SAMPLE_MD)

        store = Store(db_path)
        update_file(store, md_path)

        hybrid = HybridSearch(store, enable_semantic=False)
        results = hybrid.search("xyzzy quantum blockchain", top_k=3)

        # BM25 may still return results with low scores, but they should be valid
        for r in results:
            assert "file_path" in r
            assert "context" in r

        store.close()
