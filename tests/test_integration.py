"""End-to-end integration test with real .agent/ files."""

import os
import tempfile

from poma_memory.chunker import indent_light
from poma_memory.tree import parse_indented_text, normalize_depths
from poma_memory.chunksets import chunks_to_chunksets
from poma_memory.store import Store
from poma_memory.bm25_search import BM25Search


SAMPLE_FULL_CONTEXT = """# Full Context Log

Append-only. Record decisions, key prompts, and important tool outputs here.

---

## 2026-02-20 — Session-scoped context for concurrent sessions

**Goal:** Make megavibe safe for multiple simultaneous Claude Code sessions in the same project.

**Analysis:** Identified race conditions in shared `.agent/LOGS/` state — counter, rehydration flag, tool-events.jsonl, and WORKING_CONTEXT.md all written by concurrent sessions without coordination.

**Decision:** Split files into shared (FULL_CONTEXT, DECISIONS, TASKS) vs session-scoped (WORKING_CONTEXT, counter, flag, tool-events). Use session_id from hook stdin.

## 2026-02-20 — AI backend fallback (Gemini to Codex)

**Problem:** Gemini MCP is geo-blocked in Andorra (403 PERMISSION_DENIED).

**Decision:** Add backend availability check + bidirectional fallback routing. At session start, Claude pings Gemini MCP. If unavailable, tasks route to Codex.
"""


def test_full_pipeline():
    """Index sample content and search for it."""
    # 1. Chunk
    arrow_text = indent_light(SAMPLE_FULL_CONTEXT)
    assert "Full Context Log" in arrow_text

    # 2. Parse into chunks
    chunks = parse_indented_text(arrow_text)
    chunks = normalize_depths(chunks)
    assert len(chunks) > 5

    # Title
    assert chunks[0]["depth"] == 0
    assert chunks[0]["content"] == "Full Context Log"

    # 3. Build chunksets
    chunksets = chunks_to_chunksets(chunks)
    assert len(chunksets) > 0

    # Each chunkset should have contents
    for cs in chunksets:
        assert cs["contents"]
        assert cs["chunk_ids"]

    # 4. Store and search
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = Store(db_path)

        store.insert_chunks("FULL_CONTEXT.md", chunks)
        store.insert_chunksets("FULL_CONTEXT.md", chunksets)

        # FTS search
        hits = store.fts_search_chunksets("session concurrent", limit=5)
        assert len(hits) > 0
        assert any("concurrent" in h["contents"].lower() for h in hits)

        # BM25 search
        bm25 = BM25Search(store)
        results = bm25.search("geo-blocked Andorra fallback", top_k=3)
        assert len(results) > 0
        assert any("andorra" in r["contents"].lower() for r in results)

        # Status
        status = store.status()
        assert status["total_chunks"] == len(chunks)
        assert status["total_chunksets"] == len(chunksets)

        store.close()


def test_chunksets_contain_ancestors():
    """Verify chunksets include ancestor (heading) context."""
    arrow_text = indent_light(SAMPLE_FULL_CONTEXT)
    chunks = parse_indented_text(arrow_text)
    chunks = normalize_depths(chunks)
    chunksets = chunks_to_chunksets(chunks)

    # Find a chunkset that contains "race conditions"
    race_cs = [cs for cs in chunksets if "race conditions" in cs["contents"].lower()]
    if race_cs:
        # It should also contain the section heading
        assert any("session-scoped" in cs["contents"].lower() for cs in race_cs)
