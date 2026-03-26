"""Tests for the heuristic markdown chunker."""

from poma_memory.chunker import indent_light


def test_basic_heading_hierarchy():
    md = """# Full Context Log

## 2026-02-20 — Session-scoped context

**Goal:** Make megavibe safe for concurrent sessions.

**Decision:** Split files into shared + session-scoped.

1. log-tool-event.sh — extract SID
2. on-compact.sh — extract SID
"""
    result = indent_light(md)
    lines = result.split("\n")

    # Title at depth 0 (no arrow)
    assert lines[0] == "Full Context Log"

    # ## heading at depth 1
    assert lines[1].startswith("→") and not lines[1].startswith("→→")
    assert "Session-scoped context" in lines[1]

    # Body text at depth 2
    goal_line = [l for l in lines if "Goal:" in l][0]
    assert goal_line.startswith("→→")

    # List items at depth 3
    item_lines = [l for l in lines if "log-tool-event" in l]
    assert item_lines
    assert item_lines[0].startswith("→→→")


def test_extract_title_false():
    md = """## Section heading

Some body text here.
"""
    result = indent_light(md, extract_title=False)
    lines = [l for l in result.split("\n") if l.strip()]

    # With extract_title=False, the heading should NOT be consumed as title
    # It should appear as a regular heading at depth >= 1
    assert any("Section heading" in l for l in lines)
    # No depth-0 line (no title emitted)
    assert not any(l and not l.startswith("→") for l in lines)


def test_markdown_table():
    md = """# Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Split files | Concurrent sessions corrupt state |
| 2 | Keep shared + flock | Append-only works with locking |
"""
    result = indent_light(md)
    lines = result.split("\n")

    assert lines[0] == "Decisions"
    # Table should be kept as a single block
    table_lines = [l for l in lines if "|" in l]
    assert len(table_lines) >= 1  # at least the table block


def test_code_fence():
    md = """# Example

```python
def hello():
    print("world")
```

Some text after.
"""
    result = indent_light(md)
    lines = result.split("\n")

    assert lines[0] == "Example"
    code_lines = [l for l in lines if "def hello" in l]
    assert code_lines


def test_sentence_splitting():
    md = """# Doc

This is sentence one. This is sentence two. And sentence three.
"""
    result = indent_light(md)
    lines = [l for l in result.split("\n") if l.strip()]

    # Should split into 3 separate lines
    assert len(lines) >= 4  # title + 3 sentences


def test_empty_input():
    result = indent_light("")
    assert result.strip() == ""


def test_no_heading():
    md = "Just a plain text line without any heading."
    result = indent_light(md)
    lines = [l for l in result.split("\n") if l.strip()]
    # First line should be treated as title (depth 0)
    assert lines[0] == "Just a plain text line without any heading."
