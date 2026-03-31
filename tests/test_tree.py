"""Tests for tree parsing and depth normalization."""

from poma_primecut_nano.tree import parse_indented_text, normalize_depths


def test_parse_basic():
    arrow_text = "Title\n→Section\n→→Detail one.\n→→Detail two."
    chunks = parse_indented_text(arrow_text)

    assert len(chunks) == 4
    assert chunks[0]["depth"] == 0
    assert chunks[0]["content"] == "Title"
    assert chunks[1]["depth"] == 1
    assert chunks[1]["content"] == "Section"
    assert chunks[2]["depth"] == 2
    assert chunks[3]["depth"] == 2


def test_normalize_parents():
    arrow_text = "Title\n→Section\n→→Detail\n→→→Subdetail"
    chunks = parse_indented_text(arrow_text)
    chunks = normalize_depths(chunks)

    # Title has no parent
    assert chunks[0]["parent_chunk_index"] is None
    # Section's parent is Title
    assert chunks[1]["parent_chunk_index"] == 0
    # Detail's parent is Section
    assert chunks[2]["parent_chunk_index"] == 1
    # Subdetail's parent is Detail
    assert chunks[3]["parent_chunk_index"] == 2


def test_sibling_parents():
    arrow_text = "Title\n→Section\n→→Item A\n→→Item B\n→→Item C"
    chunks = parse_indented_text(arrow_text)
    chunks = normalize_depths(chunks)

    # All items should have Section as parent
    assert chunks[2]["parent_chunk_index"] == 1
    assert chunks[3]["parent_chunk_index"] == 1
    assert chunks[4]["parent_chunk_index"] == 1


def test_empty_input():
    chunks = parse_indented_text("")
    assert chunks == []
