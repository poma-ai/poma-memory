"""Path rules, front-matter parsing, and how the two combine."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from poma_memory import api
from poma_memory.frontmatter import parse
from poma_memory.metadata import (
    MetadataRulesError, load_rules, merge, matches, normalize_where,
    resolve_paths, rules_hash,
)
from poma_memory.store import Store


# --- front-matter grammar ---

def test_flat_scalars_and_inline_list():
    meta, ok = parse("---\nkind: decision\ntags: [a, b]\n---\nbody\n")
    assert ok and meta == {"kind": "decision", "tags": ["a", "b"]}


def test_block_list():
    meta, ok = parse("---\ntags:\n  - a\n  - b\n---\n")
    assert ok and meta == {"tags": ["a", "b"]}


def test_one_level_nesting_flattens_to_dotted_key():
    meta, ok = parse("---\nmetadata:\n  type: feedback\nname: x\n---\n")
    assert ok and meta == {"metadata.type": "feedback", "name": "x"}


def test_no_fence_is_not_a_failure():
    assert parse("# heading\n\ntext\n") == ({}, True)


def test_fence_must_be_at_byte_zero():
    assert parse("\n---\nkind: x\n---\n") == ({}, True)


def test_crlf():
    meta, ok = parse("---\r\nkind: note\r\n---\r\nbody\r\n")
    assert ok and meta == {"kind": "note"}


def test_unterminated_fence_is_unparsed():
    assert parse("---\nkind: note\n") == ({}, False)


@pytest.mark.parametrize("block", [
    "---\nbody: |\n  text\n---\n",          # block scalar
    "---\nref: &anchor x\n---\n",           # anchor
    "---\nm: {a: 1}\n---\n",                # flow mapping
    "---\nk: v\nk: w\n---\n",               # duplicate key
    "---\nk:\n  a: 1\n   b: 2\n---\n",      # inconsistent indentation
    "---\n- a\n- b\n---\n",                 # top-level sequence
])
def test_out_of_grammar_is_unparsed(block):
    assert parse(block) == ({}, False)


def test_values_that_look_like_booleans_or_numbers_stay_strings():
    meta, ok = parse("---\nactive: true\ncount: 3\nver: 1.0\n---\n")
    assert ok and meta == {"active": "true", "count": "3", "ver": "1.0"}


def test_quotes_are_stripped():
    meta, ok = parse('---\nk: "v"\nj: \'w\'\n---\n')
    assert ok and meta == {"k": "v", "j": "w"}


# --- predicate ---

def test_predicate_and_across_keys_or_within_list():
    assert matches({"kind": "event", "status": "active"},
                   {"kind": ["event", "note"], "status": "active"})
    assert not matches({"kind": "event", "status": "old"},
                       {"kind": "event", "status": "active"})


def test_predicate_list_valued_field_matches_on_intersection():
    assert matches({"tags": ["a", "b"]}, {"tags": "b"})
    assert not matches({"tags": ["a", "b"]}, {"tags": "c"})


def test_predicate_is_case_sensitive_and_cannot_express_absence():
    assert not matches({"kind": "Event"}, {"kind": "event"})
    assert not matches({"other": "x"}, {"kind": "event"})


@pytest.mark.parametrize("bad", [{"k": 1}, {"k": []}, {"k": ["a", 2]}, {"": "v"}])
def test_normalize_where_rejects_shapes_outside_the_grammar(bad):
    with pytest.raises(ValueError):
        normalize_where(bad)


def test_merge_prefers_frontmatter_over_path_rule():
    assert merge({"kind": "note", "a": "1"}, {"kind": "event"}) == {
        "kind": "event", "a": "1"}


# --- rules file ---

def _write_rules(root: Path, rules: list[dict]) -> None:
    (root / ".poma-metadata.json").write_text(json.dumps({"rules": rules}))


def test_first_matching_rule_wins_and_catch_all_sits_last():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "events").mkdir()
        (root / "events" / "e1.md").write_text("x")
        (root / "DECISIONS.md").write_text("x")
        (root / "other.md").write_text("x")
        _write_rules(root, [
            {"glob": "events/**/*.md", "metadata": {"kind": "event"}},
            {"glob": "DECISIONS.md", "metadata": {"kind": "decision"}},
            {"glob": "**/*.md", "metadata": {"kind": "note"}},
        ])
        rules, _ = load_rules(root)
        resolved = resolve_paths(root, rules)
        by_name = {Path(k).name: v for k, v in resolved.items()}
        assert by_name["e1.md"] == {"kind": "event"}
        assert by_name["DECISIONS.md"] == {"kind": "decision"}
        assert by_name["other.md"] == {"kind": "note"}


def test_absent_rules_file_is_not_an_error():
    with tempfile.TemporaryDirectory() as td:
        assert load_rules(Path(td)) == ([], "")


@pytest.mark.parametrize("body", [
    "{not json",
    '{"rules": "nope"}',
    '{"rules": [{"glob": "", "metadata": {"a": "b"}}]}',
    '{"rules": [{"glob": "*.md"}]}',
    '{"rules": [{"glob": "*.md", "metadata": {"a": 1}}]}',
])
def test_malformed_rules_name_the_file_instead_of_being_skipped(body):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / ".poma-metadata.json").write_text(body)
        with pytest.raises(MetadataRulesError) as e:
            load_rules(root)
        assert ".poma-metadata.json" in str(e.value)


# --- indexing and backfill ---

def _corpus(root: Path) -> None:
    (root / "events").mkdir()
    (root / "events" / "e1.md").write_text("# Event one\n\nThe daemon restarted.\n")
    (root / "DECISIONS.md").write_text("# Decisions\n\nUse SQLite for storage.\n")


def test_index_records_metadata_from_path_rules():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _corpus(root)
        _write_rules(root, [
            {"glob": "events/**/*.md", "metadata": {"kind": "event"}},
            {"glob": "**/*.md", "metadata": {"kind": "note"}},
        ])
        api.index(path=root)
        store = Store(root / ".poma-memory.db")
        kinds = {Path(p).name: m["kind"]
                 for p, m in store.get_file_metadata_map().items()}
        assert kinds == {"e1.md": "event", "DECISIONS.md": "note"}
        assert store.count_files_without_metadata() == 0
        store.close()


def test_frontmatter_overrides_a_path_rule_on_the_same_key():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "a.md").write_text("---\nkind: special\n---\n# A\n\ntext\n")
        _write_rules(root, [{"glob": "**/*.md", "metadata": {"kind": "note"}}])
        api.index(path=root)
        store = Store(root / ".poma-memory.db")
        assert list(store.get_file_metadata_map().values()) == [{"kind": "special"}]
        store.close()


def test_unparsed_frontmatter_is_named_in_status():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "bad.md").write_text("---\nbody: |\n  block scalar\n---\n# B\n\ntext\n")
        api.index(path=root)
        info = api.status(path=root)
        assert [Path(p).name for p in info["unparsed_frontmatter"]] == ["bad.md"]


def test_editing_the_rules_file_re_resolves_without_re_chunking():
    """A rule edit touches no document, so mtime cannot carry the change."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _corpus(root)
        _write_rules(root, [{"glob": "**/*.md", "metadata": {"kind": "note"}}])
        api.index(path=root)

        store = Store(root / ".poma-memory.db")
        before = store.status()
        chunk_ids_before = [c["chunk_id"]
                            for c in store.get_chunks_for_file(
                                os.path.realpath(root / "DECISIONS.md"))]
        store.close()

        _write_rules(root, [
            {"glob": "events/**/*.md", "metadata": {"kind": "event"}},
            {"glob": "**/*.md", "metadata": {"kind": "note"}},
        ])
        result = api.index(path=root)
        assert result["metadata_refreshed"] is True

        store = Store(root / ".poma-memory.db")
        kinds = {Path(p).name: m["kind"]
                 for p, m in store.get_file_metadata_map().items()}
        assert kinds["e1.md"] == "event"
        after = store.status()
        # Re-resolution is an UPDATE. Nothing was deleted, re-chunked or
        # re-embedded — on the OpenAI embedder that distinction is money.
        assert after["total_chunks"] == before["total_chunks"]
        assert after["total_chunksets"] == before["total_chunksets"]
        assert [c["chunk_id"] for c in store.get_chunks_for_file(
            os.path.realpath(root / "DECISIONS.md"))] == chunk_ids_before
        store.close()


def test_rules_hash_is_stable_so_an_unchanged_ruleset_does_not_refresh():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _corpus(root)
        _write_rules(root, [{"glob": "**/*.md", "metadata": {"kind": "note"}}])
        api.index(path=root)
        assert api.index(path=root)["metadata_refreshed"] is False


def test_rules_hash_differs_when_rules_differ():
    assert rules_hash('{"rules": []}') != rules_hash('{"rules": [1]}')


def test_legacy_rows_are_healed_by_a_plain_index_run():
    """Simulates an index written before metadata existed."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _corpus(root)
        api.index(path=root)

        db = root / ".poma-memory.db"
        store = Store(db)
        store._conn.execute("UPDATE files SET metadata = ''")
        store._conn.commit()
        assert store.count_files_without_metadata() == 2
        store.close()

        _write_rules(root, [{"glob": "**/*.md", "metadata": {"kind": "note"}}])
        api.index(path=root)

        store = Store(db)
        assert store.count_files_without_metadata() == 0
        store.close()


def test_a_row_whose_file_vanished_is_recorded_rather_than_left_unscanned():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _corpus(root)
        api.index(path=root)

        db = root / ".poma-memory.db"
        store = Store(db)
        store._conn.execute("UPDATE files SET metadata = ''")
        store._conn.commit()
        store.close()
        (root / "DECISIONS.md").unlink()

        api.index(path=root)
        store = Store(db)
        # Left at '', every filtered search would refuse forever over a file
        # nobody can fix.
        assert store.count_files_without_metadata() == 0
        store.close()


def test_single_file_indexing_still_applies_path_rules():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _corpus(root)
        _write_rules(root, [{"glob": "events/**/*.md", "metadata": {"kind": "event"}}])
        api.index_file(root / "events" / "e1.md", path=root)
        store = Store(root / ".poma-memory.db")
        assert list(store.get_file_metadata_map().values()) == [{"kind": "event"}]
        store.close()
