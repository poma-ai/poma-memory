"""Path rules, front-matter parsing, and how the two combine."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import pytest

from poma_memory import api
from poma_memory.frontmatter import parse
from poma_memory.metadata import (
    MetadataNotIndexed, MetadataStale,
    stale_files,
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
    assert rules_hash([]) != rules_hash([{"glob": "*.md", "metadata": {"k": "v"}}])


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


def test_a_row_whose_file_vanished_is_pruned_not_left_unscanned():
    """A deleted file's row cannot be refreshed by any later run -- no glob
    matches a file that does not exist -- so whatever is left on it is
    permanent. It is removed entirely instead, which also stops the deleted
    document answering searches."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _corpus(root)
        _write_rules(root, [
            {"glob": "events/**/*.md", "metadata": {"kind": "event"}},
        ])
        api.index(path=root)

        db = root / ".poma-memory.db"
        store = Store(db)
        store._conn.execute("UPDATE files SET metadata = ''")
        store._conn.commit()
        store.close()
        gone = os.path.realpath(root / "DECISIONS.md")
        (root / "DECISIONS.md").unlink()

        result = api.index(path=root)
        assert result["pruned"] == [gone]
        store = Store(db)
        try:
            assert store.count_files_without_metadata() == 0
            assert "DECISIONS.md" not in {
                Path(p_).name for p_ in store.all_file_paths()}
            by_name = {Path(p_).name: m
                       for p_, m in store.get_file_metadata_map().items()}
            assert by_name["e1.md"] == {"kind": "event"}
            assert not [fp for _, fp in store.get_chunkset_files()
                        if Path(fp).name == "DECISIONS.md"]
        finally:
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


# --- silent wrong parse: the failure this parser exists to avoid ---

def test_a_trailing_comment_is_not_part_of_the_value():
    meta, ok = parse("---\nkind: event   # written by agent-log.sh\n---\n")
    assert ok and meta == {"kind": "event"}


def test_a_hash_inside_quotes_is_literal():
    meta, ok = parse('---\ntitle: "issue #42"\n---\n')
    assert ok and meta == {"title": "issue #42"}


def test_a_comment_only_value_is_empty_not_the_comment_text():
    meta, ok = parse("---\nkind:  # to be decided\n---\n")
    assert ok and meta == {"kind": ""}


def test_an_inline_list_does_not_split_inside_quotes():
    meta, ok = parse('---\ntags: [api, "auth, login"]\n---\n')
    assert ok and meta == {"tags": ["api", "auth, login"]}


def test_an_unterminated_quote_is_unparsed_rather_than_guessed():
    assert parse('---\nk: "open\n---\n') == ({}, False)
    assert parse('---\ntags: [a, "open]\n---\n') == ({}, False)


def test_a_utf8_bom_does_not_silently_disable_frontmatter():
    """A BOM would otherwise read as 'no front-matter': no metadata, not
    flagged unparsed, invisible in status()."""
    meta, ok = parse("﻿---\nkind: note\n---\nbody\n")
    assert ok and meta == {"kind": "note"}


# --- hostile rules ---

@pytest.mark.parametrize("glob", ["/etc/**/*.md", "../**/*.md", "a/../../b/*.md"])
def test_absolute_or_escaping_globs_are_a_named_rules_error(glob):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_rules(root, [{"glob": glob, "metadata": {"kind": "x"}}])
        with pytest.raises(MetadataRulesError) as e:
            load_rules(root)
        assert ".poma-metadata.json" in str(e.value)


# --- the documented status() shape ---

def test_status_shape_is_the_same_with_no_database():
    with tempfile.TemporaryDirectory() as td:
        info = api.status(path=Path(td))
        assert info["files_without_metadata"] == 0
        assert info["unparsed_frontmatter"] == []


# --- a narrower glob must not claim files it did not look at ---

def test_a_narrower_glob_leaves_untouched_files_unscanned_rather_than_empty():
    """`fp not in seen` means 'outside this run's glob', not 'gone from disk'.

    Recording those as scanned-and-empty is a lie that no later run corrects,
    because '{}' looks done -- and a filtered search then returns [] that
    cannot be told apart from 'nothing matches'.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _corpus(root)
        api.index(path=root)
        db = root / ".poma-memory.db"

        store = Store(db)
        store._conn.execute("UPDATE files SET metadata = ''")
        store._conn.commit()
        store.close()

        _write_rules(root, [{"glob": "**/*.md", "metadata": {"kind": "note"}}])
        api.index(path=root, glob="events/*.md")

        store = Store(db)
        rows = dict(store._conn.execute(
            "SELECT file_path, metadata FROM files").fetchall())
        by_name = {Path(p).name: m for p, m in rows.items()}
        assert by_name["e1.md"] != ""           # covered by the narrow glob
        assert by_name["DECISIONS.md"] == ""    # exists on disk, not looked at
        assert store.count_files_without_metadata() == 1
        store.close()

        # And the command the error message tells you to run must heal it.
        api.index(path=root)
        store = Store(db)
        assert store.count_files_without_metadata() == 0
        kinds = {Path(p).name: m["kind"]
                 for p, m in store.get_file_metadata_map().items()}
        assert kinds["DECISIONS.md"] == "note"
        store.close()


# --- the rules hash may only advance on a run that covered everything ---

def test_a_partial_run_does_not_claim_the_new_rules_were_applied():
    """Otherwise the rest of the corpus keeps superseded metadata forever.

    Those rows are not '', so the legacy heal never touches them, and the
    hash says the rules are current, so the refresh path never does either.
    `status` reports complete and the filter answers from stale rules.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _corpus(root)
        _write_rules(root, [{"glob": "**/*.md", "metadata": {"kind": "note"}}])
        api.index(path=root)

        _write_rules(root, [
            {"glob": "events/**/*.md", "metadata": {"kind": "event"}},
            {"glob": "**/*.md", "metadata": {"kind": "decision"}},
        ])
        api.index(path=root, glob="events/*.md")

        # The narrow run must not have banked the new hash...
        assert api.index(path=root)["metadata_refreshed"] is True

        store = Store(root / ".poma-memory.db")
        kinds = {Path(p_).name: m["kind"]
                 for p_, m in store.get_file_metadata_map().items()}
        assert kinds == {"e1.md": "event", "DECISIONS.md": "decision"}
        store.close()

        # ...and once it has, an unchanged rule set stops re-resolving.
        assert api.index(path=root)["metadata_refreshed"] is False


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="root ignores directory permissions")
def test_an_unreadable_file_is_left_unscanned_rather_than_called_deleted():
    """`os.path.exists` is False for a permission error too, not just deletion."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "sub").mkdir()
        (root / "sub" / "a.md").write_text("# A\n\nSQLite notes.\n")
        (root / "b.md").write_text("# B\n\nMore notes.\n")
        db = Path(td + "-db") / "index.db"
        _write_rules(root, [{"glob": "**/*.md", "metadata": {"kind": "note"}}])
        api.index(path=root, db_path=db)

        store = Store(db)
        store._conn.execute("UPDATE files SET metadata = ''")
        store._conn.commit()
        store.close()

        (root / "sub").chmod(0o000)
        try:
            api.index(path=root, db_path=db)
        finally:
            (root / "sub").chmod(0o755)

        store = Store(db)
        rows = {Path(p_).name: m for p_, m in store._conn.execute(
            "SELECT file_path, metadata FROM files").fetchall()}
        # Unreachable, not gone: recording '{}' here is permanent and silent.
        assert rows["a.md"] == ""
        store.close()

        api.index(path=root, db_path=db)
        store = Store(db)
        assert store.count_files_without_metadata() == 0
        store.close()


# --- an apostrophe is not an opening quote ---

def test_an_apostrophe_does_not_swallow_the_rest_of_the_line():
    meta, ok = parse("---\ntitle: Don't ship  # decided 2026-09-21\n---\n")
    assert ok and meta == {"title": "Don't ship"}


def test_an_apostrophe_does_not_merge_inline_list_items():
    meta, ok = parse("---\ntags: [don't, can't]\n---\n")
    assert ok and meta == {"tags": ["don't", "can't"]}


def test_a_quote_still_opens_at_the_start_of_a_value():
    meta, ok = parse('---\nk: "a # b"\n---\n')
    assert ok and meta == {"k": "a # b"}


# --- more shapes the grammar cannot represent ---

def test_a_quoted_key_is_unwrapped_not_kept_with_its_quotes():
    meta, ok = parse('---\n"kind": event\n---\n')
    assert ok and meta == {"kind": "event"}


@pytest.mark.parametrize("block", [
    "---\nk: [a, [b, c]]\n---\n",   # nested flow sequence
    "---\nk: [a, b\n---\n",         # unterminated bracket
    "---\nk: [a, , b]\n---\n",      # empty item
    "---\n<<: base\n---\n",         # merge key
])
def test_flow_shapes_outside_the_grammar_are_unparsed(block):
    assert parse(block) == ({}, False)


def test_a_trailing_comma_in_an_inline_list_is_not_a_phantom_item():
    meta, ok = parse("---\ntags: [a, ]\n---\n")
    assert ok and meta == {"tags": ["a"]}


# --- the rules guards are not posix-only ---

@pytest.mark.parametrize("glob", ["..\\\\..\\\\etc\\\\*.md", "C:\\\\Users\\\\x\\\\*.md",
                                  "D:/data/*.md"])
def test_windows_style_globs_are_rejected_rather_than_bypassing_the_guards(glob):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_rules(root, [{"glob": glob, "metadata": {"kind": "x"}}])
        with pytest.raises(MetadataRulesError):
            load_rules(root)


def test_a_colon_inside_a_path_component_is_legal_on_posix():
    """A directory really can be called `notes:2026`; refusing it was collateral
    damage from the Windows guard."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "notes:2026").mkdir()
        (root / "notes:2026" / "a.md").write_text("# A\n\ntext\n")
        _write_rules(root, [{"glob": "notes:2026/**/*.md",
                             "metadata": {"kind": "dated"}}])
        rules, _ = load_rules(root)
        resolved = resolve_paths(root, rules)
        assert [v for v in resolved.values()] == [{"kind": "dated"}]


# --- the bounded head read must agree with the full read ---

def test_a_frontmatter_block_straddling_the_head_read_boundary_agrees():
    """Both directions: cut-short-looks-unparsed and cut-short-looks-closed."""
    from poma_memory.frontmatter import MAX_BYTES, parse as fm_parse
    from poma_memory.incremental import _resolve_metadata

    for pad in (MAX_BYTES - 40, MAX_BYTES + 40):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "big.md"
            f.write_text(f"---\nkind: event\np: {'x' * pad}\n---\n# Body\n\ntext\n")
            head_path = _resolve_metadata(str(f), None)          # bounded read
            full_path = _resolve_metadata(str(f), None, f.read_text())
            assert head_path == full_path, f"disagreement at pad={pad}"
            assert json.loads(head_path[0])["kind"] == "event"


# --- an unreadable file must not be recorded as scanned ---

@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="root ignores file permissions")
def test_an_unreadable_file_keeps_its_metadata_rather_than_losing_frontmatter():
    """The file itself is unreadable but its directory is traversable, so
    `os.stat` succeeds and the mtime short-circuit is taken. Swallowing the
    read error there stamped the row with path-rule metadata alone, flagged it
    parsed, and counted it complete -- permanently and silently wrong."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "a.md").write_text("---\nkind: decision\n---\n# A\n\ntext\n")
        _write_rules(root, [{"glob": "**/*.md", "metadata": {"kind": "note"}}])
        api.index(path=root)

        store = Store(root / ".poma-memory.db")
        before = store.get_file_metadata_map()
        store.close()
        assert list(before.values()) == [{"kind": "decision"}]

        _write_rules(root, [{"glob": "**/*.md", "metadata": {"kind": "note2"}}])
        (root / "a.md").chmod(0o000)
        try:
            result = api.index(path=root)
        finally:
            (root / "a.md").chmod(0o644)

        store = Store(root / ".poma-memory.db")
        # Unchanged, not overwritten with the path rule alone.
        assert store.get_file_metadata_map() == before
        store.close()
        assert [Path(p_).name for p_ in result["unreadable"]] == ["a.md"]

        # And the next run that can read it resolves it properly.
        api.index(path=root)
        store = Store(root / ".poma-memory.db")
        assert list(store.get_file_metadata_map().values()) == [{"kind": "decision"}]
        store.close()


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="root ignores file permissions")
def test_one_unreadable_file_does_not_abort_the_whole_index_run():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "good.md").write_text("# Good\n\ntext\n")
        (root / "bad.md").write_text("# Bad\n\ntext\n")
        (root / "bad.md").chmod(0o000)
        try:
            result = api.index(path=root)   # must not raise
        finally:
            (root / "bad.md").chmod(0o644)
        assert [Path(p_).name for p_ in result["unreadable"]] == ["bad.md"]
        assert result["files_indexed"] == 1


def test_a_dangling_symlink_does_not_abort_the_run():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "good.md").write_text("# Good\n\ntext\n")
        (root / "dangling.md").symlink_to(root / "nowhere.md")
        result = api.index(path=root)       # must not raise
        assert result["files_indexed"] == 1


# --- rows an index run could not reach are reported, not hidden ---

def test_rows_left_on_an_earlier_rule_set_are_named():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _corpus(root)
        _write_rules(root, [{"glob": "**/*.md", "metadata": {"kind": "note"}}])
        api.index(path=root)

        _write_rules(root, [{"glob": "**/*.md", "metadata": {"kind": "changed"}}])
        result = api.index(path=root, glob="events/*.md")

        assert [Path(p_).name for p_ in result["stale_rules"]] == ["DECISIONS.md"]
        # Per-file hashes mean the next full run heals it, unlike the global flag.
        assert api.index(path=root)["stale_rules"] == []


def test_a_dot_prefixed_file_is_still_refreshable_through_index_file():
    """`index()` skips it forever, so the per-file hash is what keeps it
    healable at all -- a global flag left it permanently stale."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "a.md").write_text("# A\n\ntext\n")
        (root / ".hidden.md").write_text("# H\n\ntext\n")
        _write_rules(root, [{"glob": "**/*.md", "metadata": {"kind": "note"}}])
        api.index(path=root)
        api.index_file(root / ".hidden.md", path=root)

        _write_rules(root, [{"glob": "**/*.md", "metadata": {"kind": "changed"}}])
        api.index(path=root)
        # The visible file heals and the run does not re-resolve forever.
        assert api.index(path=root)["metadata_refreshed"] is False

        api.index_file(root / ".hidden.md", path=root)
        store = Store(root / ".poma-memory.db")
        kinds = {Path(p_).name: m["kind"]
                 for p_, m in store.get_file_metadata_map().items()}
        assert kinds == {"a.md": "changed", ".hidden.md": "changed"}
        store.close()


# --- parser: the two remaining silent mis-parses ---

def test_a_nested_sub_key_is_unquoted_like_a_top_level_one():
    meta, ok = parse('---\nmeta:\n  "sub": v\n---\n')
    assert ok and meta == {"meta.sub": "v"}


@pytest.mark.parametrize("block", [
    '---\ntitle: "say \\\\"hi\\\\""\n---\n',
    '---\ntitle: "a \\\\" # b"\n---\n',
    "---\ntitle: 'it''s'\n---\n",
])
def test_escaped_quotes_are_refused_rather_than_half_processed(block):
    """Returning the backslashes literally is a wrong value, and `\\"` also
    closes the quote early and truncates the rest with no flag."""
    assert parse(block) == ({}, False)


# --- the COALESCE ordering, which a green suite did not defend ---

def test_metadata_survives_a_content_change_and_reindex():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "a.md").write_text("# A\n\nfirst.\n")
        _write_rules(root, [{"glob": "**/*.md", "metadata": {"kind": "note"}}])
        api.index(path=root)

        (root / "a.md").write_text("# A\n\nfirst. second, rewritten.\n")
        api.index(path=root)

        store = Store(root / ".poma-memory.db")
        rows = store._conn.execute(
            "SELECT metadata, fm_unparsed FROM files").fetchall()
        assert [dict(r) for r in rows] == [
            {"metadata": '{"kind": "note"}', "fm_unparsed": 0}]
        assert store.get_file_metadata_map() != {}
        assert store.count_files_without_metadata() == 0
        store.close()
        assert api.search("rewritten", path=root, where={"kind": "note"}) != []


def test_a_quoted_key_containing_a_colon_is_split_at_the_right_colon():
    """`partition(":")` cut inside the key and the whole block was refused."""
    meta, ok = parse('---\n"a:b": v\nplain: w\n---\n')
    assert ok and meta == {"a:b": "v", "plain": "w"}


def test_a_key_that_opens_a_quote_and_never_closes_it_is_unparsed():
    from poma_memory.frontmatter import _unquote_key
    assert parse('---\n"abc: v\n---\n') == ({}, False)
    assert _unquote_key('"abc') is None


# --- Stale rules: a rule edit changes what a file means, without touching it ---
#
# Every test here fails against a build that refuses only on `metadata = ''`.
# That was the shape of the bug through four rounds: the row is scanned, so
# nothing on the search path notices, and the filter answers from rules that no
# longer exist -- in both directions, since the predicate that used to match
# still does and the one that should now match does not.

_V1 = [{"glob": "DECISIONS.md", "metadata": {"kind": "decision"}},
       {"glob": "events/*.md", "metadata": {"kind": "event"}}]
_V2 = [{"glob": "DECISIONS.md", "metadata": {"kind": "architecture"}},
       {"glob": "events/*.md", "metadata": {"kind": "event"}}]


def _two_kinds(root: Path) -> None:
    (root / "events").mkdir()
    (root / "DECISIONS.md").write_text("# Decisions\n\nWe chose sqlite here.\n")
    (root / "events" / "e1.md").write_text("# Event\n\nDeployed sqlite today.\n")


def test_a_narrow_glob_after_a_rules_edit_refuses_instead_of_answering():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _two_kinds(root)
        _write_rules(root, _V1)
        api.index(root)
        _write_rules(root, _V2)
        api.index(root, glob="events/*.md")

        # Both directions of the wrong answer the old build gave: DECISIONS.md
        # answering to its deleted kind, and not answering to its current one.
        for where in ({"kind": "decision"}, {"kind": "architecture"}):
            with pytest.raises(MetadataStale):
                api.search("sqlite", path=root, where=where)
        # And the refusal clears once the rules are actually applied.
        api.index(root)
        hits = api.search("sqlite", path=root, where={"kind": "architecture"})
        assert [Path(h["file_path"]).name for h in hits] == ["DECISIONS.md"]


def test_adding_a_rules_file_without_reindexing_refuses():
    """No `index` run happens after the edit, so nothing can mark rows stale.

    Only comparing the rows against the rules file on disk catches this; a fix
    that invalidates rows during `index()` cannot, because `index()` never runs.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _two_kinds(root)
        api.index(root)
        _write_rules(root, _V1)
        with pytest.raises(MetadataStale):
            api.search("sqlite", path=root, where={"kind": "decision"})


def test_index_file_after_a_rules_edit_refuses():
    """The quietest variant: `index_file` prints nothing at all."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _two_kinds(root)
        _write_rules(root, _V1)
        api.index(root)
        _write_rules(root, _V2)
        api.index_file(root / "events" / "e1.md", path=root)
        with pytest.raises(MetadataStale):
            api.search("sqlite", path=root, where={"kind": "decision"})


def test_a_database_outside_the_indexed_directory_is_still_checked():
    """The rules root is stored on the row, so where the database lives is
    irrelevant. Inferring the root from the db location instead left every
    `--db` caller with no check at all, and the row-versus-row fallback that
    stood in for it was blind exactly when every row was stale together --
    which is the normal shape of a rule edit."""
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as td2:
        root, db = Path(td), Path(td2) / "x.sqlite"
        _two_kinds(root)
        _write_rules(root, _V1)
        api.index(root, db_path=db)
        _write_rules(root, _V2)
        # No re-index at all: every row is stale together, the case with no
        # internal disagreement to notice.
        for where in ({"kind": "decision"}, {"kind": "architecture"}):
            with pytest.raises(MetadataStale):
                api.search("sqlite", path=root, db_path=db, where=where)
        api.index(root, db_path=db)
        hits = api.search("sqlite", path=root, db_path=db,
                          where={"kind": "architecture"})
        assert [Path(h["file_path"]).name for h in hits] == ["DECISIONS.md"]


def test_two_indexed_roots_can_share_one_database():
    """Each row carries its own rules root, so two roots no longer look like
    one corpus disagreeing with itself -- which used to refuse forever, since
    neither run could restamp the other's rows."""
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, \
            tempfile.TemporaryDirectory() as td2:
        ra, rb, db = Path(a), Path(b), Path(td2) / "shared.sqlite"
        _two_kinds(ra)
        (rb / "OTHER.md").write_text("# Other\n\nsqlite elsewhere.\n")
        _write_rules(ra, _V1)
        _write_rules(rb, [{"glob": "*.md", "metadata": {"kind": "other"}}])
        api.index(ra, db_path=db)
        api.index(rb, db_path=db)
        hits = api.search("sqlite", path=ra, db_path=db, where={"kind": "decision"})
        assert [Path(h["file_path"]).name for h in hits] == ["DECISIONS.md"]
        hits = api.search("sqlite", path=ra, db_path=db, where={"kind": "other"})
        assert [Path(h["file_path"]).name for h in hits] == ["OTHER.md"]


@pytest.mark.parametrize("scenario", ["no_rules_ever", "db_elsewhere", "deleted_file"])
def test_consistent_indexes_are_not_refused(scenario):
    """The staleness check must not cost a correct index its answers.

    `deleted_file` is the one that bit: an orphaned row was stamped with the
    default empty `rules_hash`, which differs from every real hash forever, so
    a check like this one would refuse that index permanently.
    """
    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as td2:
        root = Path(td)
        _two_kinds(root)
        db = None
        if scenario != "no_rules_ever":
            _write_rules(root, _V1)
        if scenario == "db_elsewhere":
            db = Path(td2) / "x.sqlite"
        api.index(root, db_path=db)
        if scenario == "deleted_file":
            os.remove(root / "events" / "e1.md")
            api.index(root, db_path=db)
            assert api.index(root, db_path=db)["stale_rules"] == []
        # No exception, and an unfiltered search is unaffected either way.
        api.search("sqlite", path=root, db_path=db, where={"kind": "decision"})
        assert api.search("sqlite", path=root, db_path=db)


def test_a_dot_prefixed_row_can_be_healed_by_index():
    """`index` skips dot-names, so a row one put there was unreachable by any
    glob: every filtered search refused forever and the remediation printed
    could not work."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _two_kinds(root)
        _write_rules(root, _V1)
        (root / ".hidden.md").write_text("# Hidden\n\nsqlite notes.\n")
        api.index_file(root / ".hidden.md", path=root)
        # A row written before metadata existed at all (0.5.0), which is how
        # one of these gets stranded in practice.
        store = Store(root / ".poma-memory.db")
        store.set_file_metadata(os.path.realpath(root / ".hidden.md"), "")
        store.close()
        assert api.status(root)["files_without_metadata"] == 1
        api.index(root)
        assert api.status(root)["files_without_metadata"] == 0
        api.search("sqlite", path=root, where={"kind": "decision"})


def test_the_refusal_names_the_file_route_for_a_dot_prefixed_file():
    e = MetadataNotIndexed(1, "/db", ["/x/.hidden.md"])
    assert "index --file /x/.hidden.md" in str(e)
    assert "pass the glob" not in str(e)


def test_index_survives_a_file_that_is_not_utf8():
    """UnicodeDecodeError is a ValueError, not an OSError. Catching OSError
    alone let one byte abort the run and lose every file already indexed."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "a_good.md").write_text("# A\n\nreadable sqlite content.\n")
        (root / "z_bad.md").write_bytes(b"# Bad\n\n\xff\xfe not utf8\n")
        result = api.index(root)
        assert [Path(p).name for p in result["unreadable"]] == ["z_bad.md"]
        assert result["files_indexed"] == 1
        assert [Path(f).name for f in api.status(root)["files"]] == ["a_good.md"]


def test_a_deleted_document_stops_answering_filtered_searches():
    """The point of pruning rather than stamping: a caller narrowing to
    `kind=decision` reads the result as the current decision set, and a deleted
    document has no business in it."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _two_kinds(root)
        _write_rules(root, _V1)
        api.index(root)
        assert [Path(h["file_path"]).name for h in
                api.search("sqlite", path=root, where={"kind": "decision"})] \
            == ["DECISIONS.md"]

        (root / "DECISIONS.md").unlink()
        api.index(root)

        assert api.search("sqlite", path=root, where={"kind": "decision"}) == []
        assert not [h for h in api.search("sqlite", path=root)
                    if Path(h["file_path"]).name == "DECISIONS.md"]
        assert api.status(root)["stale_rules"] == []


@pytest.mark.parametrize("falsy", [[], "", 0, ()])
def test_a_falsy_non_dict_predicate_is_rejected_not_read_as_no_predicate(falsy):
    """Testing falsiness before type made `[]`, `""` and `0` mean "unfiltered",
    so the caller got the ENTIRE corpus back as a successful filtered answer --
    indistinguishable from a correct one. `[1, 2]` raised; `[]` did not."""
    with pytest.raises(ValueError, match="expected a dict"):
        normalize_where(falsy)


def test_an_empty_dict_is_a_predicate_that_constrains_nothing():
    assert normalize_where({}) is None
    assert normalize_where(None) is None


def test_a_falsy_non_dict_predicate_does_not_return_the_corpus_over_the_daemon():
    from poma_memory.server import _IndexCache, _handle
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _two_kinds(root)
        _write_rules(root, _V1)
        api.index(root)
        resp = _handle({"op": "search", "query": "sqlite", "path": str(root),
                        "db_path": None, "where": []}, _IndexCache())
        assert resp["ok"] is False and resp["code"] == "bad_where"
        assert "results" not in resp


@pytest.mark.parametrize("closing", ["--- ", "---\t", "---  \t"])
def test_trailing_whitespace_on_the_closing_fence_still_closes_a_block(closing):
    """The symmetry the opening-fence fix is justified by. Untested, it could
    be tightened to `line == FENCE` and take the opening tolerance's reason
    away without anything failing."""
    meta, ok = parse(f"---\nkind: decision\n{closing}\nbody\n")
    assert ok and meta == {"kind": "decision"}


@pytest.mark.parametrize("opening", ["--- ", "---\t", "---  \t"])
def test_trailing_whitespace_on_the_opening_fence_still_opens_a_block(opening):
    """The closing fence has always been `line.rstrip() == FENCE`; the opening
    one was not, so an invisible trailing space made the entire block vanish
    with ok=True -- no metadata, not flagged unparsed, invisible in status().
    PyYAML accepts it, so the document plainly has front-matter."""
    meta, ok = parse(f"{opening}\nkind: decision\n---\nbody\n")
    assert ok and meta == {"kind": "decision"}


def test_a_horizontal_rule_is_still_not_front_matter():
    """The tolerance must not swallow ordinary markdown."""
    assert parse("---foo\nkind: decision\n---\nbody\n") == ({}, True)
    assert parse("---") == ({}, True)


def test_a_file_whose_fence_has_trailing_space_is_filterable_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "a.md").write_text("--- \nkind: decision\n---\n\nsqlite notes.\n")
        (root / "b.md").write_text("# B\n\nother sqlite notes.\n")
        api.index(root)
        assert api.status(root)["unparsed_frontmatter"] == []
        hits = api.search("sqlite", path=root, where={"kind": "decision"})
        assert [Path(h["file_path"]).name for h in hits] == ["a.md"]


def test_a_renamed_file_does_not_refuse_the_index_forever():
    """R5's blocker. A row scanned under one rule set whose file is then
    renamed can be reached by NO glob -- the file does not exist -- so leaving
    its hash behind refused every filtered search permanently, with remediation
    text that could not be followed and `index --file` crashing on the vanished
    path. Gone rows are stamped against the current rules unconditionally."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _two_kinds(root)
        (root / "NOTES.md").write_text("# N\n\nsqlite notes.\n")
        _write_rules(root, _V1)
        api.index(root)
        os.rename(root / "NOTES.md", root / "NOTES-2026.md")
        api.index(root)
        _write_rules(root, _V2)
        api.index(root)
        assert api.index(root)["stale_rules"] == []
        hits = api.search("sqlite", path=root, where={"kind": "architecture"})
        assert [Path(h["file_path"]).name for h in hits] == ["DECISIONS.md"]


def test_a_deleted_file_does_not_refuse_the_index_forever():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _two_kinds(root)
        _write_rules(root, _V1)
        api.index(root)
        os.remove(root / "events" / "e1.md")
        api.index(root)
        _write_rules(root, _V2)
        api.index(root)
        assert api.status(root)["stale_rules"] == []
        api.search("sqlite", path=root, where={"kind": "architecture"})


@pytest.mark.parametrize("body,exc", [
    ('{"rules":[,]}', MetadataRulesError),
    ('{"rules": "nope"}', MetadataRulesError),
])
def test_a_broken_rules_file_refuses_at_search_time_without_a_traceback(body, exc):
    """Staleness is checked per query, so the rules file is now read on the
    search path. Left outside `MetadataIncomplete` it reached the CLI as a raw
    traceback; every surface catches the base."""
    from poma_memory.metadata import MetadataIncomplete
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _two_kinds(root)
        _write_rules(root, _V1)
        api.index(root)
        (root / ".poma-metadata.json").write_text(body)
        with pytest.raises(exc) as e:
            api.search("sqlite", path=root, where={"kind": "decision"})
        assert isinstance(e.value, MetadataIncomplete)
        # An unfiltered search does not touch the rules at all.
        assert api.search("sqlite", path=root)


def test_a_broken_rules_file_crosses_the_daemon_as_bad_rules():
    from poma_memory.server import _IndexCache, _handle
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _two_kinds(root)
        _write_rules(root, _V1)
        api.index(root)
        (root / ".poma-metadata.json").write_text("{not json")
        resp = _handle({"op": "search", "query": "sqlite", "path": str(root),
                        "db_path": None, "where": {"kind": "decision"}},
                       _IndexCache())
        assert resp["ok"] is False and resp["code"] == "bad_rules"


def test_the_cli_treats_bad_rules_as_an_answer_not_a_daemon_fault():
    """`bad_rules` was unreachable for the only shipped client: the code test
    matched `metadata_*`, so the CLI fell through to the in-process path and
    tracebacked anyway."""
    from poma_memory.cli import _REFUSAL_CODES
    assert "bad_rules" in _REFUSAL_CODES


@pytest.mark.parametrize("mutate", [
    lambda s: s + "\n",
    lambda s: s.replace("\n", "\r\n"),
    lambda s: json.dumps(json.loads(s), indent=2),
    lambda s: "﻿" + s,
])
def test_reformatting_the_rules_file_does_not_refuse(mutate):
    """The hash is over the parsed rules, not the bytes, so a trailing newline,
    a checkout's CRLF, a formatter's re-indent or an editor's BOM does not cost
    a re-index. A BOM also used to make the file invalid JSON outright."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _two_kinds(root)
        _write_rules(root, _V1)
        api.index(root)
        rf = root / ".poma-metadata.json"
        rf.write_text(mutate(rf.read_text()))
        hits = api.search("sqlite", path=root, where={"kind": "decision"})
        assert [Path(h["file_path"]).name for h in hits] == ["DECISIONS.md"]


def test_status_reports_staleness_rather_than_saying_complete():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _two_kinds(root)
        _write_rules(root, _V1)
        api.index(root)
        _write_rules(root, _V2)
        info = api.status(root)
        assert info["files_without_metadata"] == 0
        assert len(info["stale_rules"]) == 2
        api.index(root)
        assert api.status(root)["stale_rules"] == []


@pytest.mark.parametrize("ws", [" ", "\t", "\xa0", "\x0c", "\x0b", " "])
def test_any_trailing_whitespace_on_the_opening_fence_still_opens_a_block(ws):
    """The first cut allowed only space and tab, leaving NBSP, form feed,
    vertical tab and U+2003 silently vanishing -- the same defect one character
    class over. The closing fence's rstrip() accepts all of them."""
    meta, ok = parse(f"---{ws}\nkind: decision\n---\nbody\n")
    assert ok and meta == {"kind": "decision"}


@pytest.mark.parametrize("ws", ["\xa0", "\x0c", " "])
def test_the_closing_fence_accepts_the_same_whitespace_class(ws):
    meta, ok = parse(f"---\nkind: decision\n---{ws}\nbody\n")
    assert ok and meta == {"kind": "decision"}


def test_the_status_surfaces_name_both_ways_the_index_can_be_behind():
    """`status` is where a user goes when a filtered search refuses. Reporting
    only unscanned files printed "complete" in exactly that state."""
    from poma_memory import cli, mcp_server
    import io, contextlib
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _two_kinds(root)
        _write_rules(root, _V1)
        api.index(root)
        _write_rules(root, _V2)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli._cmd_status(argparse.Namespace(path=str(root), db=None))
        assert "earlier rule set" in buf.getvalue()
        assert "Metadata:  complete" not in buf.getvalue()

        (root / ".poma-metadata.json").write_text("{not json")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli._cmd_status(argparse.Namespace(path=str(root), db=None))
        assert "rules file unusable" in buf.getvalue()


def test_index_file_records_the_rules_root_too():
    """`index_file` is a full writer of rows, not a helper. Leaving the root off
    made rows it created unverifiable, and an unverifiable row is trusted only
    by guessing."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _two_kinds(root)
        _write_rules(root, _V1)
        api.index_file(root / "DECISIONS.md", path=root)
        store = Store(root / ".poma-memory.db")
        rows = store.scanned_rows_rules()
        store.close()
        assert [r[1] for r in rows] == [os.path.realpath(root)]
        # And the recorded root is what makes a later rules edit detectable.
        _write_rules(root, _V2)
        with pytest.raises(MetadataStale):
            api.search("sqlite", path=root, where={"kind": "decision"})


def test_a_row_with_no_recorded_root_counts_as_stale():
    """A scanned row that does not say which rules produced it cannot be
    verified. Trusting it is a guess; one re-index settles it."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _two_kinds(root)
        _write_rules(root, _V1)
        api.index(root)
        target = os.path.realpath(root / "DECISIONS.md")
        store = Store(root / ".poma-memory.db")
        # What a database written before provenance was recorded looks like:
        # metadata present, root blank.
        store._conn.execute(
            "UPDATE files SET rules_root = '' WHERE file_path = ?", (target,))
        store._conn.commit()
        rows = store.scanned_rows_rules()
        store.close()
        assert stale_files(rows) == [target]
        with pytest.raises(MetadataStale):
            api.search("sqlite", path=root, where={"kind": "decision"})
        api.index(root)
        api.search("sqlite", path=root, where={"kind": "decision"})


def test_a_held_hybrid_search_does_not_answer_from_a_stale_metadata_map():
    """The metadata map is read per query, not snapshotted in __init__. Split
    from the freshness check that guards it, a reindex landing between them
    passed the check against new rows while resolving the predicate from the
    old map.

    The metadata is changed WITHOUT touching chunks, because the BM25 corpus and
    embedding matrix are snapshots by design -- a re-chunk would empty the
    result either way and hide the difference.
    """
    from poma_memory.search import HybridSearch
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _two_kinds(root)
        _write_rules(root, _V1)
        api.index(root)

        store = Store(root / ".poma-memory.db")
        held = HybridSearch(store)
        assert held.search("sqlite", where={"kind": "decision"})

        target = os.path.realpath(root / "DECISIONS.md")
        record = store.get_file_record(target)
        store.set_file_metadata(target, json.dumps({"kind": "superseded"}),
                                False, record["rules_hash"], record["rules_root"])

        assert held.search("sqlite", where={"kind": "decision"}) == []
        assert held.search("sqlite", where={"kind": "superseded"})
        store.close()


@pytest.mark.parametrize("break_it", [
    lambda p: os.chmod(p, 0o000),
    lambda p: p.write_bytes(json.dumps({"rules": _V1}).encode("utf-16")),
])
def test_an_unreadable_rules_file_refuses_on_every_surface(break_it):
    """The READ needs the same guard the parse has. Left bare it raised
    PermissionError or UnicodeDecodeError, neither a MetadataIncomplete, so no
    surface caught it -- and `index`, the remedy the message recommends,
    tracebacked the same way."""
    from poma_memory.metadata import MetadataIncomplete
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _two_kinds(root)
        _write_rules(root, _V1)
        api.index(root)
        rf = root / ".poma-metadata.json"
        break_it(rf)
        try:
            for call in (lambda: api.search("sqlite", path=root,
                                            where={"kind": "decision"}),
                         lambda: api.index(root)):
                with pytest.raises(MetadataRulesError) as e:
                    call()
                assert isinstance(e.value, MetadataIncomplete)
                assert "cannot be read" in str(e.value)
        finally:
            os.chmod(rf, 0o644)


def test_the_daemon_reports_an_unreadable_rules_file_as_bad_rules():
    from poma_memory.server import _IndexCache, _handle
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _two_kinds(root)
        _write_rules(root, _V1)
        api.index(root)
        rf = root / ".poma-metadata.json"
        rf.write_bytes(json.dumps({"rules": _V1}).encode("utf-16"))
        resp = _handle({"op": "search", "query": "sqlite", "path": str(root),
                        "db_path": None, "where": {"kind": "decision"}},
                       _IndexCache())
        assert resp["ok"] is False and resp["code"] == "bad_rules"


def test_index_file_refuses_a_path_that_does_not_contain_the_file():
    """`path` supplies the rules AND is recorded as their source, so a file
    outside it would be stamped `{}` against that root's CURRENT hash: up to
    date forever, silently absent from every filtered result, `status`
    complete. The MCP tool reaches this with its own defaults."""
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        ra, rb = Path(a), Path(b)
        _two_kinds(ra)
        _write_rules(ra, _V1)
        _write_rules(rb, _V1)
        with pytest.raises(ValueError, match="is not inside"):
            api.index_file(ra / "DECISIONS.md", path=rb,
                           db_path=ra / ".poma-memory.db")
        # The legitimate call is untouched.
        assert api.index_file(ra / "DECISIONS.md", path=ra)["status"]


def test_a_second_root_going_stale_is_noticed():
    """Two roots sharing one database is the configuration round five fixed,
    and nothing exercised it: a `stale_files` that checked only the FIRST root
    passed the whole suite."""
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, \
            tempfile.TemporaryDirectory() as c:
        ra, rb, db = Path(a), Path(b), Path(c) / "shared.sqlite"
        _two_kinds(ra)
        (rb / "OTHER.md").write_text("# Other\n\nsqlite elsewhere.\n")
        _write_rules(ra, _V1)
        _write_rules(rb, [{"glob": "*.md", "metadata": {"kind": "other"}}])
        api.index(ra, db_path=db)
        api.index(rb, db_path=db)
        assert api.search("sqlite", path=ra, db_path=db, where={"kind": "decision"})

        # Edit ONLY the second root's rules. The first root is still current,
        # so a check that stops after one root sees nothing wrong.
        _write_rules(rb, [{"glob": "*.md", "metadata": {"kind": "moved"}}])
        with pytest.raises(MetadataStale):
            api.search("sqlite", path=ra, db_path=db, where={"kind": "decision"})
        assert [Path(p).name for p in api.status(ra, db_path=db)["stale_rules"]] \
            == ["OTHER.md"]


def test_reordering_keys_in_the_rules_file_does_not_refuse():
    """`sort_keys=True` is what makes the canonical hash survive a formatter,
    and `jq -S` reordering object keys is the case it exists for."""
    a = [{"glob": "*.md", "metadata": {"kind": "note", "area": "x"}}]
    b = [{"metadata": {"area": "x", "kind": "note"}, "glob": "*.md"}]
    assert rules_hash(a) == rules_hash(b)
    assert rules_hash(a) != rules_hash(
        [{"glob": "*.md", "metadata": {"kind": "note", "area": "y"}}])


def test_stale_files_checks_every_root_not_just_the_first():
    """Deterministic version of the two-root case. The integration test above
    feeds rows in `ORDER BY file_path`, so whether the stale root comes first
    depends on the temp directory names -- a check that stopped after one root
    passed it about half the time. Here the CURRENT root is first by
    construction, so stopping early can only return [].
    """
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        ra, rb = Path(a), Path(b)
        _write_rules(ra, _V1)
        _write_rules(rb, [{"glob": "*.md", "metadata": {"kind": "other"}}])
        current_a = rules_hash(_V1)
        rows = [
            (str(ra / "a.md"), str(ra), current_a),      # current, listed first
            (str(rb / "b.md"), str(rb), "0" * 64),       # stale, listed second
        ]
        assert stale_files(rows) == [str(rb / "b.md")]


def test_a_corrupt_rules_file_in_another_root_does_not_abort_this_index_run():
    """The end-of-run staleness warning re-reads OTHER roots' rules files,
    because rows in a shared database point wherever they came from. It used to
    raise past `store.close()` AFTER every file had been indexed and committed,
    so a corrupt file over there took down an unrelated run over here."""
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, \
            tempfile.TemporaryDirectory() as c:
        ra, rb, db = Path(a), Path(b), Path(c) / "shared.sqlite"
        _two_kinds(ra)
        (rb / "OTHER.md").write_text("# Other\n\nsqlite elsewhere.\n")
        _write_rules(ra, _V1)
        _write_rules(rb, [{"glob": "*.md", "metadata": {"kind": "other"}}])
        api.index(ra, db_path=db)
        api.index(rb, db_path=db)

        (rb / ".poma-metadata.json").write_text("{not json")
        result = api.index(ra, db_path=db)          # must not raise
        assert result["files_indexed"] >= 0
        assert result["stale_rules"] == []

        # The run completed and closed its store, so the database is usable.
        store = Store(db)
        assert store.all_file_paths()
        store.close()


def test_the_id_lookup_batches_correctly_across_the_parameter_limit():
    """`IN (?, ?, …)` is capped by SQLITE_MAX_VARIABLE_NUMBER, so the lookup
    batches. Driving the batch size down to 2 exercises the boundary on a small
    corpus instead of needing a thousand files."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for i in range(5):
            (root / f"f{i}.md").write_text(f"# F{i}\n\nsqlite content {i}\n")
        _write_rules(root, [{"glob": "*.md", "metadata": {"kind": "note"}}])
        api.index(root)
        store = Store(root / ".poma-memory.db")
        try:
            keep = set(store.get_file_metadata_map())
            assert len(keep) == 5
            whole = {cs for cs, _ in store.get_chunkset_files()}
            store._IN_BATCH = 2          # forces three batches
            assert store.chunkset_ids_for_files(keep) == whole
            assert store.chunkset_ids_for_files(set()) == set()
            one = next(iter(keep))
            assert store.chunkset_ids_for_files({one}) == {
                cs for cs, fp in store.get_chunkset_files() if fp == one}
        finally:
            store.close()


def test_the_id_lookup_matches_the_whole_table_scan_it_replaced():
    """Equivalence with the version that pulled every chunkset and filtered in
    Python -- the behaviour must be identical, only the cost different."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _two_kinds(root)
        _write_rules(root, _V1)
        api.index(root)
        store = Store(root / ".poma-memory.db")
        try:
            for meta in ({"kind": "decision"}, {"kind": "event"},
                         {"kind": "nothing-matches"}):
                keep = {p for p, m in store.get_file_metadata_map().items()
                        if matches(m, meta)}
                assert store.chunkset_ids_for_files(keep) == {
                    cs for cs, fp in store.get_chunkset_files() if fp in keep}
        finally:
            store.close()


def test_metadata_rows_answers_every_question_from_one_read():
    """Completeness, staleness and the predicate map all came from separate
    queries that could straddle a concurrent write and describe three different
    moments."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _two_kinds(root)
        _write_rules(root, _V1)
        api.index(root)
        store = Store(root / ".poma-memory.db")
        try:
            rows = store.metadata_rows()
            assert len(rows) == len(store.all_file_paths())
            assert [r[0] for r in rows] == sorted(r[0] for r in rows)
            # Same answers as the single-purpose queries it replaces.
            assert len([r for r in rows if not r[1]]) == \
                store.count_files_without_metadata()
            assert [(r[0], r[2], r[3]) for r in rows if r[1]] == \
                store.scanned_rows_rules()
        finally:
            store.close()


def test_the_id_lookup_really_issues_one_query_per_batch():
    """`_IN_BATCH` has to be honoured, not merely present: a version that put
    every path in one statement passed the behavioural test, because a corpus
    small enough to test by hand never reaches SQLITE_MAX_VARIABLE_NUMBER.
    Counting the statements is what pins it."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for i in range(5):
            (root / f"f{i}.md").write_text(f"# F{i}\n\nsqlite content {i}\n")
        _write_rules(root, [{"glob": "*.md", "metadata": {"kind": "note"}}])
        api.index(root)
        store = Store(root / ".poma-memory.db")
        try:
            keep = set(store.get_file_metadata_map())
            assert len(keep) == 5
            seen: list[str] = []
            store._conn.set_trace_callback(
                lambda sql: seen.append(sql) if "IN (" in sql else None)
            store._IN_BATCH = 2          # 5 paths -> 3 statements
            store.chunkset_ids_for_files(keep)
            store._conn.set_trace_callback(None)
            assert len(seen) == 3, seen
            # And each carries no more placeholders than the batch allows.
            assert all(sql.count("?") <= 2 for sql in seen), seen
        finally:
            store.close()


def test_a_row_whose_metadata_is_not_an_object_matches_nothing():
    """`metadata` is opaque JSON, and nothing stops a hand-edited row or a
    third-party writer putting a list there. `matches()` would raise on it, so
    the guard has to exclude rather than admit."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _two_kinds(root)
        _write_rules(root, _V1)
        api.index(root)
        target = os.path.realpath(root / "DECISIONS.md")
        store = Store(root / ".poma-memory.db")
        record = store.get_file_record(target)
        store.set_file_metadata(target, "[1, 2]", False,
                                record["rules_hash"], record["rules_root"])
        store.close()
        # No raise, and the malformed row is simply not a match.
        hits = api.search("sqlite", path=root, where={"kind": "decision"})
        assert [Path(h["file_path"]).name for h in hits] == []
        assert api.search("sqlite", path=root, where={"kind": "event"})


def test_the_batch_size_stays_under_the_oldest_parameter_limit():
    """SQLITE_MAX_VARIABLE_NUMBER is 32766 on current builds but 999 on older
    ones, and the failure is a hard `OperationalError` on exactly the corpora
    big enough to need batching. Raising this constant is safe only against the
    SQLite you happen to have."""
    assert Store._IN_BATCH <= 999


def test_an_edit_that_preserves_mtime_is_still_noticed():
    """Archive and sync tools that preserve timestamps carry a CHANGED file
    across with its old mtime. The row then keeps metadata the document no
    longer says, while `status` reports the index complete. A recorded size
    costs nothing -- the stat already happened -- and catches it."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "a.md").write_text("# A\n\nsqlite notes.\n")
        _write_rules(root, [{"glob": "*.md", "metadata": {"kind": "note"}}])
        api.index(root)
        assert [Path(h["file_path"]).name for h in
                api.search("sqlite", path=root, where={"kind": "note"})] == ["a.md"]

        before = os.stat(root / "a.md")
        (root / "a.md").write_text(
            "---\nkind: superseded\n---\n\n# A\n\nsqlite notes, revised.\n")
        os.utime(root / "a.md", (before.st_atime, before.st_mtime))
        assert os.stat(root / "a.md").st_mtime == before.st_mtime

        api.index(root)
        assert api.search("sqlite", path=root, where={"kind": "note"}) == []
        assert [Path(h["file_path"]).name for h in
                api.search("sqlite", path=root, where={"kind": "superseded"})] \
            == ["a.md"]


def test_a_legacy_row_with_no_recorded_size_is_healed_without_re_chunking():
    """A row from before the column records 0, so it reads as changed and is
    re-read once. It must NOT re-chunk or re-embed: the content hash still
    matches on the append path, so the file comes back "unchanged" and the
    size is recorded for good."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "a.md").write_text("# A\n\nsqlite notes.\n")
        _write_rules(root, [{"glob": "*.md", "metadata": {"kind": "note"}}])
        api.index(root)

        target = os.path.realpath(root / "a.md")
        store = Store(root / ".poma-memory.db")
        store._conn.execute("UPDATE files SET size_bytes = 0")
        store._conn.commit()
        before = store.get_file_record(target)
        store.close()

        result = api.index(root)
        assert result["files_indexed"] == 0, "must not re-chunk a legacy row"
        store = Store(root / ".poma-memory.db")
        after = store.get_file_record(target)
        store.close()
        assert after["content_hash"] == before["content_hash"]
        # Healed: the size is recorded now, so the check applies from here on.
        assert after["size_bytes"] == os.stat(root / "a.md").st_size


def test_the_size_check_does_not_re_chunk_an_unchanged_file():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "a.md").write_text("# A\n\nsqlite notes.\n")
        _write_rules(root, [{"glob": "*.md", "metadata": {"kind": "note"}}])
        api.index(root)
        for _ in range(3):
            assert api.index(root)["files_indexed"] == 0


def test_an_unmounted_or_unreadable_parent_never_prunes():
    """Pruning keys on FileNotFoundError alone. `_disk_state` answers "unknown"
    for a permission error or a dead mount, and deleting a corpus because a
    disk was not mounted is not a recoverable mistake."""
    from poma_memory.api import _disk_state
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        sub = root / "sub"
        sub.mkdir()
        (sub / "a.md").write_text("# A\n\nsqlite notes.\n")
        _write_rules(root, [{"glob": "**/*.md", "metadata": {"kind": "note"}}])
        api.index(root)
        target = os.path.realpath(sub / "a.md")

        os.chmod(sub, 0o000)
        try:
            assert _disk_state(target) == "unknown"
            result = api.index(root)
            assert result["pruned"] == []
        finally:
            os.chmod(sub, 0o755)
        store = Store(root / ".poma-memory.db")
        assert target in store.all_file_paths()
        store.close()


def _chunkset_indices(root: Path) -> dict:
    store = Store(root / ".poma-memory.db")
    try:
        rows = store._conn.execute(
            "SELECT file_path, local_index FROM chunksets "
            "ORDER BY file_path, local_index").fetchall()
    finally:
        store.close()
    out: dict = {}
    for r in rows:
        out.setdefault(Path(r["file_path"]).name, []).append(r["local_index"])
    return out


def test_appending_after_a_prune_does_not_collide_on_local_index():
    """`local_index` is UNIQUE per file, but the append path offset it by the
    GLOBAL chunkset count. While the corpus only grew that produced gaps; once
    pruning removes chunksets the count falls, later appends reuse indices the
    file already holds, and the insert dies mid-run with an IntegrityError."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "a.md").write_text("# A\n\nalpha one\n\n## S2\n\nalpha two\n")
        (root / "b.md").write_text(
            "# B\n\nbeta one\n\n## S2\n\nbeta two\n\n## S3\n\nbeta three\n")
        api.index(root)
        with open(root / "a.md", "a") as f:
            f.write("\n## S3\n\nalpha three\n")
        api.index(root)

        os.remove(root / "b.md")
        api.index(root)                      # prunes b.md, count drops

        for n, sec in enumerate(["S4", "S5", "S6", "S7"], start=4):
            with open(root / "a.md", "a") as f:
                f.write(f"\n## {sec}\n\nalpha {n}\n")
            api.index(root)                  # raised IntegrityError at S6

        got = _chunkset_indices(root)["a.md"]
        assert got == list(range(len(got))), got
        assert api.search("alpha", path=root)


def test_each_file_keeps_its_own_chunkset_sequence():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for name in ("a.md", "b.md"):
            (root / name).write_text(f"# {name}\n\n{name} one\n")
        api.index(root)
        for name in ("a.md", "b.md", "a.md"):
            with open(root / name, "a") as f:
                f.write(f"\n## More\n\n{name} again\n")
            api.index(root)
        got = _chunkset_indices(root)
        for name, indices in got.items():
            assert indices == list(range(len(indices))), (name, indices)


def test_a_database_with_gapped_indices_heals_instead_of_colliding():
    """`max + 1` is correct over a sequence that already has holes, so an
    index written by the old code continues rather than needing a migration."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "a.md").write_text("# A\n\nalpha one\n\n## S2\n\nalpha two\n")
        api.index(root)
        store = Store(root / ".poma-memory.db")
        store._conn.execute(
            "UPDATE chunksets SET local_index = 99 WHERE local_index = 1")
        store._conn.commit()
        store.close()
        assert _chunkset_indices(root)["a.md"] == [0, 99]

        with open(root / "a.md", "a") as f:
            f.write("\n## S3\n\nalpha three\n")
        api.index(root)
        assert _chunkset_indices(root)["a.md"] == [0, 99, 100]


def test_the_chunkset_index_sentinel_matches_the_chunk_one():
    """-1 for a file with no chunksets, so the first one it gets is 0. Mirrors
    `get_max_local_index`; a 0 sentinel would start the sequence at 1 and leave
    a permanent hole at the front."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "a.md").write_text("# A\n\nalpha one\n")
        api.index(root)
        store = Store(root / ".poma-memory.db")
        try:
            target = os.path.realpath(root / "a.md")
            assert store.get_max_chunkset_local_index("/no/such/file.md") == -1
            assert store.get_max_local_index("/no/such/file.md") == -1
            assert store.get_max_chunkset_local_index(target) == 0
            assert _chunkset_indices(root)["a.md"][0] == 0
        finally:
            store.close()
