# Metadata filtering (0.6.0)

The design document for `where`-filtered search: why the filter has to run
before ranking, how per-file metadata is resolved and kept honest, and what six
rounds of review changed about it.

Status: implemented on `feat/metadata-filtering` (PR #3). Written 2026-09-21
against 0.5.0 (c2de15e); rev 8.

It is kept because §2 and §8 are the parts that do not fit in code comments:
§2 is the argument for the whole mechanism, and §8 is a record of five
consecutive rounds where a fix created the next blocker, which is the most
useful thing this change has to teach.

Revision history: rev 4 recorded two corrections found while building it (§2,
§6); rev 5 narrowed an overclaim in the first and recorded the early review
findings; rev 6 replaced the storage design after round five (`files.rules_root`,
§3.1) and corrected §3.6/§3.7, which had described a cache that round five
deleted; rev 7 moved this file out of the (gitignored) `.agent/PLANS/` into the
repo, where the PR that cites it can actually reach it; rev 8 records round
seven, which found this document already contradicting the code on the one
behaviour that deletes data, two commits after rev 7 corrected it for the same
reason.

Revs 1-2 were built around a runbook repo as the driving consumer; the user has
since said that repo may not use poma-memory at all and that megavibe is the
case that matters. That inverts the design: **path-derived metadata is the
core, front-matter is a separable second phase.**

## 1. Problem

`api.search()` cannot restrict a query to a subset of the index. `path` only
computes the default `db_path`; the sole scoping unit is one database per
directory.

The driver is megavibe's `.agent/` corpus. Two things about it decide the
design. It carries **no front-matter** — `DECISIONS.md`, `LESSONS.md`,
`TASKS.md`, `FULL_CONTEXT.md` and `events/*.md` all open with a heading, none
with a `---` fence. And it already encodes type **structurally**, in filenames
and directories. So the metadata source that serves it needs no producer change
at all.

What is *not* the motivation: "only decisions". `DECISIONS.md`, `LESSONS.md`
and `TASKS.md` are three small tables, and filtering a search down to one file
a caller could `cat` does not justify a schema change. The win is where the
corpus is large and mixed:

- **Intent-scoped hook injection.** The Grep/Glob hook injects whatever
  `.agent/` matches, so a planning query competes against stale narrative for
  the same slots. Scoping lets the hook ask for lessons and decisions while
  planning, narrative while debugging. Zero producer change; improves the
  surface megavibe touches on every Grep.
- **`events/` provenance.** Write-once files, one producer (`agent-log.sh`).
  `kind: event` comes free from the path; `session`, `machine` and `date` need
  phase 2, and buy "what did the other Mac do while I was offline".
- **Umbrella multi-root sessions.** One database over several repos with a
  `repo` key would give unified ranking across roots. Today that is N databases
  and N searches whose RRF scores are not comparable — the only item here that
  is impossible rather than merely awkward.
- **The Claude memory directory** (`~/.claude/projects/*/memory/`), whose
  format already specifies `metadata.type: user | feedback | project |
  reference`, one fact per file. Front-matter by construction, needs phase 2.

## 2. The correctness crux

`HybridSearch.search()` (search.py:78-83) takes the empty gate from top-1
cosine over the **unfiltered** corpus:

    vec_hits = self._semantic.search(query, top_k=top_k * 3)
    top1 = vec_hits[0]["score"] if vec_hits else 0.0
    if top1 < gate: return []

While an index is homogeneous this is correct by accident. Once a filter exists,
applying it *after* ranking breaks both directions:

- **Phantom empty.** An out-of-scope document clears the gate, the filter then
  removes it, and the caller gets `[]` indistinguishable from "the corpus has
  no answer".
- **False suppression.** Out-of-scope bulk drags top-1 under the gate and an
  in-scope hit that was genuinely there is suppressed.

There is no defect today, because there is no filter. This is a constraint on
how the filter is added, not an independent reason to add one.

**Correction, found during implementation (rev 4, narrowed in rev 5).**
"False suppression" *by the gate* cannot happen. `vec_hits[0]` is a maximum
over the corpus, and a maximum over a superset is never smaller than over a
subset — so out-of-scope documents can only push top-1 *up*, never under the
gate. **At the gate comparison, and only there**, the divergence is
one-directional: post-filtering is too permissive, never too strict.

Rev 4 stated that one-directionality of the whole search, which is wrong. The
second failure is real, and it is the strict direction: the candidate list is
`top_k * 3` deep, so enough out-of-scope documents fill it and in-scope
documents never enter the ranking at all — **crowding out**. Post-filtering
returns nothing where pre-filtering returns the document. It affects BM25 and
the semantic side equally, and pre-filtering fixes it for the same reason.

Both are covered by tests, and both were confirmed to fail against a
deliberately post-filtering build of `HybridSearch.search`.

**The gate is semantic, so masking BM25 fixes nothing on its own.** §3.6 masks
the cosine vector before `argsort` and the gate reads the masked result. BM25
masking is for recall.

## 3. Core

### 3.1 Storage

    ALTER TABLE files ADD COLUMN metadata TEXT NOT NULL DEFAULT ''
    ALTER TABLE files ADD COLUMN fm_unparsed INTEGER NOT NULL DEFAULT 0
    ALTER TABLE files ADD COLUMN rules_hash TEXT NOT NULL DEFAULT ''
    ALTER TABLE files ADD COLUMN rules_root TEXT NOT NULL DEFAULT ''
    ALTER TABLE files ADD COLUMN size_bytes INTEGER NOT NULL DEFAULT 0
    ALTER TABLE files ADD COLUMN ctime REAL NOT NULL DEFAULT 0

`files` is the natural home: metadata is per-file, and unlike
`chunks`/`chunksets` it has no FTS5 external-content triggers hanging off it.
Nothing is added to `chunksets` — they already carry `file_path`.

Named `metadata`, not `frontmatter`: two sources populate it and a rename after
release costs a migration.

**Superseded: there is no `index_meta` table.** Rev 4 put the rules hash in a
one-key `index_meta` table, as a property of the index. Round three deleted it:
one global flag produced four different wrong answers in three rounds (§8), and
a per-row column makes that class unrepresentable rather than guarded four
times.

**`rules_root` is a fourth column, added in round five.** A per-row hash still
leaves the question of *which* rules the row should be compared against. Round
four inferred that from the database's location — rules live beside the
database — which left every `--db` caller unchecked and made two roots sharing
one database refuse forever, since neither run could restamp the other's rows.
The row records the directory its rules came from, so the answer is carried
rather than deduced, and the database's location stops being evidence. All six
columns migrate through `_migrate()`.

**Deviation from rev 3, which said one new column.** `fm_unparsed` is a second.
`status()` has to name the files whose front-matter fell outside the grammar
(§4), and that is per-file state: on the row it is consistent with
`delete_file_data` for free, where a JSON list in a global table would need a
read-modify-write on every file and could lose entries when two sessions index
at once. The same argument later took the rules hash onto the row too.

Three states on the column, and the distinction is what makes §3.2 possible:

| Value | Meaning |
|---|---|
| `''` | never processed by a metadata-aware indexer (legacy row) |
| `'{}'` | processed; no rule matched and no front-matter |
| `'{"k": ...}'` | processed; opaque JSON |

### 3.2 Completeness is derived, not flagged

A per-database "metadata enabled" flag can lie: set it true, let an older
poma-memory write rows, it still reads true. Derive it:

    SELECT COUNT(*) FROM files WHERE metadata = ''

Non-zero **and** a predicate supplied → raise `MetadataNotIndexed`, naming the
count and the command that fixes it, before any ranking. Zero → an empty result
is an honest "no document matches", not "we never looked".

A predicate on a key no file carries returns `[]`, not an error.

### 3.3 Source 1: path rules — the core of this PR

A JSON file at the root of the indexed directory. For `.agent/`:

    // .agent/.poma-metadata.json
    {
      "rules": [
        {"glob": "events/**/*.md", "metadata": {"kind": "event"}},
        {"glob": "DECISIONS.md",   "metadata": {"kind": "decision"}},
        {"glob": "LESSONS.md",     "metadata": {"kind": "lesson"}},
        {"glob": "TASKS.md",       "metadata": {"kind": "task"}},
        {"glob": "FULL_CONTEXT.md","metadata": {"kind": "narrative"}},
        {"glob": "**/*.md",        "metadata": {"kind": "note"}}
      ]
    }

- **First matching rule wins**, in document order. Predictable, and it lets a
  catch-all sit last. "Most specific wins" has no crisp definition.
- **Globs are resolved with `Path.glob`, the same engine the indexer already
  uses** — one `set(root.glob(rule["glob"]))` per rule at index time, then each
  file takes the first rule whose set contains it. No hand-rolled `**` matching
  and therefore no semantic divergence from the `--glob` the caller passed.
  Cost is one walk per rule; with six rules over a `.agent/` tree it is noise.
- **The library interprets nothing.** `"kind"` and `"event"` are the caller's
  words. poma-memory stores and compares them.
- JSON, not TOML: `tomllib` is 3.11+, and `requires-python` is `>=3.10`.
- The file is not indexed — `index()` already skips names starting with `.`,
  and `**/*.md` would not match it anyway.

New module `poma_memory/metadata.py`: load the rules file, resolve globs to a
`path -> dict` map, merge sources.

This is what serves the hook-scoping and umbrella cases with no producer change.

### 3.4 The predicate — `where`

    search(query, where={"kind": ["decision", "lesson"]})

- AND across keys, OR within a list value.
- A file matches a key if its metadata has that key and the value is in the
  allowed set. A list-valued metadata field matches on non-empty intersection.
- Case-sensitive string equality. Case-folding would be interpretation.
- No operators, no negation, no ranges, no nesting.

A filter DSL turns poma-memory into a worse Qdrant; its differentiator is
hierarchical chunking and context assembly.

**Documented limit: the surface cannot express "key is absent" or
"kind != narrative".** The answer is a producer convention — always emit the
key, filter positively. Recorded so it is a decision rather than something the
first user discovers. Negation stays out until a case appears that a convention
cannot cover.

### 3.5 Rule edits are a new staleness source

Front-matter edits move a file's mtime, so `update_file()` re-reads them.
Editing `.poma-metadata.json` does not touch any indexed file, and
`update_file()` short-circuits on mtime equality (incremental.py:30), so every
row would keep metadata resolved from the old rules — silently.

`index()` records, on each row it reaches, `sha256` of the **parsed** rules
(`files.rules_hash`) and the directory they came from (`files.rules_root`).
Hashing the parsed form rather than the bytes means a reformat with no change
of meaning does not force a re-index. Storing the root means a search re-reads
*that* directory's rules: inferring it from the database's location left `--db`
callers unchecked and made two roots sharing one database refuse forever. When a row's hash differs from the current one, or its
metadata is `''`, re-resolve and `UPDATE` the column for that file. Per row, not
per index: a run given a narrower `--glob` reaches part of the corpus, and a
global flag cannot express "this row was covered but that one was not" — which
is exactly how rounds one to three each produced a different wrong answer (§8). **No re-chunking and no re-embedding** — `_full_reindex` would delete,
re-chunk and re-embed, which is real money on the OpenAI backend.

That makes both the upgrade path and the rule-edit path the same command
callers already run: `poma-memory index`. No `--backfill-metadata` flag.

**A row whose file no longer exists on disk is PRUNED** — chunks, chunksets
and the row. Earlier revisions stamped it `'{}'`; that left a deleted document
answering searches, and a row no glob can reach can never be corrected, so
whatever was stamped on it was permanent.

Pruning is the only destructive operation in the package and is gated three
ways, each of which was a reproduced way to delete live data: the run's own
root must be present (an unmounted volume yields ENOENT, not "some other
OSError", so every row under it read as deleted and a whole corpus went); only
rows under that root are candidates (two roots may share one database, and a
run given A must not delete B); and the disk state is re-checked immediately
before each delete, because it was sampled for every row before any were
removed. `size_bytes` and `ctime` are what let an edit be noticed at all when
the timestamp was restored — see `incremental._stat_agrees`.

### 3.6 The pre-filter mechanism

`HybridSearch.search(query, ..., where=None)`:

1. Resolve `where` → `allowed_ids: set[int]` of chunkset ids. Everything
   metadata reads comes from **one `metadata_rows()` scan per query** (round
   five/six): completeness, staleness and the file→metadata map describe the
   same moment, where three separate queries could straddle a concurrent write
   and describe three. The ids themselves are then resolved in SQL rather than
   by scanning every chunkset in Python — a whole-table scan cost 22 ms per
   filtered query however narrow the predicate was, against 0.12 ms for a 1%%
   filter (#4).
2. If `where` is set and any `files.metadata = ''` → raise
   `MetadataNotIndexed`; if any row's `rules_hash` differs from what its
   `rules_root` says now → raise `MetadataStale`. Both before any ranking, and
   both `MetadataIncomplete`, which is what every surface catches.
3. **Semantic — the step that fixes §2.** In `_EmbedderBase.search`, set
   disallowed rows of the cosine vector to `-inf` **before** `np.argsort`.
   `vec_hits[0]` is then the top-1 of the narrowed corpus and the gate in
   `HybridSearch.search` reads that value with its own code unchanged.
4. **BM25 — recall, not the gate.** Build a `weight_mask` in
   `BM25Search._chunksets` order (1.0 allowed, 0.0 not) and pass it to
   `bm25s.BM25.retrieve(..., weight_mask=...)`; bm25s multiplies it into the
   score vector before top-k, so ranking among in-scope documents is exact.
   Masked documents can still surface with score 0.0 when fewer than k score
   positive, so hits are dropped by **id-set membership, never by score** — an
   in-scope document can legitimately score 0.0.

`weight_mask` verified present in **bm25s 0.3.0**, not merely the installed
0.3.3: the 0.3.0 wheel was pulled from PyPI and its `bm25s/__init__.py` parsed
with `ast`, showing the parameter on `retrieve`, `get_scores`,
`get_scores_from_ids` and `_get_top_k_results`, and `scores *= weight_mask` in
the body. `bm25s>=0.3,<1` is already the pin: no dependency change, no floor
bump.

### 3.7 `_IndexCache` interaction — none, by construction

Masking at query time means neither the BM25 index nor the embedding matrix is
rebuilt per predicate, so `server.py`'s long-lived `HybridSearch` per database
is untouched. Per-query cost is one O(n_chunksets) numpy array. This is the
main reason to prefer `weight_mask` over rebuilding a narrowed index.

**Revised in round five: the file→metadata map is not cached at all.** Rev 4
put it on `HybridSearch`, rebuilt with the cache on a `PRAGMA data_version`
change. That is wrong in both directions. Editing the rules file writes nothing
to the database, so `data_version` never moves for it and the cached map would
keep vouching for rules that are gone — the daemon would answer where a cold
search refuses. And splitting the freshness check from the map it guards let a
reindex landing between them pass the check against new rows while resolving the
predicate from the old map. It is one read per query now, which is also cheaper
than the three separate queries it replaced.

The BM25 index and the embedding matrix remain snapshots on `HybridSearch`:
they are expensive, and `data_version` is the right signal for them because
they only change when the database does.

### 3.8 Surfaces

| Layer | Change |
|---|---|
| `api.search()` | `where: dict \| None = None` |
| `HybridSearch.search()` | `where` → `allowed_ids` before ranking |
| `BM25Search.search()` | `allowed_ids: set[int] \| None` |
| `_EmbedderBase.search()` | `allowed_ids: set[int] \| None` |
| `server.py _handle` | unpack `where`; error shape below |
| `cli.py` | `--where key=value`, repeatable; repeats of one key → OR, different keys → AND |
| `mcp_server.py` | `where: dict \| None` on `poma_search` |
| `status()` | `files_without_metadata: int`, `unparsed_frontmatter: [paths]` |

`MetadataNotIndexed` exported from `poma_memory`.

The CLI and daemon surfaces are in v1 even though the in-process API would
serve a Python caller alone, because megavibe drives both the CLI and the MCP
server, and because `cli.py` already goes to visible trouble resolving
`empty_gate` client-side to stop the daemon and in-process paths disagreeing.

**Daemon error shape.** `MetadataNotIndexed` has to cross a JSON socket:

    {"ok": false, "code": "metadata_not_indexed", "error": "<message>",
     "files_without_metadata": N}

`cli.py`'s daemon client re-raises on that `code` rather than falling through
to the in-process path — falling through would raise the same thing ~0.5s later
after a second model load. `code` is additive; every other failure keeps the
existing shape and old clients ignore it.

Version 0.5.0 → 0.6.0: additive API, additive schema.

## 4. Phase 2: front-matter (separable, may slip)

A second source for the same column, for the `events/` provenance and Claude
memory-directory cases. Nothing in §3 depends on it, and it can ship in 0.6.1
without rework.

New module `poma_memory/frontmatter.py`. A `---` fence at byte 0, terminated by
a `---` line. Grammar:

- `key: scalar`
- `key:` followed by indented `- item` lines → list
- `key:` followed by one level of indented `sub: value` → flattened to
  `key.sub`

**One nesting level stays in.** Rev 2 recommended flat-only to kill indentation
ambiguity; the Claude memory format writes `metadata:` / `  type: feedback`, so
flat-only would exclude one of the four motivating cases. Flattening to a
dotted key is structural, not semantic — the library still interprets no key.

**Every value stays a string. No coercion of `true`/`yes`/`1`/numbers.** The
predicate is case-sensitive string equality, so coercion opens a silent gap
between what an author wrote and what they can filter on.

Out-of-grammar input — block scalars (`|`, `>`), anchors, flow mappings,
multi-document streams — marks the file **unparsed**: stored as `'{}'`, with
`status()` returning `unparsed_frontmatter: [paths]`. The paths, not a count: a
count says something is wrong, the paths say what to fix. Silent mis-parse is
what normally makes a hand-rolled YAML subset a bad trade; a visible failure is
what makes it defensible.

No PyYAML. Four dependencies is a deliberate property of this package.

**Merge order: front-matter wins per key over path rules.** A file that matches
`{"kind": "event"}` and also declares `kind:` in its own front-matter keeps its
own.

Backfill: on the mtime short-circuit path, read only the leading fence block
(bounded at 8 KiB) and `UPDATE`.

**Corrected:** rev 4 said "editing front-matter always moves mtime, so
`update_file()` covers every case except legacy rows". It does not.
`update_file` short-circuits on mtime equality, and `cp -p`, `rsync --times`,
`tar -p` and restore tooling all preserve mtime across a content change — so an
edited document keeps its old metadata while `status` reports complete. Legacy
rows and rule edits are covered (§3.5); a content edit that does not move mtime
is not. Open, with the rest of the follow-ups, in #4.

## 5. Out of scope

**Capture groups in path rules.** `events/` filenames encode a timestamp and an
agent name (`20260921T082109Z-AgentBetaX-09443f.md`), so a regex with named
groups would yield `date` and `session` with no producer change. Rejected: it
is the first step to a matching DSL, and `agent-log.sh` writes those files and
already knows both values, so stamping front-matter (phase 2) is both simpler
and more honest.

**Stripping front-matter from indexed text.** Confirmed: nothing parses it
today, so it is chunked as prose and pollutes BM25 tokens and embeddings.
Stripping changes chunk contents and indices and forces a full reindex of every
corpus — re-embedding is real money on OpenAI. Own PR, own version bump.

**`PRAGMA user_version` retrofit.** §3.2 derives completeness from data, so
this feature needs nothing from it, and the new columns arrive by
`CREATE TABLE IF NOT EXISTS` rather than a third blind ALTER. Inferring
existing databases' versions by probing columns is cleanup with its own risk.

**Producer-side megavibe work.** Authoring `.agent/.poma-metadata.json` into
the megavibe template, and stamping `agent-log.sh` events with front-matter,
are changes to the megavibe repo, not this one.

**`_incremental_update` global-count bug.** See §7.

## 6. Tests

`tests/test_metadata.py`
- rules file: first-match-wins ordering, catch-all last, a file matching no
  rule → `'{}'`, absent rules file → every file `'{}'`, malformed JSON → error
  naming the file rather than a silent skip.
- glob semantics match `Path.glob`, including `events/**/*.md` and a bare
  filename.
- rules-hash staleness: edit the rules file without touching any `.md`,
  re-run `index()`, assert metadata re-resolved and chunk/chunkset rows
  unchanged (no re-chunk, no re-embed).

`tests/test_filtering.py`
- migration: open a 0.5.0-created db; columns added, old
  rows still read.
- `MetadataNotIndexed` raised when filtering a legacy db; not raised after
  `index()` heals it; crosses the daemon socket as `code:
  metadata_not_indexed` and the client re-raises rather than falling through.
- **Crux A (phantom empty)**: unfiltered top-1 clears the gate but is out of
  scope, and an in-scope document also answers. Must return the in-scope hit,
  not `[]`.
- **Crux B (crowding out)**: enough out-of-scope documents to fill the
  `top_k * 3` candidate window, plus one in-scope document that answers. Must
  return the in-scope hit. (Replaces "false suppression", which §2 shows is
  impossible.)
- **Equivalence**: `search(q, where=W)` over the full corpus == `search(q)`
  over a corpus built from only the W-matching files. Catches gate-ordering by
  construction rather than by a test someone remembered to write.
  **Membership, not order**: BM25 IDF is computed over whatever corpus is
  present, so the two runs can order the same in-scope documents differently.
  The result set and the gate decision are what must not depend on excluded
  documents. Order equality is asserted separately on the semantic path, where
  a cosine is absolute and corpus-independent.
- **Equivalence, no-answer case**: same pairing where nothing in the W subset
  is relevant. Both return `[]`, and the filtered one must not reach `[]` via a
  gate that saw out-of-scope documents.
- BM25 mask: an in-scope document that legitimately scores 0.0 is not dropped.
- daemon: `where` round-trips; two different `where` values against one
  database leave `cache.builds` unchanged.
- CLI `--where` parsing, including repeated keys.

`tests/test_frontmatter.py` (phase 2)
- flat scalars, lists, one-level nesting, absent block, `---` not at byte 0,
  CRLF, unterminated fence, out-of-grammar → unparsed and named.
- values that look like booleans or numbers stay strings.
- front-matter overrides a path rule on the same key.

## 7. Reported, then fixed here

**Superseded.** This section recorded `_incremental_update`'s global-count
offset as latent, narrow and unreproduced, to be handled in a separate PR. It
was none of those once pruning landed: removing a deleted file's chunksets is
exactly what makes the global count fall, so the append path started reusing
`local_index` values the file already held. Reproduced to a hard
`sqlite3.IntegrityError` in six steps and fixed on this branch with the
per-file `MAX(local_index) + 1` this section proposed.

Worth keeping as written rather than deleting: "narrow, unreproduced" was a
judgement about reachability made before a later change altered it, and the
lesson is that such judgements expire.


## 8. Review rounds

Reviewed by the `reviewer` subagent and Gemini (`--as-reviewer --pro`). Codex
is installed but outside `MEGAVIBE_REVIEWERS` on this machine, so it did not
run — a setting, not an outage.

**Round 1 — do-not-ship, 13 findings.** Two substantive.

1. `index()` erased metadata for files outside the run's glob. `fp not in seen`
   means "not matched by this run's `--glob`", not "gone from disk", so a
   partial reindex stamped live files `'{}'` and printed a false "missing from
   disk" warning. Nothing corrected it, because `'{}'` looks done.
2. Neither mask had a test. Both the semantic `-inf` mask and the BM25
   `weight_mask` could be deleted with all 101 tests green, because the four
   crux tests drive a stub embedder that reimplements the masking they check.

Plus three silent wrong parses (trailing `# comment` kept in the value, inline
list split inside quotes, UTF-8 BOM disabling front-matter unflagged), a
non-dict `where` escaping the daemon's `code:` contract, absolute and `..`
globs, MCP `poma_status` not surfacing completeness, `status()` omitting keys
with no database, and a head/full read disagreement.

**Round 2 — do-not-ship, 9 findings.** The blocker was introduced by round 1's
own fix, from the other side.

1. `index()` still advanced the rules hash unconditionally. Edit
   `.poma-metadata.json`, then run `index` with a narrower `--glob`, and the
   rest of the corpus keeps metadata resolved against superseded rules: those
   rows are not `''` so the legacy heal skips them, and the hash is current so
   the refresh path skips them too. `status` reported complete. Before the
   round-1 fix those rows were wrongly stamped `'{}'`; after it they were
   wrongly left stale — the same indistinguishable-wrong-answer, relocated.
   The hash now advances only when the run reached every live row.
2. `os.path.exists` is False for a permission error, an unmounted volume and a
   dead network mount, not only for deletion. Disk state is three-valued now;
   `unknown` leaves the row unscanned rather than calling it deleted.
3. `strip_comment` and `_split_inline` treated an apostrophe **anywhere** as an
   opening quote, re-opening two of round 1's three parser fixes:
   `title: Don't ship  # decided` kept the comment, `tags: [don't, can't]`
   collapsed to one item. A quote opens only at the start of a value or item.
4. Three round-1 fixes claimed "covered by a test" had none, and
   `MetadataNotIndexed.examples` was dead code whose comment described an
   outcome no user ever saw.

Plus more flow shapes (quoted keys, nested flow sequences, unterminated
brackets, phantom empty items, YAML merge keys), Windows-style globs bypassing
the posix-only rules guards, the head/full disagreement in the remaining
direction, and an `except ValueError` wide enough to label a corrupt
`chunk_ids` blob as a bad predicate.

**Coverage is mutation-verified, not asserted.** Every fix above was re-checked
by deleting it from a scratch copy and running the suite, each mutant killed by
a test named for the defect. That check is the reason this section can claim
coverage at all — round 1 made the same claim from a green suite and was wrong
about three of them. Rounds 1–2: 12 of 12. Round 4: 21 of 21, over three
passes. Round 5: 19 of 19, over two. Round 6: 8 of 8, then 10 of 10 for the
follow-up work in #4.

Almost every one of those passes needed a second run, and the reasons are worth
keeping: a mutant that changed no behaviour and had to be rebuilt as a
multi-point revert; a test that passed by luck of row ordering; a coverage gap
in a branch reached only by a corpus larger than any test builds. A first pass
that kills everything is the outcome to distrust.

**Round 3 — do-not-ship, both reviewers.** `_resolve_metadata` caught `OSError`
and set `text = ""`, so an unreadable file parsed as "has no front-matter" and
was stamped with path-rule metadata alone, flagged parsed, counted scanned.
Round 2 had fixed the out-of-glob branch and left the in-glob one producing the
same outcome. The response was the redesign in §3.5: the rules hash moves onto
the row and `index_meta` is deleted.

**Round 4 — do-not-ship, both reviewers, and the redesign did not close the
class.** `index()` warned about rows it could not reach, but `search` never
consulted the rules file, so a filtered query still answered from superseded
rules while `status` reported complete — in both directions, since the deleted
predicate still matched and the current one did not. Three triggers: a narrower
`--glob` after a rules edit, `index_file` after one (silent, no warning at all),
and a rules file *added* after indexing, which no index-time invalidation can
reach because `index` never runs. `search` now compares the rows against the
rules file (`metadata.current_rules_hash`) and raises `MetadataStale`, falling
back to row-versus-row hash disagreement when `path` does not own the database.
**Both of those were replaced in round five and no longer exist** — see below
for why the fallback was the wrong shape.

The fix had the same trap inside it. Captured in `HybridSearch.__init__`, the
hash would have gone stale in the daemon forever: the cache is keyed on
`PRAGMA data_version`, and editing the rules file writes nothing to the
database, so the object is deliberately never rebuilt. It is a per-call
argument, pinned by a test that asserts `cache.builds` does not move while the
refusal fires.

Also round 4: `UnicodeDecodeError` is a `ValueError`, so one Latin-1 byte still
aborted the whole run and lost every file already indexed; `index()` skipped
dot-prefixed names unconditionally, so a row `index --file` created was
reachable by no glob and refused every filtered search forever while printing
remediation that could not work; orphan rows were stamped `rules_hash=''`, which
differs from every real hash and would have refused those indexes permanently
under the new check; `normalize_where` tested falsiness before type, so `[]`,
`""` and `0` meant "no predicate" and returned the entire corpus as `ok: true`;
and the opening `---` fence rejected trailing whitespace while the closing one
`rstrip`s it, so `"--- "` made a whole block vanish with `ok=True`, invisible in
`status()`. The README sentence promising refusal on a rules edit was corrected
— it had described an outcome the code did not produce.

**Round 5 — do-not-ship, both reviewers, and the round-4 fix produced both
blockers.** Inferring the rules root from the database's location meant
`current_rules_hash` returned `None` whenever the database was not beside the
rules file, and the row-versus-row fallback that stood in for it is blind
exactly when every row is stale together — which is the normal shape of a rule
edit. So any caller passing `--db` got no check at all: the deleted predicate
kept matching and the current one returned `[]`. The same commit's orphan fix
stamped only rows that had *never* been scanned, so a row scanned under earlier
rules whose file was then deleted or renamed kept that hash forever, reachable
by no glob, refusing every filtered search permanently while `status` said
complete and `index --file` crashed on the vanished path.

The answer was to replace the mechanism rather than patch it a fifth time:
`files.rules_root` (§3.1), which closes both at once and deletes the fallback
and its "which side is current" hedging. Gone rows were stamped
unconditionally at that point; round seven replaced that with pruning (§3.5).

That fix had the same shape of bug inside it too: `rules_root` was left out of
`update_file`'s refresh test, so a row with metadata and a matching hash but no
recorded root was refused by search *and* skipped by the refresh — unhealable.
Caught by the test written for the blank-root case, not by reading the code.

Also round 5: the canonical hash moved from the file's bytes to the parsed
rules, so a reformat with no change of meaning stopped forcing a re-index;
`load_rules` reads `utf-8-sig`; `MetadataRulesError` became a
`MetadataIncomplete`; `status` reports staleness instead of printing
"complete" in exactly the state where every filtered search exits 2; and the
fence tolerance widened from space-and-tab to `str.strip()`, because the
narrower set left NBSP, form feed, vertical tab and U+2003 silently vanishing.

**Round 6 — do-not-ship, both reviewers, converging on one blocker.**
`load_rules` read the file *outside* the `try` that produces
`MetadataRulesError`, so `OSError` and `UnicodeDecodeError` escaped raw. Round
five had converted the `JSONDecodeError` branch of that same function into the
clean `bad_rules` refusal and left the read branch two lines above it throwing
— then routed it onto every surface by making staleness a per-query read. A
rules file saved as UTF-16 by a Windows shell tracebacked out of `status`,
`search` and `index`, the last being the remedy the message recommends. Over
the daemon the generic handler returned an error with no `code`, so the CLI did
not recognise it, fell through and tracebacked too.

Round 6 also found `index_file` never checking that the file lies under `path`:
since `path` supplies the rules and is recorded as their source, a file outside
it was stamped `{}` against that root's *current* hash — up to date forever,
silently absent from every filtered result, `status` complete. Reachable from
the MCP tool with its own defaults.

Two tests were added for claims the suite could not fail on: two roots sharing
one database, and `sort_keys` in the canonical hash. The first attempt at the
multi-root test passed by luck, because rows arrive `ORDER BY file_path` and
whether the stale root came first depended on the temp directory names; the
deterministic version builds the rows directly.

**The pattern, stated once.** Five of six rounds found a blocker that the
previous round's fix had created. Three mechanisms were tried before one held:
a global flag, a per-row hash, and finally per-row provenance. What separates
the last from the first two is not care — all three were written carefully —
but that it makes the wrong answer unrepresentable instead of guarding against
it. The recurring tell was a fix that narrowed one branch of a condition and
left its sibling: the out-of-glob branch without the in-glob one, the parse
branch without the read branch, space-and-tab without the rest of the
whitespace class.

Two mutants survive deliberately and are not defects: the id-membership drop in
`_EmbedderBase.search` (unreachable while the `-inf` mask stands, since
`-inf < min_score`) and the empty-`allowed_ids` early return in
`HybridSearch.search` (an all-zero mask yields no hits, so the gate returns
`[]` anyway). Both are redundant defence, kept because they hold if the other
mechanism is ever changed.

Confirmed clean across rounds 1–2, with work shown: the `COALESCE` parameter
ordering in `upsert_file_record`; `-inf` safety against `min_score`, `argsort`,
NaN and an all-excluded corpus; that the BM25 mask order cannot diverge from the
indexed corpus; `_IndexCache` invalidation versus the then-cached
`HybridSearch._file_meta` map (round five deleted that cache — §3.7); `index_file` versus `index` on path rules; `status()` versus
`MetadataNotIndexed`; the append path's metadata handling; and the rules-hash
refresh across edit/narrow/delete/re-add.

One latent caveat recorded rather than fixed: `bm25s.get_scores_from_ids` adds
its `nonoccurrence_array` *after* `scores *= weight_mask`, so under
`method="bm25l"` or `"bm25+"` a masked document would carry a non-zero score.
`bm25s.BM25()` defaults to lucene, where that array is `None`, and the id-set
membership drop catches it regardless. Commented at the call site.

## 9. Pre-existing, found during review, not this branch's to fix

- Concurrent `index()` runs on one database raise
  `sqlite3.OperationalError: database is locked` (end state stays consistent).
- `server.py` keys the per-index lock on the unresolved `db_path or path`, so
  two clients naming one index differently take different locks over a shared
  `Store`.
- `_incremental_update` offsets a per-file `local_index` by a global chunkset
  count that `delete_file_data` can decrease. Predated c2de15e, but pruning
  made it reachable, so it was fixed here rather than left — see §7.

The first two predate c2de15e and are still open.
