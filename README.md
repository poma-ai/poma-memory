# poma-memory

**Your AI agent loses context every session. poma-memory gives it back.**

AI coding agents (Claude Code, Cursor, Copilot) accumulate valuable project knowledge — decisions, architecture notes, task history — in markdown files (`.claude/`, `.cursor/`, `.github/copilot/`). But when context windows fill up or sessions restart, that knowledge becomes invisible. Grep finds strings; it doesn't understand structure.

poma-memory indexes those markdown files and returns **complete, readable context** — not disconnected snippets. When you search for "auth middleware", you get the matching paragraph _plus_ its parent headings and surrounding context, assembled into a coherent cheatsheet with `[...]` gap markers. The result reads like a compressed version of the original document.

**No POMA account or API key required.** Free, local, open-source.

---

## The Problem

Standard search (grep, embeddings over flat chunks) breaks document structure:

- **Orphaned content** — a paragraph arrives without its heading, so the agent doesn't know what section it belongs to
- **Lost hierarchy** — a nested list item loses its parent context
- **Fragment soup** — five hits from the same file come back as five disconnected blocks instead of one readable summary

For agent memory files — which are deeply hierarchical by design — this means the agent retrieves *text* but not *understanding*.

## The Solution

poma-memory preserves the full document hierarchy during chunking. Every retrieval unit is a root-to-leaf path through the heading tree, so results always carry complete context. Multiple hits from the same file are merged into a single cheatsheet.

This is an open-source extraction of [POMA](https://poma-ai.com)'s heuristic chunking engine, optimized for the clean, consistent markdown that AI agents produce.

---

## Install

```bash
pip install poma-memory                         # BM25 keyword search (always works)
pip install 'poma-memory[semantic]'               # + model2vec local embeddings (30MB, no API key)
pip install 'poma-memory[openai]'                 # + OpenAI text-embedding-3-large
pip install 'poma-memory[mcp]'                    # + MCP server for Claude Code
pip install 'poma-memory[semantic,mcp]'           # recommended combo
```

## Quick start

```bash
poma-memory index .claude/                                    # index your context files
poma-memory search "authentication middleware" --path .claude/ # search
poma-memory search "rollback" --path .claude/ --where kind=runbook  # search one kind
```

## MCP server (Claude Code)

Add poma-memory as an MCP server so Claude Code can search your project memory automatically:

```bash
claude mcp add --transport stdio --scope user poma-memory -- poma-memory-mcp
# Exposes poma_search, poma_index, poma_status tools
```

Once added, Claude Code can call `poma_search` during planning and exploration to surface relevant decisions, patterns, and context from prior sessions.

## Python API

```python
from poma_memory import index, search, status

index(path=".claude/")
results = search("session context", path=".claude/", top_k=5)
for r in results:
    print(f"{r['file_path']} (score: {r['score']:.4f})")
    print(r['context'])
```

---

## Filtering by metadata

One index can hold more than one kind of document. To search a subset, give
each file some metadata and pass a predicate.

Metadata comes from two places. **Path rules** name what your layout already
says, and need no change to the files themselves — put a
`.poma-metadata.json` at the root of the indexed directory:

```json
{
  "rules": [
    {"glob": "events/**/*.md", "metadata": {"kind": "event"}},
    {"glob": "DECISIONS.md",   "metadata": {"kind": "decision"}},
    {"glob": "**/*.md",        "metadata": {"kind": "note"}}
  ]
}
```

First matching rule wins, so put the catch-all last. **Front-matter** is the
second source, and wins per key over a path rule:

```markdown
---
kind: event
session: laptop-2
---
```

Front-matter is read by a **restricted** parser, not a full YAML one: `key:
value`, `key: [a, b]`, a block list, and one level of nesting flattened to
`parent.child`. Values stay strings, so `true` and `1` are the strings `"true"`
and `"1"`. A trailing `# comment` outside quotes is dropped.

Anything else — block scalars (`|`, `>`), anchors, flow mappings, duplicate
keys, an unterminated fence — makes the whole block **unparsed**: the file gets
no front-matter metadata, and `poma-memory status` names it. If a filter misses
a file you expected, check `status` first.

Then filter:

```bash
poma-memory search "disk pressure" --path .agent/ --where kind=event
poma-memory search "disk pressure" --path .agent/ --where kind=event --where kind=decision
```

```python
search("disk pressure", path=".agent/", where={"kind": ["event", "decision"]})
```

Keys are AND-ed, values within a list are OR-ed, and comparison is
case-sensitive string equality. There are no operators and no negation: the
predicate cannot express "kind is not X" or "key is absent", so emit the key
on every document and filter positively.

Filtering narrows the corpus **before** anything is ranked. That matters
because the relevance gate reads the top semantic score, and a gate that saw
documents you filtered out would answer for a corpus you did not ask about.

Run `poma-memory index` after adding metadata or editing the rules file — it
re-reads metadata in place, with no re-chunking and no re-embedding. Until
then, a filtered search **refuses** rather than returning an empty list you
could not tell apart from "nothing matches". That covers both ways the index
can be behind: a file never scanned for metadata, and a file still carrying
what an earlier version of the rules file said about it. `poma-memory status`
shows both.

Each row records which directory's rules produced it, so this holds however the
database is addressed — including `--db` pointing somewhere else, and two
directories sharing one database. A run given a narrower `--glob` reaches only
part of the corpus and leaves the rest on the old rules; re-run over each
directory to clear it.

`index` also **removes** documents that are gone from disk, so a deleted or
renamed file stops appearing in results. It only does this for files under the
directory it was given, and only when that directory itself is present — a run
against an unmounted drive removes nothing. It also stops short of removing
most of an index in one go, since that looks more like a directory that failed
to mount than a deletion; `--prune` says do it anyway, `--no-prune` never
removes. A change is detected by mtime, size
or ctime, so an edit restored from a backup with its timestamp intact is still
picked up.

Why the filter runs before ranking rather than after, and what the refusals are
protecting against: **[`docs/metadata-filtering.md`](docs/metadata-filtering.md)**.

---

## How it works

1. **Hierarchical chunking.** Markdown is parsed into depth-annotated chunks that preserve heading hierarchy, lists, code blocks, and tables.
2. **Chunkset assembly.** Leaf chunks are paired with their ancestors into self-contained retrieval units (root-to-leaf paths), so every result carries full context.
3. **Hybrid search.** BM25 keyword matching (always available) + optional semantic vectors, merged via Reciprocal Rank Fusion.
4. **Empty gate.** RRF scores are rank-based — something always tops the list, so a fused-score floor can't express "the corpus has no answer." The gate can: when even the *best* semantic hit's cosine similarity is below a per-embedder calibrated threshold (model2vec: 0.35 — calibrated on a real corpus where irrelevant queries topped out at 0.25 and answerable ones started at 0.44), *all* results are suppressed instead of returning the best of a bad lot. Override per call (`--empty-gate`, `empty_gate=`), via `POMA_MEMORY_EMPTY_GATE`, or disable with `0`. BM25-only mode has no cosine and is ungated.
5. **Cheatsheet merging.** Multiple hits from the same file are merged into one block with `[...]` gap markers — reads like a summary, not a list of excerpts.
6. **Incremental indexing.** Append-only files (like agent context logs) only process new content on re-index.

### Search backends

| Backend | Install | Requires | Best for |
|---------|---------|----------|----------|
| BM25 | included | nothing | exact term matching |
| model2vec | `[semantic]` | 30MB local model | general-purpose, no API key |
| OpenAI | `[openai]` | `OPENAI_API_KEY` | highest quality |

Multiple backends are automatically fused via Reciprocal Rank Fusion when available.

---

## What this is (and isn't)

poma-memory extracts the **heuristic chunking and retrieval** logic from POMA's document processing platform. It works well on clean, predictable markdown — exactly what agents produce.

It does **not** include POMA's ML-powered indentation analysis, fine-tuned embedding models, or cloud processing pipeline. For complex document processing (scanned PDFs, inconsistent formatting, enterprise scale), see [poma-ai.com](https://poma-ai.com).

## Built for

- AI coding agents that persist context in markdown (`.claude/`, `.cursor/`, `.github/copilot/`)
- [Megavibe](https://github.com/poma-ai/megavibe) multi-agent framework (ships with poma-memory integration)
- Claude Code hook pipelines (augment Grep results with semantic context)

## License

MIT
