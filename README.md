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

## How it works

1. **Hierarchical chunking.** Markdown is parsed into depth-annotated chunks that preserve heading hierarchy, lists, code blocks, and tables.
2. **Chunkset assembly.** Leaf chunks are paired with their ancestors into self-contained retrieval units (root-to-leaf paths), so every result carries full context.
3. **Hybrid search.** BM25 keyword matching (always available) + optional semantic vectors, merged via Reciprocal Rank Fusion.
4. **Cheatsheet merging.** Multiple hits from the same file are merged into one block with `[...]` gap markers — reads like a summary, not a list of excerpts.
5. **Incremental indexing.** Append-only files (like agent context logs) only process new content on re-index.

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
