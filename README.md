# poma-memory

Free, local, structure-preserving memory for AI agents. No API key. No cloud. Just markdown.

poma-memory is an open-source extraction of [POMA](https://poma-ai.com)'s heuristic chunking engine, optimized for the kind of well-structured markdown that AI agents produce. It indexes `.agent/` context files into hierarchical chunks and returns complete root-to-leaf context paths — not disconnected snippets.

**No POMA account or API key required.** This is a standalone tool.

## Why agent memory specifically?

AI coding agents (Claude Code, Cursor, Copilot) write remarkably clean markdown: consistent heading hierarchies, predictable list structures, uniform formatting. This is the sweet spot for heuristic chunking — no ML models needed to parse the structure correctly.

poma-memory exploits this by preserving the full document hierarchy during chunking. When you search for "auth middleware", you get the matching paragraph _plus_ its parent headings and surrounding context, assembled into a coherent cheatsheet with `[...]` gap markers. The result reads like a compressed version of the original document, not a bag of fragments.

## Install

```bash
pip install poma-memory                         # BM25 keyword search (always works)
pip install poma-memory[semantic]               # + model2vec local embeddings (30MB, no API key)
pip install poma-memory[openai]                 # + OpenAI text-embedding-3-large
pip install poma-memory[mcp]                    # + MCP server for Claude Code
pip install poma-memory[semantic,mcp]           # recommended combo
```

## Quick start

```bash
poma-memory index .agent/                                    # index your context files
poma-memory search "authentication middleware" --path .agent/ # search
poma-memory status --path .agent/                            # check what's indexed
```

## MCP server (Claude Code)

```bash
claude mcp add --transport stdio --scope user poma-memory -- poma-memory-mcp
# Exposes poma_search, poma_index, poma_status tools
```

## Python API

```python
from poma_memory import index, search, status

index(path=".agent/")
results = search("session context", path=".agent/", top_k=5)
for r in results:
    print(f"{r['file_path']} (score: {r['score']:.4f})")
    print(r['context'])
```

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

## What this is (and isn't)

poma-memory extracts the **heuristic chunking and retrieval** logic from POMA's document processing platform. It works well on clean, predictable markdown — exactly what agents produce.

It does **not** include POMA's ML-powered indentation analysis, fine-tuned embedding models, or cloud processing pipeline. For complex document processing (scanned PDFs, inconsistent formatting, enterprise scale), see [poma-ai.com](https://poma-ai.com).

## Built for

- AI coding agents that persist context in markdown (`.agent/`, `.cursor/`, project notes)
- [Megavibe](https://github.com/poma-ai/megavibe) multi-agent framework (ships with poma-memory integration)
- Claude Code hook pipelines (augment Grep results with semantic context)

## License

MIT
