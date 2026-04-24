# Security Policy

## Reporting a vulnerability

Please **do not** report security issues via public GitHub issues.

Use GitHub's [Private Vulnerability Reporting](https://github.com/poma-ai/poma-memory/security/advisories/new) to send a confidential report directly to the maintainers.

We aim to acknowledge reports within 72 hours and issue fixes or mitigations within 14 days for confirmed high-severity issues.

## Scope

poma-memory runs locally — it reads files from a configured directory, writes a SQLite index, and serves search over stdio (CLI + MCP). Security-relevant areas include:

- Parsing of `.agent/` file content and frontmatter
- SQLite index construction and queries
- The MCP server surface (`poma-memory mcp`)
- Optional integrations: OpenAI embeddings, model2vec semantic search

Bugs in documentation, test fixtures, or unrelated features are not security issues — report those as regular issues.

## Supported versions

Only the latest released version on PyPI receives security updates.
