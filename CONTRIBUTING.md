# Contributing to poma-memory

Thanks for your interest! poma-memory is a small project maintained by POMA AI GmbH. Contributions are welcome.

## Bug reports

Open a [GitHub Issue](https://github.com/poma-ai/poma-memory/issues/new) with:

- What you expected to happen
- What actually happened (include any error output or stack trace)
- Minimal steps to reproduce
- Your OS and `python --version`
- The output of `pip show poma-memory`

## Pull requests

1. Fork the repo and create a branch off `main` (e.g. `fix/search-ranking`, `feat/new-tokenizer`).
2. Keep changes focused — one concern per PR.
3. Install dev dependencies: `pip install -e '.[dev,semantic,mcp]'`
4. Run tests: `pytest` — all existing tests must pass. Add tests for new behavior.
5. Match existing style (4-space indent, type hints where practical, no heavy frameworks).
6. Write a clear commit message — brief imperative is great.
7. Open the PR against `main`.

Expect a review within a few days. If a PR sits longer than a week, feel free to ping in a comment.

## Releases

Releases are published to PyPI via GitHub Actions when a version tag (`0.3.4`, etc.) is pushed. Maintainers bump `version` in `pyproject.toml` AND `__version__` in `poma_memory/__init__.py` — both in lockstep.

## Security

Please **do not** file security vulnerabilities as public issues. See [SECURITY.md](./SECURITY.md).

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](./LICENSE).
