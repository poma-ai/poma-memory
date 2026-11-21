# Changelog

## 0.7.0

- Removed the OpenAI embedding backend: the `[openai]` extra, `POMA_EMBEDDER=openai`
  and `OPENAI_API_KEY` handling. model2vec is the supported embedder.
- A non-empty `POMA_EMBEDDER` other than `model2vec` is now reported on stderr and
  ignored.
- Existing indexes that hold vectors of another size (3072 dimensions from the
  removed backend) are re-embedded locally on the next build, with no action
  needed. Indexes built with model2vec are unchanged.
- Added a test that fails if a reference to a removed provider returns.

## 0.6.0

- Metadata filtering that narrows the corpus before ranking. See
  `docs/metadata-filtering.md`.
