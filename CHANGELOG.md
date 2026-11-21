# Changelog

## 0.7.0

- Removed the optional third-party embedding backend and its extra; model2vec is
  the only embedder.
- A non-empty `POMA_EMBEDDER` other than `model2vec` is now reported on stderr and
  ignored.
- Existing indexes with vectors of another size are re-embedded locally on the
  next build. Indexes built with model2vec are unchanged.
- Added a test that fails if a reference to a removed provider returns.

## 0.6.0

- Metadata filtering that narrows the corpus before ranking. See
  `docs/metadata-filtering.md`.
