# Changelog

All notable changes to Remem will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [1.1.0.dev3] - Unreleased

### Added

- Ordered `StorageInterface.get_many()` record lookup with a compatible default
  for existing custom backends and optimized built-in implementations.

### Changed

- ANN indexes are built at client initialization and refreshed after client
  mutations, outside the query path.
- ANN queries resolve only returned candidate IDs instead of calling
  `StorageInterface.all()` to locate candidate records.

---

## [1.1.0.dev2] - 2026-07-13

### Added

- Configurable HNSW candidate retrieval through `AnnConfig.candidate_count`.
- Exact cosine reranking for every ANN candidate set.

### Changed

- HNSW is used only for candidate discovery. Public ordering, thresholds, and
  similarity scores now come from exact cosine calculations.

---

## [1.1.0.dev1] - 2026-07-12

### Added

- Public `auto`, `exact_cosine`, and `hnsw_cosine` search modes.
- Observable automatic-mode resolution through `Client.resolved_search_mode`,
  `Client.search_fallback_reason`, and `Client.search_resolution`.

### Changed

- `auto` is the default search mode. It uses HNSW when the optional USearch
  dependency is installed and otherwise falls back to exact cosine search.
- The legacy `similarity_backend` argument remains compatible and now emits a
  `DeprecationWarning` in favor of `search_mode`.

---

## [1.1.0.dev0] - 2026-07-12

### Added

- Optional HNSW approximate-nearest-neighbor search via the `ann` extra.
- Configurable ANN search settings through `AnnConfig`.

### Changed

- Exact cosine search remains the default; ANN preserves its similarity-score and threshold semantics.
- Reconciled release documentation and corrected lint configuration for current Ruff versions.

### Known limitations

- HNSW indexes are rebuilt in memory from authoritative storage records and are not persisted.
- Query-time metadata filtering still enumerates stored records before ANN search.
- Index insertion, replacement, and deletion are not yet incremental.

---

## [1.0.0] - 2026-07-12

### Changed

- Promoted the first public beta to the stable `1.0.0` release.

---

## [1.0.0-beta] - 2026-07-04

### 🎉 First Public Beta Release

This is the first public release of **Remem**.

Remem introduces an AI-native approach to semantic work reuse, enabling Retrieval-Augmented Generation (RAG) systems, AI agents, and LLM applications to intelligently reuse previous executions instead of repeatedly performing expensive inference.

Although feature-complete for its initial vision, the project remains in beta while the API, architecture, and performance continue to evolve.

### Added

- AI-native semantic cache
- Embedding-based similarity search
- Intelligent execution reuse
- Metadata-aware reuse policies
- Pluggable storage architecture
- Local persistence support
- Observability and metrics
- Python package (`remem-ai`)
- Example applications
- Unit tests
- Comprehensive documentation
- GitHub community standards
- Apache License 2.0

### Documentation

- Architecture guide
- Quick Start guide
- Getting Started guide
- FAQ
- Roadmap
- Contribution guide
- Security policy
- Citation metadata

### Notes

This release is intended for:

- Early adopters
- Community feedback
- Research
- Experimental production evaluation

Breaking API changes may occur before the first stable (`1.0.0`) release.

---

Future releases follow Semantic Versioning.

- **MAJOR** — Breaking API changes
- **MINOR** — Backward-compatible features
- **PATCH** — Bug fixes
