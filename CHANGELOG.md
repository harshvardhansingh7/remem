# Changelog

All notable changes to Remem will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [1.2.0] - 2026-07-20

### Added

- Optional Redis distributed semantic cache through `remem-ai[redis]`, using
  versioned record envelopes and application-scoped key prefixes.
- Local read-through/fallback coordination with in-process write replay after
  Redis reconnect, configurable retries, timeouts, fail-open/fail-closed mode,
  and logical record expiration.
- Best-effort duplicate-work prevention for `get_or_compute()` using expiring
  token-owned Redis locks and deterministic distributed record IDs.
- Cross-node response and retrieval reuse through the existing `ReusePolicy`,
  with distributed decision traces and backend/lock/synchronization metrics.
- Real Redis 7 integration tests in CI, a two-node example, deployment guide,
  and 1.1-to-1.2 migration guide.
- Dependency-light multi-signal response safeguards for intent, critical
  entities and values, temporal scope, negation, direction, requested output
  format, freshness, required metadata, and candidate-score ambiguity.
- Structured reuse diagnostics with per-check results and tier-specific
  rejection reasons, while preserving the existing public decision fields.
- Paired similarity-only and multi-signal benchmarks for Banking77, PAWS-Wiki,
  SQuAD, and BEIR SciFact, including a streamed SciFact preprocessor and pinned
  dataset manifest.

### Changed

- Distributed `search_mode="auto"` resolves to exact cosine for coherent remote
  visibility; explicit distributed HNSW is rejected because its per-process
  index cannot safely observe remote writes.
- Local-only behavior, imports, storage defaults, and search-mode resolution
  remain unchanged when `DistributedConfig` is omitted.
- A response-ineligible but retrieval-compatible match now falls back to
  retrieval reuse instead of serving the cached response. Existing callers
  without query metadata retain threshold-only behavior.
- The benchmark runner now passes query text through `ExecutionContext` and
  accepts optional policy configuration without changing search architecture.

---

## [1.1.0] - 2026-07-14

### Added

- Optional USearch HNSW candidate retrieval through `remem-ai[ann]`, with
  inspectable `auto`, `exact_cosine`, and `hnsw_cosine` modes.
- Exact cosine reranking of ANN candidates, preserving public score and
  threshold semantics from `-1.0` to `1.0`.
- Direct ordered record-ID lookup for ANN candidates without query-time full
  storage scans in the built-in backends.
- Incremental insert, embedding update, compacting delete, clear, and
  namespace-move synchronization using stable internal keys.
- Opt-in persistent namespace indexes with versioned metadata, checksums,
  validated fast reload, atomic replacement, and deterministic recovery.
- Namespace-partitioned retrieval and conservative pre-storage filtering for
  `kb_version`, `prompt_version`, and `model` policy constraints.
- ANN lifecycle telemetry through `client.ann_index_stats` and recovery details
  through `client.ann_persistence_recovery_reason`.

### Changed

- `auto` is the default search mode. It selects HNSW only when USearch is
  installed and otherwise falls back safely to exact cosine.
- Storage remains authoritative. ANN mutation or persistence failures roll back
  client-mediated storage changes and rebuild affected derived indexes.
- CI now covers Python 3.10, 3.11, and 3.12, plus ANN-enabled tests, Ruff,
  formatting, mypy, distribution builds, and metadata validation.

### Performance

- Normal inserts, updates, and deletes avoid full ANN rebuilds.
- Valid persistent indexes load without re-adding every vector.
- Strict namespace queries avoid unrelated namespace indexes, and storage
  resolves only policy-eligible candidate IDs.

### Fixed

- Corrected public and internal type annotations across exact and ANN search,
  and added a passing mypy gate for all package modules.
- Modernized package license metadata to the SPDX form used by current
  setuptools builds.

### Compatibility and upgrade notes

- Python 3.10 or newer is required.
- The base installation remains dependency-light and does not require USearch;
  install `remem-ai[ann]` to enable HNSW.
- The legacy `similarity_backend="exact"|"hnsw"` argument remains temporarily
  supported with a `DeprecationWarning`; new code should use `search_mode`.
- Existing `1.0.0` JSON storage files load without migration. ANN files are
  derived caches and are created only when `persistence_path` is configured.
- Arbitrary `ExecutionContext.metadata` remains descriptive and is not an
  implicit filter. Only fields represented by `ReusePolicy` affect eligibility.
- Persistent ANN paths support one process owner; cross-process index writers
  and shared-filesystem coordination are not provided in this release.

---

## [1.1.0.dev6] - 2026-07-14

### Added

- Namespace-partitioned HNSW indexes for scalable default tenant/application
  isolation.
- Conservative adaptive candidate expansion for `kb_version`,
  `prompt_version`, and `model` policy constraints.
- Independent persistence and corruption recovery for each namespace partition.

### Changed

- ANN filtering now happens before storage candidate lookup and exact reranking,
  preventing incompatible global neighbors from crowding out valid records.
- Namespace-only context changes move stable record IDs between partitions
  incrementally.

### Not supported

- Arbitrary `ExecutionContext.metadata` values remain descriptive only; they are
  not implicit filters because `ReusePolicy` defines no matching semantics for
  them.

---

## [1.1.0.dev5] - 2026-07-14

### Added

- Opt-in native HNSW persistence through `AnnConfig.persistence_path`.
- Versioned metadata, native-file checksums, storage fingerprints, and stable-key
  mappings for validated fast reload.
- Public ANN lifecycle telemetry through `client.ann_index_stats` and an
  inspectable `client.ann_persistence_recovery_reason`.

### Changed

- Valid persistent indexes load without a full vector rebuild at startup.
- Missing, stale, incompatible, corrupt, or interrupted artifacts rebuild
  automatically from authoritative storage and are atomically replaced.

---

## [1.1.0.dev4] - 2026-07-14

### Added

- Incremental HNSW insertion, embedding replacement, and native deletion using
  stable internal integer keys.
- Storage rollback and deterministic index recovery when ANN mutation fails.
- Client-level lifecycle locking for concurrent ANN queries and mutations.

### Changed

- Non-vector record updates refresh cached record state without mutating HNSW.
- Full ANN rebuilds are now limited to startup, explicit reload, and recovery.

---

## [1.1.0.dev3] - 2026-07-14

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
