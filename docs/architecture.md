# Architecture

## Overview

Remem is an AI-native semantic cache that reduces the cost and latency of LLM applications by intelligently reusing previous executions.

Unlike traditional caches that operate on exact keys, Remem determines whether previously executed work can be safely reused based on **semantic similarity** combined with **execution metadata**.

```
Application
      │
      ▼
   Remem SDK
      │
      ▼
 Metadata Filter  ──► rejects incompatible candidates (namespace, KB version, model, prompt)
      │
      ▼
 Similarity Search ──► cosine similarity over remaining candidates
      │
      ▼
 Reuse Decision    ──► RESPONSE_REUSED · RETRIEVAL_REUSED · MISS
      │
      ▼
 Storage Layer
```

## Request Flow

```
Incoming Request
      │
      ▼
Generate Embedding  (your embedding model)
      │
      ▼
client.check(embedding)
      │
      ├──► RESPONSE_REUSED   ──► Return cached LLM response immediately
      │    (similarity ≥ 0.95)      Skip vector DB and LLM entirely
      │
      ├──► RETRIEVAL_REUSED  ──► Use cached document IDs, call LLM only
      │    (similarity ≥ 0.80)      Skip the vector DB search
      │
      └──► MISS              ──► Run the full pipeline
           (similarity < 0.80)      Store the result with client.remember()
```

Exact search filters all stored records through `ReusePolicy` before cosine calculation. ANN search first discovers a bounded set of record IDs in the global HNSW index, resolves only those records through `StorageInterface.get_many()`, rejects metadata-incompatible candidates, and then calculates exact cosine similarity. Incompatible records are never reranked or returned. Because metadata is currently applied after bounded ANN discovery, heavily partitioned datasets may require a larger `candidate_count`; filter-aware indexing remains a later roadmap item.

## Core Components

### SDK / Client

The primary interface used by application code (`remem/client.py`). Responsibilities:

- Expose `check`, `remember`, `get_or_compute`, and lower-level storage operations
- Coordinate the reuse engine and storage layer
- Hide all internal implementation details behind a small, stable surface

### Reuse Engine

Located in `remem/reuse/engine.py`. Orchestrates the end-to-end decision process: it invokes the metadata matcher, then the similarity engine, and combines their outputs into a single `ReuseOutcome`. The decision process is deterministic and entirely policy-driven — the same inputs always produce the same decision.

Possible outcomes:

- **Full reuse** (`RESPONSE_REUSED`) — the cached response is returned as-is
- **Retrieval reuse** (`RETRIEVAL_REUSED`) — cached documents are reused, generation re-runs
- **No reuse** (`MISS`) — nothing usable was found

### Metadata Matcher

Located in `remem/reuse/matcher.py`. Applies `ReusePolicy` constraints before exact cosine scoring and reuse decisions. Examples of what it checks:

- Same namespace
- Same knowledge-base version
- Same prompt version
- Same model

### Similarity Engine

Located in `remem/similarity/engine.py`, backed by `remem/similarity/metrics.py`. Finds semantically related requests among metadata-compatible candidates.

- **Current implementation:** automatic, exact cosine, and optional HNSW cosine search
- **Planned:** incremental mutation, index persistence, and filter-aware indexing — see the [Roadmap](roadmap.md)

The default `search_mode="auto"` selects USearch HNSW when the optional ANN extra is installed and exact cosine otherwise. `search_mode="exact_cosine"` forces exhaustive deterministic search, while `search_mode="hnsw_cosine"` forces ANN and reports a clear installation error if USearch is missing. The requested mode, resolved mode, and fallback reason are exposed on `Client`; automatic fallback is therefore inspectable without changing result structures. No dataset-size threshold is used because the project does not yet have benchmark evidence for one.

HNSW performs candidate discovery only. Remem resolves candidate records with an ordered batch ID lookup, recalculates exact cosine similarity, sorts by those exact scores, and then applies reuse thresholds. This preserves the existing `[-1.0, 1.0]` thresholds and `ReuseOutcome.similarity_score` semantics without claiming exhaustive nearest-neighbor recall. `AnnConfig.candidate_count` controls the recall-versus-reranking-latency trade-off and defaults to 50.

Execution records remain authoritative in storage. Each UUID maps to a stable monotonic integer HNSW key. Inserts use native `add`; embedding changes use compacting native removal followed by insertion with the same key; deletions use compacting native removal; and non-vector changes update cached record state without touching the graph. Storage is mutated first under a client lifecycle lock. If ANN mutation fails, storage is rolled back and the graph is deterministically rebuilt; recovery failure raises `AnnMutationError`. Full rebuilds remain for startup, explicit reload, and recovery. This protects threads using one `Client`, but direct external mutation of the injected storage and multi-process coordination are not supported.

USearch was selected because it provides lightweight prebuilt Python wheels across common platforms while using HNSW internally. Keeping it optional leaves the base package and legacy exact behavior unchanged. Higher `AnnConfig.ef_search` improves approximate-search recall at the cost of query latency; no benchmark claim is made here.

### Storage Layer

Abstracted behind `StorageInterface` (`remem/storage/storage.py`), so the reuse engine never depends on a specific persistence mechanism.

`get_many(entry_ids)` returns available records in requested order. Both built-in backends use their UUID-keyed in-memory dictionaries directly, so ANN candidate resolution does not enumerate all stored records. Existing custom `StorageInterface` implementations inherit a compatibility implementation that calls their required `get()` method once per candidate; they can override `get_many()` with a native batch query such as SQL `WHERE id IN (...)`.

| Implementation | File | Persistence |
|---|---|---|
| `JsonStorage` | `remem/storage/json_storage.py` | Durable — atomic file writes |
| `InMemoryStorage` | `remem/storage/memory_storage.py` | Volatile — in-process only |
| `Serializer` | `remem/storage/serializer.py` | JSON round-trip for `ExecutionRecord` |

Planned backends include SQLite, PostgreSQL, Redis, and cloud object storage.

### Observability

`remem/metrics/collector.py` accumulates telemetry throughout the process lifetime and produces an immutable `MetricsSnapshot` (`remem/metrics/snapshot.py`) on demand — hit rate, reuse breakdown, and average similarity.

## Data Models

**`ExecutionRecord`** — the unit stored in the persistence layer:

```
id           UUID
embedding    list[float]         # the query embedding
response     Any                 # LLM output (string, dict, DataFrame, ...)
references   list[str]           # document / chunk IDs from the vector DB
context      ExecutionContext    # metadata used for filtering
hit_count    int                 # number of times this record has been reused
created_at   datetime
```

**`ExecutionContext`** — metadata scoping for a request:

```
namespace       str              # e.g. "support-bot", "tenant:acme"
kb_version      str              # knowledge-base version string
prompt_version  str              # prompt template version
model           str | None       # LLM identifier, e.g. "gpt-4o"
metadata        dict[str, Any]   # custom key-value pairs
```

Full field-by-field documentation lives in the [API Reference](api.md).

## Design Principles

**Local-first.** Remem should work fully without any external infrastructure — no database, no network calls, no API keys.

**Modular.** Every subsystem (similarity, metadata filtering, storage, observability) can evolve independently behind its own interface.

**Extensible.** New storage engines, embedding providers, and reuse policies should be straightforward to add without touching the core decision logic.

**Explainable.** Every `ReuseOutcome` carries a `reason` and a `similarity_score` — reuse decisions are never a black box.

**Production-ready.** The architecture is designed to scale from a single local script to a production service, without a rewrite in between.

## Future Architecture

The longer-term direction includes:

- ANN indexes for large-scale similarity search
- Distributed, multi-process, and multi-node cache coordination
- Cloud-native storage and synchronization
- Multi-tenant deployment primitives
- A dedicated benchmarking suite

See the [Roadmap](roadmap.md) for version-by-version planning.
