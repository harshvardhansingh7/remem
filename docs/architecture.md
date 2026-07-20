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

Exact search filters all stored records through `ReusePolicy` before cosine calculation. ANN records are physically partitioned by namespace. The default strict policy searches only the requested namespace; relaxing `require_same_namespace` searches all partitions. Within each selected partition, policy eligibility for namespace, knowledge-base version, prompt version, and model is computed from index-side record metadata before storage lookup. Candidate discovery starts at `candidate_count` and expands geometrically until it has enough eligible IDs or exhausts that partition. Storage resolves only those eligible IDs, and Remem then calculates exact cosine similarity. Incompatible records are never fetched for reranking or returned.

## Distributed request flow

Distributed mode composes the existing layers instead of adding a second reuse
engine:

```text
Client A local state ----+
                         +--> DistributedStorage --> RedisStorage
Client B local state ----+             |
                                       v
                              existing exact search
                                       |
                              MetadataMatcher + ReusePolicy
                                       |
                    RESPONSE_REUSED / RETRIEVAL_REUSED / MISS
```

`DistributedStorage` implements the existing `StorageInterface`. It writes Redis
first, maintains optional local read-through/fallback state, and replays
in-process pending mutations after reconnect. Redis stores versioned JSON
envelopes in a namespaced hash; malformed, expired, or incompatible envelopes
are removed. Record UUIDs provide last-write-wins identity.

The consistency model is eventual. A healthy read sees current Redis records on
the next request. During an outage, fallback-enabled nodes use local state and
queue writes, but that queue is not durable across process termination.
`get_or_compute()` adds best-effort coalescing with expiring Redis token locks
and deterministic record IDs. Lock timeout or backend failure falls back to
normal computation, preserving availability.

Distributed mode uses exact cosine. Per-process HNSW cannot safely discover
another node's writes, so distributed `auto` resolves to exact and explicit
HNSW is rejected. This keeps the existing policy authoritative rather than
serving from a stale graph.

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
- **Planned:** explicitly policy-driven arbitrary metadata indexing — see the [Roadmap](roadmap.md)

The default `search_mode="auto"` selects USearch HNSW when the optional ANN extra is installed and exact cosine otherwise. `search_mode="exact_cosine"` forces exhaustive deterministic search, while `search_mode="hnsw_cosine"` forces ANN and reports a clear installation error if USearch is missing. The requested mode, resolved mode, and fallback reason are exposed on `Client`; automatic fallback is therefore inspectable without changing result structures. No dataset-size threshold is used because the project does not yet have benchmark evidence for one.

HNSW performs candidate discovery only. Remem resolves eligible candidate records with an ordered batch ID lookup, recalculates exact cosine similarity, sorts by those exact scores, and then applies reuse thresholds. This preserves the existing `[-1.0, 1.0]` thresholds and `ReuseOutcome.similarity_score` semantics without claiming exhaustive nearest-neighbor recall. `AnnConfig.candidate_count` controls the initial eligible-candidate target per selected partition and defaults to 50. Adaptive expansion is conservative: it removes fixed-overfetch false misses for structured policy fields, but may search an entire namespace partition when compatible records are sparse.

Execution records remain authoritative in storage. Within each namespace partition, every UUID maps to a stable monotonic integer HNSW key. Inserts use native `add`; embedding changes use compacting native removal followed by insertion with the same key; deletes use compacting native removal; and KB/prompt/model or response-only changes update cached record state without touching the graph. A namespace change removes the UUID from its old partition and inserts it into the new partition. Storage is mutated first under a client lifecycle lock. If ANN mutation fails, storage is rolled back and affected graphs are deterministically rebuilt; recovery failure raises `AnnMutationError`. Full rebuilds remain for unpersisted startup, explicit reload, validation failure, and recovery. This protects threads using one local HNSW `Client`, but direct external mutation of its injected storage and multi-process HNSW coordination are not supported. Distributed mode uses the separate exact-search flow described above.

Native persistence is opt-in through `AnnConfig.persistence_path`. The default empty namespace uses that base path; named namespaces use SHA-256-derived filenames under `<path>.partitions/`, avoiding raw tenant names in filenames. Each native USearch graph and versioned JSON metadata file form one independently recoverable derived generation. The metadata records the engine/configuration identity, vector dimension, storage fingerprint, UUID-to-key mapping, next monotonic key, native size, and SHA-256 of the graph. It deliberately excludes responses, references, and arbitrary context metadata. The graph temporary file is atomically replaced first and metadata is committed last; a crash between replacements leaves a checksum mismatch rather than silently accepting mixed generations. Missing metadata, stale storage, incompatible configuration or dimension, invalid mappings, missing native data, size mismatch, parse errors, and checksum failures trigger a rebuild of the affected namespace from storage and an atomic rewrite.

Arbitrary `ExecutionContext.metadata` is not indexed or compared. It remains descriptive because `ReusePolicy` has no operators, key allowlist, normalization rules, or missing-value semantics for it. Treating dictionary equality as an implicit security boundary would be surprising and incompatible; future support requires an explicit policy contract.

Fast reload still reads authoritative record metadata to validate identity and populate exact-reranking records, but it avoids re-adding every vector to HNSW. `AnnIndexStats.load_count` and `rebuild_count` make that path observable without timing-sensitive claims. Persistent artifacts are local cache files, not a trust boundary: applications should give them the same filesystem protection as storage. Native parsing is delegated to USearch only after metadata and checksum validation. One process may own a configured path; cross-process writers and shared-network-filesystem coordination are not supported.

USearch was selected because it provides lightweight prebuilt Python wheels across common platforms while using HNSW internally. Keeping it optional leaves the base package and legacy exact behavior unchanged. Higher `AnnConfig.ef_search` improves approximate-search recall at the cost of query latency; no benchmark claim is made here.

### Storage Layer

Abstracted behind `StorageInterface` (`remem/storage/storage.py`), so the reuse engine never depends on a specific persistence mechanism.

`get_many(entry_ids)` returns available records in requested order. Both built-in backends use their UUID-keyed in-memory dictionaries directly, so ANN candidate resolution does not enumerate all stored records. Existing custom `StorageInterface` implementations inherit a compatibility implementation that calls their required `get()` method once per candidate; they can override `get_many()` with a native batch query such as SQL `WHERE id IN (...)`.

| Implementation | File | Persistence |
|---|---|---|
| `JsonStorage` | `remem/storage/json_storage.py` | Durable — atomic file writes |
| `InMemoryStorage` | `remem/storage/memory_storage.py` | Volatile — in-process only |
| `RedisStorage` | `remem/distributed/redis_storage.py` | Shared Redis record hash |
| `DistributedStorage` | `remem/distributed/storage.py` | Local/remote coordination and fallback |
| `Serializer` | `remem/storage/serializer.py` | JSON round-trip for `ExecutionRecord` |

Future backends may include SQLite, PostgreSQL, cloud object storage, and a
distributed vector index.

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

**Production-conscious.** Local and distributed modes publish their different
consistency, ownership, recovery, and scaling boundaries rather than implying
database-grade guarantees.

## Future Architecture

The longer-term direction includes:

- Remote vector candidate discovery for large distributed caches
- Cloud-native storage, Redis Cluster, and multi-region synchronization
- Multi-tenant deployment primitives
- A dedicated benchmarking suite

See the [Roadmap](roadmap.md) for version-by-version planning.
