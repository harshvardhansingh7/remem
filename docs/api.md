# API Reference

This document describes every public class, method, and configuration option in Remem. It assumes you've already completed the [Quickstart](QuickStart.md).

The stable `1.2.0` public API follows [Semantic Versioning](https://semver.org/). Breaking changes occur only in major releases. Local users upgrading from 1.1 require no data migration; see the [v1.2 migration guide](migration-1.2.md) before enabling Redis.

## Table of Contents

1. [Client](#client)
2. [ExecutionContext](#executioncontext)
3. [ReusePolicy](#reusepolicy)
4. [ReuseOutcome](#reuseoutcome)
5. [ReuseDecision](#reusedecision)
6. [ExecutionResult](#executionresult)
7. [ExecutionRecord](#executionrecord)
8. [Distributed Mode](#distributed-mode)
9. [Storage Backends](#storage-backends)
10. [Environment Variables](#environment-variables)

---

## Client

The single entry point for all Remem operations.

```python
from remem import Client

Client(
    storage_backend: StorageInterface | None = None,
    policy: ReusePolicy | None = None,
    similarity_backend: Literal["exact", "hnsw"] | None = None,
    ann_config: AnnConfig | None = None,
    *,
    search_mode: SearchMode | str = SearchMode.AUTO,
    distributed: DistributedConfig | None = None,
    distributed_backend: DistributedBackend | None = None,
)
```

| Parameter | Default | Description |
|---|---|---|
| `storage_backend` | `JsonStorage("remem_store.json")` | Where execution records are persisted. |
| `policy` | `ReusePolicy()` | Similarity thresholds and metadata constraints. |
| `similarity_backend` | `None` | Deprecated compatibility alias for `"exact"` or `"hnsw"`. Do not combine it with a non-auto `search_mode`. |
| `ann_config` | `None` | Optional `AnnConfig` tuning for the HNSW backend. Ignored for exact search. |
| `search_mode` | `"auto"` | `"auto"`, `"exact_cosine"`, or `"hnsw_cosine"`. Enum members from `SearchMode` are also accepted. |
| `distributed` | `None` | Optional Redis and coordination configuration. Omit for unchanged local mode. |
| `distributed_backend` | `None` | Advanced backend injection point used with an enabled distributed config. The built-in default is `RedisStorage`. |

`auto` uses HNSW when USearch is installed and otherwise resolves to exact cosine. Resolution is observable through `client.search_mode`, `client.resolved_search_mode`, `client.search_fallback_reason`, and the combined immutable `client.search_resolution` value. No dataset-size threshold is applied.

### ANN configuration

`AnnConfig` is exported from `remem` and configures the optional HNSW backend:

```python
from remem import AnnConfig, Client

client = Client(
    search_mode="hnsw_cosine",
    ann_config=AnnConfig(
        m=16,
        ef_construction=200,
        ef_search=100,
        candidate_count=50,
        persistence_path=".remem/records.usearch",
    ),
)
```

`m` and `ef_construction` tune index construction. `ef_search` increases candidate-discovery recall when raised, at the cost of query time. `candidate_count` is the initial eligible-candidate target per selected namespace partition; its default is 50. Larger candidate sets can improve recall but increase exact-reranking latency. Final ordering, score values, and threshold filtering use exact cosine. Candidate discovery remains approximate and does not guarantee the same neighbors as exhaustive search.

`persistence_path` is an optional filesystem base path for native USearch indexes. Its default is `None`, preserving the in-memory behavior. Exact search ignores this option. The empty namespace uses `<path>` and `<path>.meta.json`; named namespaces use hashed filenames under `<path>.partitions/`, so raw namespace values are not placed in filenames. Metadata files contain only index metadata, UUID/key mappings, and hashes, not responses or arbitrary context payloads. Parent directories are created automatically.

On startup, Remem fast-loads only when the metadata format, engine identity, HNSW configuration, dimension, record/vector fingerprint, UUID/key mapping, native size, and SHA-256 checksum all match authoritative storage. Otherwise it rebuilds and atomically replaces the derived cache. Native data is written to a temporary file and committed before its metadata commit marker, so interruption at either boundary is detected on restart. Temporary files are ignored during reads.

`client.ann_index_stats` is an `AnnIndexStats` value for HNSW and `None` for exact search. It exposes aggregate `record_count`, `rebuild_count`, `load_count`, and `persistence_enabled` values across namespace partitions. A successful persistent restart reports one load per non-empty namespace partition and zero rebuilds. `client.ann_persistence_recovery_reason` is `None` after valid loads and identifies the namespace and validation failure for any automatic rebuild.

HNSW indexes are partitioned by `ExecutionContext.namespace`. The default `ReusePolicy` searches only the matching namespace. When namespace matching is relaxed, all partitions are searched and their eligible candidates are exact-reranked together. Within a selected partition, `kb_version`, `prompt_version`, and `model` constraints are applied before storage lookup. If closer incompatible records fill the initial ANN result, discovery expands geometrically until enough compatible IDs are found or the partition is exhausted; there is no fixed global overfetch multiplier. Sparse compatibility can therefore increase query work up to the size of one selected partition, while storage still fetches only eligible IDs.

Client-mediated record mutations synchronize HNSW incrementally. Inserts allocate stable internal keys; embedding replacements remove and reinsert the native vector under the same key; deletes compact native graph state; and response, reference, KB, prompt, or model-only changes avoid graph mutation. Namespace changes move the record between partitions. Storage commits first. An ANN or persistence failure rolls storage back and rebuilds the affected derived graphs, raising `AnnMutationError` so callers can observe the failed operation. Full rebuilds remain for unpersisted startup, explicit reload, validation failure, and recovery. Operations through one `Client` are lifecycle-locked; direct storage mutation and multi-process writers are outside the supported consistency boundary.

Existing `1.0.0` JSON snapshots remain readable without migration. ANN native and metadata files are optional derived caches, not replacements for record storage. Do not configure multiple processes to write the same `persistence_path`.

### `check(query_embedding, context)`

Checks whether previous work can be reused, **without running any of your pipeline code**.

```python
outcome: ReuseOutcome = client.check(
    query_embedding: list[float],
    context: ExecutionContext | None = None,
)
```

| Parameter | Description |
|---|---|
| `query_embedding` | Dense vector from your embedding model for the current request. |
| `context` | Optional scoping metadata. If omitted, a default `ExecutionContext()` is used and normal policy matching still applies. |

**Returns:** [`ReuseOutcome`](#reuseoutcome).

```python
from remem import ReuseDecision

if outcome.decision == ReuseDecision.RESPONSE_REUSED:
    return outcome.result

if outcome.decision == ReuseDecision.RETRIEVAL_REUSED:
    answer = call_llm(query, outcome.references)
    client.remember(embedding, answer, outcome.references, context=context)
    return answer

# MISS
docs   = vector_db.search(embedding)
answer = call_llm(query, docs)
client.remember(embedding, answer, references=docs, context=context)
return answer
```

### `remember(query_embedding, response, references, context)`

Stores the result of a pipeline execution so it can be reused on future requests.

```python
client.remember(
    query_embedding: list[float],
    response: Any,
    references: list[str] | None = None,
    context: ExecutionContext | None = None,
)
```

| Parameter | Description |
|---|---|
| `query_embedding` | The same vector passed to `check()`. |
| `response` | The LLM response or pipeline output — string, dict, or any serializable object. |
| `references` | Document or chunk IDs from your vector DB, stored for potential retrieval reuse. |
| `context` | The same `ExecutionContext` used in the preceding `check()` call. |

Call `remember()` after every `MISS` or `RETRIEVAL_REUSED` decision to grow the reuse store.

### `get_or_compute(query_embedding, compute_callback, context)`

An all-in-one alternative to `check()` + `remember()`. Calls `compute_callback` only when necessary and stores the result automatically.

```python
outcome: ReuseOutcome = client.get_or_compute(
    query_embedding: list[float],
    compute_callback: Callable[[], ExecutionResult],
    context: ExecutionContext | None = None,
)
```

| Parameter | Description |
|---|---|
| `query_embedding` | Dense vector for the current request. |
| `compute_callback` | Zero-argument callable returning `ExecutionResult`. Called only on `MISS` or `RETRIEVAL_REUSED`. |
| `context` | Optional metadata scoping. |

**When to use `get_or_compute` vs. `check` + `remember`:**

- Use `get_or_compute` when your pipeline is a single function and you don't need to distinguish between `RETRIEVAL_REUSED` and `MISS`.
- Use `check` + `remember` when you want to skip only the vector-DB call on `RETRIEVAL_REUSED` and reuse cached document IDs directly — the recommended approach for RAG pipelines.

### `store / delete / all / flush_storage`

Lower-level operations for direct record management.

```python
client.store(record: ExecutionRecord) -> None   # store a fully constructed record
client.delete(entry_id: UUID) -> bool           # delete a specific record
client.all() -> list[ExecutionRecord]           # return all stored records
client.flush_storage() -> None                  # wipe everything
```

### `save_snapshot / load_snapshot`

Explicit persistence control. `JsonStorage` writes to disk on every `put`, `delete`, and `flush`, so these are rarely needed directly.

```python
client.save_snapshot()   # force-write in-memory state to disk
client.load_snapshot()   # reload records from disk
```

`InMemoryStorage` silently ignores both calls.

### `distributed_status`

In distributed mode, this read-only property returns `enabled`, `backend`,
`node_id`, current `healthy` state, in-process `pending_operations`, and
`last_error`. Local clients return `{"enabled": False}`.

---

## ExecutionContext

Metadata that scopes which stored records are valid candidates for reuse.

```python
from remem import ExecutionContext

ExecutionContext(
    namespace: str = "",
    kb_version: str = "1.0",
    prompt_version: str = "1.0",
    model: str | None = None,
    metadata: dict[str, Any] = {},
)
```

| Field | Purpose | Example |
|---|---|---|
| `namespace` | Isolates results by application, tenant, or team. | `"support-bot"`, `"tenant:acme"` |
| `kb_version` | Tracks the knowledge-base version. Bump it when documents change. | `"2024-Q4"`, `"v2.1"` |
| `prompt_version` | Tracks the prompt template version. Bump it when the prompt changes materially. | `"v3"`, `"2024-06"` |
| `model` | The LLM used to generate the response. Prevents cross-model reuse. | `"gpt-4o"`, `"claude-3-5-sonnet"` |
| `metadata` | Query text, explicit signals, and application fields. Only policy-configured keys affect compatibility. | `{"query": "...", "region": "eu-west-1"}` |

All fields are optional. Namespace, KB version, prompt version, and model use
the built-in policy switches. Arbitrary metadata is not an implicit filter;
configure `required_metadata_keys` when a field is a reuse dependency.

---

## ReusePolicy

Controls similarity thresholds and which metadata fields must match before reuse is considered.

```python
from remem import ReusePolicy

ReusePolicy(
    retrieval_threshold: float = 0.80,
    response_threshold: float = 0.95,
    require_same_namespace: bool = True,
    require_same_kb_version: bool = True,
    require_same_prompt_version: bool = True,
    require_same_model: bool = True,
    required_metadata_keys: tuple[str, ...] = (),
    enable_required_metadata_check: bool = True,
    enable_intent_check: bool = True,
    enable_entity_check: bool = True,
    enable_numeric_check: bool = True,
    enable_temporal_check: bool = True,
    enable_negation_check: bool = True,
    enable_directional_check: bool = True,
    enable_output_format_check: bool = True,
    enable_freshness_check: bool = True,
    enable_candidate_ambiguity_check: bool = True,
    max_response_age_seconds: float | None = None,
    max_retrieval_age_seconds: float | None = None,
    minimum_response_score_margin: float | None = None,
    query_metadata_key: str = "query",
)
```

| Field | Default | Description |
|---|---|---|
| `retrieval_threshold` | `0.80` | Minimum cosine similarity to consider documents reusable. |
| `response_threshold` | `0.95` | Minimum cosine similarity to return the full cached response. |
| `require_same_namespace` | `True` | Reject candidates from a different namespace. |
| `require_same_kb_version` | `True` | Reject candidates built against a different knowledge-base version. |
| `require_same_prompt_version` | `True` | Reject candidates generated with a different prompt template. |
| `require_same_model` | `True` | Reject responses from a different LLM. |
| `required_metadata_keys` | `()` | Metadata keys that must exist and match for either reuse tier. |
| `enable_required_metadata_check` | `True` | Enable configured required-metadata comparisons. |
| `enable_intent_check` | `True` | Reject clear response-operation or question-focus changes. |
| `enable_entity_check` | `True` | Compare explicit or lightweight critical entities. |
| `enable_numeric_check` | `True` | Compare numbers, percentages, currencies, and years. |
| `enable_temporal_check` | `True` | Compare explicit or detected temporal scope. |
| `enable_negation_check` | `True` | Detect negation and common opposite operations. |
| `enable_directional_check` | `True` | Detect `from A to B` reversals. |
| `enable_output_format_check` | `True` | Compare requested formats, languages, and item counts. |
| `enable_freshness_check` | `True` | Apply configured or request-provided maximum ages. |
| `enable_candidate_ambiguity_check` | `True` | Apply the configured top-score margin. |
| `max_response_age_seconds` | `None` | Optional maximum record age for full responses. |
| `max_retrieval_age_seconds` | `None` | Optional maximum record age for retrieval artifacts. |
| `minimum_response_score_margin` | `None` | Optional minimum gap between the two best exact scores. |
| `query_metadata_key` | `"query"` | Metadata key containing query text for lightweight checks. |

Similarity is cosine similarity in the range `[-1.0, 1.0]`. For well-trained text embedding models, values above `0.90` typically represent near-identical meaning.

**Example — broader reuse:**

```python
from remem import Client, ReusePolicy

client = Client(
    policy=ReusePolicy(
        retrieval_threshold=0.70,
        response_threshold=0.90,
    )
)
```

**Example — allow cross-model or cross-prompt-version reuse** (useful during experimentation):

```python
policy = ReusePolicy(require_same_model=False, require_same_prompt_version=False)
```

---

## ReuseOutcome

Returned by `check()` and `get_or_compute()`.

```python
outcome.decision          # ReuseDecision — RESPONSE_REUSED | RETRIEVAL_REUSED | MISS
outcome.result             # cached response (RESPONSE_REUSED) or None
outcome.references         # list[str] — cached document IDs, if any
outcome.similarity_score   # float — cosine similarity to the best match (0.0 for MISS)
outcome.reason              # str — human-readable explanation of the decision
outcome.matched_record_id  # UUID | None — ID of the matched record, if any
outcome.diagnostics        # dict[str, Any] — structured policy/distributed trace
```

---

## ReuseDecision

```python
from remem import ReuseDecision

ReuseDecision.RESPONSE_REUSED    # full cache hit — LLM and vector DB both skipped
ReuseDecision.RETRIEVAL_REUSED   # partial hit — documents cached, response recomputed
ReuseDecision.MISS               # no usable previous work found
```

---

## ExecutionResult

The return type expected from the `compute_callback` passed to `get_or_compute()`.

```python
from remem import ExecutionResult

ExecutionResult(
    response: Any,
    references: list[str] = [],
    metadata: dict[str, Any] = {},
)
```

---

## ExecutionRecord

The data model stored internally. Exposed for advanced use cases such as direct `store()` calls or inspecting the full store via `all()`.

```python
from remem.models.execution_record import ExecutionRecord

record.id           # UUID
record.embedding    # list[float]
record.response     # Any
record.references   # list[str]
record.context       # ExecutionContext
record.hit_count    # int — incremented each time this record is reused
record.created_at   # datetime
```

---

## Distributed Mode

Install `remem-ai[redis]`, then pass `DistributedConfig` to `Client`:

```python
from remem import Client, DistributedConfig

client = Client(
    distributed=DistributedConfig(
        redis_url="redis://localhost:6379/0",
        key_prefix="my-app:prod:remem",
        node_id="worker-1",
    )
)
```

| Field | Default | Description |
|---|---|---|
| `enabled` | `True` | Enables distributed behavior when supplied to `Client`. |
| `backend` | `"redis"` | Backend selector; 1.2 supports Redis. |
| `redis_url` | environment or localhost | Redis URL, including database and optional TLS/authentication. |
| `key_prefix` | `"remem:v1.2"` | Redis key isolation prefix. |
| `node_id` | generated | Source node recorded in distributed envelopes. |
| `local_cache` | `True` | Cache successful remote reads/writes in the local backend. |
| `sync_on_read` | `True` | Replace local read-through state with the remote snapshot. |
| `fallback_to_local` | `True` | Continue locally and queue operations during Redis failures. |
| `socket_timeout_seconds` | `1.0` | Redis connect/read timeout. |
| `retry_attempts` | `1` | Additional attempts for each Redis operation. |
| `retry_delay_seconds` | `0.05` | Delay between attempts. |
| `record_ttl_seconds` | `None` | Optional logical maximum Redis record age. |
| `duplicate_work_prevention` | `True` | Enables `get_or_compute()` coalescing. |
| `lock_ttl_seconds` | `60.0` | Automatic computation-lock expiration. |
| `lock_wait_timeout_seconds` | `10.0` | Maximum time a contender polls for a result. |
| `lock_poll_interval_seconds` | `0.05` | Polling interval during contention. |

Distributed `auto` mode resolves to exact cosine; explicit HNSW is rejected to
avoid stale per-process indexes. `DistributedBackend` extends `StorageInterface`
with health and token-lock primitives; `RedisStorage` is the built-in
implementation. See the [distributed deployment guide](distributed.md) for the
consistency, failure, scale, and security guarantees.

---

## Storage Backends

| Backend | Import | Persistence | Best For |
|---|---|---|---|
| `JsonStorage` (default) | `from remem import JsonStorage` | Durable — survives restarts | Local long-running applications |
| `InMemoryStorage` | `from remem import InMemoryStorage` | Volatile — lost on exit | Tests, notebooks, short-lived jobs |
| `RedisStorage` | `from remem import RedisStorage` | Shared Redis records | Distributed mode through `DistributedConfig` |
| Custom | Subclass `StorageInterface` | Your choice | Postgres, S3, or another system you implement |

**`JsonStorage` details:**

- Default path: `remem_store.json` in the current working directory
- Uses an atomic write cycle (write to `.tmp` → `os.replace`), so a crash mid-save cannot corrupt the store
- Includes retry logic on Windows for transient `PermissionError` raised by antivirus or file-indexing software
- Records are loaded automatically on instantiation

```python
from remem import Client, JsonStorage

client = Client(storage_backend=JsonStorage(filepath="/var/data/remem_cache.json"))
```

**Implementing a custom backend:**

```python
from remem import StorageInterface

class MyStorage(StorageInterface):
    def put(self, record: ExecutionRecord) -> None: ...
    def get(self, entry_id: UUID) -> ExecutionRecord | None: ...
    def get_many(self, entry_ids: Sequence[UUID]) -> list[ExecutionRecord]: ...
    def delete(self, entry_id: UUID) -> bool: ...
    def update(self, record: ExecutionRecord) -> None: ...
    def all(self) -> list[ExecutionRecord]: ...
    def flush(self) -> None: ...
    def load(self) -> None: ...
    def increment_hit(self, entry_id: UUID) -> None: ...
```

`get_many()` must preserve requested candidate order and omit unavailable IDs. Existing custom backends do not need an immediate migration: `StorageInterface` provides a non-scanning compatibility implementation using the already-required `get()` method. Database backends should override it with a native batch lookup and explicitly restore input order when the database does not guarantee it.

---

## Environment Variables

Local Remem does not require environment variables. Distributed mode reads
`REMEM_REDIS_URL` only when `DistributedConfig.redis_url` is not explicitly
provided. The fallback value is `redis://localhost:6379/0`.

---

## Metrics

Every `check()` and `get_or_compute()` call updates telemetry, available at any time:

```python
print(client.metrics.snapshot())
```

```
========== Metrics ==========
Requests:           100
Hits:                73
Misses:              27
Response Reused:     51
Retrieval Reused:    22
Hit Rate:          73.0%
Average Similarity: 0.941
```

| Field | Description |
|---|---|
| `requests` | Total `check()` / `get_or_compute()` calls |
| `hits` | Requests that found a usable previous execution |
| `misses` | Requests with no usable previous execution |
| `response_reused` | Full cache hits — LLM and vector DB both skipped |
| `retrieval_reused` | Partial hits — documents cached, LLM re-run |
| `hit_rate` | `hits / requests`, as a percentage |
| `average_similarity` | Mean cosine similarity across all hits |

Distributed snapshots additionally expose `local_cache_hits`,
`distributed_cache_hits`, `remote_response_reused`, `remote_retrieval_reused`,
`distributed_misses`, `synchronization_events`, `synchronization_failures`,
`backend_failures`, `invalid_distributed_records`, `fallback_to_local`,
`lock_acquisitions`, `lock_contentions`, `lock_timeouts`, and
`duplicate_work_avoided`. These remain zero for local-only clients.

Metrics are in-process only and reset when `Client` is re-instantiated. To persist them, forward `snapshot()` values to your own observability system.
