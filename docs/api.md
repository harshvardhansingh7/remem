# API Reference

This document describes every public class, method, and configuration option in Remem. It assumes you've already completed the [Quickstart](QuickStart.md).

The public API follows [Semantic Versioning](https://semver.org/). Breaking changes will only occur in major releases.

## Table of Contents

1. [Client](#client)
2. [ExecutionContext](#executioncontext)
3. [ReusePolicy](#reusepolicy)
4. [ReuseOutcome](#reuseoutcome)
5. [ReuseDecision](#reusedecision)
6. [ExecutionResult](#executionresult)
7. [ExecutionRecord](#executionrecord)
8. [Storage Backends](#storage-backends)
9. [Environment Variables](#environment-variables)

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
)
```

| Parameter | Default | Description |
|---|---|---|
| `storage_backend` | `JsonStorage("remem_store.json")` | Where execution records are persisted. |
| `policy` | `ReusePolicy()` | Similarity thresholds and metadata constraints. |
| `similarity_backend` | `None` | Deprecated compatibility alias for `"exact"` or `"hnsw"`. Do not combine it with a non-auto `search_mode`. |
| `ann_config` | `None` | Optional `AnnConfig` tuning for the HNSW backend. Ignored for exact search. |
| `search_mode` | `"auto"` | `"auto"`, `"exact_cosine"`, or `"hnsw_cosine"`. Enum members from `SearchMode` are also accepted. |

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
    ),
)
```

`m` and `ef_construction` tune index construction. `ef_search` increases candidate-discovery recall when raised, at the cost of query time. `candidate_count` controls how many likely neighbors HNSW returns for exact cosine reranking; its default is 50. Larger candidate sets can improve recall but increase exact-reranking latency. Final ordering, score values, and threshold filtering use exact cosine. Candidate discovery remains approximate and does not guarantee the same neighbors as exhaustive search. The index is in-memory and rebuilds from the configured storage backend after reload, record insertion, deletion, or update.

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
| `context` | Optional scoping metadata. If omitted, all stored records are candidates. |

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
| `metadata` | Custom key-value pairs, reserved for future policy logic. | `{"region": "eu-west-1"}` |

All fields are optional. An omitted field is simply not filtered on.

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

## Storage Backends

| Backend | Import | Persistence | Best For |
|---|---|---|---|
| `JsonStorage` (default) | `from remem import JsonStorage` | Durable — survives restarts | Production, any long-running app |
| `InMemoryStorage` | `from remem import InMemoryStorage` | Volatile — lost on exit | Tests, notebooks, short-lived jobs |
| Custom | Subclass `StorageInterface` | Your choice | Redis, Postgres, S3, or another system you implement |

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

Remem does not require or read any environment variables. All configuration is code-based, via `Client`, `ReusePolicy`, and `ExecutionContext`.

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

Metrics are in-process only and reset when `Client` is re-instantiated. To persist them, forward `snapshot()` values to your own observability system.
