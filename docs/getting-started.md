# Remem — Getting Started Guide

> **Remember expensive AI work. Reuse it intelligently.**

Remem is an **AI Work Reuse Engine**. It sits between your application and its expensive AI operations — vector-DB searches, LLM calls, tool executions — and determines how much of a previous execution can be safely reused for the current request, based on **semantic similarity** rather than exact key matching.

This guide walks you through installation, first integration, and everything you need to run Remem in your own project.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Verification](#3-verification)
4. [How Remem Works](#4-how-remem-works)
5. [Configuration Files and Environment](#5-configuration-files-and-environment)
6. [Step-by-Step First Integration](#6-step-by-step-first-integration)
7. [Folder Overview](#7-folder-overview)
8. [API Reference](#8-api-reference)
9. [Advanced Customisation](#9-advanced-customisation)
10. [Metrics and Telemetry](#10-metrics-and-telemetry)
11. [Common Patterns](#11-common-patterns)
12. [Common Mistakes](#12-common-mistakes)
13. [Frequently Asked Questions](#13-frequently-asked-questions)
14. [Running Tests from Source](#14-running-tests-from-source)

---

## 1. Prerequisites

Before installing Remem, ensure you have:

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10 or later | f-strings, `match` statements, and `list[type]` hints require 3.10+ |
| pip | any recent version | Used for installation |
| NumPy | installed automatically | The only runtime dependency |

You do **not** need:
- A running database or external service
- API keys or environment variables
- Any specific operating system (Windows, macOS, and Linux are all supported)

---

## 2. Installation

### From PyPI (recommended)

```bash
pip install remem-ai
```

This installs Remem and its only runtime dependency (`numpy`).

For optional HNSW candidate retrieval, install the ANN extra. The default
`auto` mode will then select HNSW; without it, `auto` safely uses exact cosine.

```bash
pip install "remem-ai[ann]"
```

### Inside a virtual environment (recommended for projects)

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows

pip install remem-ai
```

### From source (for contributors or to run the examples)

```bash
git clone https://github.com/hrsvd/remem.git
cd remem
pip install -e ".[dev]"
```

The `[dev]` extra installs pytest, Ruff, mypy, and distribution-validation tools. The `-e` flag installs the package in editable mode so code changes are picked up immediately without reinstalling.

---

## 3. Verification

After installation, confirm the package is available and on the expected version:

```python
import remem
print(remem.__version__)   # 1.2.0
```

Or from a terminal:

```bash
python -c "import remem; print(remem.__version__)"
```

Expected output:

```
1.2.0
```

If you see `ModuleNotFoundError`, double-check that the correct virtual environment is activated.

---

## 4. How Remem Works

### The core idea

Every time your application runs an expensive AI operation — calling an LLM, searching a vector database, executing an agent tool — Remem stores the result alongside the embedding for the original request. On future requests, Remem checks whether similar work has already been done.

### The decision flow

```
Incoming Request
      │
      ▼
Generate Embedding          ← your embedding model (OpenAI, Cohere, local, etc.)
      │
      ▼
client.check(embedding)     ← Remem checks stored executions
      │
      ├── RESPONSE_REUSED   ── similarity ≥ response_threshold (default 0.95)
      │                          Return cached LLM response immediately.
      │                          Skip vector DB and LLM entirely.
      │
      ├── RETRIEVAL_REUSED  ── similarity ≥ retrieval_threshold (default 0.80)
      │                          Use cached document IDs.
      │                          Skip vector DB, still call the LLM.
      │                          Store the new response with remember().
      │
      └── MISS              ── no usable previous work found
                                 Run the full pipeline.
                                 Store the result with remember().
```

### The three reuse decisions

| Decision | Condition | What Remem gives you | What you should do |
|---|---|---|---|
| `RESPONSE_REUSED` | Similarity ≥ `response_threshold` | `outcome.result` — the cached answer | Return it. Done. |
| `RETRIEVAL_REUSED` | Similarity ≥ `retrieval_threshold` | `outcome.references` — cached document IDs | Call your LLM with those documents. Store the new response. |
| `MISS` | Similarity below both thresholds | Nothing reusable | Run your full pipeline. Store the result. |

### Metadata filtering (before similarity is computed)

Remem does not compare every stored record against every new request. Before computing any similarity, it filters candidates by metadata compatibility using `ReusePolicy`:

- `namespace` — prevents cross-tenant reuse
- `kb_version` — automatically excludes results that predate a knowledge-base update
- `prompt_version` — excludes results generated with a different prompt template
- `model` — excludes responses from a different LLM

This means you do not need to delete records to invalidate a knowledge-base generation. Bump `kb_version` in your `ExecutionContext` and Remem will route new requests to the full pipeline until that version has been queried. Arbitrary `context.metadata` values are not implicit filters; configure `ReusePolicy.required_metadata_keys` for application dependencies.

---

## 5. Configuration Files and Environment

### Environment variables

Remem does **not** read any environment variables. All configuration is code-based via `Client`, `ReusePolicy`, and `ExecutionContext`.

### Storage file

By default, Remem writes to a file named `remem_store.json` in the current working directory. You can change this path:

```python
from remem import Client, JsonStorage

client = Client(storage_backend=JsonStorage(filepath="/var/data/my_app_cache.json"))
```

The file is created on first write and loaded automatically on startup.

### `pyproject.toml`

If you are working from source, `pyproject.toml` contains:
- Package metadata (`name = "remem-ai"`, with the version read from `remem.__version__`)
- Dependency declarations (`numpy>=1.21`)
- Development extras (`pytest`, `ruff`)
- Ruff linting and formatting configuration

No changes to this file are needed for normal use.

---

## 6. Step-by-Step First Integration

This section walks you through integrating Remem into a simple question-answering application from scratch.

### Step 1 — Install

```bash
pip install remem-ai
```

### Step 2 — Create a client

```python
from remem import Client

# Durable mode: writes to remem_store.json, survives process restarts
client = Client()
```

For development and testing, use ephemeral storage that writes nothing to disk:

```python
from remem import Client, InMemoryStorage

client = Client(storage_backend=InMemoryStorage())
```

### Step 3 — Embed your query

Remem works with any embedding model. It expects a `list[float]` — the dense vector your model produces for the input text.

```python
# Example using the OpenAI client
embedding = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=user_query,
).data[0].embedding
```

Replace this with whichever embedding model your application uses (Cohere, a local sentence-transformer, etc.). The important thing is consistency — the same model must be used both when storing and when querying.

### Step 4 — Check for reusable work

```python
from remem import ReuseDecision

outcome = client.check(embedding)

if outcome.decision == ReuseDecision.RESPONSE_REUSED:
    # Full hit: a very similar question was answered before.
    # Return the cached answer immediately.
    return outcome.result

if outcome.decision == ReuseDecision.RETRIEVAL_REUSED:
    # Partial hit: the same documents are likely relevant.
    # Skip the vector-DB search and call the LLM directly.
    answer = llm.generate(user_query, documents=outcome.references)
    client.remember(embedding, answer, outcome.references)
    return answer

# MISS: no useful previous work. Run the full pipeline.
docs   = vector_db.search(embedding)
answer = llm.generate(user_query, documents=docs)
client.remember(embedding, answer, references=docs)
return answer
```

### Step 5 — Add execution context (recommended)

An `ExecutionContext` scopes which stored records are valid candidates for the current request. If omitted, Remem uses `ExecutionContext()` and still applies the normal policy against its empty namespace, default versions, and `model=None` values.

```python
from remem import ExecutionContext

context = ExecutionContext(
    namespace="support-bot",     # isolates this bot's cache from others
    kb_version="2024-Q4",        # invalidates results when the KB is updated
    model="gpt-4o",              # prevents reusing results from a different model
    prompt_version="v3",         # invalidates results when the prompt changes
)

outcome = client.check(embedding, context=context)
# ... same pattern as Step 4
client.remember(embedding, answer, references=docs, context=context)
```

### Step 6 — Verify the first successful run

After two or more requests with semantically similar content, you should observe `RESPONSE_REUSED` or `RETRIEVAL_REUSED` decisions. Print the metrics to confirm:

```python
print(client.metrics.snapshot())
```

Expected output after a few requests:

```
========== Metrics ==========
Requests:           4
Hits:               2
Misses:             2
Response Reused:    1
Retrieval Reused:   1
Hit Rate:          50.0%
Average Similarity: 0.921
```

A hit rate above 0% with meaningful similarity scores confirms Remem is working correctly.

---

## 7. Folder Overview

```
remem/
├── docs/                        # Documentation
│   ├── getting-started.md       #   ← you are here
│   └── QuickStart.md            # Minimal fast-start guide
├── examples/                    # Runnable demos
│   ├── rag_reuse.py             #   End-to-end RAG demo
│   └── persistent_storage.py   #   Durable storage demo
├── remem/                       # Main Python package
│   ├── __init__.py              #   Public exports and __version__
│   ├── client.py                #   Client — all public methods
│   ├── models/                  #   Data models
│   │   ├── execution_context.py
│   │   ├── execution_record.py
│   │   └── execution_result.py
│   ├── reuse/                   #   Decision logic
│   │   ├── engine.py            #   ReuseEngine
│   │   ├── matcher.py           #   MetadataMatcher
│   │   └── policy.py            #   ReusePolicy
│   ├── similarity/              #   Vector math
│   │   ├── engine.py            #   SimilarityEngine
│   │   └── metrics.py           #   cosine_similarity(), etc.
│   ├── storage/                 #   Persistence layer
│   │   ├── storage.py           #   StorageInterface (abstract)
│   │   ├── json_storage.py      #   JsonStorage
│   │   ├── memory_storage.py    #   InMemoryStorage
│   │   ├── serializer.py        #   JSON round-trip
│   │   ├── snapshot.py          #   StorageSnapshot
│   │   └── exceptions.py        #   Custom exceptions
│   └── metrics/                 #   Telemetry
│       ├── collector.py
│       ├── events.py
│       └── snapshot.py
├── tests/
│   ├── test_persistence.py
│   ├── test_policy.py
│   └── test_similarity.py
├── pyproject.toml
└── README.md
```

---

## 8. API Reference

### `Client`

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
| `storage_backend` | `JsonStorage("remem_store.json")` | Where to persist execution records. |
| `policy` | `ReusePolicy()` | Similarity thresholds and metadata constraints. |
| `similarity_backend` | `None` | Deprecated compatibility alias. Prefer `search_mode`. |
| `ann_config` | `None` | Optional HNSW and persistence configuration. Ignored by exact search. |
| `search_mode` | `"auto"` | Automatic, forced exact cosine, or forced HNSW cosine. |

---

### `check(query_embedding, context)`

Checks whether previous work can be reused — **without running any of your pipeline code**.

```python
outcome: ReuseOutcome = client.check(
    query_embedding: list[float],
    context: ExecutionContext | None = None,
)
```

| Parameter | Description |
|---|---|
| `query_embedding` | Dense vector from your embedding model for the current request. |
| `context` | Optional scoping metadata. If omitted, `ExecutionContext()` is used and normal policy matching still applies. |

**Returns:** `ReuseOutcome`.

Inspect `outcome.decision` to decide what to do next:

```python
from remem import ReuseDecision

if outcome.decision == ReuseDecision.RESPONSE_REUSED:
    return outcome.result                                # done

if outcome.decision == ReuseDecision.RETRIEVAL_REUSED:
    answer = call_llm(query, outcome.references)         # skip vector DB
    client.remember(embedding, answer, outcome.references, context=context)
    return answer

# MISS
docs   = vector_db.search(embedding)
answer = call_llm(query, docs)
client.remember(embedding, answer, references=docs, context=context)
return answer
```

---

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
| `query_embedding` | The same vector you passed to `check()`. |
| `response` | The LLM response or pipeline output. Can be a string, dict, list, or any serialisable Python object. |
| `references` | Document or chunk IDs returned by your vector DB. Stored for potential `RETRIEVAL_REUSED` scenarios. |
| `context` | The same `ExecutionContext` used in the preceding `check()` call. |

Call `remember()` after every `MISS` or `RETRIEVAL_REUSED` decision to grow the reuse store.

---

### `get_or_compute(query_embedding, compute_callback, context)`

All-in-one alternative to `check()` + `remember()`. Calls `compute_callback` only when necessary, stores the result automatically, and always returns a `ReuseOutcome`.

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
| `compute_callback` | A zero-argument callable returning `ExecutionResult`. Called only on `MISS` or `RETRIEVAL_REUSED`. |
| `context` | Optional metadata scoping. |

**When to use `get_or_compute` vs `check` + `remember`:**
- Use `get_or_compute` when your entire pipeline is a single function and you do not need to distinguish between `RETRIEVAL_REUSED` and `MISS` paths (i.e. you always run the same callback).
- Use `check` + `remember` when you want to skip only the vector-DB call on `RETRIEVAL_REUSED` and use the cached document IDs directly — this is the recommended approach for RAG pipelines.

---

### `store / delete / all / flush_storage`

Lower-level storage operations for direct record management.

```python
# Store a fully constructed record (useful for cache warm-up or migration)
client.store(record: ExecutionRecord) -> None

# Delete a specific record
client.delete(entry_id: UUID) -> bool

# Return all stored records
client.all() -> list[ExecutionRecord]

# Wipe everything
client.flush_storage() -> None
```

---

### `save_snapshot / load_snapshot`

Explicit persistence control. Normally you do not need these — `JsonStorage` writes to disk on every `put`, `delete`, and `flush` operation.

```python
client.save_snapshot()   # force-write in-memory state to disk
client.load_snapshot()   # reload records from disk
```

`InMemoryStorage` silently ignores both calls.

---

### `ExecutionContext`

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
| `kb_version` | Tracks the knowledge-base version. Bump when documents change. | `"2024-Q4"`, `"v2.1"` |
| `prompt_version` | Tracks the prompt template version. Bump when the prompt changes significantly. | `"v3"`, `"2024-06"` |
| `model` | The LLM used to generate the response. Prevents cross-model reuse. | `"gpt-4o"`, `"claude-3-5-sonnet"` |
| `metadata` | Query signals and custom values; only policy-configured keys affect compatibility. | `{"query": "...", "region": "eu-west-1"}` |

All fields are optional. An omitted field means the policy does not filter on it.

---

### `ReusePolicy`

Controls thresholds and which metadata fields must match before reuse is considered.

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
| `require_same_kb_version` | `True` | Reject candidates built against a different KB version. |
| `require_same_prompt_version` | `True` | Reject candidates generated with a different prompt template. |
| `require_same_model` | `True` | Reject responses from a different LLM. |

Similarity uses **cosine similarity** in the range `[-1.0, 1.0]`. For well-trained text embedding models, values above `0.90` typically represent near-identical meaning.

---

### `ReuseOutcome`

Returned by `check()` and `get_or_compute()`.

```python
outcome.decision          # ReuseDecision enum
outcome.result            # cached response (RESPONSE_REUSED) or None
outcome.references        # list[str] — cached document IDs
outcome.similarity_score  # float — cosine similarity to the best match (0.0 for MISS)
outcome.reason            # str — human-readable explanation of the decision
outcome.matched_record_id # UUID | None — ID of the matched record, if any
```

---

### `ReuseDecision`

```python
from remem import ReuseDecision

ReuseDecision.RESPONSE_REUSED    # full cache hit — LLM and vector DB skipped
ReuseDecision.RETRIEVAL_REUSED   # partial hit — documents cached, response recomputed
ReuseDecision.MISS               # no usable previous work found
```

---

### `ExecutionResult`

The return type for the `compute_callback` in `get_or_compute()`.

```python
from remem import ExecutionResult

ExecutionResult(
    response: Any,                    # LLM response or any pipeline output
    references: list[str] = [],       # document / chunk IDs used
    metadata: dict[str, Any] = {},    # optional extra data
)
```

---

### `ExecutionRecord`

The data model stored internally. Exposed for advanced use cases such as direct `store()` calls or inspecting the full store with `all()`.

```python
from remem.models.execution_record import ExecutionRecord

record.id           # UUID
record.embedding    # list[float]
record.response     # Any
record.references   # list[str]
record.context      # ExecutionContext
record.hit_count    # int — incremented each time this record is reused
record.created_at   # datetime
```

---

### Storage backends

| Backend | Import | Persistence | Use case |
|---|---|---|---|
| `JsonStorage` | `from remem import JsonStorage` | Durable (file on disk) | Production, any persistent workload |
| `InMemoryStorage` | `from remem import InMemoryStorage` | Volatile (RAM only) | Tests, notebooks, CI pipelines |
| `RedisStorage` | `from remem import RedisStorage` | Shared | Optional distributed mode |
| Custom | Subclass `StorageInterface` | Your choice | Postgres, S3, or another system you implement |

**`JsonStorage` details:**
- Default file path: `remem_store.json` in the current working directory
- Uses atomic writes: data is written to a `.tmp` file, then renamed — a crash mid-save cannot corrupt the store
- On Windows, includes retry logic for transient `PermissionError` from antivirus or file indexers
- Records are loaded automatically when `JsonStorage` is instantiated

**Custom backend interface:**

```python
from remem import StorageInterface

class MyStorage(StorageInterface):
    def put(self, record: ExecutionRecord) -> None: ...
    def get(self, entry_id: UUID) -> ExecutionRecord | None: ...
    def delete(self, entry_id: UUID) -> bool: ...
    def update(self, record: ExecutionRecord) -> None: ...
    def all(self) -> list[ExecutionRecord]: ...
    def flush(self) -> None: ...
    def load(self) -> None: ...
    def increment_hit(self, entry_id: UUID) -> None: ...
```

---

## 9. Advanced Customisation

### Loosen thresholds for broader reuse

Lower `retrieval_threshold` to reuse cached documents for a wider range of queries. Lower `response_threshold` to serve cached responses more aggressively.

```python
from remem import Client, ReusePolicy

client = Client(
    policy=ReusePolicy(
        retrieval_threshold=0.70,    # broader document reuse
        response_threshold=0.90,     # serve cached responses more often
    )
)
```

Start with the defaults and tune based on observed hit rate and response quality.

### Allow cross-model reuse

If your embedding model is the same but you are comfortable reusing responses generated by a different LLM version:

```python
policy = ReusePolicy(require_same_model=False)
```

### Allow cross-prompt-version reuse

During prompt experimentation, you may want document reuse even when the prompt template changed:

```python
policy = ReusePolicy(require_same_prompt_version=False)
```

### Isolate development and production

Use distinct namespaces or KB versions per environment. Results from one environment are never served to the other:

```python
dev_context  = ExecutionContext(namespace="dev",  kb_version="dev-snapshot")
prod_context = ExecutionContext(namespace="prod", kb_version="2024-Q4")
```

### Durable storage at a custom path

```python
from remem import Client, JsonStorage

client = Client(
    storage_backend=JsonStorage(filepath="/mnt/shared/remem_cache.json")
)
```

### Configure optional ANN persistence

```python
from remem import AnnConfig, Client

client = Client(
    search_mode="hnsw_cosine",
    ann_config=AnnConfig(
        candidate_count=50,
        ef_search=100,
        persistence_path=".remem/records.usearch",
    ),
)
```

HNSW performs candidate discovery only; final scores, ordering, and thresholds
use exact cosine. Indexes are partitioned by namespace. Persistence is a derived
cache owned by one process and rebuilds automatically from authoritative storage
when stale, corrupt, or incompatible. See the [API reference](api.md#ann-configuration)
and [v1.2 migration guide](migration-1.2.md).

### Bypass Remem for a single request

Simply do not call `check()`. Run your pipeline normally and optionally call `remember()` to update the store with the fresh result.

---

## 10. Metrics and Telemetry

Remem tracks telemetry automatically for every `check()` and `get_or_compute()` call. Retrieve a snapshot at any time:

```python
snapshot = client.metrics.snapshot()
print(snapshot)
```

Example output:

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
| `requests` | Total number of `check()` or `get_or_compute()` calls |
| `hits` | Requests that found a similar previous execution (above `retrieval_threshold`) |
| `misses` | Requests with no usable previous execution |
| `response_reused` | Full cache hits — LLM and vector DB both skipped |
| `retrieval_reused` | Partial hits — documents cached, LLM re-run |
| `hit_rate` | `hits / requests` expressed as a percentage |
| `average_similarity` | Mean cosine similarity across all hits |

Metrics are in-process only — they reset when the `Client` is re-instantiated. For persistent metrics, record `snapshot()` values to your own observability system.

---

## 11. Common Patterns

### Reset the entire cache

```python
client.flush_storage()   # removes all records; JsonStorage also clears the file
```

### Inspect what is stored

```python
for record in client.all():
    print(record.id, record.context.namespace, record.hit_count, record.response)
```

### Delete a single record

```python
client.delete(record.id)
```

### Warm up the cache at startup

Pre-populate known question-answer pairs so the first real users already benefit from reuse:

```python
from remem import ExecutionContext

KNOWN_QA = [
    ("What is our refund policy?",   "14 business days.", ["policy#returns"]),
    ("How do I reset my password?",  "Settings > Security > Reset.", ["help#auth"]),
]

context = ExecutionContext(namespace="support", kb_version="2024-Q4")

for question, answer, docs in KNOWN_QA:
    client.remember(embed(question), answer, references=docs, context=context)
```

### Use Remem in tests

```python
import pytest
from remem import Client, InMemoryStorage

@pytest.fixture
def remem_client():
    # Isolated per test — nothing written to disk
    return Client(storage_backend=InMemoryStorage())
```

### Multi-tenant SaaS isolation

```python
for tenant_id, user_query in incoming_requests:
    context = ExecutionContext(
        namespace=f"tenant:{tenant_id}",
        kb_version=tenant_kb_version(tenant_id),
        model="gpt-4o",
    )
    outcome = client.check(embed(user_query), context=context)
    # Results from tenant A are never served to tenant B
```

---

## 12. Common Mistakes

**Using a different embedding model when querying vs. when storing**
Remem compares raw vector values. If you stored records using `text-embedding-ada-002` and query with `text-embedding-3-small`, similarity scores will be meaningless. Always use the same model throughout.

**Not passing `context` consistently**
If you pass a `context` to `remember()` but omit it on `check()` (or vice versa), the metadata filter will not match and you will always get `MISS`. Use the same `ExecutionContext` object (or identical field values) in both calls.

**Expecting cache hits before any data is stored**
Remem cannot reuse what it has not seen. The first time a query is processed it is always a `MISS`. Hits appear only after similar queries have been stored with `remember()`.

**Treating all `MISS` outcomes as errors**
`MISS` is the expected outcome for any genuinely new or novel request. Optimise for a reasonable hit rate rather than 100% hits, which would indicate over-caching.

**Mutating the KB without bumping `kb_version`**
If you update your documents but keep `kb_version` unchanged, Remem may serve stale cached responses. Always bump `kb_version` after a knowledge-base update.

**Forgetting to call `remember()` after a `MISS` or `RETRIEVAL_REUSED` decision**
Remem does not observe your pipeline automatically when using `check()`. You must call `remember()` explicitly for the store to grow. If you prefer automatic storage, use `get_or_compute()`.

---

## 13. Frequently Asked Questions

**Do I need Redis or a separate database to run Remem?**
No. Local mode still ships with `JsonStorage` and `InMemoryStorage` and needs no
external service. Install `remem-ai[redis]` only when multiple instances should
share a distributed cache; see the [distributed guide](distributed.md).

**Does Remem work with any LLM or embedding model?**
Yes. Remem is model-agnostic. It takes a `list[float]` embedding vector as input — any embedding model that produces dense float vectors is compatible.

**Does Remem send my data anywhere?**
No. Remem is a local Python library. All data is stored in your own process memory or in the local JSON file. Nothing is sent to any external server.

**What similarity score should I use for my use case?**
The defaults (`retrieval_threshold=0.80`, `response_threshold=0.95`) are calibrated for modern text embedding models. If your hit rate is very low, try lowering `retrieval_threshold` to `0.70`. If you are getting incorrect responses served from cache, raise `response_threshold` to `0.97` or `0.98`.

**Is Remem thread-safe?**
Client-mediated queries and mutations are lifecycle-serialized across threads sharing one `Client`. Direct storage calls bypass that boundary, and multiple clients or processes writing the same storage or ANN path are not coordinated. For those workloads, implement a custom backend with appropriate locking and give each persistent ANN path one process owner.

**Can I pre-populate the cache before users arrive?**
Yes — use `client.remember()` or `client.store()` to seed the store with known question-answer pairs at startup. See [Common Patterns — Warm up the cache](#11-common-patterns).

**Why do I get `MISS` even for identical queries?**
Check that `ExecutionContext` fields match exactly. A difference in `namespace`, `kb_version`, `model`, or `prompt_version` between the stored record and the current request will cause the record to be filtered out before similarity is computed, resulting in `MISS`.

---

*Apache License 2.0 — Remem is open source. Contributions and feedback welcome.*

---

## 14. Running Tests from Source

If you cloned the repository, install the development extras before running the full suite:

```bash
pip install -e ".[dev]"
python -m pytest -v
```

Install `.[ann,dev]` to execute the ANN-enabled cases instead of skipping them.

The current suite covers:

- JSON persistence and serializer round trips
- Similarity scoring and threshold behavior
- Metadata policy compatibility and candidate filtering
- ANN candidate reranking and direct record lookup
- Incremental mutation, persistence, corruption recovery, and namespace isolation

CI runs the base install and RAG example on Python 3.10, 3.11, and 3.12, plus an ANN-enabled suite on Python 3.12. A quality job runs Ruff, formatting, mypy, wheel/sdist builds, and distribution metadata validation.
