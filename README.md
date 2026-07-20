# Remem

> **Remember expensive AI work. Reuse it intelligently.**

<p align="center">
  <strong>An AI-native work reuse engine for RAG pipelines, AI agents, and LLM applications</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-blue.svg">
  <img alt="Version" src="https://img.shields.io/badge/Version-v1.2.0-orange">
  <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green.svg">
  <img alt="PyPI" src="https://img.shields.io/badge/PyPI-remem--ai-blue">
</p>

<p align="center">
  <a href="docs/QuickStart.md">Quickstart</a> ·
  <a href="docs/api.md">API Reference</a> ·
  <a href="docs/architecture.md">Architecture</a> ·
  <a href="docs/benchmarks.md">Benchmarks</a> ·
  <a href="docs/roadmap.md">Roadmap</a> ·
  <a href="docs/faq.md">FAQ</a>
</p>

---

## Overview

Modern AI applications repeatedly execute expensive operations — vector database searches, LLM calls, tool executions, reranking passes — for requests that are often semantically identical or very similar to ones already served.

Remem sits between your application and its expensive AI operations. It observes what has already been computed, and for each new request it determines the highest level of previous work that can be safely reused — based on **semantic similarity**, not exact key matching.

```
User asks: "What is our company's vacation policy?"
User asks: "How many paid leaves do employees get?"   <-- Remem reuses the first answer
User asks: "What are the PTO rules?"                  <-- Remem reuses the first answer
```

## Why Not Just Use Redis?

Redis answers: *"Have I seen this exact key before?"*
Remem answers: *"Have I already done similar expensive AI work before?"*

| | Redis | Remem |
|---|---|---|
| Matching strategy | Exact key | Semantic similarity (embeddings) |
| Scope | Key-value lookup | Full AI pipeline reuse |
| Deployment | Separate service | Local Python library, optionally backed by Redis |
| AI-aware | No | Yes — understands embeddings, models, and prompts |

## How It Works

Remem makes a three-way decision for every incoming request, based on how similar it is to work that has already been done:

| Decision | When it happens | Effect |
|---|---|---|
| **Full reuse** | The request is nearly identical to a past one | The cached response is returned immediately — no vector search, no LLM call |
| **Partial reuse** | The request is related, but not identical | Cached retrieval results are reused; only the generation step re-runs |
| **Miss** | Nothing sufficiently similar has been seen | The full pipeline runs, and the result is stored for next time |

Similarity is only ever compared between compatible executions — Remem is aware of concepts like namespace, knowledge-base version, and model, so updating your data or switching models never results in stale or incorrect reuse.

For the full decision model, thresholds, and configuration options, see the [Architecture guide](docs/architecture.md) and [API Reference](docs/api.md).

## Features

- **Three-level reuse engine** — full response reuse, partial retrieval reuse, or a clean miss
- **Semantic similarity matching** — cosine similarity over embedding vectors, no exact-match required
- **Context-aware filtering** — namespace, knowledge-base version, prompt version, and model constraints are respected before anything is reused
- **Optional production ANN path** — HNSW candidate discovery with exact cosine reranking and safe exact-only fallback
- **Incremental, persistent indexes** — client-mediated mutations synchronize immediately; validated indexes fast-load across restarts
- **Namespace-partitioned retrieval** — strict namespace queries avoid unrelated partitions before policy filtering and storage lookup
- **Configurable reuse policy** — tune thresholds and constraints to match your application
- **Durable persistence** — a file-backed store that survives process restarts, with atomic writes
- **In-memory mode** — zero-disk-I/O storage for tests, notebooks, and short-lived jobs
- **Pluggable storage** — bring your own backend by implementing `StorageInterface`
- **Optional distributed cache** — share safe response and retrieval reuse across application instances through Redis
- **Duplicate-work coalescing** — expiring token locks reduce simultaneous identical `get_or_compute()` work
- **Graceful fallback** — continue locally during Redis outages and replay pending writes after reconnect
- **Built-in telemetry** — hit rate, reuse breakdown, and average similarity, out of the box
- **Minimal footprint** — a single runtime dependency (`numpy`)

## Installation

**Requirements:** Python 3.10 or later.

```bash
pip install remem-ai
```

Remem has a single runtime dependency (`numpy`), which is installed automatically.

For optional HNSW approximate-nearest-neighbor search, install the ANN extra:

```bash
pip install "remem-ai[ann]"
```

For the optional Redis distributed cache, install the Redis extra:

```bash
pip install "remem-ai[redis]"
```

<details>
<summary>Installing from source</summary>

```bash
git clone https://github.com/hrsvd/remem.git
cd remem
pip install -e ".[dev]"    # pytest, Ruff, mypy, build, and metadata checks
```
</details>

## Quickstart

```python
from remem import Client, ExecutionResult

client = Client()   # durable persistence by default

def my_pipeline():
    docs = search_vector_db(query)
    answer = call_llm(query, docs)
    return ExecutionResult(response=answer, references=docs)

outcome = client.get_or_compute(
    query_embedding=embed(query),
    compute_callback=my_pipeline,
)

print(outcome.result)   # cached or freshly computed — Remem decides
```

New to Remem? Start with the [5-minute Quickstart](docs/QuickStart.md). For the complete method-by-method reference, see the [API Reference](docs/api.md).

### Multi-signal reuse policy

Provide the current query in context metadata to let Remem reject a cached
response when high cosine similarity hides a meaning-changing difference:

```python
from remem import ExecutionContext, ReusePolicy

policy = ReusePolicy(
    retrieval_threshold=0.90,
    response_threshold=0.98,
    required_metadata_keys=("retrieval_filter_hash",),
    max_response_age_seconds=15 * 60,
    minimum_response_score_margin=0.10,
)
client = Client(policy=policy)

context = ExecutionContext(
    metadata={
        "query": query,
        "retrieval_filter_hash": active_filter_hash,
    }
)
outcome = client.check(embed(query), context=context)
print(outcome.decision, outcome.reason, outcome.diagnostics)
```

The dependency-light policy checks operation/intent, critical entities and
values, time scope, negation, direction, and requested output format. Explicit
metadata such as `operation`, `entities`, `temporal_scope`, and `output_format`
overrides text heuristics. A failed response-only check downgrades a compatible
match to retrieval reuse; failed required metadata or retrieval freshness
returns a miss. Each check has an `enable_*_check` switch, and deployments that
omit query metadata retain threshold-only behavior. These signals are
conservative guardrails, not proof that two responses are interchangeable;
calibrate them on application labels and place dependency state in
`required_metadata_keys`.

### Optional distributed cache

Point multiple clients at the same Redis key prefix to share reusable work:

```python
from remem import Client, DistributedConfig, ExecutionContext

distributed = DistributedConfig(
    redis_url="redis://localhost:6379/0",  # or set REMEM_REDIS_URL
    key_prefix="my-app:remem",
    node_id="worker-1",
)
client = Client(distributed=distributed)

context = ExecutionContext(
    namespace="support",
    metadata={"query": query},
)
outcome = client.get_or_compute(embed(query), my_pipeline, context)
```

Distributed mode is optional and preserves the existing reuse policy. Redis is
the shared record and coordination layer; Remem still performs semantic
matching and response/retrieval/miss decisions. `auto` resolves to exact cosine
in distributed mode so every query sees current remote records. Redis outages
fall back to local operation by default, with in-process pending writes replayed
on reconnect. See the [distributed deployment guide](docs/distributed.md) for
consistency, locking, security, and scale limits.

### Optional ANN search

`auto` is the default: it uses HNSW when the optional dependency is installed and otherwise falls back to exact cosine. You can also force either strategy:

```python
from remem import AnnConfig, Client

client = Client()  # auto
exact_client = Client(search_mode="exact_cosine")
hnsw_client = Client(
    search_mode="hnsw_cosine",
    ann_config=AnnConfig(
        ef_search=100,
        candidate_count=50,
        persistence_path=".remem/records.usearch",
    ),
)
```

Inspect `client.resolved_search_mode` and `client.search_fallback_reason` to see what `auto` selected. Forced HNSW mode fails with an installation hint when the ANN extra is unavailable. The older `similarity_backend="exact"|"hnsw"` argument remains temporarily supported with a deprecation warning.

HNSW retrieves candidate record IDs, storage resolves only those records through ordered batch lookup, and Remem then recalculates exact cosine similarity before sorting and applying reuse thresholds. Built-in ANN query retrieval therefore does not call `storage.all()`. Final scores retain exact cosine semantics (`-1.0` to `1.0`), although ANN candidate discovery does not guarantee exact nearest-neighbor recall. Increasing `candidate_count` can improve recall at the cost of reranking latency; higher `ef_search` can improve HNSW discovery recall at the cost of search latency.

Client-mediated inserts, embedding replacements, and deletions update HNSW incrementally using stable internal keys. Non-vector updates do not touch the native graph. Storage is written first; if ANN mutation fails, Remem rolls back storage and rebuilds the derived index or raises an explicit recovery error.

`persistence_path` is optional and exact search ignores it. When configured, Remem atomically saves each native namespace index plus checksummed, versioned JSON metadata after graph mutations. Startup validates the format, HNSW configuration, vector dimension, storage identity, stable-key mapping, native size, and checksum before loading. Invalid or interrupted artifacts are derived caches: Remem rebuilds only affected namespaces from authoritative storage and exposes the reason through `client.ann_persistence_recovery_reason`. `client.ann_index_stats` reports aggregate record, load, and rebuild counts; a clean restart has one load per non-empty namespace partition and zero rebuilds.

HNSW is partitioned by `ExecutionContext.namespace`. With the default strict policy, queries search only the matching namespace. Within it, Remem applies `kb_version`, `prompt_version`, and `model` eligibility before storage lookup and expands ANN discovery geometrically until it has enough eligible candidates or exhausts the partition. This avoids false misses caused by a fixed global overfetch multiplier while keeping final scoring exact. If `require_same_namespace=False`, all namespace partitions are searched and their eligible candidates are exact-reranked together. Arbitrary `ExecutionContext.metadata` is not an implicit filter; only fields explicitly represented by `ReusePolicy` affect eligibility.

Local users need no migration. See [Migrating to v1.2](docs/migration-1.2.md) for distributed-mode adoption and compatibility notes.

## Documentation

| Guide | Description |
|---|---|
| [Quickstart](docs/QuickStart.md) | Get a working integration in under five minutes |
| [API Reference](docs/api.md) | Every class, method, and configuration option |
| [Architecture](docs/architecture.md) | How Remem is designed internally |
| [Distributed deployment](docs/distributed.md) | Redis setup, consistency, failures, and multi-node operation |
| [Benchmarks](docs/benchmarks.md) | What Remem measures and why |
| [Roadmap](docs/roadmap.md) | Where the project is headed |
| [FAQ](docs/faq.md) | Common questions |
| [Troubleshooting](docs/troubleshooting.md) | Fixes for common issues |
| [Migrating to v1.1](docs/migration-1.1.md) | Upgrade, compatibility, persistence, and operational notes |
| [Migrating to v1.2](docs/migration-1.2.md) | Optional Redis adoption and distributed guarantees |

## Real-world evaluation

The repository includes a reproducible [benchmark framework](benchmarks/README.md)
for labeled response/retrieval reuse quality, hard-negative safety, threshold
sweeps, exact-key and no-reuse baselines, and exact-versus-HNSW performance.
Downloaded datasets, embedding caches, and generated results are not committed.
No benchmark number is promoted here until its dataset, model revision,
configuration, hardware, and result file have been reviewed.

## Testing

From a source checkout, install development dependencies and run the test suite:

```bash
pip install -e ".[ann,redis,dev]"
python -m pytest -v
```

The tests cover exact and ANN search, persistence, serialization, policy
filtering, multi-client coordination, outage fallback, lock contention, and
Redis record validation. Real Redis tests run when `REMEM_TEST_REDIS_URL` is
configured:

```bash
REMEM_TEST_REDIS_URL=redis://localhost:6379/15 \
  python -m pytest tests/integration/test_redis_distributed.py -v
```

Stable-release quality checks are:

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy remem
python -m build
python -m twine check dist/*
```

## Contributing

Contributions are welcome. Architecture discussions and design feedback are especially valuable. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening issues or pull requests.

## License

Licensed under the [Apache License 2.0](LICENSE).

## Author

**Harshvardhan Singh** — building Remem as an open-source AI infrastructure project to explore distributed systems, storage engines, and high-performance backend engineering.

If this project is useful to you, consider giving it a star and following its progress.

---

<p align="center">⭐ Star the repository if you find Remem useful.</p>
