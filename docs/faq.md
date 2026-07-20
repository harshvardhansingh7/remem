# Frequently Asked Questions (FAQ)

## What problem does Remem solve?

Modern LLM applications repeatedly perform expensive operations such as retrieval, reranking, and generation—even when similar requests have already been processed.

Remem reduces latency and inference cost by identifying semantically similar requests and intelligently reusing previous executions where appropriate.

---

## How is Remem different from Redis?

Redis is a traditional key-value cache.

A cache hit only occurs when the exact key already exists.

Remem performs semantic lookup instead of exact-key lookup.

Instead of asking:

> "Have I seen this exact request before?"

Remem asks:

> "Have I already completed similar work that can be safely reused?"

This makes Remem particularly useful for LLM applications where semantically equivalent requests rarely produce identical cache keys.

---

## Is Remem a vector database?

No.

Remem may use vector search internally, but it is **not** intended to replace vector databases.

Its primary responsibility is deciding whether previous LLM executions can be safely reused.

Vector search is only one component of that decision process.

---

## Does Remem replace Retrieval-Augmented Generation (RAG)?

No.

Remem complements RAG systems.

It sits alongside existing LLM pipelines and determines whether expensive operations—such as retrieval or generation—can be reused instead of executed again.

---

## Does Remem replace Redis?

Not necessarily.

Redis and Remem solve different problems.

Redis remains the data service; Remem supplies embeddings, compatibility policy,
and response/retrieval/miss decisions. Since 1.2, Remem can optionally use Redis
as its shared record and coordination backend while local mode remains the
zero-service default.

---

## Which LLM providers are supported?

Remem is designed to be model-agnostic.

Any provider capable of generating embeddings and running LLM inference can be integrated.

Support for additional providers will continue to expand.

---

## Which storage backends are supported?

The storage layer is intentionally modular.

Currently implemented storage options are:

* `JsonStorage` for local JSON-file persistence
* `InMemoryStorage` for volatile in-process storage
* `RedisStorage` for optional multi-instance distributed mode
* Custom storage implementations via `StorageInterface`

SQLite, PostgreSQL, and cloud/object storage remain future or custom backends.

---

## Can I use Remem in production?

Remem `1.2.0` is stable for workloads that fit the documented mode boundaries.
Local exact and ANN behavior retains the 1.1 ownership model. Redis distributed
mode provides bounded shared caches, eventual consistency, local fallback, and
best-effort request coalescing; it does not provide distributed ANN, durable
fallback queues, multi-region consistency, or database-grade transactions.

Always refer to the latest release notes before deploying Remem in production environments.

---

## How does Remem decide whether something can be reused?

Reuse decisions are based on multiple factors, including:

* Semantic similarity
* Metadata compatibility
* Execution policies
* User-defined constraints

A high similarity score alone does not guarantee reuse.

The built-in policy supports `namespace`, `kb_version`, `prompt_version`,
`model`, lightweight query signals, freshness, and explicitly required metadata
keys. Arbitrary `ExecutionContext.metadata` values are stored but are not
implicit filters.

---

## Is HNSW required?

No. `pip install remem-ai` keeps exact cosine search and has no USearch
dependency. `pip install "remem-ai[ann]"` enables HNSW. The default `auto` mode
uses HNSW when installed and otherwise exposes a safe exact-cosine fallback;
forced `hnsw_cosine` mode fails with installation guidance when unavailable.
HNSW discovers candidates only. Exact cosine determines final ordering, scores,
and thresholds.

---

## Does Remem reuse the entire LLM response?

Not always.

Depending on compatibility, Remem may reuse:

* Retrieval results
* Final generated responses

In the current API, retrieval reuse means Remem returns cached `references` such as document or chunk IDs. Full response reuse means Remem returns the cached `response`. Arbitrary intermediate pipeline stages are not modeled as separate first-class cache entries yet.

---

## Is Remem open source?

Yes.

Remem is released under the Apache 2.0 License and welcomes community contributions.

Please see the CONTRIBUTING guide before opening pull requests.

---

## How can I contribute?

You can contribute by:

* Reporting bugs
* Suggesting new features
* Improving documentation
* Adding examples
* Implementing new storage backends
* Improving similarity algorithms
* Optimizing performance

Every contribution—large or small—is appreciated.

---

## Where is the project heading?

The long-term vision for Remem includes:

* Distributed semantic caching
* Advanced policy engines
* Cloud-native deployments
* Multi-tenant support
* AI agent memory optimization
* Research-driven execution reuse

See the project roadmap for upcoming milestones.
