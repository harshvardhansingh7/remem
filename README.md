# Remem

> **Remember expensive AI work. Reuse it intelligently.**

<p align="center">
  <strong>An AI-native work reuse engine for RAG pipelines, AI agents, and LLM applications</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-blue.svg">
  <img alt="Version" src="https://img.shields.io/badge/Version-v1.0.0-orange">
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
| Deployment | Separate service | Python library — import and go |
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
- **Configurable reuse policy** — tune thresholds and constraints to match your application
- **Durable persistence** — a file-backed store that survives process restarts, with atomic writes
- **In-memory mode** — zero-disk-I/O storage for tests, notebooks, and short-lived jobs
- **Pluggable storage** — bring your own backend by implementing `StorageInterface`
- **Built-in telemetry** — hit rate, reuse breakdown, and average similarity, out of the box
- **Minimal footprint** — a single runtime dependency (`numpy`)

## Installation

**Requirements:** Python 3.10 or later.

```bash
pip install remem-ai
```

Remem has a single runtime dependency (`numpy`), which is installed automatically.

<details>
<summary>Installing from source</summary>

```bash
git clone https://github.com/hrsvd/remem.git
cd remem
pip install -e ".[dev]"    # editable install with pytest and ruff
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

## Documentation

| Guide | Description |
|---|---|
| [Quickstart](docs/QuickStart.md) | Get a working integration in under five minutes |
| [API Reference](docs/api.md) | Every class, method, and configuration option |
| [Architecture](docs/architecture.md) | How Remem is designed internally |
| [Benchmarks](docs/benchmarks.md) | What Remem measures and why |
| [Roadmap](docs/roadmap.md) | Where the project is headed |
| [FAQ](docs/faq.md) | Common questions |
| [Troubleshooting](docs/troubleshooting.md) | Fixes for common issues |

## Testing

From a source checkout, install development dependencies and run the test suite:

```bash
pip install -e ".[dev]"
python -m pytest -v
```

The tests cover persistence, serialization, similarity scoring, and policy filtering. If `pytest` is unavailable in a local environment, the current tests can also be run with:

```bash
python -m unittest discover -s tests -v
```

## Contributing

Contributions are welcome. As the project is in beta, architecture discussions and design feedback are especially valuable. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening issues or pull requests.

## License

Licensed under the [Apache License 2.0](LICENSE).

## Author

**Harshvardhan Singh** — building Remem as an open-source AI infrastructure project to explore distributed systems, storage engines, and high-performance backend engineering.

If this project is useful to you, consider giving it a star and following its progress.

---

<p align="center">⭐ Star the repository if you find Remem useful.</p>
