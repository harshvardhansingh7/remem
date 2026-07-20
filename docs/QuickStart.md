# Quickstart

Get from zero to a working integration in under five minutes.

## 1. Install

```bash
pip install remem-ai
```

The base install uses exact cosine search. To let the default `auto` mode use
HNSW candidate retrieval when available, install the optional ANN extra:

```bash
pip install "remem-ai[ann]"
```

To share one semantic cache across application instances, install the optional
Redis extra:

```bash
pip install "remem-ai[redis]"
```

To also run the bundled examples and tests, clone the repository instead:

```bash
git clone https://github.com/hrsvd/remem.git
cd remem
pip install -e ".[dev]"
```

**Requirements:** Python 3.10+. Local mode needs no service, API key, or
environment variable. Distributed mode additionally needs Redis.

## 2. Verify the Installation

```bash
python -c "import remem; print(remem.__version__)"
```

If this raises `ModuleNotFoundError`, confirm the correct virtual environment is active.

## 3. Create a Client

```python
from remem import Client

client = Client()   # persists records; auto selects HNSW only when installed
```

Prefer nothing written to disk (useful for tests and notebooks)?

```python
from remem import Client, InMemoryStorage

client = Client(storage_backend=InMemoryStorage())
```

## 4. Scope Your Cache

An `ExecutionContext` tells Remem which stored results are valid candidates for a given request — isolating tenants, knowledge-base versions, and models from one another.

```python
from remem import ExecutionContext

context = ExecutionContext(
    namespace="my-app",
    kb_version="v1",
    model="gpt-4o",
)
```

## 5. Integrate

```python
from remem import ReuseDecision

def handle_query(user_query: str) -> str:
    embedding = my_embedding_model.embed(user_query)
    outcome   = client.check(embedding, context=context)

    if outcome.decision == ReuseDecision.RESPONSE_REUSED:
        return outcome.result                                     # cached — skip everything

    if outcome.decision == ReuseDecision.RETRIEVAL_REUSED:
        answer = my_llm.generate(user_query, outcome.references)  # skip vector DB
        client.remember(embedding, answer, outcome.references, context=context)
        return answer

    # MISS — run the full pipeline
    docs   = my_vector_db.search(embedding)
    answer = my_llm.generate(user_query, docs)
    client.remember(embedding, answer, references=docs, context=context)
    return answer
```

Prefer a single-callback style? Use `get_or_compute` instead:

```python
from remem import Client, ExecutionResult

outcome = client.get_or_compute(
    query_embedding=my_embedding_model.embed(user_query),
    compute_callback=lambda: ExecutionResult(
        response=my_llm.generate(user_query, my_vector_db.search(user_query)),
        references=my_vector_db.last_doc_ids,
    ),
)

return outcome.result
```

## 6. Confirm It's Working

```python
print(client.metrics.snapshot())
```

After a couple of semantically similar requests you should see a non-zero hit rate:

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

## 7. Run the Bundled Examples

Requires a source install (`pip install -e ".[dev]"`).

```bash
python examples/rag_reuse.py           # end-to-end RAG demo — MISS, RESPONSE_REUSED, RETRIEVAL_REUSED
python examples/persistent_storage.py  # durable storage across a simulated restart
```

Both examples use fake embeddings and a fake LLM, so they run with no external dependencies.

For explicit search modes, persistent ANN configuration, and namespace
behavior, see [ANN configuration](api.md#ann-configuration). HNSW discovers
candidates, but final ordering and thresholds always use exact cosine scores.

## What to Read Next

- [API Reference](api.md) — every class, method, and configuration option
- [Architecture](architecture.md) — how Remem decides what to reuse
- [FAQ](faq.md) and [Troubleshooting](troubleshooting.md) — common questions and fixes
