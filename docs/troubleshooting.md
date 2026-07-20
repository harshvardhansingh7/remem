# Troubleshooting

## Errors

**Forced HNSW mode says USearch is unavailable**
Install the optional dependency with `pip install "remem-ai[ann]"`, or use
`search_mode="exact_cosine"`. The default `auto` mode does not fail when
USearch is absent; it exposes `client.search_fallback_reason` and uses exact
cosine safely.

**`CorruptedSnapshotException` on startup**
The JSON store file is malformed. Delete `remem_store.json` (or your custom path) to start fresh. This should not happen under normal operation — `JsonStorage` uses atomic writes to prevent mid-save corruption.

**Persistent ANN index rebuilds at startup**
Inspect `client.ann_persistence_recovery_reason`. Remem rebuilds automatically when configured metadata or a native namespace index is missing, stale relative to storage, corrupt, interrupted, or incompatible with the current HNSW configuration or embedding dimension. Only affected namespace partitions rebuild. After recovery, `client.ann_index_stats.rebuild_count` is incremented and the cache is rewritten; the next clean restart should report one load per non-empty namespace partition and zero rebuilds.

Do not share one `persistence_path` among concurrent processes. Use one path per process/storage owner, or disable ANN persistence. If the configured directory is not writable, mutations raise `AnnMutationError` after attempting storage rollback and deterministic index recovery.

Persistent ANN files are derived caches. Keep the authoritative JSON or custom
storage data during recovery; deleting only the ANN files is safe because Remem
will rebuild them on the next HNSW client initialization.

**Redis extra is unavailable**
Install `pip install "remem-ai[redis]"`. The base package intentionally does not
install or import `redis-py` for local users.

**Distributed status reports unhealthy or pending operations**
Inspect `client.distributed_status["last_error"]`, verify the Redis URL, TLS,
credentials, ACLs, DNS, and network timeout, then call `client.all()` or process
another request after connectivity returns. With fallback enabled, pending
in-process writes replay before the next successful remote snapshot. They do
not survive process termination.

**Distributed mode rejects HNSW**
Use `search_mode="auto"` or `"exact_cosine"`. Remem 1.2 intentionally avoids a
per-process HNSW graph in distributed mode because it cannot observe another
node's writes coherently.

**Redis memory or query latency grows with record count**
Use `record_ttl_seconds`, distinct bounded key prefixes, and application-level
cleanup. Version 1.2 performs an O(n) Redis hash snapshot for exact semantic
search and is not a large-scale distributed vector database.

## Unexpected Behavior

**Unexpected `MISS` decisions**
Check that `ExecutionContext` fields (`namespace`, `kb_version`, `model`, `prompt_version`) match exactly between the `check()` call and the stored records. A mismatch causes the record to be filtered out before similarity is computed.

**All decisions are `MISS`, even for identical queries**
Verify you're passing output from the *same* embedding model for similar queries. Remem compares raw embedding vectors — if your embedding model is non-deterministic, or you've changed models between runs, similarity scores won't reflect true semantic similarity.

**Similarity scores seem low**
The defaults are `retrieval_threshold=0.80` and `response_threshold=0.95`, calibrated for well-trained text embedding models. If you're using low-dimensional or custom embeddings, lower the thresholds via `ReusePolicy` — see [API Reference](api.md#reusepolicy).

## Common Mistakes

**Using a different embedding model when querying vs. when storing**
Remem compares raw vector values directly. If records were stored using one embedding model and queries use another, similarity scores become meaningless. Always use the same model throughout.

**Not passing `context` consistently**
If `context` is passed to `remember()` but omitted on `check()` (or vice versa), the metadata filter won't match and you'll always get `MISS`. Use the same `ExecutionContext` values in both calls.

**Expecting cache hits before any data has been stored**
Remem can't reuse what it hasn't seen. The first time a query is processed, it is always a `MISS`. Hits only appear after similar queries have been stored via `remember()`.

**Treating every `MISS` as an error**
`MISS` is the expected outcome for any genuinely new request. Optimize for a reasonable hit rate, not 100% — a 100% hit rate would indicate over-caching.

**Mutating the knowledge base without bumping `kb_version`**
If documents are updated but `kb_version` stays the same, Remem may serve stale cached responses. Always bump `kb_version` after a knowledge-base update.

**Forgetting to call `remember()` after `MISS` or `RETRIEVAL_REUSED`**
`check()` does not observe your pipeline automatically — you must call `remember()` explicitly for the store to grow. If you'd rather this happen automatically, use `get_or_compute()` instead.

---

If none of the above resolves your issue, please open an issue with a minimal reproduction — see [CONTRIBUTING.md](../CONTRIBUTING.md).
