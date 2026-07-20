# Migrating from 1.1.0 to 1.2.0

Remem 1.2 is backward compatible for local users. Existing `Client`,
`JsonStorage`, `InMemoryStorage`, `ReusePolicy`, exact search, HNSW, JSON
snapshots, and `ExecutionRecord` serialization continue to work without a data
migration.

## Local-only upgrade

```bash
pip install --upgrade remem-ai==1.2.0
```

No code changes are required. Redis is not imported or installed by the base
package, and local `search_mode="auto"` retains its 1.1 resolution behavior.

## Enabling Redis

```bash
pip install --upgrade "remem-ai[redis]==1.2.0"
```

```python
from remem import Client, DistributedConfig

client = Client(
    distributed=DistributedConfig(
        redis_url="redis://localhost:6379/0",
        key_prefix="my-service:prod:remem",
        node_id="worker-1",
    )
)
```

The injected `storage_backend`, if any, becomes local fallback state. With no
injected storage, distributed mode uses `InMemoryStorage`; local-only mode still
defaults to `JsonStorage`.

## Behavior differences limited to distributed mode

- `search_mode="auto"` resolves to exact cosine for current remote visibility.
- `search_mode="hnsw_cosine"` is rejected.
- `get_or_compute()` uses deterministic distributed record IDs and best-effort
  expiring locks.
- `flush_storage()` clears the shared key prefix, not just one process.
- `MetricsSnapshot` has additional fields with zero defaults, preserving
  existing construction and local behavior.

Use the same embedding model and compatibility conventions across nodes. Read
the [distributed deployment guide](distributed.md) before production rollout,
especially the eventual-consistency, fallback-queue, and O(n) scan limits.
