# Distributed semantic cache

Remem 1.2 can use Redis as an optional shared record and coordination layer.
Local mode remains the default and requires no Redis dependency or service.

## Installation and Redis setup

```bash
pip install "remem-ai[redis]"
docker run --rm -p 6379:6379 redis:7-alpine
```

Use TLS, authentication, Redis ACLs, and a private network in production. Pass
credentials in `redis_url` or `REMEM_REDIS_URL`; never commit them.

```python
from remem import Client, DistributedConfig, ExecutionContext

config = DistributedConfig(
    redis_url="redis://localhost:6379/0",
    key_prefix="acme-support:remem",
    node_id="api-1",
)
client = Client(distributed=config)

context = ExecutionContext(
    namespace="tenant:acme",
    kb_version="2026-07",
    prompt_version="v4",
    model="my-model",
    metadata={"query": query, "retrieval_filter_hash": filter_hash},
)
outcome = client.get_or_compute(embed(query), run_pipeline, context)
```

Every participating node must use the same embedding model, key prefix, reuse
policy, and compatibility metadata conventions.

## Request flow

```text
request
  |
  v
local fallback state <-- reconnect replay --> Redis record hash
  |                                           |
  +--------------- exact candidates <--------+
                          |
                          v
                 existing ReusePolicy
                          |
              +-----------+-----------+
              |           |           |
           response    retrieval      miss
              |           |           |
              +-----------+------ compute lock (get_or_compute only)
                                      |
                                      v
                          compute once, publish to Redis
```

Redis does not decide semantic equivalence. Remem loads the shared records,
applies existing namespace/version/model/metadata policy checks, performs exact
cosine scoring, and retains the normal response/retrieval/miss behavior.

## Consistency model

The first distributed release is deliberately eventual and last-write-wins by
record UUID:

- successful writes reach Redis before the local cache;
- healthy nodes observe committed remote records on their next request;
- a failed remote write is retained in an in-process pending queue and replayed
  before the next successful read;
- a successful remote snapshot replaces read-through local cache state, so
  remote deletions do not live forever locally;
- `get_or_compute()` uses a deterministic UUID for the same embedding and
  compatibility context, making repeated concurrent publication idempotent.

The pending queue is not durable across a process crash. Use
`fallback_to_local=False` when losing an unpropagated write is less acceptable
than failing the request.

## Duplicate-work prevention

`get_or_compute()` takes a single-Redis expiring lock keyed by the normalized
embedding plus namespace, KB version, prompt version, model, and metadata. The
lock uses `SET NX PX` with a random ownership token and compare-and-delete
release. A contending node polls the shared cache and reuses the first published
response. If the wait expires or Redis is unavailable, it computes normally.

This is best-effort request coalescing, not a global transaction. Work can still
duplicate if computation exceeds `lock_ttl_seconds`, a process pauses past the
lease, Redis fails over without preserving a lock, or callers use separate
`check()` and `remember()` calls. Locks always expire and cannot permanently
block a key.

## Configuration

| Field | Default | Meaning |
|---|---:|---|
| `enabled` | `True` | Enable distributed mode when the config is supplied |
| `backend` | `"redis"` | Distributed backend; Redis is the only 1.2 backend |
| `redis_url` | `REMEM_REDIS_URL` or localhost | Redis connection URL |
| `key_prefix` | `"remem:v1.2"` | Isolates applications/environments in one Redis database |
| `node_id` | generated | Source identity stored with published envelopes |
| `local_cache` | `True` | Cache successful remote reads/writes in the local backend |
| `sync_on_read` | `True` | Replace local read-through state from Redis snapshots |
| `fallback_to_local` | `True` | Continue locally and queue writes during outages |
| `socket_timeout_seconds` | `1.0` | Redis connect/read timeout |
| `retry_attempts` | `1` | Additional attempts per Redis operation |
| `retry_delay_seconds` | `0.05` | Delay between retries |
| `record_ttl_seconds` | `None` | Optional logical record expiration |
| `duplicate_work_prevention` | `True` | Enable `get_or_compute()` locks |
| `lock_ttl_seconds` | `60.0` | Automatic lock expiration |
| `lock_wait_timeout_seconds` | `10.0` | Maximum contention wait |
| `lock_poll_interval_seconds` | `0.05` | Cache polling interval while waiting |

`storage_backend` becomes the local cache/fallback backend when distributed mode
is enabled. If omitted, Remem uses `InMemoryStorage`, avoiding unexpected local
files on every application node.

## Search mode and scale

Distributed mode supports `auto` and `exact_cosine`; `auto` resolves to exact.
Explicit `hnsw_cosine` is rejected because the current per-process HNSW graph
cannot observe another node's writes safely. Redis 1.2 stores records in one
namespaced hash and performs an O(n) snapshot for semantic scoring. It is
intended for bounded shared caches, not millions of records. A future release
should add a remote vector index and paginated/partitioned discovery.

## Failures, reconnect, and observability

Initial connection failure, timeout, and connection loss all use the local
fallback when enabled. `client.distributed_status` reports health, node ID,
pending operation count, and the last backend error. `MetricsSnapshot` adds:

- local and distributed cache hits;
- remote response and retrieval reuse;
- distributed misses;
- synchronization events and failures;
- backend failures, invalid records, and local fallbacks;
- lock acquisition, contention, timeout, and duplicate-work avoidance.

Malformed, expired, and incompatible-schema Redis records are ignored and
removed rather than breaking request processing. Version, namespace, prompt,
model, freshness, and required-metadata policy checks remain unchanged.

## Operational cautions

- `flush_storage()` deletes all Remem records under the configured key prefix
  for every node. Give each environment a distinct prefix.
- Redis record responses and metadata are JSON serialized. Do not store secrets
  unless the Redis deployment and application access controls permit it.
- A key prefix is operational isolation, not an authorization boundary. Retain
  `ExecutionContext.namespace` and required metadata for policy isolation.
- Redis persistence, replication, backup, eviction, and disaster recovery are
  deployment responsibilities.
- The 1.2 client is synchronous; async APIs, Redis Cluster-specific testing,
  multi-region consistency, and distributed ANN are not included.

See [`examples/distributed_redis.py`](../examples/distributed_redis.py) for a
runnable two-node example.
