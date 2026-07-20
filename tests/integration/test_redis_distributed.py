from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

redis = pytest.importorskip("redis")

from remem import (  # noqa: E402
    Client,
    DistributedConfig,
    ExecutionContext,
    ExecutionRecord,
    ExecutionResult,
    InMemoryStorage,
    RedisStorage,
    ReuseDecision,
)

REDIS_URL = os.getenv("REMEM_TEST_REDIS_URL")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not REDIS_URL, reason="REMEM_TEST_REDIS_URL is not configured"),
]


@pytest.fixture
def config() -> DistributedConfig:
    return DistributedConfig(
        redis_url=REDIS_URL or "redis://localhost:6379/15",
        key_prefix=f"remem:integration:{uuid4().hex}",
        node_id="integration-node",
        retry_attempts=0,
        lock_wait_timeout_seconds=5,
        lock_poll_interval_seconds=0.01,
    )


@pytest.fixture
def storage(config: DistributedConfig):
    backend = RedisStorage(config)
    backend.flush()
    yield backend
    backend.flush()


def test_real_redis_crud_lock_and_hit_count(storage: RedisStorage) -> None:
    record = ExecutionRecord(
        embedding=[1.0, 0.0], references=["doc"], response={"answer": 42}
    )
    storage.put(record)
    storage.increment_hit(record.id)

    loaded = storage.get(record.id)
    assert loaded is not None
    assert loaded.response == {"answer": 42}
    assert loaded.hit_count == 1
    assert storage.acquire_lock("job", "token-a", 1000)
    assert not storage.acquire_lock("job", "token-b", 1000)
    assert not storage.release_lock("job", "token-b")
    assert storage.release_lock("job", "token-a")
    assert storage.delete(record.id)
    assert storage.get(record.id) is None


def test_real_redis_two_clients_share_records(config: DistributedConfig) -> None:
    first = Client(
        storage_backend=InMemoryStorage(),
        distributed=config,
        distributed_backend=RedisStorage(config),
    )
    second_config = DistributedConfig(
        **{**config.__dict__, "node_id": "integration-node-b"}
    )
    second = Client(
        storage_backend=InMemoryStorage(),
        distributed=second_config,
        distributed_backend=RedisStorage(second_config),
    )
    first.flush_storage()
    ctx = ExecutionContext(metadata={"query": "shared query"})
    first.remember([1.0, 0.0], "shared response", ["doc"], ctx)

    outcome = second.check([1.0, 0.0], ctx)

    assert outcome.decision is ReuseDecision.RESPONSE_REUSED
    assert outcome.result == "shared response"
    first.flush_storage()


def test_real_redis_invalid_and_expired_records_are_cleaned(
    config: DistributedConfig,
) -> None:
    client = redis.Redis.from_url(config.redis_url, decode_responses=True)
    records_key = f"{config.key_prefix}:records"
    client.hset(records_key, "malformed", "not-json")
    storage = RedisStorage(config, redis_client=client)

    assert storage.all() == []
    assert storage.invalid_record_count == 1
    assert not client.hexists(records_key, "malformed")

    ttl_config = DistributedConfig(**{**config.__dict__, "record_ttl_seconds": 0.01})
    ttl_storage = RedisStorage(ttl_config, redis_client=client)
    record = ExecutionRecord(embedding=[1.0], references=[], response="old")
    ttl_storage.put(record)
    time.sleep(0.02)

    assert ttl_storage.get(record.id) is None
    assert not client.hexists(records_key, str(record.id))


def test_real_redis_incompatible_schema_is_ignored(config: DistributedConfig) -> None:
    client = redis.Redis.from_url(config.redis_url, decode_responses=True)
    records_key = f"{config.key_prefix}:records"
    client.hset(
        records_key,
        "future",
        json.dumps(
            {
                "schema_version": 999,
                "stored_at": "2026-07-20T00:00:00+00:00",
                "record": {},
            }
        ),
    )
    storage = RedisStorage(config, redis_client=client)

    assert storage.all() == []
    assert storage.invalid_record_count == 1


def test_real_redis_duplicate_computation_is_coalesced(
    config: DistributedConfig,
) -> None:
    first = Client(
        storage_backend=InMemoryStorage(),
        distributed=config,
        distributed_backend=RedisStorage(config),
    )
    second_config = DistributedConfig(
        **{**config.__dict__, "node_id": "integration-node-b"}
    )
    second = Client(
        storage_backend=InMemoryStorage(),
        distributed=second_config,
        distributed_backend=RedisStorage(second_config),
    )
    first.flush_storage()
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def compute() -> ExecutionResult:
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(timeout=10)
        return ExecutionResult("one result", ["doc"])

    ctx = ExecutionContext(metadata={"query": "coalesced"})
    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(first.get_or_compute, [1.0], compute, ctx)
        assert started.wait(timeout=1)
        second_future = executor.submit(second.get_or_compute, [1.0], compute, ctx)
        contention_deadline = time.monotonic() + 5
        while (
            second.metrics.snapshot().lock_contentions == 0
            and time.monotonic() < contention_deadline
        ):
            time.sleep(0.01)
        assert second.metrics.snapshot().lock_contentions > 0
        release.set()
        first_future.result(timeout=3)
        second_outcome = second_future.result(timeout=3)

    assert calls == 1
    assert second_outcome.decision is ReuseDecision.RESPONSE_REUSED
    assert second.metrics.snapshot().duplicate_work_avoided == 1
    first.flush_storage()
