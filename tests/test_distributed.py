from __future__ import annotations

import builtins
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from remem import (
    Client,
    DistributedConfig,
    ExecutionContext,
    ExecutionResult,
    InMemoryStorage,
    ReuseDecision,
    SearchMode,
)
from tests.distributed_helpers import SharedDistributedBackend


def config(node_id: str, **overrides) -> DistributedConfig:
    options = {
        "node_id": node_id,
        "key_prefix": "tests:distributed",
        "lock_wait_timeout_seconds": 2,
        "lock_poll_interval_seconds": 0.01,
        **overrides,
    }
    return DistributedConfig(**options)


def distributed_client(
    backend: SharedDistributedBackend,
    node_id: str,
    **config_overrides,
) -> Client:
    return Client(
        storage_backend=InMemoryStorage(),
        distributed=config(node_id, **config_overrides),
        distributed_backend=backend,
    )


def context(query: str, **kwargs) -> ExecutionContext:
    return ExecutionContext(metadata={"query": query}, **kwargs)


def test_local_mode_remains_unchanged_without_redis_dependency(monkeypatch) -> None:
    real_import = builtins.__import__

    def no_redis(name, *args, **kwargs):
        if name == "redis":
            raise ImportError("redis intentionally unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_redis)
    client = Client(storage_backend=InMemoryStorage(), search_mode="exact_cosine")
    client.remember([1.0, 0.0], "local")

    assert client.check([1.0, 0.0]).result == "local"
    assert client.distributed_status == {"enabled": False}


def test_redis_dependency_is_required_only_when_enabled(monkeypatch) -> None:
    real_import = builtins.__import__

    def no_redis(name, *args, **kwargs):
        if name == "redis":
            raise ImportError("redis intentionally unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_redis)
    with pytest.raises(ImportError, match=r"remem-ai\[redis\]"):
        Client(distributed=config("node-a"))


def test_distributed_auto_uses_coherent_exact_search() -> None:
    client = distributed_client(SharedDistributedBackend(), "node-a")

    assert client.search_mode is SearchMode.AUTO
    assert client.resolved_search_mode is SearchMode.EXACT_COSINE
    assert "remote writes" in (client.search_fallback_reason or "")


def test_distributed_hnsw_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot safely observe remote writes"):
        Client(
            distributed=config("node-a"),
            distributed_backend=SharedDistributedBackend(),
            search_mode="hnsw_cosine",
        )


def test_two_nodes_share_response_and_metrics() -> None:
    backend = SharedDistributedBackend()
    first = distributed_client(backend, "node-a")
    second = distributed_client(backend, "node-b")
    ctx = context("How do I freeze my debit card?")
    first.remember([1.0, 0.0], "Freeze it in settings", ["card-guide"], ctx)

    outcome = second.check([1.0, 0.0], ctx)
    snapshot = second.metrics.snapshot()

    assert outcome.decision is ReuseDecision.RESPONSE_REUSED
    assert outcome.result == "Freeze it in settings"
    assert outcome.diagnostics["distributed"]["source"] == "remote"
    assert snapshot.distributed_cache_hits == 1
    assert snapshot.remote_response_reused == 1


def test_remote_retrieval_reuse_preserves_policy() -> None:
    backend = SharedDistributedBackend()
    first = distributed_client(backend, "node-a")
    second = distributed_client(backend, "node-b")
    first.remember(
        [1.0, 0.0],
        "It was built in 1889.",
        ["eiffel-history"],
        context("When was the Eiffel Tower built?"),
    )

    outcome = second.check([1.0, 0.0], context("Where is the Eiffel Tower located?"))

    assert outcome.decision is ReuseDecision.RETRIEVAL_REUSED
    assert outcome.result is None
    assert outcome.references == ["eiffel-history"]
    assert second.metrics.snapshot().remote_retrieval_reused == 1


@pytest.mark.parametrize(
    ("cached", "current"),
    [
        (context("query", namespace="a"), context("query", namespace="b")),
        (context("query", kb_version="1"), context("query", kb_version="2")),
        (
            context("query", prompt_version="1"),
            context("query", prompt_version="2"),
        ),
    ],
)
def test_distributed_records_preserve_isolation(cached, current) -> None:
    backend = SharedDistributedBackend()
    first = distributed_client(backend, "node-a")
    second = distributed_client(backend, "node-b")
    first.remember([1.0, 0.0], "private", context=cached)

    outcome = second.check([1.0, 0.0], current)

    assert outcome.decision is ReuseDecision.MISS
    assert second.metrics.snapshot().distributed_misses == 1


def test_duplicate_get_or_compute_is_coalesced_across_nodes() -> None:
    backend = SharedDistributedBackend()
    first = distributed_client(backend, "node-a")
    second = distributed_client(backend, "node-b")
    started = threading.Event()
    release = threading.Event()
    calls = 0
    calls_lock = threading.Lock()
    ctx = context("What is the refund policy?")

    def compute() -> ExecutionResult:
        nonlocal calls
        with calls_lock:
            calls += 1
        started.set()
        assert release.wait(timeout=2)
        return ExecutionResult(response="14 days", references=["refund-policy"])

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(first.get_or_compute, [1.0, 0.0], compute, ctx)
        assert started.wait(timeout=1)
        second_future = executor.submit(second.get_or_compute, [1.0, 0.0], compute, ctx)
        time.sleep(0.05)
        release.set()
        first_outcome = first_future.result(timeout=3)
        second_outcome = second_future.result(timeout=3)

    assert calls == 1
    assert first_outcome.decision is ReuseDecision.MISS
    assert second_outcome.decision is ReuseDecision.RESPONSE_REUSED
    assert second.metrics.snapshot().duplicate_work_avoided == 1
    assert len(backend.all()) == 1


def test_outage_falls_back_locally_and_replays_after_reconnect() -> None:
    backend = SharedDistributedBackend()
    first = distributed_client(backend, "node-a")
    backend.available = False

    outcome = first.get_or_compute(
        [1.0, 0.0],
        lambda: ExecutionResult("fallback", ["local-doc"]),
        context("outage query"),
    )

    assert outcome.result == "fallback"
    assert first.distributed_status["pending_operations"] >= 1
    assert first.metrics.snapshot().fallback_to_local > 0

    backend.available = True
    first.all()
    second = distributed_client(backend, "node-b")
    reused = second.check([1.0, 0.0], context("outage query"))

    assert reused.decision is ReuseDecision.RESPONSE_REUSED
    assert reused.result == "fallback"
    assert first.distributed_status["pending_operations"] == 0


def test_connection_loss_reuses_synchronized_local_record() -> None:
    backend = SharedDistributedBackend()
    client = distributed_client(backend, "node-a")
    ctx = context("cached before outage")
    client.remember([1.0, 0.0], "cached response", ["doc"], ctx)
    assert client.check([1.0, 0.0], ctx).result == "cached response"
    initial_local_hits = client.metrics.snapshot().local_cache_hits

    backend.available = False
    outcome = client.check([1.0, 0.0], ctx)

    assert outcome.decision is ReuseDecision.RESPONSE_REUSED
    assert outcome.result == "cached response"
    assert outcome.diagnostics["distributed"]["source"] == "local"
    assert client.metrics.snapshot().local_cache_hits == initial_local_hits + 1


def test_outage_can_be_configured_to_fail_closed() -> None:
    backend = SharedDistributedBackend()
    backend.available = False
    client = distributed_client(backend, "node-a", fallback_to_local=False)

    with pytest.raises(ConnectionError, match="unavailable"):
        client.check([1.0, 0.0], context("query"))


def test_lock_expiry_and_token_ownership_are_safe() -> None:
    backend = SharedDistributedBackend()
    assert backend.acquire_lock("work", "owner-a", 20)
    assert not backend.acquire_lock("work", "owner-b", 20)
    assert not backend.release_lock("work", "owner-b")
    time.sleep(0.03)
    assert backend.acquire_lock("work", "owner-b", 20)
    assert not backend.release_lock("work", "owner-a")
    assert backend.release_lock("work", "owner-b")


def test_lock_wait_timeout_computes_without_leaving_a_lock() -> None:
    backend = SharedDistributedBackend()
    client = distributed_client(
        backend,
        "node-a",
        lock_wait_timeout_seconds=0.03,
        lock_poll_interval_seconds=0.005,
    )
    ctx = context("locked query")
    resource = client.storage.computation_key([1.0, 0.0], ctx)
    assert backend.acquire_lock(resource, "other-node", 1000)

    outcome = client.get_or_compute(
        [1.0, 0.0], lambda: ExecutionResult("computed", ["doc"]), ctx
    )

    assert outcome.result == "computed"
    assert client.metrics.snapshot().lock_timeouts == 1
    assert not backend.release_lock(resource, "node-a")
    assert backend.release_lock(resource, "other-node")
