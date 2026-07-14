from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from remem import Client, ExecutionContext, InMemoryStorage
from remem.models.execution_record import ExecutionRecord
from remem.similarity.index import AnnMutationError, HnswSimilarityIndex


def record(
    vector: list[float],
    name: str,
    *,
    record_id=None,
    namespace: str = "",
) -> ExecutionRecord:
    return ExecutionRecord(
        id=record_id or uuid4(),
        embedding=vector,
        references=[],
        response=name,
        context=ExecutionContext(namespace=namespace),
    )


def test_incremental_insert_update_delete_without_full_rebuild() -> None:
    pytest.importorskip("usearch.index")
    index = HnswSimilarityIndex()
    stored = record([1.0, 0.0], "first")

    assert index.upsert(stored) == "inserted"
    assert index.rebuild_count == 0
    assert index.candidate_ids([1.0, 0.0], 1) == [stored.id]

    replacement = record([0.0, 1.0], "replacement", record_id=stored.id)
    assert index.upsert(replacement) == "updated"
    assert index.rebuild_count == 0
    assert index.candidate_ids([0.0, 1.0], 1) == [stored.id]

    assert index.delete(stored.id)
    assert index.candidate_ids([0.0, 1.0], 1) == []
    assert index.rebuild_count == 0


def test_non_vector_update_does_not_mutate_native_index() -> None:
    pytest.importorskip("usearch.index")
    index = HnswSimilarityIndex()
    stored = record([1.0, 0.0], "before")
    index.upsert(stored)
    key = index._key_by_record_id[stored.id]
    native_size = index._index.size

    metadata_only = record(
        [1.0, 0.0],
        "after",
        record_id=stored.id,
        namespace="moved",
    )

    assert index.upsert(metadata_only) == "unchanged"
    assert index._key_by_record_id[stored.id] == key
    assert index._index.size == native_size
    assert index._records[0].response == "after"


def test_repeated_replacement_and_delete_leave_no_duplicate_entries() -> None:
    pytest.importorskip("usearch.index")
    index = HnswSimilarityIndex()
    record_id = uuid4()

    for iteration in range(10):
        current = record(
            [1.0, float(iteration + 1)],
            f"version-{iteration}",
            record_id=record_id,
        )
        index.upsert(current)
        assert index._index.size == 1
        assert len(index._key_by_record_id) == 1

    assert index.delete(record_id)
    assert not index.delete(record_id)
    assert index._index is None
    assert index._key_by_record_id == {}


def test_repeated_reload_does_not_duplicate_native_entries() -> None:
    pytest.importorskip("usearch.index")
    storage = InMemoryStorage()
    records = [record([1.0, 0.0], "first"), record([0.0, 1.0], "second")]
    for stored in records:
        storage.put(stored)

    client = Client(storage_backend=storage, search_mode="hnsw_cosine")
    native_index = client.similarity._index
    initial_rebuild_count = native_index.rebuild_count

    client.reuse_planner.rebuild_index()
    client.reuse_planner.rebuild_index()

    assert native_index._index.size == len(records)
    assert len(native_index._key_by_record_id) == len(records)
    assert native_index.rebuild_count == initial_rebuild_count


def test_client_rolls_back_storage_when_ann_upsert_fails(monkeypatch) -> None:
    pytest.importorskip("usearch.index")
    client = Client(storage_backend=InMemoryStorage(), search_mode="hnsw_cosine")
    stored = record([1.0, 0.0], "failing")

    def fail_upsert(record):
        raise RuntimeError("simulated ANN failure")

    monkeypatch.setattr(client.similarity, "upsert", fail_upsert)

    with pytest.raises(AnnMutationError, match="rolled back"):
        client.store(stored)

    assert client.storage.get(stored.id) is None
    assert client.check([1.0, 0.0]).matched_record_id is None


def test_partial_ann_update_failure_restores_previous_record(monkeypatch) -> None:
    pytest.importorskip("usearch.index")
    client = Client(storage_backend=InMemoryStorage(), search_mode="hnsw_cosine")
    original = record([1.0, 0.0], "original")
    client.store(original)
    real_upsert = client.similarity.upsert

    def mutate_then_fail(record):
        real_upsert(record)
        raise RuntimeError("failure after native mutation")

    monkeypatch.setattr(client.similarity, "upsert", mutate_then_fail)
    replacement = record([0.0, 1.0], "replacement", record_id=original.id)

    with pytest.raises(AnnMutationError, match="rolled back"):
        client.store(replacement)

    assert client.storage.get(original.id).response == "original"
    assert client.check([1.0, 0.0]).result == "original"


def test_ann_delete_failure_restores_storage(monkeypatch) -> None:
    pytest.importorskip("usearch.index")
    client = Client(storage_backend=InMemoryStorage(), search_mode="hnsw_cosine")
    stored = record([1.0, 0.0], "keep")
    client.store(stored)

    def fail_delete(record_id):
        raise RuntimeError("simulated delete failure")

    monkeypatch.setattr(client.similarity, "delete", fail_delete)

    with pytest.raises(AnnMutationError, match="rolled back"):
        client.delete(stored.id)

    assert client.storage.get(stored.id) is stored
    assert client.check([1.0, 0.0]).matched_record_id == stored.id


def test_client_lock_serializes_concurrent_ann_mutations() -> None:
    pytest.importorskip("usearch.index")
    client = Client(storage_backend=InMemoryStorage(), search_mode="hnsw_cosine")
    records = [record([1.0, float(i + 1)], str(i)) for i in range(12)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(client.store, records))

    assert len(client.all()) == len(records)
    assert client.similarity._index._index.size == len(records)
    assert len(client.similarity._index._key_by_record_id) == len(records)
