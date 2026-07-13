from __future__ import annotations

from uuid import uuid4

import pytest

from remem import AnnConfig, Client, ExecutionContext, InMemoryStorage, ReusePolicy
from remem.models.execution_record import ExecutionRecord
from remem.similarity.index import AnnIndexStateError
from remem.storage.json_storage import JsonStorage
from remem.storage.storage import StorageInterface


def record(
    vector: list[float],
    name: str,
    *,
    namespace: str = "",
) -> ExecutionRecord:
    return ExecutionRecord(
        id=uuid4(),
        embedding=vector,
        references=[],
        response=name,
        context=ExecutionContext(namespace=namespace),
    )


@pytest.mark.parametrize("storage_type", [InMemoryStorage, JsonStorage])
def test_builtin_get_many_preserves_order_and_omits_missing_ids(
    storage_type, tmp_path
) -> None:
    storage = (
        storage_type(tmp_path / "records.json")
        if storage_type is JsonStorage
        else storage_type()
    )
    first = record([1.0, 0.0], "first")
    second = record([0.0, 1.0], "second")
    storage.put(first)
    storage.put(second)

    records = storage.get_many([second.id, uuid4(), first.id])

    assert records == [second, first]


class AllScanSpyStorage(InMemoryStorage):
    def __init__(self) -> None:
        super().__init__()
        self.all_calls = 0
        self.get_many_calls = 0
        self.fail_on_all = False

    def all(self):
        self.all_calls += 1
        if self.fail_on_all:
            raise AssertionError("ANN query performed a full storage scan")
        return super().all()

    def get_many(self, entry_ids):
        self.get_many_calls += 1
        return super().get_many(entry_ids)


def test_ann_query_fetches_candidates_by_id_without_all() -> None:
    pytest.importorskip("usearch.index")
    storage = AllScanSpyStorage()
    best = record([1.0, 0.0], "best")
    storage.put(best)
    client = Client(
        storage_backend=storage,
        search_mode="hnsw_cosine",
        ann_config=AnnConfig(candidate_count=10),
    )
    startup_all_calls = storage.all_calls
    storage.fail_on_all = True

    outcome = client.check([1.0, 0.0])

    assert outcome.matched_record_id == best.id
    assert storage.all_calls == startup_all_calls
    assert storage.get_many_calls == 1


class JsonAllScanSpyStorage(JsonStorage):
    def __init__(self, filepath) -> None:
        self.fail_on_all = False
        self.get_many_calls = 0
        super().__init__(filepath)

    def all(self):
        if self.fail_on_all:
            raise AssertionError("ANN query performed a full storage scan")
        return super().all()

    def get_many(self, entry_ids):
        self.get_many_calls += 1
        return super().get_many(entry_ids)


def test_json_ann_candidate_lookup_does_not_call_all(tmp_path) -> None:
    pytest.importorskip("usearch.index")
    storage = JsonAllScanSpyStorage(tmp_path / "records.json")
    storage.put(record([0.0, 1.0], "below-threshold"))
    client = Client(
        storage_backend=storage,
        policy=ReusePolicy(retrieval_threshold=0.8),
        search_mode="hnsw_cosine",
    )
    storage.fail_on_all = True

    outcome = client.check([1.0, 0.0])

    assert outcome.matched_record_id is None
    assert storage.get_many_calls == 1


def test_ann_query_filters_directly_fetched_candidates_by_namespace() -> None:
    pytest.importorskip("usearch.index")
    storage = AllScanSpyStorage()
    accepted = record([0.99, 0.01], "accepted", namespace="tenant-a")
    rejected = record([1.0, 0.0], "rejected", namespace="tenant-b")
    storage.put(accepted)
    storage.put(rejected)
    client = Client(
        storage_backend=storage,
        search_mode="hnsw_cosine",
        ann_config=AnnConfig(candidate_count=2),
    )
    storage.fail_on_all = True

    outcome = client.check([1.0, 0.0], context=ExecutionContext(namespace="tenant-a"))

    assert outcome.matched_record_id == accepted.id


def test_stale_ann_candidate_is_reported_without_storage_scan() -> None:
    pytest.importorskip("usearch.index")
    storage = AllScanSpyStorage()
    stale = record([1.0, 0.0], "stale")
    storage.put(stale)
    client = Client(storage_backend=storage, search_mode="hnsw_cosine")
    storage.delete(stale.id)
    storage.fail_on_all = True

    with pytest.raises(AnnIndexStateError, match=str(stale.id)):
        client.check([1.0, 0.0])


def test_client_mutations_keep_ann_lookup_and_index_synchronized() -> None:
    pytest.importorskip("usearch.index")
    client = Client(storage_backend=InMemoryStorage(), search_mode="hnsw_cosine")
    stored = record([1.0, 0.0], "original")
    client.store(stored)
    assert client.check([1.0, 0.0]).matched_record_id == stored.id

    replacement = ExecutionRecord(
        id=stored.id,
        embedding=[0.0, 1.0],
        references=[],
        response="replacement",
    )
    client.store(replacement)
    assert client.check([0.0, 1.0]).result == "replacement"

    assert client.delete(stored.id)
    assert client.check([0.0, 1.0]).matched_record_id is None

    client.store(record([1.0, 0.0], "flush-me"))
    client.flush_storage()
    assert client.check([1.0, 0.0]).matched_record_id is None


def test_json_lookup_remains_consistent_after_update_delete_clear_and_reload(
    tmp_path,
) -> None:
    filepath = tmp_path / "records.json"
    storage = JsonStorage(filepath)
    first = record([1.0, 0.0], "first")
    second = record([0.0, 1.0], "second")
    storage.put(first)
    storage.put(second)

    first.response = "updated"
    storage.update(first)
    assert storage.get_many([first.id])[0].response == "updated"

    storage.delete(second.id)
    assert storage.get_many([second.id]) == []

    reloaded = JsonStorage(filepath)
    assert [item.id for item in reloaded.get_many([first.id, second.id])] == [first.id]

    reloaded.flush()
    assert reloaded.get_many([first.id]) == []


class LegacyCustomStorage(AllScanSpyStorage):
    get_many = StorageInterface.get_many

    def __init__(self) -> None:
        super().__init__()
        self.get_calls = 0

    def get(self, entry_id):
        self.get_calls += 1
        return super().get(entry_id)


def test_legacy_custom_storage_inherits_non_scanning_batch_compatibility() -> None:
    pytest.importorskip("usearch.index")
    storage = LegacyCustomStorage()
    stored = record([1.0, 0.0], "legacy")
    storage.put(stored)
    client = Client(storage_backend=storage, search_mode="hnsw_cosine")
    storage.fail_on_all = True

    outcome = client.check([1.0, 0.0])

    assert outcome.matched_record_id == stored.id
    assert storage.get_calls >= 1
