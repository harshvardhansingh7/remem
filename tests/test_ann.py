from __future__ import annotations

import builtins
import os
from uuid import uuid4

import pytest

from remem import AnnConfig, Client, InMemoryStorage
from remem.models.execution_record import ExecutionRecord
from remem.similarity.engine import SimilarityEngine
from remem.storage.json_storage import JsonStorage


def record(vector: list[float], name: str = "doc") -> ExecutionRecord:
    return ExecutionRecord(
        id=uuid4(), embedding=vector, references=[name], response=name
    )


def hnsw_engine() -> SimilarityEngine:
    pytest.importorskip("usearch.index")
    return SimilarityEngine("hnsw")


def test_hnsw_index_returns_nearest_neighbors_in_score_order() -> None:
    entries = [record([0.0, 1.0], "orthogonal"), record([1.0, 0.0], "exact")]

    matches = hnsw_engine().find_all_matches([1.0, 0.0], entries, top_k=2)

    assert [entry.response for entry, _ in matches] == ["exact", "orthogonal"]
    assert matches[0][1] == pytest.approx(1.0)
    assert matches[1][1] == pytest.approx(0.0)


def test_hnsw_top_k_threshold_and_exact_match() -> None:
    entries = [record([1.0, 0.0]), record([0.9, 0.1]), record([0.0, 1.0])]
    engine = hnsw_engine()

    assert len(engine.find_all_matches([1.0, 0.0], entries, top_k=1)) == 1
    assert len(engine.find_all_matches([1.0, 0.0], entries, threshold=0.999)) == 1
    assert engine.find_best_match([1.0, 0.0], entries).score == pytest.approx(1.0)


def test_hnsw_handles_empty_one_item_and_duplicate_vectors() -> None:
    engine = hnsw_engine()
    one = record([1.0, 0.0])
    duplicate = record([1.0, 0.0])

    assert engine.find_all_matches([1.0, 0.0], []) == []
    assert len(engine.find_all_matches([1.0, 0.0], [one])) == 1
    assert len(engine.find_all_matches([1.0, 0.0], [one, duplicate])) == 2


def test_hnsw_rejects_invalid_vectors_and_top_k() -> None:
    engine = hnsw_engine()
    entries = [record([1.0, 0.0])]

    with pytest.raises(ValueError, match="dimension"):
        engine.find_all_matches([1.0, 0.0, 0.0], entries)
    with pytest.raises(TypeError, match="finite numbers"):
        engine.find_all_matches("not-a-vector", entries)
    with pytest.raises(ValueError, match="top_k"):
        engine.find_all_matches([1.0, 0.0], entries, top_k=0)
    with pytest.raises(ValueError, match="ef_search"):
        AnnConfig(ef_search=0)


def test_hnsw_rebuilds_when_records_are_added_or_updated() -> None:
    first = record([1.0, 0.0], "first")
    second = record([0.0, 1.0], "second")
    engine = hnsw_engine()

    assert engine.find_best_match([0.0, 1.0], [first]).entry.id == first.id
    assert engine.find_best_match([0.0, 1.0], [first, second]).entry.id == second.id

    first.embedding = [0.0, 1.0]
    assert engine.find_all_matches([0.0, 1.0], [first, second], top_k=2)[0][
        1
    ] == pytest.approx(1.0)


def test_hnsw_matches_exact_search_on_a_small_deterministic_dataset() -> None:
    entries = [
        record([1.0, 0.0, 0.0], "a"),
        record([0.8, 0.2, 0.0], "b"),
        record([0.0, 1.0, 0.0], "c"),
        record([0.0, 0.0, 1.0], "d"),
    ]
    query = [0.95, 0.05, 0.0]

    exact = SimilarityEngine("exact").find_all_matches(query, entries, top_k=3)
    ann = hnsw_engine().find_all_matches(query, entries, top_k=3)

    assert [entry.id for entry, _ in ann] == [entry.id for entry, _ in exact]
    assert [score for _, score in ann] == pytest.approx([score for _, score in exact])


def test_hnsw_rebuilds_from_persisted_storage() -> None:
    pytest.importorskip("usearch.index")
    filepath = "ann-test-store.json"
    try:
        storage = JsonStorage(filepath)
        stored = record([1.0, 0.0], "persisted")
        storage.put(stored)

        reloaded = JsonStorage(filepath)
        client = Client(storage_backend=reloaded, search_mode="hnsw_cosine")

        outcome = client.check([1.0, 0.0])
        assert outcome.matched_record_id == stored.id
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


def test_forced_hnsw_reports_how_to_install_missing_dependency(monkeypatch) -> None:
    real_import = builtins.__import__

    def import_without_usearch(name, *args, **kwargs):
        if name == "usearch.index":
            raise ImportError("simulated missing optional dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_usearch)

    with pytest.raises(ImportError, match=r"pip install remem-ai\[ann\]"):
        SimilarityEngine("hnsw")


def test_explicit_exact_client_search_remains_compatible() -> None:
    stored = record([1.0, 0.0], "legacy")
    client = Client(storage_backend=InMemoryStorage(), search_mode="exact_cosine")
    client.store(stored)

    assert client.similarity.backend == "exact"
    assert client.check([1.0, 0.0]).matched_record_id == stored.id
