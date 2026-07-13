from __future__ import annotations

import builtins
import os
from uuid import uuid4

import pytest

from remem import AnnConfig, Client, InMemoryStorage
from remem.models.execution_record import ExecutionRecord
from remem.similarity.engine import SimilarityEngine
from remem.similarity.index import (
    AnnIndexStateError,
    HnswSimilarityIndex,
    rerank_candidates,
)
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
    with pytest.raises(ValueError, match="candidate_count"):
        AnnConfig(candidate_count=0)


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


def test_hnsw_reranks_approximate_candidates_with_exact_cosine() -> None:
    pytest.importorskip("usearch.index")
    best = record([1.0, 0.0], "best")
    middle = record([0.8, 0.2], "middle")
    worst = record([0.0, 1.0], "worst")
    entries = [best, middle, worst]
    index = HnswSimilarityIndex(AnnConfig(candidate_count=3))

    index.search([1.0, 0.0], entries, threshold=-1.0, top_k=3)

    class ApproximateResults:
        keys = [2, 1, 0]
        distances = [0.0, 0.1, 0.2]

    class ApproximateIndex:
        def search(self, query, count):
            assert count == 3
            return ApproximateResults()

    index._index = ApproximateIndex()
    matches = index.search([1.0, 0.0], entries, threshold=-1.0, top_k=3)

    assert [entry.id for entry, _ in matches] == [best.id, middle.id, worst.id]
    assert [score for _, score in matches] == pytest.approx([1.0, 0.9701425, 0.0])


def test_hnsw_candidate_count_limits_discovery_but_not_small_indexes() -> None:
    pytest.importorskip("usearch.index")
    entries = [record([1.0, 0.0]), record([0.0, 1.0])]
    index = HnswSimilarityIndex(AnnConfig(candidate_count=10))

    assert len(index.search([1.0, 0.0], entries, threshold=-1.0, top_k=1)) == 1


def test_exact_reranking_deduplicates_ids_and_applies_threshold() -> None:
    best = record([1.0, 0.0], "best")
    rejected = record([0.0, 1.0], "rejected")

    matches = rerank_candidates(
        [1.0, 0.0],
        [rejected.id, best.id, best.id],
        {best.id: best, rejected.id: rejected},
        threshold=0.5,
        top_k=2,
    )

    assert matches == [(best, pytest.approx(1.0))]


def test_exact_reranking_reports_missing_candidate_ids() -> None:
    missing_id = uuid4()

    with pytest.raises(AnnIndexStateError, match=str(missing_id)):
        rerank_candidates([1.0, 0.0], [missing_id], {}, threshold=0.0, top_k=1)


def test_hnsw_reports_unknown_internal_candidate_labels() -> None:
    pytest.importorskip("usearch.index")
    entry = record([1.0, 0.0])
    index = HnswSimilarityIndex()
    index.search([1.0, 0.0], [entry], threshold=0.0, top_k=1)

    class CorruptResults:
        keys = [99]
        distances = [0.0]

    class CorruptIndex:
        def search(self, query, count):
            return CorruptResults()

    index._index = CorruptIndex()

    with pytest.raises(AnnIndexStateError, match="unknown internal label 99"):
        index.search([1.0, 0.0], [entry], threshold=0.0, top_k=1)


def test_reuse_threshold_uses_exact_reranked_score() -> None:
    pytest.importorskip("usearch.index")
    best = record([1.0, 0.0], "best")
    below_threshold = record([0.0, 1.0], "below")
    storage = InMemoryStorage()
    storage.put(best)
    storage.put(below_threshold)
    client = Client(
        storage_backend=storage,
        search_mode="hnsw_cosine",
        ann_config=AnnConfig(candidate_count=2),
    )

    client.similarity.find_all_matches(
        [1.0, 0.0], storage.all(), threshold=-1.0, top_k=2
    )

    class ApproximateResults:
        keys = [1, 0]
        distances = [0.0, 0.5]

    class ApproximateIndex:
        def search(self, query, count):
            return ApproximateResults()

    client.similarity._index._index = ApproximateIndex()
    outcome = client.check([1.0, 0.0])

    assert outcome.matched_record_id == best.id
    assert outcome.similarity_score == pytest.approx(1.0)


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
