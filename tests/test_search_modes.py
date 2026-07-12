from __future__ import annotations

import builtins

import pytest

from remem import Client, ExecutionRecord, InMemoryStorage, SearchMode


def block_usearch_import(monkeypatch) -> None:
    real_import = builtins.__import__

    def import_without_usearch(name, *args, **kwargs):
        if name == "usearch.index":
            raise ImportError("simulated missing optional dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_usearch)


def test_auto_uses_hnsw_when_optional_dependency_is_available() -> None:
    pytest.importorskip("usearch.index")

    client = Client(storage_backend=InMemoryStorage())

    assert client.search_mode is SearchMode.AUTO
    assert client.resolved_search_mode is SearchMode.HNSW_COSINE
    assert client.search_fallback_reason is None
    assert client.similarity.backend == "hnsw"

    client.store(
        ExecutionRecord(embedding=[1.0, 0.0], references=[], response="cached")
    )
    assert client.check([1.0, 0.0]).result == "cached"


def test_auto_falls_back_to_exact_with_observable_reason(monkeypatch) -> None:
    block_usearch_import(monkeypatch)

    client = Client(storage_backend=InMemoryStorage())

    assert client.search_mode is SearchMode.AUTO
    assert client.resolved_search_mode is SearchMode.EXACT_COSINE
    assert "usearch" in client.search_fallback_reason
    assert client.similarity.backend == "exact"


def test_exact_cosine_never_requires_optional_dependency(monkeypatch) -> None:
    block_usearch_import(monkeypatch)

    client = Client(storage_backend=InMemoryStorage(), search_mode="exact_cosine")

    assert client.resolved_search_mode is SearchMode.EXACT_COSINE
    assert client.similarity.backend == "exact"


def test_forced_hnsw_fails_clearly_without_optional_dependency(monkeypatch) -> None:
    block_usearch_import(monkeypatch)

    with pytest.raises(ImportError, match=r"pip install remem-ai\[ann\]"):
        Client(storage_backend=InMemoryStorage(), search_mode="hnsw_cosine")


@pytest.mark.parametrize(
    ("legacy_backend", "expected_mode"),
    [
        ("exact", SearchMode.EXACT_COSINE),
        ("hnsw", SearchMode.HNSW_COSINE),
    ],
)
def test_legacy_similarity_backend_remains_compatible(
    legacy_backend, expected_mode
) -> None:
    if legacy_backend == "hnsw":
        pytest.importorskip("usearch.index")

    with pytest.deprecated_call(match="similarity_backend is deprecated"):
        client = Client(
            InMemoryStorage(),
            None,
            legacy_backend,
        )

    assert client.resolved_search_mode is expected_mode


def test_conflicting_new_and_legacy_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be configured together"):
        Client(
            storage_backend=InMemoryStorage(),
            similarity_backend="exact",
            search_mode="hnsw_cosine",
        )


def test_invalid_search_mode_lists_supported_values() -> None:
    with pytest.raises(
        ValueError,
        match="auto, exact_cosine, hnsw_cosine",
    ):
        Client(storage_backend=InMemoryStorage(), search_mode="cosine")


def test_invalid_legacy_backend_preserves_value_error_behavior() -> None:
    with pytest.raises(ValueError, match="either 'exact' or 'hnsw'"):
        Client(storage_backend=InMemoryStorage(), similarity_backend="invalid")
