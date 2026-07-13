"""Search-index implementations used by :mod:`remem.similarity`.

Indexes deliberately store only derived vector data.  Execution records remain
the source of truth in the configured storage backend, so an ANN index can be
rebuilt safely after a process restart.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Mapping, Optional
from uuid import UUID

import numpy as np

from remem.models.execution_record import ExecutionRecord
from remem.similarity.metrics import cosine_similarity


@dataclass(frozen=True)
class AnnConfig:
    """Configuration for the optional HNSW approximate-neighbor index.

    Higher ``ef_search`` values generally improve recall at the cost of query
    latency.  The defaults favor accurate semantic-cache decisions.
    """

    m: int = 16
    ef_construction: int = 200
    ef_search: int = 50
    candidate_count: int = 50

    def __post_init__(self) -> None:
        if self.m <= 0:
            raise ValueError("m must be a positive integer.")
        if self.ef_construction <= 0:
            raise ValueError("ef_construction must be a positive integer.")
        if self.ef_search <= 0:
            raise ValueError("ef_search must be a positive integer.")
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be a positive integer.")


class AnnIndexStateError(RuntimeError):
    """Raised when ANN candidate identifiers cannot be resolved safely."""


def rerank_candidates(
    query_embedding: Sequence[float],
    candidate_ids: Sequence[UUID],
    records_by_id: Mapping[UUID, ExecutionRecord],
    threshold: float,
    top_k: Optional[int],
) -> list[tuple[ExecutionRecord, float]]:
    """Deduplicate ANN candidates and rank them using exact cosine scores."""

    unique_ids = list(dict.fromkeys(candidate_ids))
    missing_ids = [
        record_id for record_id in unique_ids if record_id not in records_by_id
    ]
    if missing_ids:
        missing = ", ".join(str(record_id) for record_id in missing_ids)
        raise AnnIndexStateError(
            f"ANN candidates reference unavailable records: {missing}. "
            "Rebuild the ANN index from authoritative storage."
        )

    matches = [
        (
            records_by_id[record_id],
            cosine_similarity(query_embedding, records_by_id[record_id].embedding),
        )
        for record_id in unique_ids
    ]
    matches = [match for match in matches if match[1] >= threshold]
    matches.sort(key=lambda item: (-item[1], str(item[0].id)))
    return matches if top_k is None else matches[:top_k]


class ExactSimilarityIndex:
    """Dependency-free exhaustive cosine search preserving legacy behavior."""

    def search(
        self,
        query_embedding: Sequence[float],
        entries: Sequence[ExecutionRecord],
        threshold: float,
        top_k: Optional[int],
    ) -> list[tuple[ExecutionRecord, float]]:
        matches = [
            (entry, cosine_similarity(query_embedding, entry.embedding))
            for entry in entries
        ]
        matches = [match for match in matches if match[1] >= threshold]
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches if top_k is None else matches[:top_k]


class HnswSimilarityIndex:
    """Optional USearch/HNSW cosine index synchronized from storage records.

    The index is intentionally rebuildable rather than persisted separately:
    JSON storage remains authoritative, avoiding stale index files and keeping
    index updates and deletes correct across existing storage backends.
    """

    def __init__(self, config: Optional[AnnConfig] = None) -> None:
        try:
            from usearch.index import Index
        except ImportError as exc:  # pragma: no cover - exercised without extra
            raise ImportError(
                "ANN search requires the optional 'usearch' dependency. "
                "Install it with: pip install remem-ai[ann]"
            ) from exc

        self._index_type = Index
        self.config = config or AnnConfig()
        self._index = None
        self._records: list[ExecutionRecord] = []
        self._fingerprint: tuple[tuple[str, tuple[float, ...]], ...] = ()
        self._dimension: Optional[int] = None

    def search(
        self,
        query_embedding: Sequence[float],
        entries: Sequence[ExecutionRecord],
        threshold: float,
        top_k: Optional[int],
    ) -> list[tuple[ExecutionRecord, float]]:
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be a positive integer when provided.")
        if not entries:
            return []

        self.rebuild(entries)
        candidate_ids = self.candidate_ids(query_embedding, top_k)
        records_by_id = {record.id: record for record in self._records}
        return rerank_candidates(
            query_embedding,
            candidate_ids,
            records_by_id,
            threshold,
            top_k,
        )

    def rebuild(self, entries: Sequence[ExecutionRecord]) -> None:
        """Synchronize the derived index from authoritative records."""

        self._synchronize(entries)

    def candidate_ids(
        self,
        query_embedding: Sequence[float],
        top_k: Optional[int],
    ) -> list[UUID]:
        """Return approximate candidate IDs without resolving stored records."""

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be a positive integer when provided.")
        if not self._records:
            return []

        query = self._validate_vector(query_embedding, "query embedding")
        if query.size != self._dimension:
            raise ValueError(
                f"Query embedding dimension {query.size} does not match index "
                f"dimension {self._dimension}."
            )

        requested = self.config.candidate_count
        if top_k is not None:
            requested = max(requested, top_k)
        count = min(requested, len(self._records))
        matches = self._index.search(query, count=count)
        candidate_ids: list[UUID] = []
        for label in matches.keys:
            position = int(label)
            if position < 0 or position >= len(self._records):
                raise AnnIndexStateError(
                    f"ANN index returned unknown internal label {position}. "
                    "Rebuild the ANN index from authoritative storage."
                )
            candidate_ids.append(self._records[position].id)

        return list(dict.fromkeys(candidate_ids))

    def _synchronize(self, entries: Sequence[ExecutionRecord]) -> None:
        fingerprint = tuple(
            (str(entry.id), tuple(float(value) for value in entry.embedding))
            for entry in entries
        )
        if fingerprint == self._fingerprint:
            return

        if not entries:
            self._index = None
            self._records = []
            self._fingerprint = ()
            self._dimension = None
            return

        vectors = [
            self._validate_vector(entry.embedding, f"embedding for {entry.id}")
            for entry in entries
        ]
        dimension = int(vectors[0].size)
        if any(vector.size != dimension for vector in vectors):
            raise ValueError(f"All index embeddings must have dimension {dimension}.")
        if len({entry.id for entry in entries}) != len(entries):
            raise ValueError("ANN index entries must have unique record IDs.")

        index = self._index_type(
            ndim=dimension,
            metric="cos",
            dtype="f32",
            connectivity=self.config.m,
            expansion_add=self.config.ef_construction,
            expansion_search=self.config.ef_search,
        )
        for label, vector in enumerate(vectors):
            index.add(label, vector)

        self._index = index
        self._records = list(entries)
        self._fingerprint = fingerprint
        self._dimension = dimension

    @staticmethod
    def _validate_vector(vector: Sequence[float], name: str) -> np.ndarray:
        if isinstance(vector, (str, bytes)):
            raise TypeError(f"{name} must be a sequence of finite numbers.")
        try:
            array = np.asarray(vector, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a sequence of finite numbers.") from exc
        if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
            raise ValueError(f"{name} must be a non-empty sequence of finite numbers.")
        return array
