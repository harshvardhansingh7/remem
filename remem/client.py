from typing import Any, Callable, Optional
from uuid import UUID

from remem.models.execution_record import ExecutionRecord
from remem.reuse.engine import ReuseDecision, ReuseEngine, ReuseOutcome
from remem.similarity.engine import SimilarityEngine
from remem.storage.in_memory_storage import InMemoryStorage
from remem.storage.storage import StorageInterface


class Client:
    """Public entry point into the Remem engine orchestrating dependencies cleanly."""

    def __init__(self, storage_backend: Optional[StorageInterface] = None):
        self.storage: StorageInterface = storage_backend or InMemoryStorage()
        self.similarity = SimilarityEngine()
        self.reuse_planner = ReuseEngine(self.storage, self.similarity)
        self._misses = 0  # Fallback counter trackable directly on missing events

    def store(self, record: ExecutionRecord) -> None:
        """Saves a rich execution record using the single domain object contract."""
        self.storage.put(record)

    def get_or_compute(
        self,
        query_embedding: list[float],
        compute_callback: Callable[[], Any],
        similarity_threshold: float = 0.8,
        response_reuse_threshold: float = 0.95,
    ) -> ReuseOutcome:
        """Flagship reuse planner endpoint."""
        outcome = self.reuse_planner.get_or_compute(
            query_embedding=query_embedding,
            compute_callback=compute_callback,
            similarity_threshold=similarity_threshold,
            response_reuse_threshold=response_reuse_threshold,
        )

        # Stricter Enum evaluation bypassing string values
        if outcome.decision is ReuseDecision.MISS:
            self._misses += 1

        return outcome

    def delete(self, entry_id: UUID) -> bool:
        """Removes an entry by ID from underlying storage."""
        return self.storage.delete(entry_id)

    def all(self) -> list[ExecutionRecord]:
        """Returns all stored execution records."""
        return self.storage.all()

    @property
    def stats(self) -> dict:
        """Exposes observability metrics."""
        entries_list = self.storage.all()
        total_hits = sum(e.hit_count for e in entries_list)

        return {
            "entries": len(entries_list),
            "hits": total_hits,
            "misses": self._misses,
        }