from typing import Callable, Optional
from uuid import UUID

from remem.models.execution_context import ExecutionContext
from remem.models.execution_record import ExecutionRecord
from remem.models.execution_result import ExecutionResult
from remem.reuse.policy import ReusePolicy
from remem.reuse.engine import ReuseEngine, ReuseOutcome
from remem.similarity.engine import SimilarityEngine
from remem.storage.storage import StorageInterface
from remem.storage.json_storage import JsonStorage
from remem.metrics.collector import MetricsCollector


class Client:
    """Public facade coordinating policy engines, telemetry, and durable storage layers."""

    def __init__(
        self,
        storage_backend: Optional[StorageInterface] = None,
        policy: Optional[ReusePolicy] = None,
    ):
        # Default directly to file-backed JSON persistence or use an injected backend
        self.storage: StorageInterface = storage_backend or JsonStorage()
        self.similarity = SimilarityEngine()
        self.policy = policy or ReusePolicy()
        self.metrics = MetricsCollector()
        self.reuse_planner = ReuseEngine(
            self.storage, self.similarity, self.policy, self.metrics
        )

    def store(self, record: ExecutionRecord) -> None:
        """Saves a rich execution record directly."""
        self.storage.put(record)

    def get_or_compute(
        self,
        query_embedding: list[float],
        compute_callback: Callable[[], ExecutionResult],
        context: Optional[ExecutionContext] = None,
    ) -> ReuseOutcome:
        """Flagship reuse planner endpoint accepting structured ExecutionContext."""
        exec_context = context or ExecutionContext()
        return self.reuse_planner.get_or_compute(
            query_embedding=query_embedding,
            compute_callback=compute_callback,
            context=exec_context,
        )

    def delete(self, entry_id: UUID) -> bool:
        return self.storage.delete(entry_id)

    def all(self) -> list[ExecutionRecord]:
        return self.storage.all()

    def save_snapshot(self) -> None:
        """Explicitly serializes internal working tables to durable disk files."""
        if hasattr(self.storage, "save"):
            getattr(self.storage, "save")()

    def load_snapshot(self) -> None:
        """Explicitly synchronizes working tables from disk files."""
        if hasattr(self.storage, "load"):
            getattr(self.storage, "load")()

    def flush_storage(self) -> None:
        """Clears all records from persistence."""
        self.storage.flush()