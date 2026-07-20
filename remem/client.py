import warnings
from typing import Callable, Literal, Optional
from uuid import UUID

from remem.distributed.backend import DistributedBackend
from remem.distributed.config import DistributedConfig
from remem.distributed.redis_storage import RedisStorage
from remem.distributed.storage import DistributedStorage
from remem.metrics.collector import MetricsCollector
from remem.models.execution_context import ExecutionContext
from remem.models.execution_record import ExecutionRecord
from remem.models.execution_result import ExecutionResult
from remem.reuse.engine import ReuseEngine, ReuseOutcome
from remem.reuse.policy import ReusePolicy
from remem.similarity.engine import SimilarityEngine
from remem.similarity.index import AnnConfig, AnnIndexStats
from remem.similarity.mode import SearchMode, SearchModeResolution, resolve_search_mode
from remem.storage.json_storage import JsonStorage
from remem.storage.memory_storage import InMemoryStorage
from remem.storage.storage import StorageInterface


class Client:
    """Public facade coordinating policy engines, telemetry, and durable storage layers."""

    def __init__(
        self,
        storage_backend: Optional[StorageInterface] = None,
        policy: Optional[ReusePolicy] = None,
        similarity_backend: Optional[Literal["exact", "hnsw"]] = None,
        ann_config: Optional[AnnConfig] = None,
        *,
        search_mode: SearchMode | str = SearchMode.AUTO,
        distributed: Optional[DistributedConfig] = None,
        distributed_backend: Optional[DistributedBackend] = None,
    ):
        self.metrics = MetricsCollector()
        self.storage: StorageInterface
        self.distributed_config = distributed
        distributed_enabled = distributed is not None and distributed.enabled
        if distributed_backend is not None and not distributed_enabled:
            raise ValueError(
                "distributed_backend requires an enabled DistributedConfig"
            )
        if similarity_backend is not None:
            if similarity_backend not in ("exact", "hnsw"):
                raise ValueError("similarity_backend must be either 'exact' or 'hnsw'.")
            if search_mode not in (SearchMode.AUTO, SearchMode.AUTO.value):
                raise ValueError(
                    "search_mode and the deprecated similarity_backend cannot "
                    "be configured together."
                )
            warnings.warn(
                "similarity_backend is deprecated; use search_mode='exact_cosine' "
                "or search_mode='hnsw_cosine' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            search_mode = {
                "exact": SearchMode.EXACT_COSINE,
                "hnsw": SearchMode.HNSW_COSINE,
            }[similarity_backend]

        if distributed_enabled:
            requested_mode = SearchMode(search_mode)
            if requested_mode is SearchMode.HNSW_COSINE:
                raise ValueError(
                    "Distributed mode requires search_mode='exact_cosine' or 'auto'; "
                    "per-process HNSW indexes cannot safely observe remote writes."
                )
            reason = None
            if requested_mode is SearchMode.AUTO:
                reason = (
                    "Distributed mode selected exact cosine search so remote writes "
                    "are visible without a stale per-process ANN index."
                )
            resolution = SearchModeResolution(
                requested=requested_mode,
                resolved=SearchMode.EXACT_COSINE,
                fallback_reason=reason,
            )
            backend: Literal["exact", "hnsw"] = "exact"
        else:
            resolution, backend = resolve_search_mode(search_mode)
        self.search_resolution: SearchModeResolution = resolution
        self.search_mode = resolution.requested
        self.resolved_search_mode = resolution.resolved
        self.search_fallback_reason = resolution.fallback_reason
        self.similarity = SimilarityEngine(backend, ann_config)
        self.policy = policy or ReusePolicy()
        if distributed_enabled:
            assert distributed is not None
            local_storage = storage_backend or InMemoryStorage()
            remote_storage = distributed_backend or RedisStorage(distributed)
            self.storage = DistributedStorage(
                local_storage, remote_storage, distributed, self.metrics
            )
        else:
            self.storage = storage_backend or JsonStorage()
        self.reuse_planner = ReuseEngine(
            self.storage, self.similarity, self.policy, self.metrics
        )

    @property
    def distributed_status(self) -> dict[str, object]:
        """Return current distributed mode, node, and fallback health state."""

        if not isinstance(self.storage, DistributedStorage):
            return {"enabled": False}
        return {
            "enabled": True,
            "backend": self.storage.config.backend,
            "node_id": self.storage.config.node_id,
            "healthy": self.storage.ping(),
            "pending_operations": self.storage.pending_operation_count,
            "last_error": self.storage.last_backend_error,
        }

    @property
    def ann_persistence_recovery_reason(self) -> Optional[str]:
        """Explain why a configured persistent ANN cache was rebuilt."""

        return self.similarity.persistence_recovery_reason

    @property
    def ann_index_stats(self) -> Optional[AnnIndexStats]:
        """Return read-only ANN load/rebuild lifecycle telemetry."""

        return self.similarity.ann_index_stats

    def check(
        self,
        query_embedding: list[float],
        context: Optional[ExecutionContext] = None,
    ) -> ReuseOutcome:
        """Check whether previous work can be reused — without running your pipeline.

        Returns a :class:`ReuseOutcome` whose ``decision`` field tells you
        exactly what to do next:

        * ``RESPONSE_REUSED`` — ``outcome.result`` is the cached answer.
          Return it directly; skip your entire pipeline.
        * ``RETRIEVAL_REUSED`` — ``outcome.references`` are the cached documents.
          Pass them straight to your LLM (skip the vector-DB search), then
          call :meth:`remember` to store the fresh response.
        * ``MISS`` — no usable previous work.  Run your full pipeline, then
          call :meth:`remember` to store the result.

        Example::

            outcome = client.check(embed(query), context=ctx)

            if outcome.decision == ReuseDecision.RESPONSE_REUSED:
                return outcome.result

            if outcome.decision == ReuseDecision.RETRIEVAL_REUSED:
                response = call_llm(query, outcome.references)  # no vector-DB call
                client.remember(embed(query), response, outcome.references, context=ctx)
                return response

            # MISS: full pipeline
            docs = search_vector_db(query)
            response = call_llm(query, docs)
            client.remember(embed(query), response, docs, context=ctx)
            return response
        """
        exec_context = context or ExecutionContext()
        return self.reuse_planner.check(query_embedding, exec_context)

    def remember(
        self,
        query_embedding: list[float],
        response: object,
        references: Optional[list[str]] = None,
        context: Optional[ExecutionContext] = None,
    ) -> None:
        """Store the result of a pipeline execution so it can be reused later.

        Call this after running your pipeline following a ``MISS`` or
        ``RETRIEVAL_REUSED`` decision from :meth:`check`.

        Example::

            docs = search_vector_db(query)
            answer = call_llm(query, docs)
            client.remember(embed(query), answer, references=docs, context=ctx)
        """
        from uuid import uuid4

        self.reuse_planner.store_record(
            ExecutionRecord(
                id=uuid4(),
                embedding=query_embedding,
                response=response,
                references=references or [],
                context=context or ExecutionContext(),
            )
        )

    def store(self, record: ExecutionRecord) -> None:
        """Saves a rich execution record directly."""
        self.reuse_planner.store_record(record)

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
        return self.reuse_planner.delete_record(entry_id)

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
            self.reuse_planner.rebuild_index()

    def flush_storage(self) -> None:
        """Clears all records from persistence."""
        self.reuse_planner.clear_records()
