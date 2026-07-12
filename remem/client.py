import warnings
from typing import Callable, Literal, Optional
from uuid import UUID

from remem.metrics.collector import MetricsCollector
from remem.models.execution_context import ExecutionContext
from remem.models.execution_record import ExecutionRecord
from remem.models.execution_result import ExecutionResult
from remem.reuse.engine import ReuseEngine, ReuseOutcome
from remem.reuse.policy import ReusePolicy
from remem.similarity.engine import SimilarityEngine
from remem.similarity.index import AnnConfig
from remem.similarity.mode import SearchMode, SearchModeResolution, resolve_search_mode
from remem.storage.json_storage import JsonStorage
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
    ):
        # Default directly to file-backed JSON persistence or use an injected backend
        self.storage: StorageInterface = storage_backend or JsonStorage()
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

        resolution, backend = resolve_search_mode(search_mode)
        self.search_resolution: SearchModeResolution = resolution
        self.search_mode = resolution.requested
        self.resolved_search_mode = resolution.resolved
        self.search_fallback_reason = resolution.fallback_reason
        self.similarity = SimilarityEngine(backend, ann_config)
        self.policy = policy or ReusePolicy()
        self.metrics = MetricsCollector()
        self.reuse_planner = ReuseEngine(
            self.storage, self.similarity, self.policy, self.metrics
        )

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

        self.storage.put(
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
