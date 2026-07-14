from threading import RLock
from typing import Callable
from uuid import UUID, uuid4

from remem.metrics.collector import MetricsCollector
from remem.metrics.events import MetricEvent
from remem.models.execution_context import ExecutionContext
from remem.models.execution_record import ExecutionRecord
from remem.models.execution_result import ExecutionResult
from remem.reuse.decision import ReuseDecision, ReuseOutcome
from remem.reuse.matcher import MetadataMatcher
from remem.reuse.policy import ReusePolicy
from remem.similarity.engine import SimilarityEngine, SimilarityMatch
from remem.similarity.index import AnnIndexStateError, AnnMutationError
from remem.storage.storage import StorageInterface


class ReuseEngine:
    """Core engine evaluating reusability boundaries and emitting metric events."""

    def __init__(
        self,
        storage: StorageInterface,
        similarity: SimilarityEngine,
        policy: ReusePolicy,
        metrics: MetricsCollector,
    ):
        self.storage = storage
        self.similarity = similarity
        self.policy = policy
        self.metrics = metrics
        self._lifecycle_lock = RLock()
        self.rebuild_index()

    def rebuild_index(self) -> None:
        """Rebuild ANN state outside the query path from authoritative storage."""

        with self._lifecycle_lock:
            if self.similarity.backend == "hnsw":
                self.similarity.rebuild(self.storage.all())

    def store_record(self, record: ExecutionRecord) -> None:
        """Store a record and refresh derived ANN state when configured."""

        with self._lifecycle_lock:
            previous = self.storage.get(record.id)
            self.storage.put(record)
            if self.similarity.backend != "hnsw":
                return
            try:
                self.similarity.upsert(record)
            except Exception as mutation_error:
                try:
                    if previous is None:
                        self.storage.delete(record.id)
                    else:
                        self.storage.put(previous)
                    self.similarity.rebuild(self.storage.all())
                except Exception as recovery_error:
                    raise AnnMutationError(
                        "ANN upsert failed and rollback recovery also failed."
                    ) from recovery_error
                raise AnnMutationError(
                    "ANN upsert failed; the storage write was rolled back and "
                    "the index was rebuilt."
                ) from mutation_error

    def delete_record(self, record_id: UUID) -> bool:
        """Delete storage and ANN state together with rollback recovery."""

        with self._lifecycle_lock:
            previous = self.storage.get(record_id)
            if previous is None or not self.storage.delete(record_id):
                return False
            if self.similarity.backend != "hnsw":
                return True
            try:
                self.similarity.delete(record_id)
            except Exception as mutation_error:
                try:
                    self.storage.put(previous)
                    self.similarity.rebuild(self.storage.all())
                except Exception as recovery_error:
                    raise AnnMutationError(
                        "ANN delete failed and rollback recovery also failed."
                    ) from recovery_error
                raise AnnMutationError(
                    "ANN delete failed; the storage deletion was rolled back and "
                    "the index was rebuilt."
                ) from mutation_error
            return True

    def clear_records(self) -> None:
        """Clear storage and derived ANN state with rollback recovery."""

        with self._lifecycle_lock:
            previous = self.storage.all()
            self.storage.flush()
            if self.similarity.backend != "hnsw":
                return
            try:
                self.similarity.clear()
            except Exception as mutation_error:
                try:
                    for record in previous:
                        self.storage.put(record)
                    self.similarity.rebuild(previous)
                except Exception as recovery_error:
                    raise AnnMutationError(
                        "ANN clear failed and rollback recovery also failed."
                    ) from recovery_error
                raise AnnMutationError(
                    "ANN clear failed; storage was restored and the index rebuilt."
                ) from mutation_error

    def _find_best_compatible(
        self,
        query_embedding: list[float],
        context: ExecutionContext,
        threshold: float,
    ):
        """Shared metadata-filter + similarity scan used by both public methods."""
        with self._lifecycle_lock:
            if self.similarity.backend == "exact":
                all_entries = self.storage.all()
                compatible = MetadataMatcher.filter_candidates(
                    all_entries, context, self.policy
                )
                return self.similarity.find_best_match(
                    query_embedding, compatible, threshold=threshold
                )

            candidate_ids = self.similarity.find_candidate_ids(query_embedding, top_k=1)
            records = self.storage.get_many(candidate_ids)
            resolved_ids = {record.id for record in records}
            missing_ids = [
                record_id
                for record_id in candidate_ids
                if record_id not in resolved_ids
            ]
            if missing_ids:
                missing = ", ".join(str(record_id) for record_id in missing_ids)
                raise AnnIndexStateError(
                    f"ANN candidates reference unavailable records: {missing}. "
                    "Reload or rebuild the client index from authoritative storage."
                )

            compatible = MetadataMatcher.filter_candidates(
                records, context, self.policy
            )
            compatible_ids = {record.id for record in compatible}
            ordered_ids = [
                record_id for record_id in candidate_ids if record_id in compatible_ids
            ]
            matches = self.similarity.rerank_candidate_records(
                query_embedding,
                ordered_ids,
                compatible,
                threshold,
                top_k=1,
            )
            if not matches:
                return None
            entry, score = matches[0]
            return SimilarityMatch(entry=entry, score=score)

    def check(
        self,
        query_embedding: list[float],
        context: ExecutionContext,
    ) -> ReuseDecision:
        """Returns the reuse decision and any cached artifacts without running any callback.

        The caller inspects the decision and routes accordingly:

        * ``RESPONSE_REUSED`` – ``outcome.result`` holds the cached LLM response.
          No pipeline work needed.
        * ``RETRIEVAL_REUSED`` – ``outcome.references`` holds the cached documents.
          Skip the vector-DB search; pass those docs to your LLM, then call
          ``remember()`` to store the result.
        * ``MISS`` – no usable previous work.  Run the full pipeline and call
          ``remember()`` to store the result.
        """
        self.metrics.record(MetricEvent.REQUEST)
        best_match = self._find_best_compatible(
            query_embedding, context, self.policy.retrieval_threshold
        )

        if not best_match:
            self.metrics.record(MetricEvent.MISS)
            return ReuseOutcome(
                result=None,
                decision=ReuseDecision.MISS,
                similarity_score=0.0,
                reason="No compatible execution found.",
            )

        matched = best_match.entry
        self.storage.increment_hit(matched.id)
        score = best_match.score

        if score >= self.policy.response_threshold and matched.response is not None:
            self.metrics.record(MetricEvent.HIT, similarity=score)
            self.metrics.record(MetricEvent.RESPONSE_REUSED)
            return ReuseOutcome(
                result=matched.response,
                decision=ReuseDecision.RESPONSE_REUSED,
                similarity_score=score,
                reason=f"Vector similarity {score:.2f} met response threshold.",
                matched_record_id=matched.id,
                references=matched.references,
            )

        self.metrics.record(MetricEvent.HIT, similarity=score)
        self.metrics.record(MetricEvent.RETRIEVAL_REUSED)
        return ReuseOutcome(
            result=None,
            decision=ReuseDecision.RETRIEVAL_REUSED,
            similarity_score=score,
            reason=f"Vector similarity {score:.2f} met retrieval threshold but fell below response threshold.",
            matched_record_id=matched.id,
            references=matched.references,
        )

    def get_or_compute(
        self,
        query_embedding: list[float],
        compute_callback: Callable[[], ExecutionResult],
        context: ExecutionContext,
    ) -> ReuseOutcome:
        """All-in-one reuse planner: runs ``compute_callback`` only when necessary.

        For the RETRIEVAL_REUSED branch the callback is still invoked (because
        it owns the full pipeline), but ``outcome.references`` carries the
        cached documents so callers that inspect the decision can skip their
        own vector-DB search.  Prefer ``check()`` + ``remember()`` when you
        want explicit control over each pipeline stage.
        """
        self.metrics.record(MetricEvent.REQUEST)
        best_match = self._find_best_compatible(
            query_embedding, context, self.policy.retrieval_threshold
        )

        # MISS: run full pipeline and store the result
        if not best_match:
            self.metrics.record(MetricEvent.MISS)
            exec_result = compute_callback()
            self.store_record(
                ExecutionRecord(
                    id=uuid4(),
                    embedding=query_embedding,
                    references=exec_result.references,
                    response=exec_result.response,
                    context=context,
                )
            )
            return ReuseOutcome(
                result=exec_result.response,
                decision=ReuseDecision.MISS,
                similarity_score=0.0,
                reason="No compatible execution found.",
                references=exec_result.references,
            )

        matched_entry = best_match.entry
        self.storage.increment_hit(matched_entry.id)
        score = best_match.score

        # Full hit: return cached LLM response, skip pipeline entirely
        if (
            score >= self.policy.response_threshold
            and matched_entry.response is not None
        ):
            self.metrics.record(MetricEvent.HIT, similarity=score)
            self.metrics.record(MetricEvent.RESPONSE_REUSED)
            return ReuseOutcome(
                result=matched_entry.response,
                decision=ReuseDecision.RESPONSE_REUSED,
                similarity_score=score,
                reason=f"Vector similarity {score:.2f} met response threshold.",
                matched_record_id=matched_entry.id,
                references=matched_entry.references,
            )

        # Partial hit: retrieval can be reused, re-run computation, store new result
        self.metrics.record(MetricEvent.HIT, similarity=score)
        self.metrics.record(MetricEvent.RETRIEVAL_REUSED)
        computed_exec = compute_callback()
        self.store_record(
            ExecutionRecord(
                id=uuid4(),
                embedding=query_embedding,
                references=computed_exec.references,
                response=computed_exec.response,
                context=context,
            )
        )
        return ReuseOutcome(
            result=computed_exec.response,
            decision=ReuseDecision.RETRIEVAL_REUSED,
            similarity_score=score,
            reason=f"Vector similarity {score:.2f} met retrieval threshold but fell below response threshold.",
            matched_record_id=matched_entry.id,
            references=matched_entry.references,
        )
