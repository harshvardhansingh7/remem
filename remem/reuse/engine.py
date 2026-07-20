import time
from threading import RLock
from typing import Callable
from uuid import UUID, uuid4

from remem.distributed.storage import DistributedStorage, LockStatus
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
        self.initialize_index()

    def initialize_index(self) -> None:
        """Load valid persistent ANN state or derive it from storage."""

        with self._lifecycle_lock:
            if self.similarity.backend == "hnsw":
                self.similarity.initialize(self.storage.all())

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

    def _find_compatible_matches(
        self,
        query_embedding: list[float],
        context: ExecutionContext,
        threshold: float,
    ) -> list[SimilarityMatch]:
        """Shared metadata-filter + similarity scan used by both public methods."""
        match_count = (
            2
            if self.policy.enable_candidate_ambiguity_check
            and self.policy.minimum_response_score_margin is not None
            else 1
        )
        with self._lifecycle_lock:
            if self.similarity.backend == "exact":
                all_entries = self.storage.all()
                compatible = MetadataMatcher.filter_candidates(
                    all_entries, context, self.policy
                )
                compatible = [
                    record
                    for record in compatible
                    if self.policy.retrieval_freshness_check(
                        context, record.created_at
                    )["passed"]
                ]
                return [
                    SimilarityMatch(entry=entry, score=score)
                    for entry, score in self.similarity.find_all_matches(
                        query_embedding,
                        compatible,
                        threshold=threshold,
                        top_k=match_count,
                    )
                ]

            namespace = (
                context.namespace if self.policy.require_same_namespace else None
            )
            candidate_ids = self.similarity.find_candidate_ids(
                query_embedding,
                top_k=match_count,
                namespace=namespace,
                predicate=lambda record: (
                    self.policy.is_compatible(context, record.context)
                    and self.policy.retrieval_freshness_check(
                        context, record.created_at
                    )["passed"]
                ),
            )
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
                top_k=match_count,
            )
            return [
                SimilarityMatch(entry=entry, score=score) for entry, score in matches
            ]

    def _evaluate_reuse(
        self,
        matches: list[SimilarityMatch],
        context: ExecutionContext,
    ) -> tuple[ReuseDecision, SimilarityMatch | None, str, dict]:
        if not matches:
            diagnostics = {
                "similarity": {
                    "score": 0.0,
                    "retrieval_threshold": self.policy.retrieval_threshold,
                    "response_threshold": self.policy.response_threshold,
                    "retrieval_passed": False,
                    "response_passed": False,
                    "top_score_margin": None,
                },
                "checks": {},
                "response_reuse": {
                    "eligible": False,
                    "rejection_reasons": [
                        "no compatible candidate met retrieval threshold"
                    ],
                },
                "retrieval_reuse": {
                    "eligible": False,
                    "rejection_reasons": [
                        "no compatible candidate met retrieval threshold"
                    ],
                },
            }
            return (
                ReuseDecision.MISS,
                None,
                "No compatible execution met the retrieval threshold.",
                diagnostics,
            )

        best = matches[0]
        margin = best.score - matches[1].score if len(matches) > 1 else None
        compatibility_passed, compatibility_failures = self.policy.compatibility_check(
            context, best.entry.context
        )
        checks = self.policy.response_checks(
            context, best.entry.context, best.entry.created_at, margin
        )
        checks["metadata_compatibility"] = {
            "passed": compatibility_passed,
            "applied": True,
            "detail": (
                "namespace, version, model, and required metadata matched"
                if compatibility_passed
                else "; ".join(compatibility_failures)
            ),
        }
        checks["cached_response"] = {
            "passed": best.entry.response is not None,
            "applied": True,
            "detail": (
                "cached response is available"
                if best.entry.response is not None
                else "cached response is unavailable"
            ),
        }
        response_failures = [
            f"{name}: {result['detail']}"
            for name, result in checks.items()
            if not result["passed"]
        ]
        response_similarity_passed = best.score >= self.policy.response_threshold
        if not response_similarity_passed:
            response_failures.insert(
                0,
                f"similarity {best.score:.4f} below response threshold {self.policy.response_threshold:.4f}",
            )
        response_eligible = response_similarity_passed and not response_failures
        retrieval_freshness = self.policy.retrieval_freshness_check(
            context, best.entry.created_at
        )
        retrieval_eligible = (
            best.score >= self.policy.retrieval_threshold
            and compatibility_passed
            and retrieval_freshness["passed"]
        )
        retrieval_failures = []
        if best.score < self.policy.retrieval_threshold:
            retrieval_failures.append(
                f"similarity {best.score:.4f} below retrieval threshold {self.policy.retrieval_threshold:.4f}"
            )
        if not compatibility_passed:
            retrieval_failures.extend(compatibility_failures)
        if not retrieval_freshness["passed"]:
            retrieval_failures.append(retrieval_freshness["detail"])
        diagnostics = {
            "similarity": {
                "score": best.score,
                "retrieval_threshold": self.policy.retrieval_threshold,
                "response_threshold": self.policy.response_threshold,
                "retrieval_passed": best.score >= self.policy.retrieval_threshold,
                "response_passed": response_similarity_passed,
                "top_score_margin": margin,
            },
            "checks": checks,
            "response_reuse": {
                "eligible": response_eligible,
                "rejection_reasons": response_failures,
            },
            "retrieval_reuse": {
                "eligible": retrieval_eligible,
                "rejection_reasons": retrieval_failures,
                "freshness": retrieval_freshness,
            },
        }
        if response_eligible:
            reason = (
                f"Response reuse selected: similarity {best.score:.4f} passed "
                "and all configured response checks passed."
            )
            return ReuseDecision.RESPONSE_REUSED, best, reason, diagnostics
        if retrieval_eligible:
            rejected = "; ".join(response_failures)
            reason = (
                f"Retrieval reuse selected: similarity {best.score:.4f} passed "
                f"the retrieval threshold; response reuse rejected because {rejected}."
            )
            return ReuseDecision.RETRIEVAL_REUSED, best, reason, diagnostics
        reason = "Miss selected: " + "; ".join(retrieval_failures)
        return ReuseDecision.MISS, best, reason, diagnostics

    def _record_distributed_miss(self, diagnostics: dict) -> None:
        if isinstance(self.storage, DistributedStorage):
            self.metrics.record(MetricEvent.DISTRIBUTED_MISS)
            diagnostics["distributed"] = {
                "enabled": True,
                "source": None,
                "fallback_active": self.storage.last_backend_error is not None,
            }

    def _cached_outcome(
        self,
        decision: ReuseDecision,
        best_match: SimilarityMatch,
        reason: str,
        diagnostics: dict,
        *,
        duplicate_work_avoided: bool = False,
    ) -> ReuseOutcome:
        matched = best_match.entry
        score = best_match.score
        self.storage.increment_hit(matched.id)
        self.metrics.record(MetricEvent.HIT, similarity=score)
        if decision is ReuseDecision.RESPONSE_REUSED:
            self.metrics.record(MetricEvent.RESPONSE_REUSED)
        else:
            self.metrics.record(MetricEvent.RETRIEVAL_REUSED)

        if isinstance(self.storage, DistributedStorage):
            source = self.storage.record_source(matched.id)
            diagnostics["distributed"] = {
                "enabled": True,
                "source": source,
                "fallback_active": self.storage.last_backend_error is not None,
                "duplicate_work_avoided": duplicate_work_avoided,
            }
            if source == "remote":
                self.metrics.record(MetricEvent.DISTRIBUTED_CACHE_HIT)
                self.metrics.record(
                    MetricEvent.REMOTE_RESPONSE_REUSED
                    if decision is ReuseDecision.RESPONSE_REUSED
                    else MetricEvent.REMOTE_RETRIEVAL_REUSED
                )
            else:
                self.metrics.record(MetricEvent.LOCAL_CACHE_HIT)
            if duplicate_work_avoided:
                self.metrics.record(MetricEvent.DUPLICATE_WORK_AVOIDED)

        return ReuseOutcome(
            result=(
                matched.response if decision is ReuseDecision.RESPONSE_REUSED else None
            ),
            decision=decision,
            similarity_score=score,
            reason=reason,
            matched_record_id=matched.id,
            references=matched.references,
            diagnostics=diagnostics,
        )

    def check(
        self,
        query_embedding: list[float],
        context: ExecutionContext,
    ) -> ReuseOutcome:
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
        matches = self._find_compatible_matches(
            query_embedding, context, self.policy.retrieval_threshold
        )
        decision, best_match, reason, diagnostics = self._evaluate_reuse(
            matches, context
        )
        if best_match is None:
            self.metrics.record(MetricEvent.MISS)
            self._record_distributed_miss(diagnostics)
            return ReuseOutcome(
                result=None,
                decision=decision,
                similarity_score=0.0,
                reason=reason,
                diagnostics=diagnostics,
            )

        matched = best_match.entry
        score = best_match.score
        if decision is ReuseDecision.MISS:
            self.metrics.record(MetricEvent.MISS)
            self._record_distributed_miss(diagnostics)
            return ReuseOutcome(
                result=None,
                decision=decision,
                similarity_score=score,
                reason=reason,
                matched_record_id=matched.id,
                diagnostics=diagnostics,
            )

        return self._cached_outcome(decision, best_match, reason, diagnostics)

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
        matches = self._find_compatible_matches(
            query_embedding, context, self.policy.retrieval_threshold
        )
        decision, best_match, reason, diagnostics = self._evaluate_reuse(
            matches, context
        )

        if decision is ReuseDecision.RESPONSE_REUSED:
            assert best_match is not None
            return self._cached_outcome(decision, best_match, reason, diagnostics)

        distributed_storage = (
            self.storage if isinstance(self.storage, DistributedStorage) else None
        )
        lock_token: str | None = None
        if distributed_storage is not None:
            status, lock_token = distributed_storage.acquire_computation_lock(
                query_embedding, context
            )
            deadline = (
                time.monotonic() + distributed_storage.config.lock_wait_timeout_seconds
            )
            while status is LockStatus.CONTENDED and time.monotonic() < deadline:
                time.sleep(distributed_storage.config.lock_poll_interval_seconds)
                refreshed_matches = self._find_compatible_matches(
                    query_embedding, context, self.policy.retrieval_threshold
                )
                (
                    refreshed_decision,
                    refreshed_match,
                    refreshed_reason,
                    refreshed_diagnostics,
                ) = self._evaluate_reuse(refreshed_matches, context)
                if refreshed_decision is ReuseDecision.RESPONSE_REUSED:
                    assert refreshed_match is not None
                    return self._cached_outcome(
                        refreshed_decision,
                        refreshed_match,
                        refreshed_reason,
                        refreshed_diagnostics,
                        duplicate_work_avoided=True,
                    )
                status, lock_token = distributed_storage.acquire_computation_lock(
                    query_embedding, context
                )
            if status is LockStatus.CONTENDED:
                self.metrics.record(MetricEvent.DISTRIBUTED_LOCK_TIMEOUT)

            if status is LockStatus.ACQUIRED:
                refreshed_matches = self._find_compatible_matches(
                    query_embedding, context, self.policy.retrieval_threshold
                )
                (
                    refreshed_decision,
                    refreshed_match,
                    refreshed_reason,
                    refreshed_diagnostics,
                ) = self._evaluate_reuse(refreshed_matches, context)
                if refreshed_decision is ReuseDecision.RESPONSE_REUSED:
                    assert refreshed_match is not None
                    distributed_storage.release_computation_lock(
                        query_embedding, context, lock_token or ""
                    )
                    return self._cached_outcome(
                        refreshed_decision,
                        refreshed_match,
                        refreshed_reason,
                        refreshed_diagnostics,
                        duplicate_work_avoided=True,
                    )

        if decision is ReuseDecision.MISS:
            self.metrics.record(MetricEvent.MISS)
            self._record_distributed_miss(diagnostics)
        else:
            assert best_match is not None
            self.storage.increment_hit(best_match.entry.id)
            self.metrics.record(MetricEvent.HIT, similarity=best_match.score)
            self.metrics.record(MetricEvent.RETRIEVAL_REUSED)
            if distributed_storage is not None:
                source = distributed_storage.record_source(best_match.entry.id)
                diagnostics["distributed"] = {
                    "enabled": True,
                    "source": source,
                    "fallback_active": (
                        distributed_storage.last_backend_error is not None
                    ),
                    "duplicate_work_avoided": False,
                }
                if source == "remote":
                    self.metrics.record(MetricEvent.DISTRIBUTED_CACHE_HIT)
                    self.metrics.record(MetricEvent.REMOTE_RETRIEVAL_REUSED)
                else:
                    self.metrics.record(MetricEvent.LOCAL_CACHE_HIT)

        try:
            computed_exec = compute_callback()
            record_id = (
                distributed_storage.deterministic_record_id(query_embedding, context)
                if distributed_storage is not None
                else uuid4()
            )
            self.store_record(
                ExecutionRecord(
                    id=record_id,
                    embedding=query_embedding,
                    references=computed_exec.references,
                    response=computed_exec.response,
                    context=context,
                )
            )
        finally:
            if distributed_storage is not None and lock_token is not None:
                distributed_storage.release_computation_lock(
                    query_embedding, context, lock_token
                )

        return ReuseOutcome(
            result=computed_exec.response,
            decision=decision,
            similarity_score=best_match.score if best_match else 0.0,
            reason=reason,
            matched_record_id=best_match.entry.id if best_match else None,
            references=(
                best_match.entry.references
                if decision is ReuseDecision.RETRIEVAL_REUSED and best_match
                else computed_exec.references
            ),
            diagnostics=diagnostics,
        )
