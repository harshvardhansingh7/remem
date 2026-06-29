from typing import Callable
from uuid import uuid4

from remem.metrics.collector import MetricsCollector
from remem.metrics.events import MetricEvent
from remem.models.execution_context import ExecutionContext
from remem.models.execution_result import ExecutionResult
from remem.models.execution_record import ExecutionRecord
from remem.reuse.matcher import MetadataMatcher
from remem.reuse.policy import ReusePolicy
from remem.similarity.engine import SimilarityEngine
from remem.storage.storage import StorageInterface
from remem.reuse.decision import ReuseDecision, ReuseOutcome


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

    def get_or_compute(
        self,
        query_embedding: list[float],
        compute_callback: Callable[[], ExecutionResult],
        context: ExecutionContext,
    ) -> ReuseOutcome:

        self.metrics.record(MetricEvent.REQUEST)

        # 1. Apply metadata compatibility filtering before vector similarity scan
        all_entries = self.storage.all()
        compatible_candidates = MetadataMatcher.filter_candidates(
            all_entries, context, self.policy
        )

        # 2. Query vector scan abstracted purely over compatible records
        best_match = self.similarity.find_best_match(
            query_embedding, compatible_candidates, threshold=self.policy.retrieval_threshold
        )

        # Cache MISS Scenario
        if not best_match:
            self.metrics.record(MetricEvent.MISS)
            exec_result = compute_callback()

            new_record = ExecutionRecord(
                id=uuid4(),
                embedding=query_embedding,
                references=exec_result.references,
                response=exec_result.response,
                context=context,
            )
            self.storage.put(new_record)

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

        # Cache HIT: High Confidence (Reuse cached response)
        if score >= self.policy.response_threshold and matched_entry.response is not None:
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

        # Partial Hit (Retrieval Reused, Computation re-run)
        self.metrics.record(MetricEvent.HIT, similarity=score)
        self.metrics.record(MetricEvent.RETRIEVAL_REUSED)
        computed_exec = compute_callback()

        return ReuseOutcome(
            result=computed_exec.response,
            decision=ReuseDecision.RETRIEVAL_REUSED,
            similarity_score=score,
            reason=f"Vector similarity {score:.2f} met retrieval threshold but fell below response threshold.",
            matched_record_id=matched_entry.id,
            references=matched_entry.references,
        )