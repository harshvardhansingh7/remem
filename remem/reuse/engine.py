from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
from uuid import uuid4

from remem.models.execution_record import ExecutionRecord
from remem.models.execution_result import ExecutionResult
from remem.similarity.engine import SimilarityEngine
from remem.storage.storage import StorageInterface


class ReuseDecision(Enum):
    RESPONSE_REUSED = "RESPONSE_REUSED"
    RETRIEVAL_USED = "RETRIEVAL_REUSED"
    COMPUTED = "COMPUTED"
    MISS = "MISS"


@dataclass
class ReuseOutcome:
    """Encapsulates structural reuse planner decisions."""

    result: Any
    decision: ReuseDecision
    similarity_score: float
    references: list[str] = field(default_factory=list)


class ReuseEngine:
    """Core engine orchestrating decision boundaries around work reuse."""

    def __init__(self, storage: StorageInterface, similarity: SimilarityEngine):
        self.storage = storage
        self.similarity = similarity

    def get_or_compute(
        self,
        query_embedding: list[float],
        compute_callback: Callable[[], ExecutionResult],
        similarity_threshold: float = 0.8,
        response_reuse_threshold: float = 0.95,
    ) -> ReuseOutcome:

        # Query index iteration is abstracted within the storage/similarity layers
        all_entries = self.storage.all()
        best_match = self.similarity.find_best_match(
            query_embedding, all_entries, threshold=similarity_threshold
        )

        # Cache MISS: Execute, persist record, and return structural payload
        if not best_match:
            exec_result = compute_callback()
            new_record = ExecutionRecord(
                id=uuid4(),
                embedding=query_embedding,
                references=exec_result.references,
                response=exec_result.response,
                metadata=exec_result.metadata,
            )
            self.storage.put(new_record)

            return ReuseOutcome(
                result=exec_result.response,
                decision=ReuseDecision.MISS,
                similarity_score=0.0,
                references=exec_result.references,
            )

        # Problem 6: Delegate hit calculation completely down to storage layer operations
        matched_entry = best_match.entry
        self.storage.increment_hit(matched_entry.id)
        score = best_match.score

        # Cache HIT: High Confidence -> Reuse precomputed response payload directly
        if score >= response_reuse_threshold and matched_entry.response is not None:
            return ReuseOutcome(
                result=matched_entry.response,
                decision=ReuseDecision.RESPONSE_REUSED,
                similarity_score=score,
                references=matched_entry.references,
            )

        # Partial Hit (Retrieve Reuse): Re-run computation using found references
        computed_exec = compute_callback()

        return ReuseOutcome(
            result=computed_exec.response,
            decision=ReuseDecision.RETRIEVAL_REUSED,
            similarity_score=score,
            references=matched_entry.references,
        )