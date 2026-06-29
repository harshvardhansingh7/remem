from enum import Enum
from typing import Any, Optional
from uuid import UUID


class ReuseDecision(Enum):
    RESPONSE_REUSED = "RESPONSE_REUSED"
    RETRIEVAL_REUSED = "RETRIEVAL_USED"
    MISS = "MISS"


class ReuseOutcome:
    """Encapsulates structural reuse planner decisions with debug traces."""

    def __init__(
        self,
        result: Any,
        decision: ReuseDecision,
        similarity_score: float,
        reason: str,
        matched_record_id: Optional[UUID] = None,
        references: Optional[list[str]] = None,
    ):
        self.result = result
        self.decision = decision
        self.similarity_score = similarity_score
        self.reason = reason
        self.matched_record_id = matched_record_id
        self.references = references or []