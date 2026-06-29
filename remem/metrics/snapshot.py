from dataclasses import dataclass


@dataclass(frozen=True)
class MetricsSnapshot:
    """Immutable payload representing system-wide execution telemetry."""

    requests: int
    hits: int
    misses: int
    response_reused: int
    retrieval_reused: int
    average_similarity: float
    hit_rate: float

    def __str__(self) -> str:
        return (
            "========== Metrics ==========\n"
            f"Requests: {self.requests}\n"
            f"Hits: {self.hits}\n"
            f"Misses: {self.misses}\n"
            f"Response Reused: {self.response_reused}\n"
            f"Retrieval Reused: {self.retrieval_reused}\n"
            f"Hit Rate: {self.hit_rate * 100:.1f}%\n"
            f"Average Similarity: {self.average_similarity:.3f}"
        )