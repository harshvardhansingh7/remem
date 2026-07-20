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
    local_cache_hits: int = 0
    distributed_cache_hits: int = 0
    remote_response_reused: int = 0
    remote_retrieval_reused: int = 0
    distributed_misses: int = 0
    synchronization_events: int = 0
    synchronization_failures: int = 0
    backend_failures: int = 0
    invalid_distributed_records: int = 0
    fallback_to_local: int = 0
    lock_acquisitions: int = 0
    lock_contentions: int = 0
    lock_timeouts: int = 0
    duplicate_work_avoided: int = 0

    def __str__(self) -> str:
        return (
            "========== Metrics ==========\n"
            f"Requests: {self.requests}\n"
            f"Hits: {self.hits}\n"
            f"Misses: {self.misses}\n"
            f"Response Reused: {self.response_reused}\n"
            f"Retrieval Reused: {self.retrieval_reused}\n"
            f"Hit Rate: {self.hit_rate * 100:.1f}%\n"
            f"Average Similarity: {self.average_similarity:.3f}\n"
            f"Distributed Cache Hits: {self.distributed_cache_hits}\n"
            f"Distributed Misses: {self.distributed_misses}\n"
            f"Synchronization Failures: {self.synchronization_failures}\n"
            f"Duplicate Work Avoided: {self.duplicate_work_avoided}"
        )
