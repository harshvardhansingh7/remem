from typing import Optional

from remem.metrics.events import MetricEvent
from remem.metrics.snapshot import MetricsSnapshot


class MetricsCollector:
    """High-performance, O(1) telemetry accumulator."""

    def __init__(self):
        self._requests: int = 0
        self._hits: int = 0
        self._misses: int = 0
        self._response_reused: int = 0
        self._retrieval_reused: int = 0
        self._similarity_sum: float = 0.0
        self._similarity_count: int = 0
        self._distributed: dict[MetricEvent, int] = {
            event: 0
            for event in MetricEvent
            if event
            not in {
                MetricEvent.REQUEST,
                MetricEvent.HIT,
                MetricEvent.MISS,
                MetricEvent.RESPONSE_REUSED,
                MetricEvent.RETRIEVAL_REUSED,
            }
        }

    def record(self, event: MetricEvent, similarity: Optional[float] = None) -> None:
        """Atomically logs state updates based on emitted MetricEvents."""
        if event == MetricEvent.REQUEST:
            self._requests += 1
        elif event == MetricEvent.MISS:
            self._misses += 1
        elif event == MetricEvent.HIT:
            self._hits += 1
            if similarity is not None:
                self._similarity_sum += similarity
                self._similarity_count += 1
        elif event == MetricEvent.RESPONSE_REUSED:
            self._response_reused += 1
        elif event == MetricEvent.RETRIEVAL_REUSED:
            self._retrieval_reused += 1
        elif event in self._distributed:
            self._distributed[event] += 1

    def snapshot(self) -> MetricsSnapshot:
        """Generates a point-in-time, read-only MetricsSnapshot."""
        hit_rate = self._hits / self._requests if self._requests > 0 else 0.0
        avg_sim = (
            self._similarity_sum / self._similarity_count
            if self._similarity_count > 0
            else 0.0
        )

        return MetricsSnapshot(
            requests=self._requests,
            hits=self._hits,
            misses=self._misses,
            response_reused=self._response_reused,
            retrieval_reused=self._retrieval_reused,
            average_similarity=avg_sim,
            hit_rate=hit_rate,
            local_cache_hits=self._distributed[MetricEvent.LOCAL_CACHE_HIT],
            distributed_cache_hits=self._distributed[MetricEvent.DISTRIBUTED_CACHE_HIT],
            remote_response_reused=self._distributed[
                MetricEvent.REMOTE_RESPONSE_REUSED
            ],
            remote_retrieval_reused=self._distributed[
                MetricEvent.REMOTE_RETRIEVAL_REUSED
            ],
            distributed_misses=self._distributed[MetricEvent.DISTRIBUTED_MISS],
            synchronization_events=self._distributed[MetricEvent.DISTRIBUTED_SYNC],
            synchronization_failures=self._distributed[
                MetricEvent.DISTRIBUTED_SYNC_FAILURE
            ],
            backend_failures=self._distributed[MetricEvent.DISTRIBUTED_BACKEND_FAILURE],
            invalid_distributed_records=self._distributed[
                MetricEvent.DISTRIBUTED_INVALID_RECORD
            ],
            fallback_to_local=self._distributed[MetricEvent.FALLBACK_TO_LOCAL],
            lock_acquisitions=self._distributed[MetricEvent.DISTRIBUTED_LOCK_ACQUIRED],
            lock_contentions=self._distributed[MetricEvent.DISTRIBUTED_LOCK_CONTENTION],
            lock_timeouts=self._distributed[MetricEvent.DISTRIBUTED_LOCK_TIMEOUT],
            duplicate_work_avoided=self._distributed[
                MetricEvent.DUPLICATE_WORK_AVOIDED
            ],
        )
