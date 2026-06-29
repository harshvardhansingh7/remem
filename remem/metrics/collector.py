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
        )