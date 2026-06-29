import unittest
from remem.metrics.collector import MetricsCollector
from remem.metrics.events import MetricEvent
from remem.models.execution_context import ExecutionContext
from remem.models.execution_result import ExecutionResult
from remem import Client


class TestMetricsObservability(unittest.TestCase):

    def test_metrics_sums_and_rates(self):
        collector = MetricsCollector()

        collector.record(MetricEvent.REQUEST)
        collector.record(MetricEvent.REQUEST)
        collector.record(MetricEvent.HIT, similarity=0.96)
        collector.record(MetricEvent.HIT, similarity=0.88)
        collector.record(MetricEvent.MISS)
        collector.record(MetricEvent.RESPONSE_REUSED)

        snapshot = collector.snapshot()

        self.assertEqual(snapshot.requests, 2)
        self.assertEqual(snapshot.hits, 2)
        self.assertEqual(snapshot.misses, 1)
        self.assertEqual(snapshot.response_reused, 1)
        self.assertEqual(snapshot.hit_rate, 1.0)
        self.assertAlmostEqual(snapshot.average_similarity, 0.92)

    def test_outcome_reasons_attached(self):
        client = Client()

        def dummy_cb():
            return ExecutionResult(response="test", references=[])

        outcome = client.get_or_compute(
            query_embedding=[0.5, 0.5],
            compute_callback=dummy_cb,
            context=ExecutionContext(namespace="test"),
        )
        
        self.assertIsNotNone(outcome.reason)
        self.assertTrue(len(outcome.reason) > 0)


if __name__ == "__main__":
    unittest.main()