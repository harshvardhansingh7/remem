"""Two Remem clients sharing one Redis semantic cache.

Run Redis first, then install ``remem-ai[redis]`` and execute this file.
"""

from remem import Client, DistributedConfig, ExecutionContext, ExecutionResult


def client(node_id: str) -> Client:
    return Client(
        distributed=DistributedConfig(
            key_prefix="remem:example:distributed",
            node_id=node_id,
        )
    )


def main() -> None:
    first = client("worker-a")
    second = client("worker-b")
    first.flush_storage()  # Safe only because this example owns its key prefix.
    context = ExecutionContext(
        namespace="example",
        metadata={"query": "What is the refund period?"},
    )
    computations = 0

    def expensive_pipeline() -> ExecutionResult:
        nonlocal computations
        computations += 1
        return ExecutionResult("Refunds are accepted for 14 days.", ["refund-policy"])

    first_outcome = first.get_or_compute([1.0, 0.0], expensive_pipeline, context)
    second_outcome = second.get_or_compute([1.0, 0.0], expensive_pipeline, context)

    print(first_outcome.decision.value, first_outcome.result)
    print(second_outcome.decision.value, second_outcome.result)
    print("pipeline computations:", computations)
    print("worker-b distributed metrics:", second.metrics.snapshot())
    first.flush_storage()


if __name__ == "__main__":
    main()
