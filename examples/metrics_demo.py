from remem import Client
from remem.models.execution_record import ExecutionRecord
from remem.models.execution_context import ExecutionContext
from remem.models.execution_result import ExecutionResult
from remem.reuse.policy import ReusePolicy


def main():
    print("🚀 Seeding Observability Demo v0.5.0...\n")

    client = Client(
        policy=ReusePolicy(
            retrieval_threshold=0.80,
            response_threshold=0.95,
        )
    )

    # Seed an entry
    client.store(
        ExecutionRecord(
            embedding=[0.1, 0.9],
            references=["sys_docs.txt"],
            response="Precalculated System Forecast.",
            context=ExecutionContext(namespace="weather"),
        )
    )

    def dummy_callback() -> ExecutionResult:
        return ExecutionResult(
            response="Dynamic computation result.",
            references=["fresh_doc.txt"],
        )

    # Trigger Scenario A: HIT (Response Reused)
    outcome_a = client.get_or_compute(
        query_embedding=[0.11, 0.89],
        compute_callback=dummy_callback,
        context=ExecutionContext(namespace="weather"),
    )
    print(f"Outcome A Reason: {outcome_a.reason}")

    # Trigger Scenario B: MISS
    outcome_b = client.get_or_compute(
        query_embedding=[0.9, 0.1],
        compute_callback=dummy_callback,
        context=ExecutionContext(namespace="weather"),
    )
    print(f"Outcome B Reason: {outcome_b.reason}\n")

    # Fetch immutable metrics snapshot
    snapshot = client.metrics.snapshot()
    print(snapshot)


if __name__ == "__main__":
    main()