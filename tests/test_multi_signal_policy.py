from datetime import datetime, timedelta, timezone

import pytest

from remem import (
    Client,
    ExecutionContext,
    ExecutionRecord,
    ExecutionResult,
    InMemoryStorage,
    ReuseDecision,
    ReusePolicy,
)


def context(query: str, **metadata) -> ExecutionContext:
    return ExecutionContext(metadata={"query": query, **metadata})


def client_with_query(
    cached_query: str,
    *,
    policy: ReusePolicy | None = None,
    created_at: datetime | None = None,
    cached_metadata: dict | None = None,
) -> tuple[Client, ExecutionRecord]:
    record = ExecutionRecord(
        embedding=[1.0, 0.0],
        references=["shared-source"],
        response="cached response",
        context=context(cached_query, **(cached_metadata or {})),
        created_at=created_at or datetime.now(timezone.utc),
    )
    client = Client(
        storage_backend=InMemoryStorage(),
        policy=policy or ReusePolicy(),
        search_mode="exact_cosine",
    )
    client.store(record)
    return client, record


def assert_response_rejected(cached: str, current: str, check: str) -> None:
    client, record = client_with_query(cached)
    outcome = client.check([1.0, 0.0], context(current))
    assert outcome.decision is ReuseDecision.RETRIEVAL_REUSED
    assert outcome.matched_record_id == record.id
    assert not outcome.diagnostics["checks"][check]["passed"]
    assert check in outcome.reason


def test_safe_paraphrase_can_reuse_response() -> None:
    client, record = client_with_query("How can I freeze my debit card?")
    outcome = client.check(
        [1.0, 0.0], context("How do I temporarily block my debit card?")
    )
    assert outcome.decision is ReuseDecision.RESPONSE_REUSED
    assert outcome.matched_record_id == record.id
    assert all(row["passed"] for row in outcome.diagnostics["checks"].values())


@pytest.mark.parametrize(
    ("cached", "current", "failed_check"),
    [
        ("Summarize this document.", "Translate this document.", "intent"),
        (
            "How many states are in India?",
            "How many states are in the USA?",
            "critical_entities",
        ),
        (
            "Summarize the 2025 annual report.",
            "Summarize the 2026 annual report.",
            "numeric_values",
        ),
        ("Flights from Delhi to London.", "Flights from London to Delhi.", "direction"),
        (
            "Enable international payments.",
            "Disable international payments.",
            "negation",
        ),
        (
            "Hyperfibrinogenemia increases thrombosis rates.",
            "Hyperfibrinogenemia decreases thrombosis rates.",
            "negation",
        ),
        (
            "Return the result as JSON.",
            "Return the result as a Markdown table.",
            "output_format",
        ),
    ],
)
def test_meaning_changing_signal_downgrades_response_reuse(
    cached: str, current: str, failed_check: str
) -> None:
    assert_response_rejected(cached, current, failed_check)


def test_different_question_focus_reuses_retrieval_not_response() -> None:
    assert_response_rejected(
        "When was the Eiffel Tower built?",
        "Where is the Eiffel Tower located?",
        "intent",
    )


def test_explicit_operation_metadata_overrides_heuristic_detection() -> None:
    client, _ = client_with_query(
        "Summarize this document.", cached_metadata={"operation": "document_task"}
    )
    outcome = client.check(
        [1.0, 0.0],
        context("Translate this document.", operation="document_task"),
    )
    assert outcome.decision is ReuseDecision.RESPONSE_REUSED
    assert outcome.diagnostics["checks"]["intent"]["passed"]


def test_explicit_metadata_is_checked_without_query_text() -> None:
    record = ExecutionRecord(
        embedding=[1.0, 0.0],
        references=["source"],
        response="cached",
        context=ExecutionContext(metadata={"operation": "summarization"}),
    )
    client = Client(storage_backend=InMemoryStorage(), search_mode="exact_cosine")
    client.store(record)

    outcome = client.check(
        [1.0, 0.0],
        ExecutionContext(metadata={"operation": "translation"}),
    )

    assert outcome.decision is ReuseDecision.RETRIEVAL_REUSED
    assert not outcome.diagnostics["checks"]["intent"]["passed"]


def test_required_metadata_mismatch_returns_miss() -> None:
    policy = ReusePolicy(required_metadata_keys=("retrieval_filter_hash",))
    client, _ = client_with_query(
        "What is the refund policy?",
        policy=policy,
        cached_metadata={"retrieval_filter_hash": "filter-a"},
    )
    outcome = client.check(
        [1.0, 0.0],
        context("What is the refund policy?", retrieval_filter_hash="filter-b"),
    )
    assert outcome.decision is ReuseDecision.MISS
    assert outcome.matched_record_id is None


def test_stale_response_downgrades_to_fresh_retrieval() -> None:
    policy = ReusePolicy(max_response_age_seconds=60)
    client, _ = client_with_query(
        "What is the current status?",
        policy=policy,
        created_at=datetime.now(timezone.utc) - timedelta(minutes=5),
    )
    outcome = client.check([1.0, 0.0], context("What is the current status?"))
    assert outcome.decision is ReuseDecision.RETRIEVAL_REUSED
    assert not outcome.diagnostics["checks"]["freshness"]["passed"]


def test_stale_retrieval_returns_miss() -> None:
    policy = ReusePolicy(max_retrieval_age_seconds=60)
    client, _ = client_with_query(
        "What is the current status?",
        policy=policy,
        created_at=datetime.now(timezone.utc) - timedelta(minutes=5),
    )
    outcome = client.check([1.0, 0.0], context("What is the current status?"))
    assert outcome.decision is ReuseDecision.MISS


def test_ambiguous_top_score_downgrades_response_reuse() -> None:
    policy = ReusePolicy(minimum_response_score_margin=0.05)
    client, first = client_with_query("How can I freeze my debit card?", policy=policy)
    client.store(
        ExecutionRecord(
            embedding=[0.999, 0.001],
            references=["other-source"],
            response="other response",
            context=context("How can I freeze my debit card?"),
        )
    )
    outcome = client.check(
        [1.0, 0.0], context("How do I temporarily block my debit card?")
    )
    assert outcome.decision is ReuseDecision.RETRIEVAL_REUSED
    assert outcome.matched_record_id == first.id
    assert not outcome.diagnostics["checks"]["candidate_margin"]["passed"]


def test_threshold_only_policy_remains_backward_compatible_without_query_text() -> None:
    policy = ReusePolicy(retrieval_threshold=0.8, response_threshold=0.95)
    record = ExecutionRecord(
        embedding=[1.0, 0.0], references=["doc"], response="cached"
    )
    client = Client(
        storage_backend=InMemoryStorage(),
        policy=policy,
        search_mode="exact_cosine",
    )
    client.store(record)
    outcome = client.check([1.0, 0.0])
    assert outcome.decision is ReuseDecision.RESPONSE_REUSED
    assert not outcome.diagnostics["checks"]["intent"]["applied"]


def test_get_or_compute_recomputes_response_after_retrieval_only_match() -> None:
    client, record = client_with_query("Summarize this document.")
    calls = 0

    def compute() -> ExecutionResult:
        nonlocal calls
        calls += 1
        return ExecutionResult(
            response="translated response", references=["fresh-source"]
        )

    outcome = client.get_or_compute(
        query_embedding=[1.0, 0.0],
        context=context("Translate this document."),
        compute_callback=compute,
    )

    assert calls == 1
    assert outcome.decision is ReuseDecision.RETRIEVAL_REUSED
    assert outcome.matched_record_id == record.id
    assert outcome.result == "translated response"
