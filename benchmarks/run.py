from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from benchmarks import BENCHMARK_SCHEMA_VERSION
from benchmarks.baselines import exact_key, no_reuse
from benchmarks.embeddings import provider_from_config
from benchmarks.io import load_workload, read_json, write_json
from benchmarks.metrics.quality import retrieval_ranking_metrics, summarize
from benchmarks.model import (
    BenchmarkCase,
    CostAssumptions,
    DecisionObservation,
    SeedRecord,
)
from benchmarks.system import machine_info, memory_rss_bytes
from remem import (
    AnnConfig,
    Client,
    ExecutionContext,
    ExecutionRecord,
    InMemoryStorage,
    ReuseDecision,
    ReusePolicy,
)
from remem.reuse.matcher import MetadataMatcher
from remem.similarity.engine import SimilarityEngine


def _context(spec: Any, query: str | None = None) -> ExecutionContext:
    return ExecutionContext(
        namespace=spec.namespace,
        kb_version=spec.kb_version,
        prompt_version=spec.prompt_version,
        model=spec.model,
        metadata={"query": query} if query is not None else {},
    )


def _prediction(decision: ReuseDecision) -> str:
    return {
        ReuseDecision.RESPONSE_REUSED: "response_reuse",
        ReuseDecision.RETRIEVAL_REUSED: "retrieval_reuse",
        ReuseDecision.MISS: "miss",
    }[decision]


def _is_safe(
    case: BenchmarkCase,
    predicted: str,
    matched: SeedRecord | None,
) -> bool:
    if predicted == "miss":
        return case.expected.value == "miss"
    if matched is None:
        return False
    if predicted == "response_reuse":
        return (
            case.response_group is not None
            and case.response_group == matched.response_group
        )
    return (
        case.retrieval_group is not None
        and case.retrieval_group == matched.retrieval_group
    )


def _observe(
    client: Client,
    cases: list[BenchmarkCase],
    vectors: dict[str, list[float]],
    seed_lookup: dict[str, SeedRecord],
    warmup: int,
    repeats: int,
) -> list[DecisionObservation]:
    observations = []
    for case in cases:
        context = _context(case.context, case.query)
        for _ in range(warmup):
            client.check(vectors[case.id], context=context)
        outcomes = []
        latencies = []
        for _ in range(repeats):
            started = time.perf_counter_ns()
            outcome = client.check(vectors[case.id], context=context)
            latencies.append(time.perf_counter_ns() - started)
            outcomes.append(outcome)
        outcome = outcomes[-1]
        matched_id = (
            str(outcome.matched_record_id) if outcome.matched_record_id else None
        )
        matched = seed_lookup.get(matched_id or "")
        predicted = _prediction(outcome.decision)
        observations.append(
            DecisionObservation(
                case_id=case.id,
                expected=case.expected.value,
                predicted=predicted,
                score=outcome.similarity_score,
                matched_record_id=matched_id,
                matched_response_group=matched.response_group if matched else None,
                matched_retrieval_group=matched.retrieval_group if matched else None,
                safe=_is_safe(case, predicted, matched),
                latency_ns=int(statistics.median(latencies)),
                tags=case.tags,
            )
        )
    return observations


def _threshold_sweep(
    raw: list[DecisionObservation],
    cases: dict[str, BenchmarkCase],
    seeds: dict[str, SeedRecord],
    retrieval_thresholds: list[float],
    response_thresholds: list[float],
) -> list[dict[str, Any]]:
    results = []
    for retrieval_threshold in retrieval_thresholds:
        for response_threshold in response_thresholds:
            if response_threshold < retrieval_threshold:
                continue
            derived = []
            for row in raw:
                case = cases[row.case_id]
                matched = seeds.get(row.matched_record_id or "")
                if matched is None or row.score < retrieval_threshold:
                    predicted = "miss"
                elif row.score >= response_threshold:
                    predicted = "response_reuse"
                else:
                    predicted = "retrieval_reuse"
                derived.append(
                    replace(
                        row,
                        predicted=predicted,
                        safe=_is_safe(case, predicted, matched),
                        latency_ns=0,
                    )
                )
            results.append(
                {
                    "retrieval_threshold": retrieval_threshold,
                    "response_threshold": response_threshold,
                    **summarize(derived),
                }
            )
    return results


def _cost_summary(
    summary: dict[str, Any], assumptions: CostAssumptions
) -> dict[str, Any]:
    response = summary["response_reuse_count"]
    retrieval = summary["retrieval_reuse_count"]
    safe_fraction = 1.0 - summary["unsafe_reuse_rate"]
    safe_response = response * summary["response_reuse_precision"]
    safe_retrieval = retrieval * summary["retrieval_reuse_precision"]
    return {
        "classification": "simulated estimate, not measured financial savings",
        "assumptions": asdict(assumptions),
        "gross_model_calls_avoided": response,
        "gross_retrieval_calls_avoided": response + retrieval,
        "estimated_safe_model_calls_avoided": safe_response,
        "estimated_safe_retrieval_calls_avoided": safe_response + safe_retrieval,
        "estimated_safe_tokens_avoided": safe_response
        * (assumptions.input_tokens_per_call + assumptions.output_tokens_per_call),
        "estimated_safe_cost_avoided": safe_response * assumptions.model_cost_per_call
        + (safe_response + safe_retrieval) * assumptions.retrieval_cost_per_call,
        "safe_reuse_fraction": safe_fraction,
    }


def _retrieval_diagnostics(
    client: Client,
    storage: InMemoryStorage,
    cases: list[BenchmarkCase],
    vectors: dict[str, list[float]],
    seed_lookup: dict[str, SeedRecord],
    k: int,
) -> dict[str, Any]:
    """Use internal search hooks only for benchmark candidate/ranking diagnostics."""

    exact_engine = SimilarityEngine("exact")
    candidate_recalls: list[float] = []
    top1_agreements: list[float] = []
    ranked_groups: list[list[str]] = []
    relevant_groups: list[str | None] = []
    for case in cases:
        context = _context(case.context, case.query)
        compatible = MetadataMatcher.filter_candidates(
            storage.all(), context, client.policy
        )
        compatible = [
            record
            for record in compatible
            if client.policy.retrieval_freshness_check(context, record.created_at)[
                "passed"
            ]
        ]
        exact_matches = exact_engine.find_all_matches(
            vectors[case.id], compatible, threshold=-1.0, top_k=k
        )
        exact_ids = [record.id for record, _ in exact_matches]
        if client.resolved_search_mode.value == "hnsw_cosine":
            namespace = (
                context.namespace if client.policy.require_same_namespace else None
            )

            def compatible_record(
                record: ExecutionRecord, current: ExecutionContext = context
            ) -> bool:
                return (
                    client.policy.is_compatible(current, record.context)
                    and client.policy.retrieval_freshness_check(
                        current, record.created_at
                    )["passed"]
                )

            candidate_ids = client.similarity.find_candidate_ids(
                vectors[case.id],
                top_k=k,
                namespace=namespace,
                predicate=compatible_record,
            )
            candidate_records = storage.get_many(candidate_ids)
            final_matches = client.similarity.rerank_candidate_records(
                vectors[case.id], candidate_ids, candidate_records, -1.0, k
            )
            candidate_recalls.append(
                len(set(candidate_ids) & set(exact_ids)) / len(exact_ids)
                if exact_ids
                else 1.0
            )
        else:
            final_matches = exact_matches
            candidate_recalls.append(1.0)
        final_ids = [record.id for record, _ in final_matches]
        top1_agreements.append(
            1.0
            if (not exact_ids and not final_ids)
            or (exact_ids and final_ids and exact_ids[0] == final_ids[0])
            else 0.0
        )
        ranked_groups.append(
            [
                seed_lookup[str(record_id)].retrieval_group
                for record_id in final_ids
                if str(record_id) in seed_lookup
            ]
        )
        relevant_groups.append(case.retrieval_group)
    ranking = retrieval_ranking_metrics(ranked_groups, relevant_groups, k)
    return {
        "diagnostic_api": (
            "internal similarity hooks; public decisions remain Client.check outputs"
        ),
        "k": k,
        "candidate_recall_relative_to_exact": statistics.fmean(candidate_recalls)
        if candidate_recalls
        else 0.0,
        "final_top1_agreement_with_exact": statistics.fmean(top1_agreements)
        if top1_agreements
        else 0.0,
        **ranking,
    }


def _directory_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    if not path.parent.exists():
        return 0
    total = 0
    for item in path.parent.glob(f"{path.name}*"):
        if item.is_file():
            total += item.stat().st_size
        elif item.is_dir():
            total += sum(
                child.stat().st_size for child in item.rglob("*") if child.is_file()
            )
    return total


def _validate_vectors(
    ids: list[str], texts: list[str], vectors: dict[str, list[float]], dimension: int
) -> dict[str, Any]:
    matrix = np.asarray([vectors[item_id] for item_id in ids], dtype=np.float64)
    if matrix.shape != (len(ids), dimension):
        raise ValueError(
            f"Embedding shape {matrix.shape} does not match {(len(ids), dimension)}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings contain NaN or infinite values")
    by_text: dict[str, list[float]] = {}
    duplicate_texts = 0
    for item_id, text in zip(ids, texts):
        if text in by_text:
            duplicate_texts += 1
            if not np.array_equal(by_text[text], vectors[item_id]):
                raise ValueError(
                    "Identical text produced inconsistent cached embeddings"
                )
        else:
            by_text[text] = vectors[item_id]
    return {
        "validated": True,
        "vector_count": len(ids),
        "shape": [len(ids), dimension],
        "finite": True,
        "duplicate_text_count": duplicate_texts,
        "duplicate_identity_consistent": True,
    }


def run_benchmark(config_path: str | Path) -> Path:
    config = read_json(config_path)
    workload = load_workload(config["workload"])
    record_limit = int(config.get("record_limit", len(workload.seeds)))
    query_limit = int(config.get("query_limit", len(workload.cases)))
    selected_seeds = workload.seeds[:record_limit]
    isolation_cases = [case for case in workload.cases if "isolation" in case.tags]
    ordinary_cases = [case for case in workload.cases if "isolation" not in case.tags]
    if query_limit < len(isolation_cases):
        selected_cases = isolation_cases[:query_limit]
    else:
        selected_cases = ordinary_cases[: query_limit - len(isolation_cases)]
        selected_cases.extend(isolation_cases)
    output_dir = Path(config.get("output_dir", "benchmarks/results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    provider = provider_from_config(
        config.get("embedding", {}), output_dir / "embedding-cache"
    )
    all_ids = [seed.id for seed in selected_seeds] + [
        case.id for case in selected_cases
    ]
    all_texts = [seed.text for seed in selected_seeds] + [
        case.query for case in selected_cases
    ]
    embedding_started = time.perf_counter_ns()
    vectors = dict(zip(all_ids, provider.encode(all_texts)))
    embedding_seconds = (time.perf_counter_ns() - embedding_started) / 1_000_000_000
    embedding_validation = _validate_vectors(
        all_ids, all_texts, vectors, provider.dimension
    )

    storage = InMemoryStorage()
    seed_lookup = {seed.id: seed for seed in selected_seeds}
    for seed_record in selected_seeds:
        storage.put(
            ExecutionRecord(
                id=UUID(seed_record.id),
                embedding=vectors[seed_record.id],
                response=seed_record.response,
                references=seed_record.references,
                context=_context(seed_record.context, seed_record.text),
            )
        )
    ann_raw = config.get("ann", {})
    persistence_path = ann_raw.get("persistence_path")
    ann_config = AnnConfig(
        m=int(ann_raw.get("m", 16)),
        ef_construction=int(ann_raw.get("ef_construction", 200)),
        ef_search=int(ann_raw.get("ef_search", 50)),
        candidate_count=int(ann_raw.get("candidate_count", 50)),
        persistence_path=persistence_path,
    )
    policy_config = config.get("thresholds", {})
    policy_options = dict(config.get("policy", {}))
    policy = ReusePolicy(
        retrieval_threshold=float(policy_config.get("retrieval", 0.80)),
        response_threshold=float(policy_config.get("response", 0.95)),
        **policy_options,
    )
    rss_before = memory_rss_bytes()
    build_started = time.perf_counter_ns()
    client = Client(
        storage_backend=storage,
        policy=policy,
        search_mode=config.get("search_mode", "exact_cosine"),
        ann_config=ann_config,
    )
    cold_start_seconds = (time.perf_counter_ns() - build_started) / 1_000_000_000
    rss_after = memory_rss_bytes()
    observations = _observe(
        client,
        selected_cases,
        vectors,
        seed_lookup,
        warmup=int(config.get("warmup", 1)),
        repeats=int(config.get("repeats", 3)),
    )
    summary = summarize(observations)
    retrieval_diagnostics = _retrieval_diagnostics(
        client,
        storage,
        selected_cases,
        vectors,
        seed_lookup,
        int(config.get("retrieval_k", 10)),
    )

    probe = Client(
        storage_backend=storage,
        policy=ReusePolicy(retrieval_threshold=-1.0, response_threshold=2.0),
        search_mode=config.get("search_mode", "exact_cosine"),
        ann_config=ann_config,
    )
    raw_observations = _observe(
        probe, selected_cases, vectors, seed_lookup, warmup=0, repeats=1
    )
    sweep = config.get("sweep", {})
    threshold_sweep = _threshold_sweep(
        raw_observations,
        {case.id: case for case in selected_cases},
        seed_lookup,
        [
            float(value)
            for value in sweep.get("retrieval", [policy.retrieval_threshold])
        ],
        [float(value) for value in sweep.get("response", [policy.response_threshold])],
    )

    mutation_vector = [0.0] * provider.dimension
    mutation_vector[0] = 1.0
    mutation_record = ExecutionRecord(
        id=uuid4(),
        embedding=mutation_vector,
        response="mutation probe",
        references=["mutation-probe"],
        context=ExecutionContext(namespace="benchmark-mutations"),
    )
    mutation_started = time.perf_counter_ns()
    client.store(mutation_record)
    insert_ns = time.perf_counter_ns() - mutation_started
    mutation_record.response = "updated mutation probe"
    mutation_started = time.perf_counter_ns()
    client.store(mutation_record)
    update_ns = time.perf_counter_ns() - mutation_started
    mutation_started = time.perf_counter_ns()
    client.delete(mutation_record.id)
    delete_ns = time.perf_counter_ns() - mutation_started

    warm_start_seconds = None
    warm_reload_validation = None
    if persistence_path and client.resolved_search_mode.value == "hnsw_cosine":
        reload_started = time.perf_counter_ns()
        warm_client = Client(
            storage_backend=storage,
            policy=policy,
            search_mode="hnsw_cosine",
            ann_config=ann_config,
        )
        warm_start_seconds = (time.perf_counter_ns() - reload_started) / 1_000_000_000
        warm_observations = _observe(
            warm_client, selected_cases, vectors, seed_lookup, warmup=0, repeats=1
        )
        warm_reload_validation = {
            "decision_agreement": statistics.fmean(
                before.predicted == after.predicted
                for before, after in zip(observations, warm_observations)
            ),
            "matched_record_agreement": statistics.fmean(
                before.matched_record_id == after.matched_record_id
                for before, after in zip(observations, warm_observations)
            ),
            "recovery_reason": warm_client.ann_persistence_recovery_reason,
            "stats": (
                asdict(warm_client.ann_index_stats)
                if warm_client.ann_index_stats
                else None
            ),
            "validated_queries": len(warm_observations),
        }

    assumptions = CostAssumptions(**config.get("cost_assumptions", {}))
    result = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "run_name": config.get("name", Path(config_path).stem),
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workload": {
            "name": workload.name,
            "dataset": workload.dataset,
            "dataset_version": workload.dataset_version,
            "license": workload.license,
            "seed": workload.seed,
            "seed_records": len(selected_seeds),
            "queries": len(selected_cases),
            "available_seed_records": len(workload.seeds),
            "available_queries": len(workload.cases),
            "notes": workload.notes,
        },
        "configuration": config,
        "embedding": {
            "provider": provider.name,
            "dimension": provider.dimension,
            "elapsed_seconds_including_cache_lookup": embedding_seconds,
            "validation": embedding_validation,
        },
        "search": {
            "requested_mode": client.search_mode.value,
            "resolved_mode": client.resolved_search_mode.value,
            "fallback_reason": client.search_fallback_reason,
            "ann": ann_raw,
            "ann_index_stats": (
                asdict(client.ann_index_stats) if client.ann_index_stats else None
            ),
            "ann_persistence_recovery_reason": client.ann_persistence_recovery_reason,
        },
        "quality": summary,
        "retrieval_quality": retrieval_diagnostics,
        "threshold_sweep": threshold_sweep,
        "performance": {
            "cold_start_seconds": cold_start_seconds,
            "warm_reload_seconds": warm_start_seconds,
            "warm_reload_validation": warm_reload_validation,
            "memory_rss_before_bytes": rss_before,
            "memory_rss_after_build_bytes": rss_after,
            "memory_rss_delta_bytes": (
                rss_after - rss_before
                if rss_before is not None and rss_after is not None
                else None
            ),
            "index_size_bytes": _directory_size(Path(persistence_path))
            if persistence_path
            else None,
            "incremental_insert_ms": insert_ns / 1_000_000,
            "update_ms": update_ns / 1_000_000,
            "deletion_ms": delete_ns / 1_000_000,
        },
        "cost_effectiveness": _cost_summary(summary, assumptions),
        "baselines": {
            "no_reuse": summarize(no_reuse(selected_cases)),
            "exact_key": summarize(exact_key(selected_seeds, selected_cases)),
        },
        "observations": [asdict(row) for row in observations],
        "raw_nearest_observations": [asdict(row) for row in raw_observations],
        "machine": machine_info(),
        "validity_notes": [
            "Cost values are configurable simulations, not measured provider billing.",
            "Query timings exclude dataset download and embedding generation.",
            "Raw-nearest probing uses the public Client with permissive thresholds.",
            "RSS is process-wide and may include embedding runtime memory.",
        ],
    }
    result_path = output_dir / f"{result['run_name']}.json"
    write_json(result_path, result)
    csv_path = result_path.with_suffix(".observations.csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(observations[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in observations)
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Remem benchmark configuration")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    path = run_benchmark(args.config)
    print(json.dumps({"result": str(path)}, indent=2))


if __name__ == "__main__":
    main()
