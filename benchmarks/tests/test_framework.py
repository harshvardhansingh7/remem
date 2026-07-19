from __future__ import annotations

import csv
import json
from pathlib import Path
from zipfile import ZipFile

from benchmarks.baselines import exact_key, no_reuse, normalize_key
from benchmarks.datasets.preprocess import preprocess_banking77, preprocess_beir_scifact
from benchmarks.embeddings import HashEmbeddingProvider
from benchmarks.io import load_workload, read_json, write_json
from benchmarks.metrics.quality import percentile, retrieval_ranking_metrics, summarize
from benchmarks.model import DecisionObservation
from benchmarks.report import generate_report
from benchmarks.run import _directory_size, run_benchmark

ROOT = Path(__file__).parents[2]
FIXTURE = ROOT / "benchmarks" / "fixtures" / "tiny-workload.json"


def test_workload_parsing_preserves_ground_truth() -> None:
    workload = load_workload(FIXTURE)
    assert len(workload.seeds) == 2
    assert [case.expected.value for case in workload.cases] == [
        "response_reuse",
        "retrieval_reuse",
        "miss",
        "miss",
    ]


def test_hash_embeddings_are_deterministic_and_normalized() -> None:
    provider = HashEmbeddingProvider(32)
    first, second = provider.encode(["Reset my password", "Reset my password"])
    assert first == second
    assert abs(sum(value * value for value in first) - 1.0) < 1e-9


def test_confusion_matrix_and_latency_percentiles() -> None:
    rows = [
        DecisionObservation(
            "a",
            "response_reuse",
            "response_reuse",
            1.0,
            "1",
            "a",
            "a",
            True,
            1_000_000,
            [],
        ),
        DecisionObservation(
            "b", "miss", "retrieval_reuse", 0.9, "2", None, "b", False, 2_000_000, []
        ),
        DecisionObservation(
            "c", "retrieval_reuse", "miss", 0.0, None, None, None, False, 3_000_000, []
        ),
        DecisionObservation(
            "d", "miss", "miss", 0.0, None, None, None, True, 4_000_000, []
        ),
    ]
    result = summarize(rows)
    assert result["confusion"] == {
        "true_positive_reuse": 1,
        "false_positive_reuse": 1,
        "true_negative_miss": 1,
        "false_negative_miss": 1,
    }
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5


def test_retrieval_metrics() -> None:
    result = retrieval_ranking_metrics(
        [["wrong", "right"], ["right", "wrong"]], ["right", "right"], 2
    )
    assert result["recall@2"] == 1.0
    assert result["precision@2"] == 0.5
    assert result["mrr"] == 0.75


def test_baselines_apply_equivalent_eligibility() -> None:
    workload = load_workload(FIXTURE)
    assert normalize_key(" Reset, MY password! ") == "reset my password"
    none_summary = summarize(no_reuse(workload.cases))
    key_rows = exact_key(workload.seeds, workload.cases)
    assert none_summary["response_reuse_count"] == 0
    assert key_rows[0].predicted == "response_reuse"
    assert key_rows[3].predicted == "miss"
    assert key_rows[3].safe is True


def test_banking_preprocessing_is_deterministic(tmp_path: Path) -> None:
    rows = [
        {"text": "reset passcode", "category": "passcode"},
        {"text": "forgot passcode", "category": "passcode"},
        {"text": "refund card", "category": "refund"},
    ]
    for filename in ["train.csv", "test.csv"]:
        with (tmp_path / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["text", "category"])
            writer.writeheader()
            writer.writerows(rows)
    first = preprocess_banking77(tmp_path, "validation", None, 7)
    second = preprocess_banking77(tmp_path, "validation", None, 7)
    assert first.to_dict() == second.to_dict()
    assert {seed.retrieval_group for seed in first.seeds} == {
        "banking77:intent:passcode",
        "banking77:intent:refund",
    }


def test_scifact_preprocessing_uses_shared_evidence_for_retrieval_only(
    tmp_path: Path,
) -> None:
    queries = [
        {"_id": "1", "text": "Claim one", "metadata": {}},
        {"_id": "2", "text": "Claim two", "metadata": {}},
        {"_id": "3", "text": "Unpaired claim", "metadata": {}},
    ]
    with ZipFile(tmp_path / "scifact.zip", "w") as archive:
        archive.writestr(
            "scifact/queries.jsonl",
            "".join(json.dumps(row) + "\n" for row in queries),
        )
        archive.writestr(
            "scifact/qrels/test.tsv",
            "query-id\tcorpus-id\tscore\n1\tdoc-a\t1\n2\tdoc-a\t1\n3\tdoc-b\t1\n",
        )
        archive.writestr(
            "scifact/qrels/train.tsv",
            "query-id\tcorpus-id\tscore\n1\tdoc-a\t1\n2\tdoc-a\t1\n",
        )

    workload = preprocess_beir_scifact(tmp_path, "test", None, 17)

    assert len(workload.seeds) == 1
    assert workload.cases[0].expected.value == "retrieval_reuse"
    assert workload.cases[0].retrieval_group == workload.seeds[0].retrieval_group
    assert workload.cases[0].response_group is None
    assert len([case for case in workload.cases if "isolation" in case.tags]) == 4


def test_json_serialization_is_atomic_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    write_json(path, {"b": 2, "a": [1]})
    assert read_json(path) == {"a": [1], "b": 2}
    assert not path.with_suffix(".json.tmp").exists()


def test_directory_size_includes_partition_siblings(tmp_path: Path) -> None:
    base = tmp_path / "index"
    partitions = tmp_path / "index.partitions"
    partitions.mkdir()
    (partitions / "namespace.usearch").write_bytes(b"index")
    (partitions / "namespace.usearch.meta.json").write_bytes(b"metadata")
    assert _directory_size(base) == 13


def test_small_end_to_end_run_and_report(tmp_path: Path) -> None:
    config = read_json(ROOT / "benchmarks" / "configs" / "smoke-exact.json")
    config["workload"] = str(FIXTURE)
    config["output_dir"] = str(tmp_path / "results")
    config["name"] = "test-smoke"
    config["record_limit"] = 2
    config["query_limit"] = 3
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    result_path = run_benchmark(config_path)
    result = read_json(result_path)
    assert result["schema_version"] == 1
    assert result["workload"]["seed_records"] == 2
    assert result["workload"]["queries"] == 3
    assert result["workload"]["available_queries"] == 4
    assert {row["case_id"] for row in result["observations"]} == {
        "tiny-case-response",
        "tiny-case-retrieval",
        "tiny-case-isolation",
    }
    assert result["search"]["resolved_mode"] == "exact_cosine"
    assert len(result["threshold_sweep"]) > 1
    report_path = generate_report([result_path], tmp_path / "report.md")
    assert "Only synthetic framework smoke tests" in report_path.read_text(
        encoding="utf-8"
    )
