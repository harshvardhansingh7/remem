from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from collections import defaultdict
from io import TextIOWrapper
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5
from zipfile import ZipFile

from benchmarks.io import write_json
from benchmarks.model import (
    BenchmarkCase,
    ContextSpec,
    ExpectedDecision,
    SeedRecord,
    Workload,
)


def _stable_id(value: str) -> str:
    return str(
        uuid5(NAMESPACE_URL, f"https://github.com/hrsvd/remem/benchmark/{value}")
    )


def _bucket(value: str, modulus: int = 2) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % modulus


def _isolation_cases(seed: SeedRecord, dataset: str, split: str) -> list[BenchmarkCase]:
    variants = [
        ("namespace", ContextSpec(namespace="isolated-tenant")),
        ("kb_version", ContextSpec(kb_version="2")),
        ("prompt_version", ContextSpec(prompt_version="2")),
        ("model", ContextSpec(model="different-model")),
    ]
    return [
        BenchmarkCase(
            id=f"{seed.id}:isolation:{name}",
            dataset=dataset,
            split=split,
            query=seed.text,
            expected=ExpectedDecision.MISS,
            response_group=None,
            retrieval_group=None,
            context=context,
            tags=["isolation", name],
        )
        for name, context in variants
    ]


def preprocess_banking77(
    raw_dir: Path, split: str, limit: int | None, seed: int
) -> Workload:
    by_intent: dict[str, list[str]] = defaultdict(list)
    with (raw_dir / "train.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            by_intent[row["category"]].append(row["text"])
    seeds: list[SeedRecord] = []
    for intent in sorted(by_intent):
        text = by_intent[intent][0]
        record_id = _stable_id(f"banking77/seed/{intent}/{text}")
        seeds.append(
            SeedRecord(
                id=record_id,
                text=text,
                response=f"Deterministic support response for intent: {intent}",
                references=[f"banking-faq:{intent}"],
                response_group=f"banking77:seed:{record_id}",
                retrieval_group=f"banking77:intent:{intent}",
            )
        )
    cases: list[BenchmarkCase] = []
    with (raw_dir / "test.csv").open("r", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            assigned = "validation" if _bucket(row["text"]) == 0 else "test"
            if assigned != split:
                continue
            cases.append(
                BenchmarkCase(
                    id=f"banking77:{assigned}:{index}",
                    dataset="banking77",
                    split=assigned,
                    query=row["text"],
                    expected=ExpectedDecision.RETRIEVAL_REUSE,
                    response_group=None,
                    retrieval_group=f"banking77:intent:{row['category']}",
                    tags=["customer-support", "intent", row["category"]],
                )
            )
            if limit is not None and len(cases) >= limit:
                break
    duplicate_seed = seeds[0]
    cases.append(
        BenchmarkCase(
            id=f"{duplicate_seed.id}:exact-duplicate",
            dataset="banking77",
            split=split,
            query=duplicate_seed.text,
            expected=ExpectedDecision.RESPONSE_REUSE,
            response_group=duplicate_seed.response_group,
            retrieval_group=duplicate_seed.retrieval_group,
            tags=["exact-duplicate"],
        )
    )
    cases.extend(_isolation_cases(duplicate_seed, "banking77", split))
    return Workload(
        name=f"banking77-{split}",
        dataset="banking77",
        dataset_version="upstream master receipt",
        license="CC BY 4.0",
        seed=seed,
        seeds=seeds,
        cases=cases,
        notes=[
            "Distinct queries sharing an intent are retrieval-reusable, not assumed response-equivalent.",
            "The official test split is deterministically divided for validation and final testing.",
        ],
    )


def _paws_rows(raw_dir: Path, split: str) -> list[dict[str, Any]]:
    parquet_path = raw_dir / f"{split}.parquet"
    if parquet_path.exists():
        import pandas as pd

        return pd.read_parquet(parquet_path).to_dict(orient="records")
    candidates = list((raw_dir / "extracted").rglob(f"{split}.tsv"))
    if not candidates:
        archive_path = raw_dir / "paws_wiki_labeled_final.tar.gz"
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(raw_dir / "extracted", filter="data")
        candidates = list((raw_dir / "extracted").rglob(f"{split}.tsv"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Could not uniquely locate PAWS {split} data")
    with candidates[0].open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def preprocess_paws(
    raw_dir: Path, split: str, limit: int | None, seed: int
) -> Workload:
    source_split = "validation" if split == "validation" else "test"
    seeds: list[SeedRecord] = []
    cases: list[BenchmarkCase] = []
    for row in _paws_rows(raw_dir, source_split):
        pair_id = f"paws:{source_split}:{row['id']}"
        record_id = _stable_id(pair_id)
        positive = int(row["label"]) == 1
        response_group = pair_id if positive else f"{pair_id}:anchor-only"
        retrieval_group = pair_id if positive else f"{pair_id}:anchor-only"
        seeds.append(
            SeedRecord(
                id=record_id,
                text=row["sentence1"],
                response=row["sentence1"],
                references=[f"paws-source:{row['id']}"],
                response_group=response_group,
                retrieval_group=retrieval_group,
            )
        )
        cases.append(
            BenchmarkCase(
                id=pair_id,
                dataset="paws_wiki",
                split=split,
                query=row["sentence2"],
                expected=(
                    ExpectedDecision.RESPONSE_REUSE
                    if positive
                    else ExpectedDecision.MISS
                ),
                response_group=pair_id if positive else None,
                retrieval_group=pair_id if positive else None,
                tags=[
                    "paraphrase" if positive else "hard-negative",
                    "high-overlap",
                ],
            )
        )
        if limit is not None and len(cases) >= limit:
            break
    cases.extend(_isolation_cases(seeds[0], "paws_wiki", split))
    return Workload(
        name=f"paws-wiki-{split}",
        dataset="paws_wiki",
        dataset_version="labeled_final",
        license="Free use for any purpose; Google attribution requested",
        seed=seed,
        seeds=seeds,
        cases=cases,
        notes=[
            "Human paraphrase labels define response equivalence independently of embeddings."
        ],
    )


def _squad_paragraphs(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    paragraphs = []
    for article in raw["data"]:
        for paragraph in article["paragraphs"]:
            paragraphs.append({"title": article["title"], **paragraph})
    return paragraphs


def preprocess_squad(
    raw_dir: Path, split: str, limit: int | None, seed: int
) -> Workload:
    paragraphs = _squad_paragraphs(raw_dir / "dev-v1.1.json")
    selected = [
        paragraph
        for paragraph in paragraphs
        if ("validation" if _bucket(paragraph["context"]) == 0 else "test") == split
    ]
    seeds: list[SeedRecord] = []
    cases: list[BenchmarkCase] = []
    for paragraph_index, paragraph in enumerate(selected):
        questions = paragraph["qas"]
        if len(questions) < 2:
            continue
        context_hash = hashlib.sha256(paragraph["context"].encode("utf-8")).hexdigest()[
            :16
        ]
        retrieval_group = f"squad:context:{context_hash}"
        anchor = questions[0]
        anchor_answer = anchor["answers"][0]["text"].strip().casefold()
        record_id = _stable_id(f"squad/{anchor['id']}")
        seeds.append(
            SeedRecord(
                id=record_id,
                text=anchor["question"],
                response=anchor["answers"][0]["text"],
                references=[retrieval_group],
                response_group=f"squad:answer:{context_hash}:{anchor_answer}",
                retrieval_group=retrieval_group,
            )
        )
        for question in questions[1:]:
            answer = question["answers"][0]["text"].strip().casefold()
            response_group = f"squad:answer:{context_hash}:{answer}"
            same_answer = response_group == seeds[-1].response_group
            cases.append(
                BenchmarkCase(
                    id=f"squad:{question['id']}",
                    dataset="squad_v1",
                    split=split,
                    query=question["question"],
                    expected=(
                        ExpectedDecision.RESPONSE_REUSE
                        if same_answer
                        else ExpectedDecision.RETRIEVAL_REUSE
                    ),
                    response_group=response_group if same_answer else None,
                    retrieval_group=retrieval_group,
                    tags=[
                        "qa",
                        "shared-passage",
                        "same-answer" if same_answer else "different-answer",
                    ],
                )
            )
            if limit is not None and len(cases) >= limit:
                break
        if limit is not None and len(cases) >= limit:
            break
    cases.extend(_isolation_cases(seeds[0], "squad_v1", split))
    return Workload(
        name=f"squad-v1-{split}",
        dataset="squad_v1",
        dataset_version="1.1",
        license="CC BY-SA 4.0",
        seed=seed,
        seeds=seeds,
        cases=cases,
        notes=[
            "Shared context identifies retrieval reuse; normalized answer text identifies response equivalence.",
            "Development contexts are deterministically divided into validation and held-out test workloads.",
        ],
    )


def preprocess_beir_scifact(
    raw_dir: Path, split: str, limit: int | None, seed: int
) -> Workload:
    source_split = "train" if split == "validation" else "test"
    archive_path = raw_dir / "scifact.zip"
    with ZipFile(archive_path) as archive:
        with archive.open("scifact/queries.jsonl") as raw_queries:
            queries = {
                str(row["_id"]): str(row["text"])
                for row in (
                    json.loads(line)
                    for line in TextIOWrapper(raw_queries, encoding="utf-8")
                )
            }
        with archive.open(f"scifact/qrels/{source_split}.tsv") as raw_qrels:
            qrels = list(
                csv.DictReader(
                    TextIOWrapper(raw_qrels, encoding="utf-8"), delimiter="\t"
                )
            )

    queries_by_document: dict[str, set[str]] = defaultdict(set)
    for row in qrels:
        if int(row["score"]) > 0:
            queries_by_document[row["corpus-id"]].add(row["query-id"])

    seeds: list[SeedRecord] = []
    cases: list[BenchmarkCase] = []
    for document_id, query_ids in sorted(queries_by_document.items()):
        ordered_queries = sorted(query_ids, key=lambda value: (int(value), value))
        if len(ordered_queries) < 2:
            continue
        anchor_id = ordered_queries[0]
        anchor_text = queries[anchor_id]
        retrieval_group = f"beir-scifact:document:{document_id}"
        record_id = _stable_id(f"beir-scifact/{source_split}/{document_id}/{anchor_id}")
        seeds.append(
            SeedRecord(
                id=record_id,
                text=anchor_text,
                response=f"Evidence-backed result for SciFact claim {anchor_id}",
                references=[retrieval_group],
                response_group=f"beir-scifact:seed:{record_id}",
                retrieval_group=retrieval_group,
            )
        )
        for query_id in ordered_queries[1:]:
            cases.append(
                BenchmarkCase(
                    id=f"beir-scifact:{source_split}:{document_id}:{query_id}",
                    dataset="beir_scifact",
                    split=split,
                    query=queries[query_id],
                    expected=ExpectedDecision.RETRIEVAL_REUSE,
                    response_group=None,
                    retrieval_group=retrieval_group,
                    tags=["scientific-claim", "shared-evidence", "retrieval-only"],
                )
            )
            if limit is not None and len(cases) >= limit:
                break
        if limit is not None and len(cases) >= limit:
            break

    if not seeds:
        raise ValueError(f"SciFact {source_split} has no shared-evidence query groups")
    duplicate_seed = seeds[0]
    cases.append(
        BenchmarkCase(
            id=f"{duplicate_seed.id}:exact-duplicate",
            dataset="beir_scifact",
            split=split,
            query=duplicate_seed.text,
            expected=ExpectedDecision.RESPONSE_REUSE,
            response_group=duplicate_seed.response_group,
            retrieval_group=duplicate_seed.retrieval_group,
            tags=["exact-duplicate"],
        )
    )
    cases.extend(_isolation_cases(duplicate_seed, "beir_scifact", split))
    return Workload(
        name=f"beir-scifact-{split}",
        dataset="beir_scifact",
        dataset_version="BEIR SciFact official archive MD5 5f7d1de60b170fc8027bb7898e2efca1",
        license="CC BY-NC 2.0",
        seed=seed,
        seeds=seeds,
        cases=cases,
        notes=[
            "Queries sharing a positive qrel document are labeled retrieval-reusable only.",
            "Shared evidence does not imply response equivalence or identical claim veracity.",
            "Only documents associated with at least two queries form reuse pairs.",
        ],
    )


def preprocess_dataset(
    dataset: str,
    data_dir: str | Path,
    split: str,
    limit: int | None,
    seed: int,
) -> Path:
    raw_dir = Path(data_dir) / "raw" / dataset
    builders = {
        "banking77": preprocess_banking77,
        "beir_scifact": preprocess_beir_scifact,
        "paws_wiki": preprocess_paws,
        "squad_v1": preprocess_squad,
    }
    workload = builders[dataset](raw_dir, split, limit, seed)
    output = Path(data_dir) / "processed" / f"{dataset}-{split}.json"
    write_json(output, workload.to_dict())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic Remem workloads")
    parser.add_argument(
        "dataset", choices=["banking77", "beir_scifact", "paws_wiki", "squad_v1"]
    )
    parser.add_argument("--data-dir", default="benchmarks/data")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    path = preprocess_dataset(
        args.dataset, args.data_dir, args.split, args.limit, args.seed
    )
    print(path)


if __name__ == "__main__":
    main()
