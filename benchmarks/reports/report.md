# Remem real-world benchmark report

**Run date:** 2026-07-15
**Product:** `remem-ai 1.1.0`
**Assessment:** **Suitable for internal testing**

This report covers real Banking77, PAWS-Wiki, and SQuAD 1.1 experiments. The
committed [JSON](summary.json), [CSV](summary.csv), and
[failure examples](failure-examples.json) are compact derivatives; raw
observations, datasets, embeddings, and indexes remain ignored.

## Executive findings

- HNSW materially improves search overhead. Across 100/1,000/8,000 PAWS
  records, exact p95 was 2.88/27.51/233.18 ms versus HNSW
  0.45/0.91/4.39 ms. At 8,000 records throughput rose from 4.4 to 255.1 QPS.
- ANN has a quality cost. With `candidate_count=10`, `ef_search=50`, candidate
  recall was 99.61% at 1,000 records and 97.15% at 8,000. PAWS top-1 agreement
  was 95.12% and 75.0% respectively.
- Defaults (`retrieval=0.80`, `response=0.95`) are not universal safe defaults.
  Validation unsafe-hit rates were 9.52% on Banking77, 68.21% on PAWS, and
  46.40% on SQuAD.
- Conservative validation thresholds achieved 100% precision for Banking77
  and SQuAD, but Banking77 held-out precision fell to 92.31%. SQuAD held-out
  precision remained 100% with only 0.90% recall. No tested PAWS threshold was
  safe; held-out unsafe reuse was 70.05%.
- All compatibility probes in the principal exact/HNSW runs missed. The suite
  found no namespace, KB-version, prompt-version, or model leakage.
- The 8,000-record persistent index was 13.52 MiB. Cold build took 4.97 s and
  warm reload 2.06 s; 200/200 post-reload decisions and matched IDs agreed.

## Environment and embeddings

| Item | Recorded value |
|---|---|
| CPU / RAM | AMD Ryzen 5 5600H, 12 logical CPUs, 7.35 GiB RAM |
| OS / Python | Windows 11, CPython 3.12.10 |
| GPU | Not used |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` |
| Revision / dimensions | `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, 384 |
| Batch / cache | 64; 13,389 vectors; 138.09 MiB ignored cache |
| Largest pass | 8,200 inputs validated in 28.36 s including cache lookup |

Vectors had the expected shape, were finite, and passed duplicate-identity
checks. Runs used CPU and offline model loading. Recorded packages include
sentence-transformers 5.6.0, Transformers 5.13.1, Torch 2.13.0, datasets 5.0.0,
NumPy 2.5.1, pandas 3.0.3, matplotlib 3.11.0, psutil 7.2.2, and USearch 2.26.0.

## Datasets and ground truth

| Dataset | Version / license | Processed workloads |
|---|---|---:|
| Banking77 | upstream snapshot, CC BY 4.0 | validation 77 records/1,504 queries; test 77/1,586 |
| PAWS-Wiki | `labeled_final` revision `161ece…6a09`, attribution requested | validation 8,000/8,004; test 8,000/8,004 |
| SQuAD | 1.1, CC BY-SA 4.0 | validation sample 178/1,004; test 1,074/4,320 |

Ignored download receipts preserve URLs, bytes, and SHA-256. Representative
checksums are Banking77 test `d12d6e…474d`, PAWS validation `7760d8…2f6d`,
PAWS test `ae342f…2f4d`, and SQuAD dev `95aa6a…72c9` (full values remain in the
dataset manifests/receipts).

PAWS label 1 permits response reuse and label 0 is a hard negative. Banking77
same-intent pairs permit retrieval only. SQuAD same-passage questions permit
retrieval; normalized equal answers permit response reuse. Labels are proxies
for application safety, not proof of answer interchangeability.

## Held-out results

Thresholds were selected from validation only. PAWS uses a fixed 1,000/1,000
held-out sample; SQuAD uses 1,074 records and an isolation-preserving 1,000
query sample; Banking77 uses all 77/1,586.

| Dataset | Mode | R/R | Precision | Recall | F1 | Unsafe | p50/p95 ms | QPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Banking77 | exact | .95/.99 | 92.31% | 0.76% | 1.51% | 7.69% | 2.07/2.12 | 481.4 |
| Banking77 | HNSW | .95/.99 | 92.31% | 0.76% | 1.51% | 7.69% | 1.49/1.55 | 668.5 |
| PAWS | exact | .95/.95 | 29.95% | 88.55% | 44.77% | 70.05% | 27.86/29.19 | 35.7 |
| PAWS | HNSW | .95/.95 | 30.07% | 88.59% | 44.91% | 69.93% | 2.03/2.41 | 492.2 |
| SQuAD | exact | .95/.99 | 100% | 0.90% | 1.79% | 0% | 29.97/30.41 | 33.4 |
| SQuAD | HNSW | .95/.99 | 100% | 0.90% | 1.79% | 0% | 2.00/2.11 | 499.8 |

Reuse-tier decisions agreed on all paired held-out queries. Candidate recall was
100% Banking77, 99.97% PAWS, and 99.82% SQuAD. PAWS top-1 agreement was 93.9%:
equal tiers can conceal a different and potentially unsafe matched record.

## Baselines and savings

| Dataset | Strategy | Precision | Recall | Hit rate | Unsafe |
|---|---|---:|---:|---:|---:|
| Banking77 | no reuse / exact key / semantic | — / 100% / 92.31% | 0 / 0.06% / 0.76% | 0 / 0.06% / 0.82% | 0 / 0 / 7.69% |
| PAWS | no reuse / exact key / semantic | — / 2.76% / 29.95% | 0 / 1.04% / 88.55% | 0 / 14.50% / 87.80% | 0 / 97.24% / 70.05% |
| SQuAD | no reuse / exact key / semantic | — / 100% / 100% | 0 / 0.10% / 0.90% | 0 / 0.10% / 0.90% | 0 / 0 / 0 |

PAWS exact-key failure is a validity warning: identical surface text can appear
in different pair-local labels. Baselines enforce compatibility context.

Measured safe avoided work was 1 model + 12 retrieval calls (Banking77), 263 +
263 (PAWS), and 2 + 9 (SQuAD). Under the explicit 1,000 input + 300 output token,
$0.01 model, and $0.001 retrieval scenario, estimated safe savings were $0.022,
$2.893, and $0.029. These are simulations, not observed billing. PAWS gross
avoided calls are not savings because 615 predictions were false positives.

## Threshold assessment

| Validation choice | R/R | Precision | Recall | F1 | Unsafe |
|---|---:|---:|---:|---:|---:|
| Banking77 default | .80/.95 | 90.48% | 11.54% | 20.47% | 9.52% |
| Banking77 conservative | .95/.99 | 100% | 0.73% | 1.46% | 0% |
| Banking77 higher recall | .70/.99 | 77.09% | 28.73% | 41.86% | 22.91% |
| PAWS default | .80/.95 | 31.79% | 99.06% | 48.13% | 68.21% |
| PAWS least-unsafe tested | .95/.95 | 32.04% | 86.69% | 46.78% | 67.96% |
| SQuAD default | .80/.95 | 53.60% | 7.11% | 12.56% | 46.40% |
| SQuAD conservative | .95/.99 | 100% | 0.50% | 1.00% | 0% |
| SQuAD higher recall | .70/.99 | 44.27% | 20.99% | 28.47% | 55.73% |

No global default change is justified. Validation precision with very few hits
did not fully generalize on Banking77. Response reuse should remain opt-in until
applications validate a false-positive budget using their own labels.

## ANN tuning, scale, and persistence

| `ef_search` (1k, candidate 10) | Candidate recall | Top-1 | p95 ms | QPS |
|---:|---:|---:|---:|---:|
| 10 | 98.45% | 95.12% | 0.843 | 1,281 |
| 50 | 99.61% | 95.12% | 0.872 | 1,235 |
| 100 | 99.63% | 95.12% | 0.918 | 1,176 |

`ef_search=50` is the measured knee, but not a universal recall guarantee.

| Records | Exact p95/QPS | HNSW p95/QPS | HNSW build | RSS delta |
|---:|---:|---:|---:|---:|
| 100 | 2.88 ms / 370.6 | 0.45 ms / 2,462 | 0.016 s | 1.19 MiB |
| 1,000 | 27.51 ms / 37.7 | 0.91 ms / 1,202 | 0.221 s | 10.67 MiB |
| 8,000 | 233.18 ms / 4.4 | 4.39 ms / 255 | 4.972 s | 74.80 MiB |

The largest safe run was 8,000 records on a 7.35 GiB machine. No 10k–100k
claim is made. RSS is process-wide. The persisted index was 14,171,324 bytes;
reload took 2.062 s with `load_count=1`, `rebuild_count=0`, and 200/200 matched
record agreement. The initial missing-artifact recovery is expected on a fresh
cold run.

## Auto mode and failures

With USearch installed, `auto` resolved to HNSW without a fallback. With its
import blocked, it resolved to exact and exposed the fallback reason. Forced
HNSW emitted `pip install remem-ai[ann]` guidance.

The failure artifact preserves concise query/match/score examples. Banking77's
conservative setting still made one unsafe response reuse. PAWS produces many
high-overlap non-paraphrase false hits. SQuAD safety is achieved mostly by
abstention. HNSW can change the matched PAWS record without changing the tier.
Compatibility probes all missed, but do not prove distributed tenant isolation.

## Charts

- [Precision versus recall](charts/precision-recall.png)
- [Threshold versus F1 and false-positive rate](charts/threshold-quality.png)
- [Dataset size versus latency, throughput, and memory](charts/scalability.png)
- [ANN recall versus `ef_search`](charts/ann-ef-search.png)
- [Index build and reload](charts/index-build-reload.png)
- [Response and retrieval reuse rates](charts/reuse-rates.png)
- [Estimated safe cost savings](charts/estimated-cost-savings.png)

## Production assessment and limits

**Suitable for internal testing.** Search modes, exact reranking, compatibility
isolation, mutation, and persistence are coherent in this single-process suite,
and HNSW is worthwhile. Production is blocked by unsafe response reuse,
threshold instability, very low recall at safe settings, declining ANN recall
at scale, and absent concurrency/network-storage evidence.

This does not prove safety for conversational, domain-specific, multilingual,
multi-process, concurrent-writer, network-storage, GPU, or >8,000-record
workloads. It does not measure end-to-end LLM latency or actual spend.

See [recommendations](../recommendations.md) and the [harness README](../README.md).

## Multi-signal policy follow-up (2026-07-19)

This follow-up holds the dataset, embeddings, exact search, and selected
thresholds constant while comparing threshold-only decisions with the new
dependency-light policy checks. The multi-signal profile also uses a `0.10`
minimum response-score margin selected from PAWS validation only. It is an
experiment setting, not a runtime default.

| Dataset | Policy | Response / retrieval / miss | Response precision | Response unsafe | Overall precision / recall | p50 / p95 ms |
|---|---|---:|---:|---:|---:|---:|
| Banking77 | similarity only | 2 / 11 / 1,573 | 50.00% | 50.00% | 92.31% / 0.76% | 2.06 / 2.20 |
| Banking77 | multi-signal | 2 / 11 / 1,573 | 50.00% | 50.00% | 92.31% / 0.76% | 2.08 / 2.14 |
| PAWS | similarity only | 878 / 0 / 122 | 29.95% | 70.05% | 29.95% / 88.55% | 28.99 / 30.36 |
| PAWS | multi-signal | 303 / 575 / 122 | 43.56% | 56.44% | 29.95% / 88.55% | 29.33 / 29.99 |
| SQuAD | similarity only | 2 / 7 / 991 | 100% | 0% | 100% / 0.90% | 29.59 / 31.07 |
| SQuAD | multi-signal | 1 / 8 / 991 | 100% | 0% | 100% / 0.90% | 30.10 / 30.54 |
| SciFact | similarity only | 15 / 14 / 32 | 6.67% | 93.33% | 41.38% / 30.00% | 1.21 / 1.24 |
| SciFact | multi-signal | 10 / 19 / 32 | 10.00% | 90.00% | 58.62% / 37.78% | 1.28 / 1.62 |

On PAWS, the margin reduced unsafe response reuse by 13.61 percentage points
and retained 140 of 191 validation-safe response hits at the selected setting.
The validation response precision progression was 30.56% (no margin), 38.18%
(`0.01`), 42.73% (`0.02`), 49.47% (`0.05`), and 52.63% (`0.10`). The held-out
result confirms useful tiering improvement, but not response safety: 56.44% of
remaining PAWS response hits were still unsafe.

Banking77 decisions were unchanged. Its one conservatively labeled unsafe
response pairs “Why doesn't my disposable virtual card work?” with “Why isn't
my disposable virtual card working?”, an apparent benchmark-proxy false
positive rather than a clear semantic error. SQuAD retained all nine useful
hits and shifted one from response to retrieval. SciFact improved overall
precision by moving five response predictions into retrieval, but response
reuse remained unsafe because shared scientific evidence does not imply that
different claims have interchangeable answers.

The SciFact workload streams queries and qrels directly from the pinned BEIR
archive. Train supplies 248 records and 359 validation cases; held-out test
supplies 45 records and 61 cases. Positive qrels define retrieval equivalence,
while only a controlled exact duplicate permits response reuse. The archive
MD5 is `5f7d1de60b170fc8027bb7898e2efca1` and its recorded SHA-256 is
`536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`.

A direct policy microbenchmark measured 2.321 microseconds per threshold-only
decision with all text checks disabled and 174.686 microseconds with the full
multi-signal path, an added 172.365 microseconds. Public end-to-end exact-search
timings above include storage and scoring noise and should not be interpreted
as an isolated policy cost.

The result is a narrower, evidence-backed recommendation: use the checks to
block obvious meaning changes and route uncertain matches to retrieval reuse,
but require explicit metadata for authorization, filters, source versions, and
other dependencies. Do not treat regex signals or a global score margin as a
proof of response equivalence.
