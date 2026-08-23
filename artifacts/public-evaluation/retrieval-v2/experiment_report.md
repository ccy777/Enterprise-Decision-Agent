# Retrieval Benchmark v2 — Final Experiment Report

## Outcome

**Recommendation: formally adopt v2 = yes.** The Candidate is unchanged, blind relevance was frozen before ranking access, all evidence modes are explicit, and all final metrics are independently recomputable.

## Evidence semantics

- Single-window sufficient: 135 Answerable cases.
- Multi-evidence required: 25 Answerable cases, all inherited from formal v1.
- New v2 multi-evidence cases: 0.
- Multi-evidence Dataset defects: 0.
- M2C1-Q008 is a valid multi-evidence case, not a Dataset defect.

## Child relevance ranking — all 160 Answerable

| Stage | Hit@1 | Hit@5 | MRR@5 |
|---|---:|---:|---:|
| RRF | 85.62% (137/160) | 100.00% (160/160) | 91.69% (146.700000/160) |
| Reranker | 96.25% (154/160) | 100.00% (160/160) | 98.12% (157.000000/160) |

Hit@1 gain: 10.63pp. MRR@5 gain: 6.44pp.

Overall Hit@1 means Top1 is a relevant evidence window. For a multi-evidence query it does not mean that one window alone supplies the full answer.

## Single-window sufficiency — 135 Answerable

| Stage | Hit@1 | Hit@5 | MRR@5 |
|---|---:|---:|---:|
| RRF | 85.19% (115/135) | 100.00% (135/135) | 91.38% (123.366667/135) |
| Reranker | 96.30% (130/135) | 100.00% (135/135) | 98.15% (132.500000/135) |

Here a hit means that a retrieved Child independently and completely supports the core answer.

## Multi-evidence ranking and atomic coverage — 25 Answerable

| Stage | Relevance Hit@1 | Relevance Hit@5 | Relevance MRR@5 | Mean atomic coverage@5 | Fully covered@5 |
|---|---:|---:|---:|---:|---:|
| RRF | 88.00% (22/25) | 100.00% (25/25) | 93.33% (23.333333/25) | 89.67% | 72.00% (18/25) |
| Reranker | 96.00% (24/25) | 100.00% (25/25) | 98.00% (24.500000/25) | 81.40% | 52.00% (13/25) |

## Required slices

| Slice / stage | Hit@1 | Hit@5 | MRR@5 |
|---|---:|---:|---:|
| Current-corpus v1 / RRF | 82.61% (38/46) | 100.00% (46/46) | 90.58% (41.666667/46) |
| Current-corpus v1 / Reranker | 93.48% (43/46) | 100.00% (46/46) | 96.74% (44.500000/46) |
| New-only / RRF | 86.84% (99/114) | 100.00% (114/114) | 92.13% (105.033333/114) |
| New-only / Reranker | 97.37% (111/114) | 100.00% (114/114) | 98.68% (112.500000/114) |
| v2 overall / RRF | 85.62% (137/160) | 100.00% (160/160) | 91.69% (146.700000/160) |
| v2 overall / Reranker | 96.25% (154/160) | 100.00% (160/160) | 98.12% (157.000000/160) |

Historical v1 remains unchanged and separate:

- RRF Hit@1: 84.78% (39/46)
- Reranker Hit@1: 93.48% (43/46)
- RRF MRR@5: 91.67%
- Reranker MRR@5: 96.74%
- Ranking digest: `0acbcdea406fa6d9dc10d7818479a2ee9b0892fa1bd5241d68daf637d4b3b75e`

## Candidate vs adjudicated relevance

- Additional legitimate evidence windows: 12.
- Reranker previous Top1 misses converted to valid evidence hits: 3 — `['RBV2-N010', 'RBV2-N024', 'RBV2-N112']`.
- Corrected single-window semantics excluded 0 former Top1 hits — `[]`.
- Remaining Reranker Top1 misses: 6.

## Movement and failures

- Improvements: 20.
- Regressions: 3.
- Unchanged correct: 134.
- Unchanged wrong: 3.
- Final failure patterns: `{'ranking ambiguity': 2, 'window-boundary issue': 4}`.

See `RETRIEVAL_BENCHMARK_V2_FINAL_FAILURE_CASES.md` for every remaining Reranker Top1 miss.

## Bootstrap

10,000 paired resamples, seed 20260820, N=160. Hit@1 delta 10.62pp, 95% CI [5.00, 16.25]pp. MRR@5 delta 6.44pp, 95% CI [3.33, 9.78]pp.

## Reproducibility and boundaries

- Candidate SHA-256: `fec7a1ddd89d7d703ea19d732b00fec65a0e58a30f6215752fc6e17dbc16ddd3`
- Relevance Dataset SHA-256: `1a42163309bbf4f5fb4675b8fa2a07288a305263d08cd00ef0fd9e28a453f598`
- Current-run ranking digest: `ba03edc91b0573c0b01ab498dad563e85a7710107ba13c6906786b473a8e6740`
- Runtime: 133.58s for 200 queries; 0.668s/query, CPU, strictly offline.
- Corpus, chunking, Pipeline, RRF/Top-K, KnowledgeScope, Parent Expansion, Runtime, Provider, Embedding and Reranker were not modified.
- `.env` was not read; no commit, push, PR, merge, or formal Provider evaluation occurred.

## Recommended resume wording

> 构建 Dense + BM25 → RRF → Cross-Encoder → Parent Expansion 混合检索链；构建并冻结 200-query 分层 Retrieval Benchmark，覆盖 12 个企业知识文档与 101 个 Child evidence windows；在完整 relevance-set 口径下，Child Hit@1 从 85.62% 提升至 96.25%，MRR@5 从 91.69% 提升至 98.12%。
