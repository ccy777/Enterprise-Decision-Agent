# Evaluation

## Evidence policy

Delivery and M9 figures below come from frozen v1.0.2 artifacts. Retrieval Benchmark v2 is a separately frozen, adopted 200-query package. These are engineering evaluation records, not production accuracy, coverage, commercial security rates, cost estimates or service-level objectives. Public verification does not rerun a Provider evaluation.

## Delivery checks

The public repository verification reports:

| Check | Reproducible public result |
| --- | ---: |
| Unit | 1,802 passed |
| Stable offline integration | 235 passed |
| Deterministic security evaluation | 28 / 28 |

The private v1.0.2 frozen delivery package recorded 1,822 Unit tests. The public repository excludes 33 private release, freeze and delivery-documentation tests; its count also includes the public corpus-initialization utility tests added after the frozen private snapshot.

The CI workflow defines exactly six checks for pull requests and pushes to `main`: `quality`, `unit`, `security-evaluation`, `secret-scan`, `dependency-scan` and `offline-integration`. GitHub branch protection registers those same six checks as required status checks for `main`.

## Retrieval benchmark

The adopted Retrieval Benchmark v2 contains 200 synthetic-enterprise queries: 160 Answerable and 40 Unanswerable. Ranking metrics use the 160 Answerable queries as their denominator.

| Child metric | RRF | Cross-Encoder |
| --- | ---: | ---: |
| Hit@1 | 137 / 160 = 85.62% | 154 / 160 = 96.25% |
| MRR@5 | 146.7 / 160 = 91.69% | 157.0 / 160 = 98.12% |
| Hit@5 | 160 / 160 = 100% | 160 / 160 = 100% |

The Cross-Encoder change is +10.63 percentage points for Hit@1 and +6.44 percentage points for MRR@5. `scripts/verify_retrieval_v2_evidence.py` verifies artifact hashes and recomputes the published ranking metrics from the committed relevance freeze and ranking records without model loading.

The formal 50-query v1 package remains unchanged as a historical baseline: 46 Answerable / 4 Unanswerable, RRF Hit@1 84.78%, Cross-Encoder Hit@1 93.48%, RRF MRR@5 91.67% and Cross-Encoder MRR@5 96.74%.

## M9 frozen system run

| Frozen-run observed metric | Result |
| --- | ---: |
| Formal Runtime | 8 / 9 |
| Deterministic Boundaries | 4 / 4 |
| Overall | 12 / 13 |
| Unanswerable | 2 / 3 |
| False positives | 1 |
| Provider calls | 45 |
| Input tokens | 38,549 |
| Output tokens | 4,258 |
| E2E P50 | 9.594 s |
| E2E P95 | 24.250 s |

The strict v1.0.2 calculator emits `numerator`, `denominator` and `value` for rates, returns `null` for 0/0, and fails closed if source manifest, artifact, dataset, run, commit, case-set or adjudication hashes drift. The original case records remain unchanged; two explicit adjudications are applied during derivation.

## Retained failure

`m9-knowledge-unanswerable-003` is the sole true failure. The system released an answer with citations for an unanswerable knowledge request, so completion, citation and answerability assertions failed. This is an Answerability/evidence-sufficiency failure, not an authorization bypass or sensitive-data leak. Keeping the case visible is part of the evidence integrity policy.

## Reproduction boundary

The v1/v2 Retrieval verifiers and M9 calculator are offline. Running `scripts/run_m9_final_evaluation.py` would invoke a configured Provider and is intentionally outside normal public verification. No Provider call, corpus re-ingestion or ranking-model load is needed to validate the retained v2 metrics.
