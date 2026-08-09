# Evaluation

## Evidence policy

All figures below come from frozen v1.0.2 artifacts. They are engineering evaluation records, not production accuracy, coverage, commercial security rates, cost estimates or service-level objectives. This public repository does not rerun the formal Provider evaluation.

## Delivery checks

The public repository verification reports:

| Check | Reproducible public result |
| --- | ---: |
| Unit | 1,802 passed |
| Stable offline integration | 235 passed |
| Deterministic security evaluation | 28 / 28 |

The private v1.0.2 frozen delivery package recorded 1,822 Unit tests. The public repository excludes 33 private release, freeze and delivery-documentation tests; its count also includes the public corpus-initialization utility tests added after the frozen private snapshot.

The CI workflow defines exactly six checks for pull requests and pushes to `main`: `quality`, `unit`, `security-evaluation`, `secret-scan`, `dependency-scan` and `offline-integration`. The repository does not describe them as branch-protection required checks unless that GitHub setting is enabled separately.

## Retrieval benchmark

The independent frozen benchmark contains 50 synthetic-document queries: 46 answerable and 4 unanswerable.

| Child metric | RRF | Cross-Encoder |
| --- | ---: | ---: |
| Hit@1 | 39 / 46 = 84.78% | 43 / 46 = 93.48% |
| MRR@5 | 42.1667 / 46 = 91.67% | 44.5 / 46 = 96.74% |
| Hit@5 | 46 / 46 = 100% | 46 / 46 = 100% |

The Cross-Encoder change is +8.70 percentage points for Hit@1 and +5.07 percentage points for MRR@5. `scripts/verify_retrieval_evidence.py` verifies hashes, counts and ranking digest without model loading.

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

The Retrieval verifier and M9 calculator are offline. Running `scripts/run_m9_final_evaluation.py` would invoke a configured Provider and is intentionally outside normal public verification. No Provider call, corpus re-ingestion or baseline optimization is needed to validate the retained evidence.
