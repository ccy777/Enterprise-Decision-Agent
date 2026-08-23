# Retrieval Benchmark v2 public evidence

This package makes the current 200-query retrieval claim independently reviewable without
loading an embedding model, loading a reranker or calling a Provider.

## Headline result

The benchmark contains 200 synthetic-enterprise queries: 160 Answerable and 40 Unanswerable.
Ranking metrics use the 160 Answerable queries as their denominator.

| Child metric | RRF | Cross-Encoder |
| --- | ---: | ---: |
| Hit@1 | 137 / 160 = 85.62% | 154 / 160 = 96.25% |
| Hit@5 | 160 / 160 = 100.00% | 160 / 160 = 100.00% |
| MRR@5 | 146.7 / 160 = 91.69% | 157.0 / 160 = 98.12% |

## Verify

From the repository root:

```powershell
python scripts/verify_retrieval_v2_evidence.py
```

The verifier checks every committed artifact hash, the 200/160/40 population, evidence-mode
counts, ordered case identity, Answerable/Unanswerable relevance invariants, stored ranking
fields and all five published metric slices. It then recomputes Hit@1, Hit@5 and MRR@5 directly
from `ranking_records.jsonl` and `relevance_freeze.jsonl`.

## Package contents

- `relevance_freeze.jsonl`: the frozen synthetic query and adjudicated Child relevance set.
- `ranking_records.jsonl`: RRF and Cross-Encoder candidate rankings for all 200 queries.
- `metrics.json`: the complete adopted metrics, slices, bootstrap and coverage analysis.
- `experiment_report.md`: metric interpretation and experiment boundaries.
- `failure_cases.md`: all six remaining Cross-Encoder Top1 misses.
- `relevance_audit.md`: freeze and relevance-review summary.
- `runtime.json`: fixed model revisions, configuration and offline runtime metadata.
- `manifest.json`: counts, identities, hashes and the public disclosure boundary.

## Interpretation boundary

This is a frozen engineering benchmark over synthetic enterprise material. Hit@1 means that the
first Child is a relevant evidence window. For a multi-evidence query, it does not mean that one
window alone supports the complete answer. These figures are not answer accuracy, production
accuracy, an SLA or evidence of generalization outside this benchmark.

The package intentionally contains no credentials, private business data, prompts, Provider
outputs or internal filesystem paths.
