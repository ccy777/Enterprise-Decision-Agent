# Hybrid RAG

## Retrieval chain

```text
Dense Retrieval + BM25
  -> KnowledgeScope filtering
  -> Reciprocal Rank Fusion (k=60)
  -> Cross-Encoder reranking
  -> parent expansion
  -> bounded evidence context
  -> evidence selection
  -> answerability review
  -> cited answer
```

The fixed-window corpus uses versioned parent and child chunks. Dense retrieval uses `BAAI/bge-small-zh-v1.5` with 512-dimensional normalized embeddings; sparse retrieval uses BM25. RRF combines child rankings without exposing out-of-scope documents. `BAAI/bge-reranker-base` reranks the fused child candidates, and parent expansion restores enough surrounding context for evidence selection and citation.

The default pipeline budgets are reviewable: Dense top 10, BM25 top 10, RRF top 10, reranker top 5, parent top 5, evidence count up to 5 and total evidence text up to 6,000 characters.

## Frozen retrieval evidence

The retained `m2c2a2` package is a frozen 50-query engineering benchmark over synthetic enterprise documents:

| Population | Count |
| --- | ---: |
| Queries | 50 |
| Answerable | 46 |
| Unanswerable | 4 |

| Child-level metric | RRF | Cross-Encoder | Change |
| --- | ---: | ---: | ---: |
| Hit@1 | 84.78% | 93.48% | +8.70 percentage points |
| MRR@5 | 91.67% | 96.74% | +5.07 percentage points |
| Hit@5 | 100% | 100% | — |

`93.48%` is specifically Cross-Encoder Child Hit@1. It is not model accuracy, answer accuracy or an online production result.

## Integrity verification

`python scripts/verify_retrieval_evidence.py` verifies the committed report, query results, failure cases, runtime profile and source snapshot against fixed SHA-256 values, validates the 50/46/4 case counts and recomputes the deterministic ranking digest without loading embedding or reranking models.

The source manifest classifies the corpus as `synthetic_enterprise_fixture_no_real_business_data`. Runtime-dependent timing bytes are not claimed deterministic; ranking integrity is represented by the ranking digest.

## Scope and limitations

Knowledge filtering occurs before fusion and subsequent stages receive only in-scope candidates. Empty authorized results do not fall back to the global corpus. The implementation does not include Neo4j, GraphRAG, RAGAS or Langfuse, and the small frozen benchmark is engineering evidence rather than a generalization claim.
