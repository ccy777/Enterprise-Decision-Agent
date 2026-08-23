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

## Adopted Retrieval Benchmark v2

The current public benchmark is a frozen 200-query engineering benchmark over synthetic enterprise documents:

| Population | Count |
| --- | ---: |
| Queries | 200 |
| Answerable | 160 |
| Unanswerable | 40 |
| Documents | 12 |
| Parent records | 36 |
| Child evidence windows | 101 |

| Child-level metric | RRF | Cross-Encoder | Change |
| --- | ---: | ---: | ---: |
| Hit@1 | 85.62% | 96.25% | +10.63 percentage points |
| MRR@5 | 91.69% | 98.12% | +6.44 percentage points |
| Hit@5 | 100% | 100% | — |

Ranking metrics use only the 160 Answerable queries as their denominator. `96.25%` is specifically Cross-Encoder Child Hit@1; it is not model accuracy, answer accuracy or an online production result. The original 50-query v1 package remains a historical baseline.

## Integrity verification

`python scripts/verify_retrieval_v2_evidence.py` verifies fixed SHA-256 values, validates the 200/160/40 counts and evidence modes, and recomputes every published Hit@1, Hit@5 and MRR@5 slice from the committed relevance freeze and ranking records without loading embedding or reranking models.

The source manifest classifies the corpus as `synthetic_enterprise_fixture_no_real_business_data`. Runtime-dependent timing bytes are not claimed deterministic; ranking integrity is represented by the ranking digest.

## Scope and limitations

Knowledge filtering occurs before fusion and subsequent stages receive only in-scope candidates. Empty authorized results do not fall back to the global corpus. The implementation does not include Neo4j, GraphRAG, RAGAS or Langfuse, and the frozen synthetic benchmark is engineering evidence rather than a generalization claim.
