# Engineering Decisions

## Controlled runtime over open-ended autonomy

The project chooses a Router–Planner–Executor–Reviewer workflow with closed contracts because enterprise decisions need reviewable authority, bounded external effects and deterministic failure behavior. The trade-off is a narrower capability surface than an open-ended autonomous agent.

## Scope before ranking or execution

`KnowledgeScope` filtering occurs before RRF and reranking, so later evidence stages cannot reintroduce unauthorized documents. `DataScope` is checked before constructing the MCP client and again against returned resource identities. Scope is an execution boundary, not a post-processing filter.

## Hybrid retrieval with explicit stage evidence

Dense and BM25 retrieval expose complementary rankings. RRF provides deterministic fusion, the Cross-Encoder improves top-rank relevance, and parent expansion restores context. The adopted frozen 200-query v2 package publishes relevance labels, ranking-only records, per-stage metrics, failures and hashes instead of reducing retrieval quality to one unsupported headline number.

## MCP as a tool boundary

The Data Agent does not hand a database connection to the model. MCP separates the model-facing skill from server-owned query validation, SQL execution and safe projection. SQL Guard and a read-only account provide independent application and database controls.

## Fail-closed Provider governance

Provider stages, data classifications, context budgets and output checks are closed policies. Recursive redaction happens before transport. Missing policy, invalid audit state, sensitive payload, budget exhaustion or output inspection failure blocks progress rather than silently relaxing controls.

## Evidence freezing

Formal Provider evaluation is expensive and non-deterministic. The release therefore preserves sanitized case records, manifests, adjudications and derived metrics, then verifies hashes and derivation offline. The sole true failure remains visible. A 0/0 rate is represented as `null`, not 100%.

## Deliberate non-goals

The frozen line does not implement Neo4j/GraphRAG, Langfuse, RAGAS, Kubernetes, autonomous multi-agent collaboration or production high-concurrency guarantees. Historical plans are not treated as current implementation claims.
