# Architecture

## System boundary

Enterprise Decision Agent is a controlled Router–Planner–Executor–Reviewer runtime. It accepts a formal request only with server-owned identity and scope context, selects one bounded route, executes registered skills and releases a response only after evidence, review, citation, provider-output and audit checks.

```mermaid
flowchart LR
    A["RequestPrincipal + SecurityContext"] --> B["FormalRequestExecutor"]
    B --> C["Router"]
    C --> D["Coordinator"]
    D --> E["Direct or controlled workflow"]
    E --> F["Knowledge / Data / Mixed skill"]
    F --> G["Evidence and answer"]
    G --> H["Reviewer + citation validation"]
    H --> I["Response release gate"]
    I --> J["Trace + AuditEvent"]
```

The router classifies; it does not grant authority. The coordinator rechecks scenario, workflow and skill permissions. Planner output is validated against closed schemas and registered capabilities. Tools inherit the immutable request context and cannot widen tenant, session, knowledge or data scope.

## Knowledge path

The knowledge path loads versioned parent/child chunks, runs Dense and BM25 retrieval, filters candidates by `KnowledgeScope`, fuses ranks with RRF, reranks child candidates with a Cross-Encoder, expands selected children to parent context, builds bounded evidence and then performs evidence selection, answerability review, answer generation and citation validation.

## Data path

The data path crosses `EnterpriseDataMCPClient` into the MCP server and `EnterpriseDataToolService`. `SafeQueryService` and SQL Guard validate one bounded, read-only query against allowlisted tables and columns before SQLAlchemy reaches a read-only MySQL account. Accessed resource identities are checked against `DataScope` before evidence is returned.

## Mixed path

The mixed inventory workflow executes registered knowledge and data subskills under the same immutable security context, then synthesizes their bounded evidence and runs workflow review. It is not a free-form conversation among independent agents.

## Runtime resources

External clients are created during explicit runtime bootstrap, never at module import. Resource ownership and shutdown are managed by the runtime builder. The default offline suites replace external I/O with deterministic substitutes; real Provider, MySQL, Milvus and Redis tests remain opt-in.

## Evidence and observability

Trace spans carry safe stage metadata and timings. Security-critical actions append payload-free `AuditEvent` records. Response release depends on both completed workflow state and a durable release-allowed audit append. Audit verification covers one Python process and one local file/anchor pair; it does not provide distributed consensus or hostile-writer protection.
