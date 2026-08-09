# Security Boundaries

## Request-owned context

The formal runtime carries immutable, server-owned `RequestPrincipal` and `SecurityContext` values. Context binds tenant, session and policy version plus optional `DataScope` and `KnowledgeScope`. Request body text, routing output, planning output and generated content cannot create identity or permissions.

## Repeated authorization

Security checks are repeated across request, workflow, skill, tool, data and provider boundaries and fail closed when required context or authorization is unavailable.

```text
Authentication
  -> tenant/session validation
  -> scenario and route authorization
  -> workflow authorization
  -> skill authorization
  -> tool authorization
  -> KnowledgeScope / DataScope
  -> ProviderPolicy and recursive redaction
  -> response release gate
  -> AuditEvent append
```

Router, Planner and Reviewer outputs cannot widen a grant. Knowledge candidates are filtered before fusion; data resources are checked before client creation and after tool results return. Provider inputs are classified, bounded and recursively redacted before transport, and Provider outputs receive the same deterministic inspection before use.

## Response release

Only a completed result with an answer and citations may be released. A failed workflow, absent citations, output-policy failure, provider-budget exhaustion or failed critical audit append produces a fixed safe failure instead of returning the blocked body.

## Audit scope

`AuditEvent` is a payload-free closed schema linked through request and trace identifiers. The JSONL sink canonicalizes and appends records under a process lock, fsyncs writes and maintains a committed-tip sidecar. Verification detects mutation, reordering, deletion, gaps and mismatched anchors for one local file pair.

The supported claim is a **single-process local-file verifiable hash chain**. It does not provide multi-process coordination, cross-host consistency, signing, remote anchoring, blockchain semantics or an absolute tamper-proof guarantee.

## Deterministic evidence

The frozen security suite contains 28 deterministic cases and records 28/28 passed, with zero unauthorized releases, sensitive leaks, provider bypasses or tool bypasses in that suite. This is engineering boundary evidence, not a commercial security rate or proof of universal safety.
