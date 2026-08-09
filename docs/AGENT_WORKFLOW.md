# Agent Workflow

## Controlled execution model

The system deliberately separates model judgment from authority. A Provider may classify, plan or generate within a stage contract, but server code owns identity, authorization, schemas, budgets, skill registration, tool registration, evidence scope and final release.

```text
Request validation
  -> authentication and SecurityContext validation
  -> Router classification
  -> scenario/workflow authorization
  -> Coordinator dispatch
  -> Planner schema validation
  -> registered Skill and Tool execution
  -> evidence selection and answerability
  -> answer generation
  -> workflow review and citation validation
  -> provider-output inspection
  -> response release and audit
```

## Routes and skills

- Knowledge: `enterprise-knowledge-qa`
- Data: `enterprise-data-analysis`
- Mixed: `inventory-risk-diagnosis`

The direct routes run one registered skill. The controlled mixed route composes registered knowledge and data subskills and a bounded synthesizer/reviewer path. Unknown routes, skills or tools fail closed.

## Provider stages

The Provider egress policy recognizes exactly nine stages:

1. `routing`
2. `planning`
3. `data_planning`
4. `data_answer`
5. `evidence_selection`
6. `answerability_review`
7. `knowledge_answer`
8. `inventory_synthesis`
9. `workflow_review`

This list is a closed allowlist and budget boundary. It is not a promise that every request invokes all nine stages or makes nine model calls. Each route activates only the stages required by its formal workflow.

## Fail-closed behavior

The formal executor rejects missing or invalid security context before memory, routing or external calls. Authorization is repeated after route selection and before workflow, skill and tool execution. Provider governance validates classification, context/evidence budgets and redaction before transport; output is inspected before downstream use. Missing evidence, citations, audit persistence or acceptable workflow state prevents response release.

## What this design excludes

There is no unlimited planner loop, arbitrary dynamic DAG, model-created permission, unrestricted tool discovery or free autonomous multi-agent collaboration. These constraints make execution reviewable and testable at the cost of a narrower capability surface.
