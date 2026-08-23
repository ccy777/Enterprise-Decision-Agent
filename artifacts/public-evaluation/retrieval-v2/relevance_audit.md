# Retrieval Benchmark v2 — Blind Relevance Audit

Status: **PASS — relevance frozen before any ranking access**

## Blindness declaration

The audit used only the frozen Candidate, Candidate Freeze, pre-evaluation audit, frozen 12-document corpus, 36 parents, 101 children, and historical v1 Gold. No prior ranking, metrics, raw results, runtime results, experiment report, bootstrap output, or failure-case artifact was read before this freeze.

## Counts

- Candidate SHA-256: `fec7a1ddd89d7d703ea19d732b00fec65a0e58a30f6215752fc6e17dbc16ddd3`
- Relevance Dataset SHA-256: `1a42163309bbf4f5fb4675b8fa2a07288a305263d08cd00ef0fd9e28a453f598`
- Queries: 200 (160 Answerable, 40 Unanswerable)
- Evidence modes: 135 single-window sufficient, 25 multi-evidence required, 40 unanswerable
- Final relevant-child assignments: 241
- Additional accepted evidence windows: 12
- Conservative-validation rejections: 6
- Historical overlap annotations excluded only from the new single-window adjudicated field: 6

## Evidence semantics

For `single_window_sufficient`, every adjudicated Child independently supports the full core answer. For `multi_evidence_required`, every adjudicated Child directly contributes at least one required atomic fact; all required historical clause atoms are covered collectively. Historical v1 fields are preserved unchanged in `original_candidate_fields` and the explicit `original_relevant_*` arrays.

M2C1-Q008 is classified as `multi_evidence_required`; its six required warranty atoms are present in the frozen corpus and it is not a Dataset defect.

## Two-pass self-review

- Discovery candidates: 18
- Accepted after conservative validation: 12
- Rejected after conservative validation: 6
- Conservative question: if the canonical window were hidden, would the proposed single-window evidence still answer the full core question?

## Multi-evidence coverage

All 25 multi-evidence cases are inherited unchanged from formal v1. Each required clause/atomic fact has at least one supporting frozen Child. New v2 multi-evidence cases: 0. Multi-evidence Dataset defects: 0.

## Unanswerable exhaustiveness

All 40 Unanswerable cases were checked against all 101 frozen Children. No Child set supplies the missing real-time facts, absent parameters, unsupported comparisons/inferences, wrong product version, or external company identity requested by those cases. Label defects: 0.

## Near-duplicate disposition

- Total pairs dispositioned across Candidate preparation and final audit: 5
- Rewritten before the original evaluation: 3
- Accepted as distinct-semantic pairs: 2
- Unresolved: 0
- `M2C1-Q011` vs `RBV2-N022`: accepted; different regions and different numeric facts.
- `M2C1-Q020` vs `RBV2-N126`: accepted; one asks the existing A-battery safety line, the other asks absent emergency/reorder lines.

## Defects and adoption blockers

Dataset defects found: 0. Unresolved relevance issues: 0. Relevance-stage adoption blockers: 0.

## Rejected discovery candidates

- `RBV2-N012` / `child_63c8476cc32b8ff4835aaadd92e0655cb325c045607bcfc94b3b2bd9501fce37`: Begins with a dangling antecedent and does not independently state the seventh-day submission condition.
- `RBV2-N021` / `child_847f6feeddf96ec901bf20d009a5a8c97a557913c51b139bfabeaac5c78c4ada`: Covers bundle pricing but not every concession mechanism asked by the query.
- `RBV2-N024` / `child_1c6cef2b2e0f80cafc312e17ccb7af994e21a666d05f95c968fae41918ffa553`: Contains a cross-region warning but not the western 9% fact.
- `RBV2-N030` / `child_28af1f4ad4fd123dd5b535c5f0f19f62664b0496d460a3f5069c2af1f45472b0`: States the general prohibition but not the complete bundle display and aggregation method.
- `RBV2-N040` / `child_e8cc822f91849770caa3485092b534f76893f4816b19408889f031ed77e327f8`: Rejects unaccepted in-transit stock but omits the complete available-stock formula.
- `RBV2-N108` / `child_3669ad4f9b732315371b1eab1df72f6a1c18d73c02768bc4cd66c684c01ceff9`: Explains Mixed limits but does not independently define Data mode.
