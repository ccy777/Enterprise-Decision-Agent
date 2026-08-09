# Local Demo

## Prerequisites

- Python 3.11 or newer
- Docker with Compose
- Enough memory for MySQL, Milvus, etcd and MinIO
- Your own OpenAI-compatible Provider credentials for formal runtime demos

The repository contains synthetic knowledge documents and synthetic MySQL schema/seed data. It contains no usable credentials, model weights, database volumes or runtime audit logs.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
Copy-Item .env.example .env
```

The copied example contains obvious local-only placeholder passwords and loopback service addresses. Edit only the ignored local `.env` file. Provider-backed answer generation requires the user to configure their own supported API credentials. Keep the audit path outside the repository and do not commit local configuration.

## Start services

```powershell
docker compose config
docker compose up -d
docker compose ps
```

The Compose stack starts MySQL plus Milvus dependencies. MySQL initializes from the synthetic schema and seed, and the application connects through a read-only account.

## Initialize the synthetic knowledge corpus

Wait until MySQL, etcd, MinIO and Milvus are healthy, then run the public initialization utility once:

```powershell
.\.venv\Scripts\python.exe scripts\initialize_knowledge_corpus.py
```

The command reads `DECISION_AGENT_KNOWLEDGE_DATASET_ROOT`, builds the existing production retrieval runtime, calls its formal `initialize_for_ingestion()` entry point, verifies that the complete generated child corpus is present, prints only bounded counts and closes the runtime. It fails with a nonzero exit code if configuration, dataset, model dependency, Milvus schema, ingestion or cleanup is unavailable.

Re-running the command is supported: stable record IDs and Milvus upsert make the synthetic corpus initialization idempotent. A successful second run reports updates rather than silently creating duplicates.

To verify the committed frozen retrieval evidence without loading models or calling a Provider:

```powershell
.\.venv\Scripts\python.exe scripts\verify_retrieval_evidence.py
```

## Fixed demo cases

The CLI accepts only three fixed cases; it does not accept an arbitrary query that could enlarge server-owned grants.

```powershell
# Knowledge QA: scoped documents, evidence selection and citations
.\.venv\Scripts\python.exe scripts\run_local_demo.py knowledge

# Data QA: MCP, DataScope, SQL Guard and read-only MySQL
.\.venv\Scripts\python.exe scripts\run_local_demo.py data

# Mixed: inventory data plus replenishment-policy evidence
.\.venv\Scripts\python.exe scripts\run_local_demo.py mixed
```

The JSON result includes request ID, completion status, selected route/skill, answer, citations and a fixed error code when execution fails. A configured demo may make Provider calls and incur cost.

The Knowledge and Mixed demos require the corpus initialization step above. All three answer-generation demos require user-supplied Provider configuration; the repository does not include a usable Provider credential. Infrastructure, corpus ingestion, frozen retrieval verification and the guarded data-path smoke can be validated without a Provider.

## What to inspect

- Citations in the CLI result should identify only authorized synthetic documents.
- Data answers should flow through MCP and contain only safe projected results.
- Trace output is operational metadata, not raw business payload.
- The audit JSONL and committed-tip sidecar live at the configured external path. Verification covers the local single-process hash chain.

## Offline evidence verification

These checks require neither Provider credentials nor model loading:

```powershell
.\.venv\Scripts\python.exe scripts\verify_retrieval_evidence.py
.\.venv\Scripts\python.exe scripts\calculate_m9_metrics.py artifacts\evaluation\m9-final-eval-v1\case_records.jsonl --dataset datasets\agent_tasks\m9_final_eval_v1.json --adjudications artifacts\evaluation\m9-final-eval-v1\adjudications.json --output $env:TEMP\m9-public-metrics.json --manifest-output $env:TEMP\m9-public-manifest.json
```

The read-only MySQL path can be checked without a Provider:

```powershell
.\.venv\Scripts\python.exe scripts\run_safe_query_demo.py
```

## Stop services

```powershell
docker compose down
```

Do not add `--volumes` unless you explicitly intend to remove the local synthetic service data.
