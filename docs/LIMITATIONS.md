# Limitations

## Retained answerability failure

The frozen system evaluation contains one real false positive: `m9-knowledge-unanswerable-003`. An unanswerable knowledge request was released with citations. This demonstrates an Answerability/evidence-sufficiency weakness; it is not an authorization bypass or sensitive-data leak. The result remains 8/9 Formal Runtime, 2/3 Unanswerable and 12/13 Overall.

## Evaluation size and meaning

The M9 set contains 13 cases, including 9 formal-runtime cases and 4 deterministic boundary cases. The retrieval benchmark contains 50 queries over synthetic enterprise material. These sets support engineering regression and evidence review but do not establish production accuracy, broad statistical generalization, production cost or a service-level objective.

## Audit threat model

The audit chain is verifiable for one Python process writing one local JSONL file and committed-tip sidecar. It does not coordinate multiple processes or hosts, sign events, anchor hashes remotely or protect against a hostile writer with filesystem access.

## Runtime and deployment scope

Real Provider, Milvus, MySQL and Redis operation requires local provisioning and user-supplied credentials. Default unit and offline-integration suites do not validate production capacity, high concurrency, disaster recovery or multi-region behavior.

## Capability scope

The project is a controlled agent, not a free autonomous multi-agent system. It does not implement unlimited dynamic planning, arbitrary database actions, Neo4j/GraphRAG, Langfuse, RAGAS or Kubernetes. SQL is one read-only, allowlisted and bounded query; this is not an unrestricted natural-language-to-SQL product.

## Publication note

No project-level open-source license is currently granted. Public source visibility does not by itself grant permission to copy, modify or redistribute the project.
