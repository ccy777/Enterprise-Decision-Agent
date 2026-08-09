# Data Agent and MCP

## Explicit data boundary

```text
Data Skill
  -> EnterpriseDataMCPClient
  -> MCP Server
  -> EnterpriseDataToolService
  -> SafeQueryService
  -> SQL Guard
  -> SQLAlchemy
  -> read-only MySQL
```

The model does not receive unrestricted database access and does not directly execute arbitrary generated SQL. The Provider produces a bounded data plan; server-owned code validates the requested operation, resource scope and tool contract before any query can reach the database.

## DataScope

`DataScope` binds allowed domains, resources and read capability to the request tenant. The client exposes only permitted tables to the planner, rejects known out-of-scope references before MCP execution and verifies returned accessed-table identities before constructing data evidence.

## Safe query controls

SQL Guard parses MySQL syntax through SQLGlot and enforces:

- exactly one query expression;
- read-only operation;
- allowlisted business tables and columns;
- no system schemas, unsafe functions, output files or locking reads;
- no unrestricted wildcard projection;
- a fixed row limit and bounded result-cell budget.

`SafeQueryService` adds execution timeout, safe projection and result bounds. The deployed account is read-only, adding a database privilege boundary beneath application validation. None of these controls is described as complete prevention of every possible SQL attack.

## Synthetic local database

`docker/mysql/init/01-schema.sql` and `02-seed.sql` define a synthetic operations fixture for products, suppliers, sales orders, inventory snapshots and purchase orders. `03-create-readonly-user.sh` provisions the application reader from local environment values. No database volume or dump is part of the repository.

## MCP lifecycle

The client and server use an explicit schema and lifecycle. External clients are created during runtime bootstrap and closed by the resource stack. Offline integration tests use deterministic process/service substitutes; real MySQL and Provider tests are opt-in.
