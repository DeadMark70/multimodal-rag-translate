# PostgreSQL-only evaluation runtime

The user confirmed that the production PostgreSQL import is verified and that evaluation reads are substantially faster, then authorized removing the old SQLite code.

Removed the SQLite connection fallback, embedded schema, additive migrations, initialization locks, WAL/PRAGMA setup, SQLite lock-error retries and direct `aiosqlite` dependency. Evaluation repositories now require PostgreSQL. Ledger transactions keep the same PostgreSQL advisory locks, expressed directly at their call sites. Persistent analysis refresh always runs. Versioned PostgreSQL migrations, source data and offline import behavior are unchanged.

Two remaining SQLite assumptions were corrected: duplicate result insertion catches PostgreSQL `UniqueViolation` and rolls back before retrieving the existing result; human-rating timestamps use PostgreSQL's timezone-aware datetime directly.

Database tests now use isolated PostgreSQL schemas, including HTTP/worker tests. Removed temporary SQLite paths and duplicate PostgreSQL wrapper suites. HTTP tests seed records on the application's event loop while it is running; pre-start seed helpers close their own pool. CI provides PostgreSQL 17.11. The read benchmark now compares PostgreSQL full-campaign, scoped and cached reads.

## Validation

- Evaluation/worker/export suites: 520 cases covered. The broad run passed 518 and exposed two stale test assumptions: ordering tied timestamps and classifying a missing provider metric as invalid dataset input. Corrected the assertions; both affected files then passed all 28 tests. Confirmed the prior committed error policy already returned `unknown`; no scoring/error-policy behavior was changed.
- Startup, health, conversations and OpenAPI: 32 cases covered. Updated the startup mock from schema migration to schema validation; all five startup tests pass, and the other 27 cases passed. Startup explicitly asserts that no migration runs.
- Duplicate-result idempotency, missing configuration without file fallback, PostgreSQL constraints, import rollback/source preservation, parallel claims, revisioned caches and full HTTP export parity are covered.
- On the user's imported 667,938,816-byte snapshot: three large campaigns retain equivalent full/scoped run and direct/cached summary output. All six minimal/full exports validate successfully (up to approximately 7.5 MB). No model calls or production-server access.
- Runtime search finds no SQLite imports, PRAGMAs, fallback paths or `aiosqlite` references. SQLite remains only in the offline importer and its tests; historical documentation and backup files remain.
- Correctness lint, modified-source syntax checks, Markdown links and OpenAPI drift checks pass. The OpenAPI snapshot changes only the database description and its hash; endpoints and response fields are unchanged. The entire repository test suite was not run.

Deployment: rebuild/recreate the backend with existing PostgreSQL configuration and volumes. There are no new SQL migration files and no repeat import is needed. The frontend is unchanged. See [deployment guide](../docs/evaluation-postgresql.md).
