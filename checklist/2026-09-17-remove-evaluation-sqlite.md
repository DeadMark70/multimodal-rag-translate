# Remove evaluation SQLite runtime

- [x] Production migration confirmed by the user before removing fallback.
- [x] Remove legacy SQLite runtime schema, connections, initialization and retries.
- [x] Remove direct `aiosqlite` dependency; retain offline import and backups.
- [x] Require PostgreSQL configuration; never silently create an empty local DB.
- [x] Verify duplicate result insertion and human-rating timestamps on PostgreSQL.
- [x] Convert affected repository, worker and HTTP tests to isolated PostgreSQL schemas.
- [x] Supply PostgreSQL in CI and remove duplicate contract-test wrappers.
- [x] Verify startup checks schema without running migrations.
- [x] Verify full/scoped reads, summaries and six exports on the imported server snapshot.
- [ ] Deploy rebuilt backend to the server; retain existing connection settings and volumes.

No schema change or repeat SQLite import is required. Existing `.db` files were not deleted.
