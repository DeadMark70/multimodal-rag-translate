# Evaluation database cutover

- [x] PostgreSQL migrations, pool and repository compatibility
- [x] Read-only import with whole-table count/checksum verification
- [x] Import rollback and populated destination refusal
- [x] Run-scoped reads and bounded campaign/run/question/behavior lists
- [x] Persistent analysis invalidation, owner checks and background refresh
- [x] Cross-connection claim and live-heartbeat recovery verification
- [x] Latest server snapshot imported; 516 analysis pages built
- [x] Existing API/SQLite contracts, frontend tests and production build
- [ ] Server: pause writers, create consistent backup, retain original volumes
- [ ] Server: start PostgreSQL, migrate with new image, verify output
- [ ] Server: connect backend network/env, start API/frontend, verify one small evaluation
- [ ] Server: measure browser timings and confirm PostgreSQL backup/restore

Commands and rollback boundary: [deployment guide](../docs/evaluation-postgresql.md).
