# Export and loading regression

- [x] Reproduce strict export validation failure on the imported server snapshot.
- [x] Exclude page-only metadata while preserving strict export validation and all rows.
- [x] Verify minimal/full exports and frontend parsing with real data.
- [x] Verify export redaction, HTTP parity and run accounting contracts.
- [x] Reproduce and fix overview/run-tab loading indicators.
- [ ] Deploy updated backend/frontend images and confirm server export/download.

Deployment requires rebuilding/recreating the application containers only; retain the PostgreSQL service, connection settings and volumes. Do not repeat SQLite import.
