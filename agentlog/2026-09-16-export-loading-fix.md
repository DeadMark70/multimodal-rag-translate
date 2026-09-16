# Export page metadata compatibility

The PostgreSQL/page-read rollout added dashboard pagination and freshness fields. The complete export passed these through to strict question/behavior schemas, raising a validation error (HTTP 500). Research-summary inheritance also leaked freshness fields into JSON rejected by the frontend parser.

Share the research-value model independently of dashboard freshness and explicitly omit page-only fields at the export boundary. Keep strict export validation, redaction, complete rows and accounting unchanged. No database migration or data edits are needed.

Reproduced on the user's imported PostgreSQL snapshot. After the fix, three campaigns exported successfully with both minimal and full/raw observability options (64, 64 and 32 runs). Both real 64-run exports passed the frontend parser. Export schema/redaction/HTTP-parity/bulk tests and dashboard schema tests pass. The frontend also clears stale loading indicators after terminal refreshes and cached run-tab changes, with regression coverage.
