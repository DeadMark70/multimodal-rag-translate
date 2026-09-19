# Document Lifecycle API

## OCR deployment profiles (2026-09-20)

The SDK 2 image supports Datalab API with explicit `USE_LOCAL_MARKER=false`.
Existing local Marker 1 support requires the alternative
`requirements-local-marker.txt` image and `USE_LOCAL_MARKER=true` (the existing
routing default). Missing configuration never enables cloud upload. The two
dependency profiles must not be installed together. Upload and page-marker
contracts remain unchanged. See [stage 1 deployment](../dependency-upgrade-stage1.md).

## User Outcomes

- Upload a PDF for OCR processing.
- Poll or fetch processing status by `doc_id`.
- Download original or translated files.
- Trigger translation, fetch summaries, retry indexing, and delete documents.

## Acceptance Notes

- Current user-facing document lifecycle endpoints should be auth-protected.
- Document file/status/delete routes should use UUID `doc_id` contracts.
- Mixed-success background processing must remain visible through status and error messaging.
