# Document Lifecycle API

## User Outcomes

- Upload a PDF for OCR processing.
- Poll or fetch processing status by `doc_id`.
- Download original or translated files.
- Trigger translation, fetch summaries, retry indexing, and delete documents.

## Acceptance Notes

- Current user-facing document lifecycle endpoints should be auth-protected.
- Document file/status/delete routes should use UUID `doc_id` contracts.
- Mixed-success background processing must remain visible through status and error messaging.
