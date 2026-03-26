# Generated API Surface

Human-maintained inventory of the current backend surface.

## Router Prefixes

| Prefix | Area | High-value endpoints |
|---|---|---|
| `/pdfmd` | document lifecycle | `/list`, `/upload_pdf_md`, `/ocr`, `/file/{doc_id}/status`, `/file/{doc_id}`, `/file/{doc_id}/translate`, `/file/{doc_id}/retry-index`, `/file/{doc_id}/summary`, `/file/{doc_id}` DELETE |
| `/rag` | ask and research | `/ask`, `/ask/stream`, `/research`, `/plan`, `/execute`, `/execute/stream` |
| `/graph` | graph state and maintenance | `/status`, `/data`, `/documents`, `/optimize`, `/rebuild`, `/rebuild/full`, document retry/purge endpoints |
| `/api/evaluation` | evaluation runtime | `/test-cases`, `/models`, `/model-configs`, `/campaigns`, `/campaigns/{id}/results`, `/campaigns/{id}/traces`, `/campaigns/{id}/metrics`, `/campaigns/{id}/evaluate`, `/campaigns/{id}/cancel`, `/campaigns/{id}/stream` |
| `/api/conversations` | conversation persistence | list/create/detail/update/delete, `/{conversation_id}/messages` |
| `/stats` | dashboard stats | `/dashboard` |
| `/multimodal` | multimodal extraction | `/extract`, `/file/{doc_id}` DELETE |
| `/imagemd` | image translation | `/translate_image` |

## Shared Runtime Contracts

- Request-id middleware returns `X-Request-Id`.
- Errors normalize to `{ error: { code, message, request_id, details? } }`.
- Startup warmups are skipped when `TEST_MODE` or `USE_FAKE_PROVIDERS` is enabled.
- Evaluation persists to SQLite and supports result, trace, metric, cancel, and stream recovery flows.
