# Evaluation Phase 1 and 2 Completion

## Summary

Phase 1 and Phase 2 of the web-based evaluation system are complete on the backend.
The service now supports both evaluation setup workflows and long-running campaign execution.

## Phase 1 Delivered

- Added the `evaluation/` backend module and router registration
- Added per-user JSON storage for test cases and model config presets
- Added dynamic Gemini model discovery
- Added protected REST APIs for:
  - `GET/POST/PUT/DELETE /api/evaluation/test-cases`
  - `GET/POST/PUT/DELETE /api/evaluation/model-configs`
  - `GET /api/evaluation/models`
- Added `tests/test_evaluation_api.py`

## Phase 2 Delivered

- Added `evaluation/db.py` with SQLite persistence and WAL mode
- Added `evaluation/retry.py` for tenacity retry handling and request budget control
- Added `evaluation/rag_modes.py` to reuse benchmark logic as an importable module
- Added `evaluation/campaign_engine.py` for async campaign execution and incremental result writes
- Added `evaluation/campaign_schemas.py` for campaign request, status, result, and SSE payloads
- Added campaign APIs for create, list, stream, results, and cancel
- Added request-scoped LLM overrides so concurrent campaigns do not share model settings
- Added `tests/test_campaign_engine.py`

## Validation

- `pytest tests/test_evaluation_api.py`
- `pytest tests/test_campaign_engine.py tests/test_evaluation_api.py`
- Concurrent SQLite write stress test:
  - 4 questions
  - 4 modes
  - 2 runs
  - `batch_size=4`
  - verified no missing result rows and `PRAGMA journal_mode=wal`

## Notes

- Runtime SQLite artifacts are ignored via `.gitignore`
- `Phase 3` remains the next implementation target: RAGAS evaluation and result visualization support
