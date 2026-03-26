# BACKEND

## Stack

- Python 3.10+
- FastAPI + Uvicorn
- SSE-Starlette
- LangChain ecosystem + Gemini
- FAISS + Sentence-Transformers + GraphRAG tooling
- Pytest + pytest-asyncio

## Backend Architecture

- App entry: `main.py`
- App factory and lifecycle: `core/app_factory.py`
- Routers:
  - `pdfserviceMD/router.py`
  - `data_base/router.py`
  - `image_service/router.py`
  - `multimodal_rag/router.py`
  - `graph_rag/router.py`
  - `conversations/router.py`
  - `stats/router.py`
  - `evaluation/router.py`

## Gemini Layering

- Control plane uses the direct Google GenAI SDK through `core/google_genai_client.py`.
- `evaluation/model_discovery.py` stays on the control plane and only handles model listing, filtering, normalization, and fallback behavior.
- Runtime LLM access stays behind `core/providers.py`, with `core/llm_factory.py` as the only `ChatGoogleGenerativeAI` construction point.
- Runtime embeddings stay centralized in `data_base/vector_store_manager.py`, which remains the only `GoogleGenerativeAIEmbeddings` construction point.
- Business logic modules should not mix direct `google-genai` client creation with runtime `get_llm(...)` usage.

## Evaluation Backend

### Phase 1 delivered

- `evaluation/schemas.py`: test case, import/export, model config, available model schemas
- `evaluation/storage.py`: per-user JSON storage under `uploads/<user_id>/evaluation/`
- `evaluation/model_discovery.py`: dynamic Gemini model listing with cache via the Google GenAI SDK
- `evaluation/router.py`: `/api/evaluation/test-cases`, `/models`, `/model-configs`

### Phase 2 delivered

- `evaluation/db.py`: SQLite repository layer backed by `pdftopng/data/evaluation.db`
- `evaluation/retry.py`: tenacity-based retry wrapper for 429/503 and RPM budget helper
- `evaluation/rag_modes.py`: importable benchmark execution core extracted from Bergen flow
- `evaluation/campaign_engine.py`: async campaign orchestration, incremental result persistence, cancellation
- `evaluation/router.py`: `/api/evaluation/campaigns`, `/campaigns/{id}/stream`, `/results`, `/cancel`
- `core/llm_factory.py`: request-scoped LLM overrides to avoid cross-campaign model leakage

### Runtime behavior

- Campaign progress is persisted in SQLite and survives browser refreshes
- SSE clients reconnect by campaign id and recover from the latest DB snapshot
- SQLite runs in WAL mode and is ignored from git via `data/evaluation.db*`

### Focused verification

- `tests/test_evaluation_api.py`: Phase 1 CRUD and dynamic model discovery coverage
- `tests/test_campaign_engine.py`: smoke campaign, cancel path, retry behavior, concurrent SQLite write stress test

## API Boundary Rules

1. Keep request/response schemas explicit and typed.
2. Keep auth dependencies on protected endpoints.
3. Keep file path handling UUID-safe and traversal-safe.
4. Keep heavy CPU tasks off the event loop.

## Runtime Notes

- CORS defaults are local-dev friendly and overrideable with `CORS_ORIGINS`.
- Environment is loaded from `config.env` in app factory bootstrap.
