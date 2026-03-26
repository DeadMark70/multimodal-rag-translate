# BACKEND

## Stack

- Python 3.10+
- FastAPI + Uvicorn
- SSE-Starlette
- Google GenAI SDK + LangChain runtime integrations
- FAISS + GraphRAG + Supabase + SQLite
- Pytest + pytest-asyncio

## Runtime Entry

- Thin entrypoint: `main.py`
- App factory and lifecycle: `core/app_factory.py`
- Shared error envelope: `core/errors.py`
- Shared provider/auth/upload helpers live under `core/`

## Router Prefixes

- `/pdfmd`
- `/rag`
- `/imagemd`
- `/multimodal`
- `/stats`
- `/graph`
- `/api/evaluation`
- `/api/conversations`

## Lifecycle Responsibilities

- Load env from `config.env`
- Configure logging and provider selection
- Attach request-id middleware and global error handlers
- Initialize Supabase client handle
- Initialize evaluation SQLite database
- Warm up RAG components unless fake/test providers are enabled
- Warm up PDF OCR unless fake/test providers are enabled

## Subsystem Ownership

- `pdfserviceMD/`: document ingestion, OCR artifacts, translation, summaries, retry-index lifecycle
- `data_base/`: ordinary ask, streamed ask, Deep Research orchestration, retrieval, reranking, indexing
- `graph_rag/`: graph extraction, graph store, optimize/rebuild/retry/purge maintenance
- `conversations/`: conversation and message persistence
- `evaluation/`: test cases, model presets, campaigns, traces, metrics, rerun/evaluate flows
- `stats/`: dashboard aggregates
- `multimodal_rag/` and `image_service/`: multimodal extraction and image translation support

## Runtime-Critical Behaviors

- Protected routes depend on `get_current_user_id`.
- `core/errors.py` returns a standard `{ error: { code, message, request_id, details? } }` envelope.
- Request middleware attaches `X-Request-Id` to the response.
- `TEST_MODE` or `USE_FAKE_PROVIDERS` skip real warmups and provider calls during startup-sensitive paths.
- Evaluation persists campaign state in SQLite with WAL mode and supports results, traces, metrics, manual evaluate, cancel, and SSE reconnect.
- Canonical metadata writes use `doc_id`; `original_doc_uid` remains compatibility fallback on read/delete paths only.

## Focused Verification Surface

- Contract and router tests under `tests/`
- Evaluation engine and persistence tests
- GraphRAG extractor/store/router tests
- PDF service repository/background/manual-translation tests
- Full backend acceptance: `.\.venv\Scripts\python.exe -m pytest`
