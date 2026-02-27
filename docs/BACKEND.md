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

## API Boundary Rules

1. Keep request/response schemas explicit and typed.
2. Keep auth dependencies on protected endpoints.
3. Keep file path handling UUID-safe and traversal-safe.
4. Keep heavy CPU tasks off the event loop.

## Runtime Notes

- CORS defaults are local-dev friendly and overrideable with `CORS_ORIGINS`.
- Environment is loaded from `config.env` in app factory bootstrap.

