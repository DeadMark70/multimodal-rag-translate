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

## Evaluation Dataset Model

- `ragasfullqa.json` is the master dataset shape.
- Master test cases now support:
  - `ground_truth` as the long-form research answer
  - `ground_truth_short` as the RAGAS-ready canonical answer
  - `key_points` as structured diagnostic facts
  - `ragas_focus` as metadata for grouping and analysis only
- `evaluation/dataset_generator.py` merges `ragasshortqa.json` into the master file and derives `ragas_ready.json` deterministically.
- Derived dataset metadata now includes:
  - `dataset_version`
  - `dataset_role`
  - `derived_from`
  - `derived_at`

## Runtime-Critical Behaviors

- Protected routes depend on `get_current_user_id`.
- Evaluation model discovery (`GET /api/evaluation/models`) is protected by the same auth dependency and exposes HTTP bearer security in OpenAPI.
- `core/errors.py` returns a standard `{ error: { code, message, request_id, details? } }` envelope.
- Request middleware attaches `X-Request-Id` to the response.
- `TEST_MODE` or `USE_FAKE_PROVIDERS` skip real warmups and provider calls during startup-sensitive paths.
- Evaluation persists campaign state in SQLite with WAL mode and supports results, traces, metrics, manual evaluate, cancel, and SSE reconnect.
- Vector-store hot paths now run through an async coordination seam in `data_base/vector_store_manager.py`:
  - FAISS load/save/create/delete, BM25 construction, synchronous retriever `invoke(...)`, and short-chunk expansion are offloaded from request/background coroutines
  - per-user async locks serialize same-user FAISS mutations so ask/upload/retry-index/delete do not race the same index directory
- Upload, retry-index, visual-summary indexing, and document-vector deletion now share that async vector-store seam instead of calling synchronous FAISS work directly from route/background coroutines.
- Production markdown indexing is profile-aware:
  - default compatibility profile in `data_base/indexing_service.py` remains `recursive_baseline`
  - document upload / retry-index currently opt into `semantic_contextual`
  - formal profiles also include `hierarchical_parent_child` and `hierarchical_parent_child_proposition` for controlled A/B use
- `/rag/ask` and `/rag/ask/stream` now reuse one RAG pipeline execution even when evaluation is enabled:
  - retrieval/documents from the first pass are reused for evaluator metrics
  - chat no longer pays for a second `rag_answer_question(...)` call just to compute `return_docs=True`
- Evaluation `agentic` uses a dedicated baseline (`agentic_eval_v6_semantic_contextual`) distinct from user-facing Deep Research: tightened numeric benchmark routing, figure-flow first-task anchoring to the original question (plus at most one gap-focused auxiliary task), single-task synthesis-lite normalization, lightweight retrieval-quality gating before drill-down, deep image analysis kept enabled, and versioning aligned to the semantic-contextual indexing baseline.
- Canonical metadata writes use `doc_id`; `original_doc_uid` remains compatibility fallback on read/delete paths only.

## RAGAS Evaluation Contract

- Formal reference selection is `ground_truth_short ?? ground_truth`.
- Each metric score row records `reference_source` in `ragas_scores.details_json`.
- Phase 1 metrics are enabled by default:
  - `faithfulness`
  - `answer_correctness`
  - `answer_relevancy`
- Phase 2 metrics are implemented behind `ENABLE_RAGAS_CONTEXT_METRICS`:
  - `context_precision`
  - `context_recall`
- Metric execution is non-blocking per metric; one metric failure does not fail the campaign.
- Metrics API now returns:
  - `available_metrics`
  - per-row `metric_values`
  - `summary_by_mode`
  - `summary_by_category`
  - `summary_by_focus`

## Evaluator Context Policy

- Evaluation no longer reuses the UI preview truncation path.
- Formal evaluator policy is deterministic and answer-aware (`context_policy_version = v3_answer_aware_pack`):
  - normalize whitespace and deduplicate repeated chunks
  - keep at most 8 contexts per sample
  - cap each context at 1800 characters
  - rank contexts against `question + final answer` lexical overlap
  - preserve at least one candidate per agentic subtask when task-tagged evidence is available

## Focused Verification Surface

- Contract and router tests under `tests/`
- Evaluation engine, dataset generator, and persistence tests
- GraphRAG extractor/store/router tests
- PDF service repository/background/manual-translation tests
- Full backend acceptance: `.\.venv\Scripts\python.exe -m pytest`





