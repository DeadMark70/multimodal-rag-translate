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
- GraphRAG now has two contract surfaces:
  - `graph_raw_current` remains the legacy rollback and ablation baseline.
  - the evidence-locator path treats graph summaries, community summaries, and inferred relations as retrieval hints only; final answer evidence must resolve back to source chunks, tables, figures, formulas, captions, or full provenance anchors.
  - `data_base.RAG_QA_service._get_graph_context(...)` preserves its legacy return contracts: a context string by default, `(context, evidence)` with `return_evidence=True`, and `(context, evidence, details)` when `return_details=True`. The default remains `graph_raw_current`; enabling `graph_evidence_locator_enabled` uses `_get_graph_evidence_bundle(...)`, resolves eligible anchors through a lazy vector-document lookup, and renders only verified source quotes for the legacy prompt.
  - Enabling `graph_to_chunk_enabled` bypasses graph-summary prompt injection. It reuses one lazy vector-document lookup to build the structured bundle, re-resolves eligible source chunks, applies bounded graph boosts, and interleaves the resulting source documents with vector retrieval under a 35% graph cap. Lookup or expansion failures retain the original vector documents.
- GraphRAG local search now supports vector-first node seed retrieval with safe fallback:
  - local node-vector sidecars live under `uploads/<user>/rag_index/` (`node_index.faiss/.pkl`, `node_index_map.json`, `node_index.meta.json`)
  - extraction and graph maintenance paths mark node-vector state dirty and trigger autosync (`GRAPH_NODE_VECTOR_AUTOSYNC`, default enabled)
  - query path tries vector node seeds first (`GRAPH_NODE_VECTOR_SEARCH_ENABLED`, default enabled), then falls back to legacy `LLM entity extraction + fuzzy label match` when index/embedding/score conditions are not met
  - main knobs: `GRAPH_NODE_VECTOR_TOP_K` (default `12`), `GRAPH_NODE_VECTOR_MIN_SCORE` (default `0.35`), `GRAPH_NODE_VECTOR_BATCH_SIZE` (default `64`)
  - manual sync APIs now support legacy graph backfill without blocking long requests:
    - `POST /graph/node-vector/sync` starts a background sync job and persists progress state
    - `GET /graph/node-vector/sync/status` exposes `idle/running/completed/failed`, processed counts, and error/details for polling UIs
  - embedding provider calls in node-vector sync/search are guarded by a process-local per-user limiter (`GRAPH_NODE_VECTOR_EMBEDDING_RPM_LIMIT`, default `1000` RPM) with wait-queue behavior and retry/backoff on 429/transport errors
- Production markdown indexing is profile-aware:
  - default compatibility profile in `data_base/indexing_service.py` remains `recursive_baseline`
  - document upload / retry-index currently opt into `semantic_contextual`
  - formal profiles also include `hierarchical_parent_child` and `hierarchical_parent_child_proposition` for controlled A/B use
- `/rag/ask` and `/rag/ask/stream` now reuse one RAG pipeline execution even when evaluation is enabled:
  - retrieval/documents from the first pass are reused for evaluator metrics
  - chat no longer pays for a second `rag_answer_question(...)` call just to compute `return_docs=True`
- Evaluation `agentic` uses a dedicated baseline (`agentic_eval_v7_semantic_router_semantic_contextual`) distinct from user-facing Deep Research:
  - main-question semantic classifier (LLM + timeout/parse fallback to heuristic)
  - complexity-to-strategy mapping for initial tier/subtask/iteration budget
  - sub-task micro-routing (`direct_point_access`, `broad_context_rag`, `visual_evidence_path`) mapped into existing route profiles
  - semantic retrieval-quality gate + first-round dynamic tier shift (`upshift/downshift/keep`)
  - reverse-pruning for redundant follow-up tasks against structured fact-state
  - additive trace telemetry: classifier decision, complexity score, tier shift, pruned follow-ups, semantic gate score
  - evaluation context policy version for agentic rows is now `v4_semantic_router_gate` (other modes remain `v3_answer_aware_pack`)
  - rollout flag: `AGENTIC_SEMANTIC_ROUTER_MODE=off|shadow|active` (`shadow` records decisions without changing execution behavior)
- Deep Research and evaluation `agentic` drill-down now maintain a shared structured fact-state (`atomic_facts` per sub-task + response-level `fact_state`) and pass that state to planner follow-up generation instead of relying only on raw long-form answer concatenation.
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






## Evaluation Model Thinking Controls

- `evaluation/model_capabilities.py` is the backend source of truth for model-specific thinking controls.
- `GET /api/evaluation/models` returns `AvailableModel.thinking` so the UI can render only valid controls for the selected model.
- Gemini 2.5 Flash-family models use budget controls (`thinking_budget`); Gemini 3.x Flash and Gemma 4-family models use level controls (`thinking_level`). Unknown models expose `control_type="none"`.
- Model preset create/update/list paths normalize incompatible fields before storage, and campaign runtime overrides normalize again before provider execution.
- Campaign configs preserve the normalized preset snapshot so history/results can show the actual reasoning setting used for a run.

## Evaluation Research Dashboard Backend

### Run Snapshots

- `campaign_results.id` is both the legacy `campaign_result_id` and the new `run_id` for research APIs.
- Persisted run rows now carry reproducibility and replay fields used by the dashboard:
  - `question_version`
  - `request_id`
  - `started_at`
  - `completed_at`
  - `total_latency_ms`
  - `total_tokens`
  - `question_snapshot`
  - `model_config_snapshot`
  - `system_version_snapshot`
  - `derived_metrics`
  - `final_answer_hash`
- `question_snapshot` stores the full test-case snapshot captured at execution time, including research metadata such as `required_modalities`, `atomic_facts`, and `expected_evidence`.
- `system_version_snapshot` stores runtime context for comparison work:
  - always: `mode`, `run_number`, `repeat_number`
  - when present: `condition_id`, `condition_label`, `ablation_flags`, `budget`, `execution_profile`, `context_policy_version`
- `derived_metrics` is the lightweight cross-tab summary surface. Current runtime population includes:
  - `repeat_number`
  - ablation metadata such as `condition_id`, `condition_label`, `ablation_flags`
  - `context_count`, `source_doc_count`, `expected_source_count`
  - when claim data exists: `supported_claim_ratio`, `unsupported_claim_ratio`, `citation_precision`, `evidence_coverage`, `repair_count`
  - when gold facts exist: `gold_fact_attrition`
- Legacy rows remain readable. When an older row has none of the new snapshot fields populated, the loader keeps it valid and normalizes `total_tokens` back to `null` instead of treating `0` as a real snapshot value.

### Observability Tables

- Normalized run observability lives alongside legacy `agent_traces`:
  - `evaluation_trace_events`
  - `evaluation_llm_calls`
  - `evaluation_retrieval_events`
  - `evaluation_retrieval_chunks`
  - `evaluation_context_packs`
  - `evaluation_tool_calls`
  - `evaluation_routing_decisions`
  - `evaluation_claims`
  - `evaluation_human_ratings`
  - `evaluation_graph_events`
  - `evaluation_graph_evidence_items`
- Each table is keyed by `run_id` and `campaign_id` so research APIs can assemble run detail without inflating `campaign_results`.
- Table creation and additive column repair for these observability rows live in `evaluation/db.py`; repository read/write methods stay in `evaluation/observability_storage.py`.
- Current detail payload shapes:
  - trace events: stage span/event rows with `event_schema_version`, `sequence`, `stage_type`, `stage_name`, `status`, `payload`, `error`
  - LLM calls: provider/model/tokens/cost plus `prompt_hash`, `prompt_preview`, `response_hash`
  - retrieval: one request row plus chunk rows with evidence-match flags
  - context packing: selected vs dropped evidence summary and packing metadata
  - tool calls: normalized action/latency/status rows
  - routing: retrospective routing decision rows
  - claims: support status, evidence list, unsupported reason, repair metadata
  - human ratings: rubric scores plus blinded-review flags
  - graph events: route, evidence mode, feature-flag snapshot, graph schema/prompt snapshots, matched entities/communities, and graph retrieval latency/token summaries
  - graph evidence items: per-evidence lifecycle rows capturing node/edge ids, provenance status, locator usage, and context-packing state
- Recorder writes are best-effort by default. `EvaluationRunRecorder(strict=False)` logs failures and does not fail the campaign on observability write errors.

### Research And Analytics API

- Existing compatibility endpoints remain:
  - `GET /api/evaluation/campaigns/{campaign_id}/results`
  - `GET /api/evaluation/campaigns/{campaign_id}/traces`
  - `GET /api/evaluation/campaigns/{campaign_id}/results/{campaign_result_id}/trace`
  - `GET /api/evaluation/campaigns/{campaign_id}/metrics`
  - `POST /api/evaluation/campaigns/{campaign_id}/evaluate`
  - `POST /api/evaluation/campaigns/{campaign_id}/cancel`
  - `GET /api/evaluation/campaigns/{campaign_id}/stream`
- Research dashboard endpoints are all auth-protected and run off `run_id`:
  - campaign aggregates:
    - `GET /api/evaluation/campaigns/{campaign_id}/overview`
    - `GET /api/evaluation/campaigns/{campaign_id}/runs`
    - `GET /api/evaluation/campaigns/{campaign_id}/mode-comparison`
    - `GET /api/evaluation/campaigns/{campaign_id}/question-comparison`
    - `GET /api/evaluation/campaigns/{campaign_id}/cost-latency`
    - `GET /api/evaluation/campaigns/{campaign_id}/router-analysis`
    - `GET /api/evaluation/campaigns/{campaign_id}/ablation`
    - `GET /api/evaluation/campaigns/{campaign_id}/repeat-stability`
    - `GET /api/evaluation/campaigns/{campaign_id}/human-vs-auto`
    - `GET /api/evaluation/campaigns/{campaign_id}/human-eval-queue`
    - `GET /api/evaluation/campaigns/{campaign_id}/errors`
    - `POST /api/evaluation/campaigns/{campaign_id}/export`
  - run detail:
    - `GET /api/evaluation/runs/{run_id}/trace`
    - `GET /api/evaluation/runs/{run_id}/retrieval`
    - `GET /api/evaluation/runs/{run_id}/context`
    - `GET /api/evaluation/runs/{run_id}/llm-calls`
    - `GET /api/evaluation/runs/{run_id}/tools`
    - `GET /api/evaluation/runs/{run_id}/visual`
    - `GET /api/evaluation/runs/{run_id}/graph`
    - `GET /api/evaluation/runs/{run_id}/claims`
    - `GET /api/evaluation/runs/{run_id}/metrics`
    - `GET /api/evaluation/runs/{run_id}/diff?baseline_run_id=...`
    - `POST /api/evaluation/runs/{run_id}/human-ratings`
  - campaign-scoped full observability dump:
    - `GET /api/evaluation/campaigns/{campaign_id}/runs/{run_id}/observability`

### SSE And Event Ordering

- Campaign progress streaming is still coarse-grained SSE:
  - `campaign_snapshot`
  - `campaign_progress`
  - terminal `campaign_completed`, `campaign_failed`, or `campaign_cancelled`
- `campaign_snapshot` and terminal events serialize the full `CampaignStatus` model.
- `campaign_progress` serializes `CampaignProgressEvent` with counters and current question/mode.
- The current SSE envelope does not add `event_schema_version` or `sequence`.
- Versioning and ordering are instead enforced on persisted run observability:
  - every `evaluation_trace_events` row carries `event_schema_version="1.0"`
  - `EvaluationRunRecorder` emits monotonic `sequence` values per run recorder
  - repository reads order by `sequence ASC, started_at ASC`
  - running spans keep `duration_ms = null` until the closing event is written

### Legacy Compatibility And Empty States

- Legacy campaigns without normalized observability rows remain valid for:
  - results listing
  - legacy trace endpoints
  - metrics
  - research aggregate endpoints
- Empty research collections are expected for older campaigns:
  - run trace/retrieval/context/tool/claim endpoints return typed wrapper objects with empty collection fields such as `trace_events=[]`, `retrieval_events=[]`, `chunks=[]`, `context_packs=[]`, `tool_calls=[]`, or `claims=[]`
  - human ratings do not have a standalone `GET /runs/{run_id}/human-ratings` endpoint; they are exposed as `human_ratings=[]` inside `GET /campaigns/{campaign_id}/runs/{run_id}/observability`
  - `GET /campaigns/{campaign_id}/runs/{run_id}/observability` returns a wrapper object with empty detail collections
  - `router-analysis` returns retrospective summaries with `rows=[]` when no routing rows exist
  - `human-vs-auto` returns `sample_count=0` and `"No paired human/auto samples yet."`
  - `campaigns/{campaign_id}/errors` returns `rows=[]` when no sanitized errors exist
- Legacy `agentic` rows without `execution_profile` are normalized to `legacy_shared` on read so comparison views do not break.
- Router execution is still gated. Creating a campaign with `mode="router"` and `actual_router_execution_enabled=false` fails with `400` and directs callers to retrospective analysis.

### Prompt Storage And Redaction

- Default run/result APIs do not surface raw prompt bodies.
- LLM observability rows are designed around:
  - `prompt_hash`
  - `prompt_preview`
  - optional prompt payload data inside `payload_json`
- Export is POST-only for the research surface:
  - `POST /api/evaluation/campaigns/{campaign_id}/export`
  - request body controls redaction:
    - `include_raw_trace_payloads`
    - `include_prompt_previews`
    - `include_full_prompts`
    - `include_answers`
    - `include_retrieved_excerpts`
    - `format` (`json` only)
- Export behavior in current code:
  - trace event `payload` is blanked unless `include_raw_trace_payloads=true`
  - `prompt_preview` is omitted unless `include_prompt_previews=true`
  - `payload.full_prompt` is removed unless `include_full_prompts=true`
  - answer, ground truth, and final-answer hash are removed when `include_answers=false`
  - retrieval excerpts and result context/source lists are removed when `include_retrieved_excerpts=false`
  - each `retrieval_summary[]` row now also carries `graph_events`, `graph_event_count`, `graph_evidence_items`, and `graph_evidence_item_count` when GraphRAG observability rows exist for that run
  - `question_snapshot` is always partially redacted on export by removing `ground_truth`, `ground_truth_short`, `source_docs`, `atomic_facts`, and `expected_evidence`
- Sanitized errors are stored and exported instead of raw provider dumps. Multiline stack traces and obvious secrets are redacted.
