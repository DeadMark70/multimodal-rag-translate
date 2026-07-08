# Generated API Surface

Human-maintained inventory of the current backend surface.

## Router Prefixes

| Prefix | Area | High-value endpoints |
|---|---|---|
| `/pdfmd` | document lifecycle | `/list`, `/upload_pdf_md`, `/ocr`, `/file/{doc_id}/status`, `/file/{doc_id}`, `/file/{doc_id}/translate`, `/file/{doc_id}/retry-index`, `/file/{doc_id}/summary`, `/file/{doc_id}` DELETE |
| `/rag` | ask and research | `/ask`, `/ask/stream`, `/research`, `/plan`, `/execute`, `/execute/stream` |
| `/graph` | graph state and maintenance | `/status`, `/data`, `/documents`, `/optimize`, `/rebuild`, `/rebuild-full`, document retry/purge endpoints, `/node-vector/sync`, `/node-vector/sync/status` |
| `/api/evaluation` | evaluation runtime | `/test-cases`, `/models`, `/model-configs`, `/campaigns`, `/campaigns/{id}/results`, `/campaigns/{id}/overview`, `/campaigns/{id}/runs`, `/campaigns/{id}/mode-comparison`, `/campaigns/{id}/question-comparison`, `/campaigns/{id}/cost-latency`, `/campaigns/{id}/router-analysis`, `/campaigns/{id}/ablation`, `/campaigns/{id}/human-vs-auto`, `/campaigns/{id}/human-eval-queue`, `/campaigns/{id}/repeat-stability`, `/campaigns/{id}/errors`, `/campaigns/{id}/export`, `/campaigns/{id}/traces`, `/campaigns/{id}/metrics`, `/campaigns/{id}/evaluate`, `/campaigns/{id}/cancel`, `/campaigns/{id}/stream`, `/runs/{run_id}/*` |
| `/api/conversations` | conversation persistence | list/create/detail/update/delete, `/{conversation_id}/messages` |
| `/stats` | dashboard stats | `/dashboard` |
| `/multimodal` | multimodal extraction | `/extract`, `/file/{doc_id}` DELETE |
| `/imagemd` | image translation | `/translate_image` |

## Evaluation Contract Snapshot

- Test-case schema now accepts and returns:
  - `ground_truth_short`
  - `key_points`
  - `ragas_focus`
- Test-case research metadata now also accepts and returns:
  - `question_version`
  - `required_modalities`
  - `atomic_facts`
  - `expected_evidence`
- Campaign result rows now persist and return the same three fields for executed samples, plus `execution_profile` and `context_policy_version` for runtime/evaluator comparability.
- Campaign result rows also persist run snapshots for research analysis:
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
- Metrics response now exposes:
  - `available_metrics`
  - row-level `metric_values`
  - row-level `reference_source`
  - `summary_by_mode`
  - `summary_by_category`
  - `summary_by_focus`
- `GET /api/evaluation/models` is bearer-auth protected (no anonymous model discovery).
- Dataset tooling lives in `evaluation/dataset_generator.py` and derives `ragas_ready.json` from the master dataset.
- Evaluation `agentic` is a dedicated baseline profile (`agentic_eval_v7_semantic_router_semantic_contextual`), not a generic alias for user Deep Research; it now adds semantic classifier routing, complexity-based strategy budgeting, micro-routing with retrieval policy hints, dynamic tier shift, reverse-pruning, and additive trace telemetry under `AGENTIC_SEMANTIC_ROUTER_MODE=off|shadow|active`, with agentic context policy rows versioned as `v4_semantic_router_gate`.
- Deep Research + evaluation `agentic` execution responses now carry structured fact persistence fields (`sub_tasks[].atomic_facts` and top-level `fact_state`) used for follow-up planning context during drill-down.
- Normalized observability detail tables back the research dashboard:
  - `evaluation_trace_events`
  - `evaluation_llm_calls`
  - `evaluation_retrieval_events`
  - `evaluation_retrieval_chunks`
  - `evaluation_context_packs`
  - `evaluation_tool_calls`
  - `evaluation_routing_decisions`
  - `evaluation_claims`
  - `evaluation_human_ratings`
- Research analytics and run-detail endpoints currently exposed under `/api/evaluation`:
  - campaign aggregates: `/overview`, `/runs`, `/mode-comparison`, `/question-comparison`, `/cost-latency`, `/router-analysis`, `/ablation`, `/human-vs-auto`, `/human-eval-queue`, `/repeat-stability`, `/errors`, `/export`
  - run detail: `/runs/{run_id}/trace`, `/retrieval`, `/context`, `/llm-calls`, `/tools`, `/visual`, `/graph`, `/claims`, `/metrics`, `/diff`
  - campaign-scoped dump: `/campaigns/{campaign_id}/runs/{run_id}/observability`
- Human review flows are auth-protected and use hashed rater identity:
  - `GET /api/evaluation/campaigns/{campaign_id}/human-eval-queue`
  - `POST /api/evaluation/runs/{run_id}/human-ratings`
  - `GET /api/evaluation/campaigns/{campaign_id}/human-vs-auto`
- Export is redaction-aware and POST-only for the research surface:
  - `POST /api/evaluation/campaigns/{campaign_id}/export`
  - supports `include_raw_trace_payloads`, `include_prompt_previews`, `include_full_prompts`, `include_answers`, `include_retrieved_excerpts`
- Campaign SSE remains coarse-grained (`campaign_snapshot`, `campaign_progress`, terminal `campaign_*` events). `event_schema_version="1.0"` and monotonic `sequence` currently apply to persisted trace events, not the SSE envelope.
- Legacy campaigns remain readable even when research observability tables are empty; research run-detail endpoints return empty collections instead of failing.

## Shared Runtime Contracts

- Request-id middleware returns `X-Request-Id`.
- Errors normalize to `{ error: { code, message, request_id, details? } }`.
- Startup warmups are skipped when `TEST_MODE` or `USE_FAKE_PROVIDERS` is enabled.
- Evaluation persists to SQLite and supports result, trace, metric, cancel, and stream recovery flows.
- Evaluation research APIs treat `campaign_results.id` as `run_id` for new endpoints while keeping `campaign_result_id` in legacy trace and metrics surfaces.
- Vector-store runtime coordination is async-first:
  - same-user FAISS mutations are serialized behind per-user async locks
  - FAISS load/save/create/delete, BM25 construction, retriever invokes, and chunk expansion are offloaded off the event loop
- `/pdfmd` upload/retry-index and document delete paths now share the same async vector-store seam used by `/rag` retrieval/index maintenance.
- GraphRAG keeps API contracts unchanged while adding local node-vector autosync and retrieval sidecars:
  - node-vector files: `uploads/<user>/rag_index/node_index.faiss`, `node_index.pkl`, `node_index_map.json`, `node_index.meta.json`
  - upload extraction and graph maintenance mark node-vector state dirty and trigger autosync when enabled
  - graph local search now attempts vector seed retrieval first, then safely falls back to legacy `identify_query_entities + fuzzy label match`
  - manual backfill APIs now expose async sync start + polling status for large legacy graphs (`POST /graph/node-vector/sync`, `GET /graph/node-vector/sync/status`)
  - node-vector embedding calls now enforce process-local per-user request budget (`GRAPH_NODE_VECTOR_EMBEDDING_RPM_LIMIT`, default `1000` RPM) with wait-queue + retry/backoff semantics
- Production markdown ingestion now routes through named indexing profiles; compatibility default remains `recursive_baseline` while upload/retry-index paths currently opt into `semantic_contextual`.
- `/rag/ask` and `/rag/ask/stream` keep the existing schemas/SSE phases, but `enable_evaluation=true` now reuses the first RAG pass instead of issuing a second `rag_answer_question(...)` call for metrics.
- RAGAS reference selection is `ground_truth_short ?? ground_truth` and evaluator context ingestion is deterministic plus answer-aware (`v3_answer_aware_pack`: top 8 chunks, 1800 chars each, whitespace-normalized, overlap-ranked, task-aware when metadata exists).




