# Generated API Surface

Human-maintained inventory of the current backend surface.

<!-- BEGIN GENERATED OPENAPI ROUTES -->
| Method | Path | Operation ID |
|---|---|---|
| GET | `/` | `read_root__get` |
| GET | `/api/conversations` | `list_conversations_api_conversations_get` |
| POST | `/api/conversations` | `create_conversation_api_conversations_post` |
| GET | `/api/conversations/page` | `list_conversation_page_api_conversations_page_get` |
| DELETE | `/api/conversations/{conversation_id}` | `delete_conversation_api_conversations__conversation_id__delete` |
| GET | `/api/conversations/{conversation_id}` | `get_conversation_api_conversations__conversation_id__get` |
| PATCH | `/api/conversations/{conversation_id}` | `update_conversation_api_conversations__conversation_id__patch` |
| POST | `/api/conversations/{conversation_id}/messages` | `create_message_api_conversations__conversation_id__messages_post` |
| GET | `/api/conversations/{conversation_id}/messages/page` | `list_message_page_api_conversations__conversation_id__messages_page_get` |
| GET | `/api/evaluation/campaigns` | `get_campaigns_api_evaluation_campaigns_get` |
| POST | `/api/evaluation/campaigns` | `create_campaign_api_evaluation_campaigns_post` |
| POST | `/api/evaluation/campaigns/preflight` | `preflight_campaign_api_evaluation_campaigns_preflight_post` |
| GET | `/api/evaluation/campaigns/{campaign_id}/ablation` | `get_campaign_ablation_api_evaluation_campaigns__campaign_id__ablation_get` |
| GET | `/api/evaluation/campaigns/{campaign_id}/agent-behavior` | `get_campaign_agent_behavior_api_evaluation_campaigns__campaign_id__agent_behavior_get` |
| GET | `/api/evaluation/campaigns/{campaign_id}/analytics-dashboard` | `get_campaign_analytics_dashboard_api_evaluation_campaigns__campaign_id__analytics_dashboard_get` |
| POST | `/api/evaluation/campaigns/{campaign_id}/cancel` | `cancel_campaign_api_evaluation_campaigns__campaign_id__cancel_post` |
| GET | `/api/evaluation/campaigns/{campaign_id}/cost-latency` | `get_campaign_cost_latency_api_evaluation_campaigns__campaign_id__cost_latency_get` |
| GET | `/api/evaluation/campaigns/{campaign_id}/errors` | `get_campaign_errors_api_evaluation_campaigns__campaign_id__errors_get` |
| POST | `/api/evaluation/campaigns/{campaign_id}/evaluate` | `evaluate_campaign_api_evaluation_campaigns__campaign_id__evaluate_post` |
| POST | `/api/evaluation/campaigns/{campaign_id}/export` | `post_campaign_export_api_evaluation_campaigns__campaign_id__export_post` |
| GET | `/api/evaluation/campaigns/{campaign_id}/human-eval-queue` | `get_campaign_human_eval_queue_api_evaluation_campaigns__campaign_id__human_eval_queue_get` |
| GET | `/api/evaluation/campaigns/{campaign_id}/human-vs-auto` | `get_campaign_human_vs_auto_api_evaluation_campaigns__campaign_id__human_vs_auto_get` |
| GET | `/api/evaluation/campaigns/{campaign_id}/jobs` | `list_campaign_jobs_api_evaluation_campaigns__campaign_id__jobs_get` |
| GET | `/api/evaluation/campaigns/{campaign_id}/metrics` | `get_campaign_metrics_api_evaluation_campaigns__campaign_id__metrics_get` |
| GET | `/api/evaluation/campaigns/{campaign_id}/mode-comparison` | `get_campaign_mode_comparison_api_evaluation_campaigns__campaign_id__mode_comparison_get` |
| GET | `/api/evaluation/campaigns/{campaign_id}/overview` | `get_campaign_research_overview_api_evaluation_campaigns__campaign_id__overview_get` |
| GET | `/api/evaluation/campaigns/{campaign_id}/question-comparison` | `get_campaign_question_comparison_api_evaluation_campaigns__campaign_id__question_comparison_get` |
| GET | `/api/evaluation/campaigns/{campaign_id}/release-metrics` | `get_campaign_release_metrics_api_evaluation_campaigns__campaign_id__release_metrics_get` |
| GET | `/api/evaluation/campaigns/{campaign_id}/repeat-stability` | `get_campaign_repeat_stability_api_evaluation_campaigns__campaign_id__repeat_stability_get` |
| POST | `/api/evaluation/campaigns/{campaign_id}/reruns` | `create_campaign_rerun_api_evaluation_campaigns__campaign_id__reruns_post` |
| GET | `/api/evaluation/campaigns/{campaign_id}/research-question-comparison` | `get_campaign_research_question_comparison_api_evaluation_campaigns__campaign_id__research_question_comparison_get` |
| GET | `/api/evaluation/campaigns/{campaign_id}/research-summary` | `get_campaign_research_summary_api_evaluation_campaigns__campaign_id__research_summary_get` |
| GET | `/api/evaluation/campaigns/{campaign_id}/results` | `get_campaign_results_api_evaluation_campaigns__campaign_id__results_get` |
| GET | `/api/evaluation/campaigns/{campaign_id}/results/{campaign_result_id}/trace` | `get_campaign_result_trace_api_evaluation_campaigns__campaign_id__results__campaign_result_id__trace_get` |
| GET | `/api/evaluation/campaigns/{campaign_id}/router-analysis` | `get_campaign_router_analysis_api_evaluation_campaigns__campaign_id__router_analysis_get` |
| GET | `/api/evaluation/campaigns/{campaign_id}/runs` | `get_campaign_research_runs_api_evaluation_campaigns__campaign_id__runs_get` |
| GET | `/api/evaluation/campaigns/{campaign_id}/runs/{run_id}/observability` | `get_campaign_run_observability_api_evaluation_campaigns__campaign_id__runs__run_id__observability_get` |
| GET | `/api/evaluation/campaigns/{campaign_id}/stage-warnings` | `get_campaign_stage_warnings_api_evaluation_campaigns__campaign_id__stage_warnings_get` |
| GET | `/api/evaluation/campaigns/{campaign_id}/stream` | `stream_campaign_api_evaluation_campaigns__campaign_id__stream_get` |
| GET | `/api/evaluation/campaigns/{campaign_id}/traces` | `get_campaign_traces_api_evaluation_campaigns__campaign_id__traces_get` |
| GET | `/api/evaluation/jobs/{job_id}` | `get_evaluation_job_api_evaluation_jobs__job_id__get` |
| POST | `/api/evaluation/jobs/{job_id}/cancel` | `cancel_evaluation_job_api_evaluation_jobs__job_id__cancel_post` |
| GET | `/api/evaluation/jobs/{job_id}/items` | `list_evaluation_job_items_api_evaluation_jobs__job_id__items_get` |
| GET | `/api/evaluation/model-configs` | `get_model_configs_api_evaluation_model_configs_get` |
| POST | `/api/evaluation/model-configs` | `post_model_config_api_evaluation_model_configs_post` |
| DELETE | `/api/evaluation/model-configs/{config_id}` | `remove_model_config_api_evaluation_model_configs__config_id__delete` |
| PUT | `/api/evaluation/model-configs/{config_id}` | `put_model_config_api_evaluation_model_configs__config_id__put` |
| GET | `/api/evaluation/models` | `get_available_models_api_evaluation_models_get` |
| GET | `/api/evaluation/runs/{run_id}/claims` | `get_evaluation_run_claims_api_evaluation_runs__run_id__claims_get` |
| GET | `/api/evaluation/runs/{run_id}/context` | `get_evaluation_run_context_api_evaluation_runs__run_id__context_get` |
| GET | `/api/evaluation/runs/{run_id}/detail` | `get_evaluation_run_detail_api_evaluation_runs__run_id__detail_get` |
| GET | `/api/evaluation/runs/{run_id}/diff` | `get_evaluation_run_diff_api_evaluation_runs__run_id__diff_get` |
| GET | `/api/evaluation/runs/{run_id}/graph` | `get_evaluation_run_graph_tools_api_evaluation_runs__run_id__graph_get` |
| POST | `/api/evaluation/runs/{run_id}/human-ratings` | `post_run_human_rating_api_evaluation_runs__run_id__human_ratings_post` |
| GET | `/api/evaluation/runs/{run_id}/llm-calls` | `get_evaluation_run_llm_calls_api_evaluation_runs__run_id__llm_calls_get` |
| GET | `/api/evaluation/runs/{run_id}/metrics` | `get_evaluation_run_metrics_api_evaluation_runs__run_id__metrics_get` |
| GET | `/api/evaluation/runs/{run_id}/retrieval` | `get_evaluation_run_retrieval_api_evaluation_runs__run_id__retrieval_get` |
| GET | `/api/evaluation/runs/{run_id}/tools` | `get_evaluation_run_tools_api_evaluation_runs__run_id__tools_get` |
| GET | `/api/evaluation/runs/{run_id}/trace` | `get_evaluation_run_trace_api_evaluation_runs__run_id__trace_get` |
| GET | `/api/evaluation/runs/{run_id}/visual` | `get_evaluation_run_visual_tools_api_evaluation_runs__run_id__visual_get` |
| GET | `/api/evaluation/test-cases` | `get_test_cases_api_evaluation_test_cases_get` |
| POST | `/api/evaluation/test-cases` | `create_or_import_test_case_api_evaluation_test_cases_post` |
| DELETE | `/api/evaluation/test-cases/{test_case_id}` | `remove_test_case_api_evaluation_test_cases__test_case_id__delete` |
| PUT | `/api/evaluation/test-cases/{test_case_id}` | `put_test_case_api_evaluation_test_cases__test_case_id__put` |
| GET | `/api/evaluation/work-items/{work_item_id}/attempts` | `get_work_item_attempts_api_evaluation_work_items__work_item_id__attempts_get` |
| GET | `/graph/data` | `get_graph_visualization_data_graph_data_get` |
| POST | `/graph/debug/search` | `debug_graph_search_graph_debug_search_post` |
| GET | `/graph/documents` | `list_graph_documents_graph_documents_get` |
| DELETE | `/graph/documents/{doc_id}` | `purge_graph_document_graph_documents__doc_id__delete` |
| POST | `/graph/documents/{doc_id}/retry` | `retry_graph_document_graph_documents__doc_id__retry_post` |
| POST | `/graph/node-vector/sync` | `start_node_vector_sync_graph_node_vector_sync_post` |
| GET | `/graph/node-vector/sync/status` | `get_node_vector_sync_status_graph_node_vector_sync_status_get` |
| GET | `/graph/nodes/{node_key}/evidence` | `get_graph_node_evidence_graph_nodes__node_key__evidence_get` |
| POST | `/graph/optimize` | `optimize_graph_graph_optimize_post` |
| GET | `/graph/quality` | `get_graph_quality_graph_quality_get` |
| POST | `/graph/rebuild` | `rebuild_graph_graph_rebuild_post` |
| POST | `/graph/rebuild-full` | `rebuild_graph_full_graph_rebuild_full_post` |
| POST | `/graph/rebuild-full/resume` | `resume_rebuild_graph_full_graph_rebuild_full_resume_post` |
| GET | `/graph/rebuild-full/status` | `get_rebuild_graph_full_status_graph_rebuild_full_status_get` |
| GET | `/graph/runtime-quality` | `get_graph_runtime_quality_graph_runtime_quality_get` |
| GET | `/graph/status` | `get_graph_status_graph_status_get` |
| POST | `/imagemd/translate_image` | `translate_image_inplace_imagemd_translate_image_post` |
| POST | `/multimodal/extract` | `extract_from_pdf_endpoint_multimodal_extract_post` |
| DELETE | `/multimodal/file/{doc_id}` | `delete_multimodal_document_multimodal_file__doc_id__delete` |
| DELETE | `/pdfmd/file/{doc_id}` | `delete_pdf_file_pdfmd_file__doc_id__delete` |
| GET | `/pdfmd/file/{doc_id}` | `get_pdf_file_pdfmd_file__doc_id__get` |
| POST | `/pdfmd/file/{doc_id}/retry-index` | `retry_document_index_pdfmd_file__doc_id__retry_index_post` |
| GET | `/pdfmd/file/{doc_id}/status` | `get_processing_status_pdfmd_file__doc_id__status_get` |
| GET | `/pdfmd/file/{doc_id}/summary` | `get_document_summary_endpoint_pdfmd_file__doc_id__summary_get` |
| POST | `/pdfmd/file/{doc_id}/summary/regenerate` | `regenerate_summary_endpoint_pdfmd_file__doc_id__summary_regenerate_post` |
| POST | `/pdfmd/file/{doc_id}/translate` | `translate_pdf_file_pdfmd_file__doc_id__translate_post` |
| GET | `/pdfmd/list` | `list_documents_endpoint_pdfmd_list_get` |
| POST | `/pdfmd/ocr` | `upload_pdf_md_pdfmd_ocr_post` |
| POST | `/pdfmd/upload_pdf_md` | `upload_pdf_md_pdfmd_upload_pdf_md_post` |
| POST | `/rag/agentic/stream` | `execute_agentic_benchmark_stream_rag_agentic_stream_post` |
| POST | `/rag/ask` | `ask_question_with_context_rag_ask_post` |
| POST | `/rag/ask/stream` | `ask_question_with_context_stream_rag_ask_stream_post` |
| POST | `/rag/execute` | `execute_research_plan_rag_execute_post` |
| POST | `/rag/execute/stream` | `execute_research_plan_stream_rag_execute_stream_post` |
| POST | `/rag/plan` | `generate_research_plan_rag_plan_post` |
| POST | `/rag/research` | `research_question_rag_research_post` |
| GET | `/stats/dashboard` | `get_dashboard_stats_stats_dashboard_get` |
<!-- END GENERATED OPENAPI ROUTES -->

## Router Prefixes

| Prefix | Area | High-value endpoints |
|---|---|---|
| `/pdfmd` | document lifecycle | `/list`, `/upload_pdf_md`, `/ocr`, `/file/{doc_id}/status`, `/file/{doc_id}`, `/file/{doc_id}/translate`, `/file/{doc_id}/retry-index`, `/file/{doc_id}/summary`, `/file/{doc_id}` DELETE |
| `/rag` | ask and research | `/ask`, `/ask/stream`, `/research`, `/plan`, `/execute`, `/execute/stream` |
| `/graph` | graph state, quality, diagnostics, and maintenance | `/status`, `/quality`, `/runtime-quality?campaign_id=...`, `/debug/search`, `/data`, `/documents`, `/optimize`, `/rebuild`, durable `/rebuild-full`, `/rebuild-full/status`, `/rebuild-full/resume`, document retry/purge endpoints, `/node-vector/sync`, `/node-vector/sync/status` |
| `/api/evaluation` | evaluation runtime | `/test-cases`, `/models`, `/model-configs`, `/campaigns`, `/campaigns/{id}/results`, `/campaigns/{id}/overview`, `/campaigns/{id}/runs`, `/campaigns/{id}/mode-comparison`, `/campaigns/{id}/question-comparison`, `/campaigns/{id}/cost-latency`, `/campaigns/{id}/router-analysis`, `/campaigns/{id}/ablation`, `/campaigns/{id}/human-vs-auto`, `/campaigns/{id}/human-eval-queue`, `/campaigns/{id}/repeat-stability`, `/campaigns/{id}/errors`, `/campaigns/{id}/export`, `/campaigns/{id}/traces`, `/campaigns/{id}/metrics`, `/campaigns/{id}/evaluate`, `/campaigns/{id}/cancel`, `/campaigns/{id}/stream`, durable `/campaigns/{id}/reruns`, `/campaigns/{id}/jobs`, `/jobs/{job_id}`, `/jobs/{job_id}/items`, `/jobs/{job_id}/cancel`, `/work-items/{work_item_id}/attempts`, `/runs/{run_id}/*` |
| `/api/conversations` | conversation persistence | legacy list/create/detail/update/delete, `/page` summary cursor list, `/{conversation_id}/messages/page`, `/{conversation_id}/messages` |
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
- Evaluation retrieval profiles are versioned per changed mode: `advanced_eval_v2_multiquery_recursive_baseline`, `graph_eval_v2_multiquery_locator_recursive_baseline`, `<ablation_mode>_eval_v2_multiquery_recursive_baseline`, and `agentic_eval_v8_multiquery_locator_recursive_baseline`.
- Evaluation Advanced, Graph family, and Agentic disable HyDE. Main Graph and Agentic graph routes use provenance-gated locator-to-chunk evidence; `graph_raw_current` remains the explicit raw-legacy control.
- Agentic CRAG corrective retrieval uses Multi-Query and falls back to the original question when query generation fails.
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
  - `evaluation_graph_events`
  - `evaluation_graph_evidence_items`
- GraphRAG evaluation observability now snapshots:
  - graph route and router reason
  - graph evidence mode
  - graph feature flags
  - graph snapshot/schema/prompt versions
  - per-evidence provenance and context lifecycle rows
- Research analytics and run-detail endpoints currently exposed under `/api/evaluation`:
  - campaign aggregates: `/overview`, `/runs`, `/mode-comparison`, `/question-comparison`, `/cost-latency`, `/router-analysis`, `/ablation`, `/human-vs-auto`, `/human-eval-queue`, `/repeat-stability`, `/errors`, `/export`
  - run detail: `/runs/{run_id}/trace`, `/retrieval`, `/context`, `/llm-calls`, `/tools`, `/visual`, `/graph`, `/claims`, `/metrics`, `/diff`
  - campaign-scoped dump: `/campaigns/{campaign_id}/runs/{run_id}/observability`
- Ablation campaigns with at least two persisted condition IDs expose `summaries.condition_comparison` on `/ablation`. It contains immutable condition labels/flags, completed/failed counts, finite `answer_correctness`/`faithfulness`/`answer_relevancy` aggregates, tokens, latency, and matched `(question_id, repeat_number)` deltas. Failed, unpaired, missing, and non-finite observations are reported as exclusions or `null`, never zero.
- Human review flows are auth-protected and use hashed rater identity:
  - `GET /api/evaluation/campaigns/{campaign_id}/human-eval-queue`
  - `POST /api/evaluation/runs/{run_id}/human-ratings`
  - `GET /api/evaluation/campaigns/{campaign_id}/human-vs-auto`
- Export is redaction-aware and POST-only for the research surface:
  - `POST /api/evaluation/campaigns/{campaign_id}/export`
  - supports `include_raw_trace_payloads`, `include_prompt_previews`, `include_full_prompts`, `include_answers`, `include_retrieved_excerpts`
  - `retrieval_summary[]` entries may include additive GraphRAG observability fields: `graph_events`, `graph_event_count`, `graph_evidence_items`, `graph_evidence_item_count`
  - exported `runs[]` entries include finite-only `ragas_metrics`; `metrics.condition_comparison` reuses the `/ablation` condition projection
- Campaign SSE remains coarse-grained (`campaign_snapshot`, `campaign_progress`, terminal `campaign_*` events). `event_schema_version="1.0"` and monotonic `sequence` currently apply to persisted trace events, not the SSE envelope.
- Legacy campaigns remain readable even when research observability tables are empty; research run-detail endpoints return empty collections instead of failing.
- Campaign analytics reads use a bounded result projection and campaign-scoped routing bulk query; terminal campaign contexts are reused in the process-local analytics service cache while the campaign `updated_at` marker is unchanged.
- `GET /api/evaluation/campaigns/{campaign_id}/release-metrics` returns the authoritative `ReleaseMetricsReport`. `availability="available"` carries a configured benchmark report; `availability="not_applicable"`, `benchmark_kind="not_applicable"`, and `not_applicable_reason="benchmark_not_configured"` are the normal response for a campaign with no benchmark. In that state the service performs no result, score, accounting, or observability bulk load.
- A configured release report obtains one bounded result/score/work-metadata/accounting/observability snapshot per selected benchmark campaign, not one repository call per run. It never uses large answer, context, or full trace detail blobs as a list/report projection. Terminal reports are cached only for unchanged selected campaign `(id, updated_at, status)` markers; changed markers invalidate the cache and nonterminal campaigns are uncached.
- Campaign trace lists read the migrated compact `agent_traces.summary_json` projection, indexed by campaign, user, and creation time; full `trace_json` stays detail-only. Rows without a usable historical summary remain listable as `not_instrumented`.
- Evaluation result persistence rejects answers over 1,048,576 UTF-8 bytes. Durable and legacy execution expose the stable `EVALUATION_ANSWER_TOO_LARGE` failure code rather than truncating a response.

## Conversation Performance Contract

- `GET /api/conversations` remains the legacy full list for compatibility.
- `GET /api/conversations/page?limit=40&cursor=...&search=...` returns bounded summary rows and a keyset `next_cursor`; large research result metadata is excluded.
- `GET /api/conversations/{conversation_id}/messages/page?limit=50&cursor=...` returns a bounded newest message page and preserves the existing detail endpoint for compatibility.

### Durable Evaluation Jobs

- `POST /api/evaluation/campaigns/{campaign_id}/reruns`
- `GET /api/evaluation/campaigns/{campaign_id}/jobs`
- `GET /api/evaluation/jobs/{job_id}`
- `GET /api/evaluation/jobs/{job_id}/items`
- `GET /api/evaluation/work-items/{work_item_id}/attempts`
- `POST /api/evaluation/jobs/{job_id}/cancel`

Job and attempt responses include ownership-filtered status, retry, safe-error, and compatible-result provenance fields. The legacy `/evaluate` operation delegates to a durable RAGAS rerun for compatibility.

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




