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

### Loading-sensitive read paths

- Evaluation analytics uses campaign-level bulk observability reads and a bounded result projection; full answer/context blobs remain restricted to explicit result/export/detail paths.
- Terminal campaign analytics contexts are reused by the process-local analytics service while the campaign `updated_at` marker is unchanged; running campaigns continue to read live state.
- `GET /api/evaluation/campaigns/{campaign_id}/release-metrics` is a separate, backend-authoritative release report. Its `availability` is `available` when a benchmark is configured; a missing benchmark is the normal `not_applicable` state with `not_applicable_reason="benchmark_not_configured"`, not an error or a zero-valued report. That short circuit must not invoke result, score, accounting, or observability bulk loaders.
- A configured release report reads one bounded projection/snapshot set per selected benchmark campaign, independent of run count. Large answer, context, and full trace payloads stay on explicit detail, result, or export routes. Terminal release reports are cached only while every selected campaign remains terminal and its `(id, updated_at, status)` marker is unchanged; a marker change invalidates the cache, while live campaigns are never cached.
- Conversation history exposes additive summary/cursor endpoints under `/api/conversations/page` and `/messages/page`; legacy unbounded endpoints remain for compatibility during frontend migration.
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
- Agentic v9 campaign preflight derives each selected case's current route from
  the same admission-contract boundary used by runtime. Newly imported case IDs
  are not restricted to the frozen Q1-Q16 regression corpus; ownership,
  source-scope resolution, and token/call-budget feasibility remain mandatory.
- Agentic v9 Active Atomic Contract V2 Architecture:
  - **Wave 2 Quote-Qualified Execution Profile**: New open-corpus and
    explicit-scope runs end in
    `finalpack_r1_active_atomic_contract_v2_quote_qualified_v1`. This profile
    requires batch evidence extraction and validation before sufficiency gating:
    every sufficiency-supported slot must resolve to at least one verified
    `is_qualified_evidence()` packet; raw candidate count may exceed qualified
    count, but provider failure cannot increase qualified count; qualification
    provider calls equal actual persisted LLM extraction calls; positive
    controls Q5/Q23 remain answerable with qualified evidence; and the reverted
    64/64 `insufficient` pattern fails the smoke fixture.
  - **Wave 1 Retrieval-Safe Execution Profile (Historical)**: Earlier Wave 1 runs
    ended in `finalpack_r1_active_atomic_contract_v2_retrieval_safe`. This profile
    requires typed `planner_diagnostics` in canonical observability/export;
    earlier profile values remain historical and are not inferred to have it.
  - **Deterministic Route + Atomic Overlay Ownership**: The deterministic router maintains canonical route admission and feasibility gating, while the atomic contract planner generates an overlay containing sequential evidence slots (`S1..Sn`, $1 \le n \le 8$), synthesis obligations, response constraints, and optional comparison plans.
  - **Evidence Slots vs Synthesis Obligations/Constraints**: `required_slots` define verifiable source-backed retrieval goals. `synthesis_obligations` and `response_constraints` dictate answer reasoning, formatting, and structural constraints rather than raw retrieval tasks.
  - **Single Planning Seam & Zero Active Comparison Calls**: Active atomic execution unifies contract planning into at most one provider call (`atomic_planner_call_count <= 1`, `phase="contract_planning"`, `purpose="atomic_contract_planning"`), entirely replacing separate comparison planning calls (`comparison_planner_call_count == 0`, 0 active `comparison_plan` calls).
  - **Degraded V2 Contract Fallback**: Provider errors, timeouts, or schema violations during atomic planning gracefully degrade to a deterministic fallback contract (`contract_version="2"`, fallback rationale, single catch-all slot `S1`) without disrupting pipeline execution.
  - **Question-Specific Fallback Diagnostics**: A degraded Wave 1 planner
    outcome records `retrieval_query_strategy="safe_fallback_original_question"`
    and one compiled retrieval task, with matching degraded/safe-fallback
    contract provenance and exactly one `S1` whose fallback query is the
    normalized, 512-character-bounded original question. New-profile smoke
    requires `atomic_planner_call_count` in `0..1` and
    `comparison_planner_call_count=0`; historical profiles remain tolerant of
    their absence. New-profile `deterministic` diagnostics record zero planner
    calls and no provider response; `planned` records one call and a provider
    response. Offline smoke accepts safe degradation and does not claim planner
    success before the current/minimal real-server canary pair succeeds.
  - **Explicit Instrumentation Boundaries**: Slot binding is inherited from task targets (`slot_binding_method="task_target_inherited"`), and semantic qualification is recorded as `semantic_qualification="not_enabled"`. Falsified or uninstrumented values fail offline release verification.
  - **Comparison Subject Evidence Grounding**: Comparison subjects bind to evidence slot IDs declared in `required_slots`, ensuring relational and multi-subject queries maintain grounded provenance across rounds.
- Campaign trace-list reads use the migrated `agent_traces.summary_json` projection and its campaign/user/created index; full `trace_json` is reserved for trace-detail reads. Existing rows with no usable summary remain readable as a minimal `not_instrumented` summary rather than forcing the list path to deserialize a complete trace.
- Evaluation answers are capped at 1,048,576 UTF-8 bytes. Oversize answers are recorded as failed work with the stable error code `EVALUATION_ANSWER_TOO_LARGE`; callers must not retry by silently truncating or substituting the answer.
- Vector-store hot paths now run through an async coordination seam in `data_base/vector_store_manager.py`:
  - FAISS load/save/create/delete, BM25 construction, synchronous retriever `invoke(...)`, and short-chunk expansion are offloaded from request/background coroutines
  - per-user async locks serialize same-user FAISS mutations so ask/upload/retry-index/delete do not race the same index directory
- Upload, retry-index, visual-summary indexing, and document-vector deletion now share that async vector-store seam instead of calling synchronous FAISS work directly from route/background coroutines.
- GraphRAG now has two contract surfaces:
  - `graph_raw_current` remains the legacy rollback and ablation baseline.
  - Main evaluation `graph` uses provenance-gated locator-to-chunk source expansion. Evaluation also exposes `graph_locator_to_chunk` and `router_auto_graph` as explicit source-locator experiments.
  - `router_auto_graph` classifies each question from explicit execution hints: relationship/claim-scope queries may use graph retrieval, exact table/figure/formula/numeric questions use it only as a locator when both the asset feature is enabled and a request-scoped `graph_asset_probe_result` is true, and non-graph questions leave vector retrieval unchanged. Explicit manual override is locator-only.
  - Main `graph` and any auto, locator, or graph-to-chunk flag combination source-expand or skip; they never fall back to a raw graph prompt. `graph_raw_current` is the only intentional raw-legacy graph control. Planning decisions are recorded as gate events but never bundle, merge, or prompt graph evidence.
  - Locator-only decisions never inject raw or inferred graph text into the answer prompt. With `graph_to_chunk_enabled`, only resolved source chunks can be merged; graph-located chunks are filtered to caller-supplied `doc_ids` before merging.
  - Structured graph retrieval reuses evaluation graph-event recording. Events retain the mode's explicit feature-flag/evidence-mode snapshot, route and gate reason, measured graph latency, and candidate/resolved/scope-approved/scored/packed item IDs in the route reason. `packed_in_context` is set only for IDs present in the final merged documents; community hints are not final evidence.
  - the evidence-locator path treats graph summaries, community summaries, and inferred relations as retrieval hints only; final answer evidence must resolve back to source chunks, tables, figures, formulas, captions, or full provenance anchors.
  - Schema-first extraction uses the versioned `one_pass_extraction_schema_v1` prompt. Only the V1 allowlisted node and edge types with source quotes verified against the extraction chunk receive provenance anchors; unknown schema values, invalid entity references, quote mismatches, and structured-extraction failures are persisted in `graph.raw_candidates.json` for review and never become graph evidence. It does not fall back to an unverified legacy graph write.
  - Canonical entity aliases persist in `graph.aliases.json` with type and document indexes. Similarity-based automatic merging is limited to Paper, Dataset, Metric, and Method within one type; models, architecture components, and training settings require scoped review, while claims, results, tables, and formulas are never auto-merged.
  - The internal `data_base.rag_graph_runtime._get_graph_context(...)` preserves its legacy return contracts: a context string by default, `(context, evidence)` with `return_evidence=True`, and `(context, evidence, details)` when `return_details=True`. The default remains `graph_raw_current`; enabling `graph_evidence_locator_enabled` uses `data_base.rag_graph_runtime._get_graph_evidence_bundle(...)`, resolves eligible anchors through a lazy vector-document lookup, and renders only verified source quotes for the legacy prompt. External callers should use the facade's public `get_graph_evidence_bundle(...)` export.
  - Enabling `graph_to_chunk_enabled` bypasses graph-summary prompt injection. It reuses one lazy vector-document lookup to build the structured bundle, re-resolves eligible source chunks, applies bounded graph boosts, and interleaves the resulting source documents with vector retrieval under a 35% graph-only cap. Packing identity is normalized as `(doc_id, chunk_id)`, equal scores break by evidence item/document identity, and for a ratio below one graph-only additions are capped at `floor(r * vector_count / (1 - r))`; overlaps do not consume that allowance. Lookup or expansion failures retain the original vector documents.
- GraphRAG local search now supports vector-first node seed retrieval with safe fallback:
  - local node-vector sidecars live under `uploads/<user>/rag_index/` (`node_index.faiss/.pkl`, `node_index_map.json`, `node_index.meta.json`)
  - extraction and graph maintenance paths mark node-vector state dirty and trigger autosync (`GRAPH_NODE_VECTOR_AUTOSYNC`, default enabled)
  - query path tries vector node seeds first (`GRAPH_NODE_VECTOR_SEARCH_ENABLED`, default enabled), then falls back to legacy `LLM entity extraction + fuzzy label match` when index/embedding/score conditions are not met
  - main knobs: `GRAPH_NODE_VECTOR_TOP_K` (default `12`), `GRAPH_NODE_VECTOR_MIN_SCORE` (default `0.35`), `GRAPH_NODE_VECTOR_BATCH_SIZE` (default `64`)
  - manual sync APIs now support legacy graph backfill without blocking long requests:
    - `POST /graph/node-vector/sync` starts a background sync job and persists progress state
    - `GET /graph/node-vector/sync/status` exposes `idle/running/completed/failed`, processed counts, and error/details for polling UIs
  - embedding provider calls in node-vector sync/search are guarded by a process-local per-user limiter (`GRAPH_NODE_VECTOR_EMBEDDING_RPM_LIMIT`, default `1000` RPM) with wait-queue behavior and retry/backoff on 429/transport errors
- Full GraphRAG rebuilds are durable, user-scoped jobs rather than disposable background tasks:
  - `POST /graph/rebuild-full` creates a frozen source-document snapshot and persistent staging graph, or returns the active job instead of starting duplicate work
  - control manifests, locks, staging graphs, and frozen OCR markdown live below the stable per-user graph root rather than an immutable `versions/v*` snapshot; resume remains available after live snapshot promotion
  - every OCR markdown input is copied and SHA-256 checked inside its rebuild job before scheduling, so retries and resumes never consume later OCR edits
  - each document is checkpointed after staging persistence, retryable provider/transport failures receive at most three attempts, and exhausted failures do not stop later documents
  - timeout, transport, HTTP 408, HTTP 429, and HTTP 5xx provider failures are retryable; the durable status exposes `max_attempts` for the UI
  - full rebuild, legacy rebuild, optimize, document retry/purge, and node-vector sync share one stable per-user maintenance lease; `active_job_state` is display metadata rather than the concurrency primitive
  - `GET /graph/rebuild-full/status` is the recovery-safe polling projection; a stale runner becomes `interrupted` but startup/status reads never invoke model work
  - `POST /graph/rebuild-full/resume` manually continues an interrupted job or retries only its failed/partial documents
  - live graph files remain queryable until all snapshotted documents succeed and staging validation/optimization complete; only then is the new snapshot published atomically
- Graph quality and query diagnostics are protected, user-scoped graph APIs:
  - `GET /graph/quality` reports deterministic store quality, including provenance coverage, generic relations, duplicate methods, orphan nodes, and unscoped claims.
  - `GET /graph/runtime-quality?campaign_id=...` aggregates only persisted `evaluation_graph_events` and `evaluation_graph_evidence_items`; it does not infer runtime success from `GraphStore`.
  - `POST /graph/debug/search` returns entity links, graph hints, candidate evidence, and only independently eligible final-context items. Graph hints and unresolved evidence are diagnostic output, not answer evidence.
  - `GET /graph/nodes/{node_key}/evidence` returns only source documents still owned by the authenticated user. Incident-edge text is emitted as a quote only when its persisted anchor has `quote_match` verification and its document row is still owned; unchecked, stale, deleted, and foreign anchors can never become quoted evidence. Graph labels, summaries, and inferred relations are never presented as source quotes.
- Graph asset links persist in `graph.asset_links.json` and are locator metadata, not final answer text:
  - explicit Markdown tables, display formulas, and captions are parsed with their page marker, source text, and hash after Markdown indexing succeeds;
  - visual assets are registered only after their summaries have been successfully indexed, with matching `asset_id` and `chunk_id` metadata for later source resolution;
  - exact table/figure/formula questions probe only the parsed assets belonging to the request's candidate documents. Feature flags and caller-provided hints alone cannot claim asset availability;
  - extraction adds an asset anchor only when its evidence quote matches registered asset source text. The asset still must resolve to a source chunk or asset before it can support final context.
- Graph evaluation modes carry an explicit `ablation_family`: `graph_evidence`, `graph_usage_policy`, or `graph_query_strategy`. Campaign analytics report each family separately rather than treating routing policy as evidence-mechanism quality.
  - `oracle_graph_router` accepts a per-question `graph_oracle_decisions` map in the campaign condition flags; it is an upper-bound intervention, not a production routing policy.
  - The ablation response includes the graph retrieval/evidence/router metric keys per family. Metrics without the required gold annotation remain `null` rather than being inferred from an unrelated score.
- Production markdown indexing is profile-aware:
  - default compatibility profile in `data_base/indexing_service.py` remains `recursive_baseline`
  - document upload / retry-index currently opt into `semantic_contextual`
  - formal profiles also include `hierarchical_parent_child` and `hierarchical_parent_child_proposition` for controlled A/B use
- `/rag/ask` and `/rag/ask/stream` now reuse one RAG pipeline execution even when evaluation is enabled:
  - retrieval/documents from the first pass are reused for evaluator metrics
  - chat no longer pays for a second `rag_answer_question(...)` call just to compute `return_docs=True`
- Evaluation retrieval expansion is mode-scoped:
  - `naive` keeps the plain no-expansion baseline
  - `advanced` uses Multi-Query plus hybrid retrieval/reranking
  - main `graph` uses Multi-Query plus provenance-gated locator-to-chunk evidence
  - every Graph ablation disables HyDE while retaining its named graph intervention
  - `graph_raw_current` is the only intentional raw-legacy graph control
- Evaluation `agentic` uses `agentic_eval_v8_multiquery_locator_recursive_baseline`:
  - no Agentic route invokes HyDE
  - compare/visual/generic-graph routes use Multi-Query
  - CRAG corrective retrieval uses Multi-Query with original-question fallback
  - selected graph routes resolve graph evidence to source chunks
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
- Metric execution is non-blocking per metric; one metric failure produces a failed attempt and may yield `completed_with_errors` without erasing successful metrics.
- A provider response can arrive after a process interruption but before its checkpoint transaction commits. This unavoidable billing/checkpoint window is recorded as an attempt outcome and is safe to replay; retry policy never assumes the provider did not charge.
- Metrics API now returns:
  - `available_metrics`
  - per-row `metric_values`
  - `summary_by_mode`
  - `summary_by_category`
  - `summary_by_focus`

## Version-2 Evaluation Research Accounting

- Version-2 accounting is durable and scope-based. Each execution attempt owns
  one `execution_run` scope; each compatible RAGAS batch owns one
  campaign-level `ragas_batch` scope. Every observed provider call is stored as
  a normalized usage event with a non-overlapping token breakdown and a phase.
- Every new execution target durably records the mode frozen in its work-item
  input. This attributes successful, failed, and retried attempts to the same
  mode without relying on a promoted result. Older targets with `mode=null`
  are used only when an existing result ID or official attempt ID determines
  exactly one mode; otherwise campaign and per-mode accounting become partial,
  the modes are non-comparable, and no mode or cost is guessed. Affected modes
  retain their known benchmark cost but expose `operational_usd=null` with
  partial pricing; the campaign can still report complete operational pricing
  because it includes every execution event without assigning it to a mode.
- Each fresh `ragas_batch` accounting scope starts with `retry_count=0` and
  durably increments once for every scheduled evaluator retry (not for the
  terminal attempt). Historical scopes migrated before this counter existed
  retain `retry_count=null`; research-summary aggregation then returns a null
  overhead retry count and the `unknown_ragas_retry_count` warning rather than
  guessing from batch keys.
- Historical campaigns are deliberately not backfilled. A completed legacy
  result without a version-2 official execution scope remains readable, but
  reports `token_accounting_status="incomplete_legacy"` and is not comparable.
- Token category and phase values are calculated only from events with
  `usage_status="measured"` and `reconciliation_status="balanced"`. When no
  such event exists, every token category and total is `null`, `by_phase` is
  empty, accounting is partial, and phase attribution is unavailable. Mixed
  measured/missing usage retains only the known measured subtotals while the
  total remains `null` and accounting remains partial.
- Costs have separate meanings:
  - **benchmark inference cost** includes only successful official execution
    scopes for the displayed results;
  - **operational execution cost** includes all execution attempts, including
    retries and failed attempts;
  - **RAGAS overhead** is a campaign-level evaluator cost. It is never assigned
    to individual runs or included in either execution-cost total.
- Quality labels in the research summary are official only when they come from
  compatible durable `ragas_scores` rows matching the result's official source
  attempt. Missing, failed, evaluating, or incompatible work remains nullable;
  it is never converted to a score of zero.
- RAGAS keeps two distinct signatures. `evaluation_signature` is a per-result
  currentness key that includes result/content identity and drives durable work
  idempotency. `compatibility_signature` identifies only the evaluator policy
  (model/configuration, metric name/version, context-policy version, and policy
  knobs) and is used to form comparable research cohorts. Cohort compatibility
  is checked independently per metric so valid metric-specific signatures do
  not conflict. Legacy scores without the compatibility signature fall back
  strictly to identical raw evaluation signatures.
- For every metric, analytics selects one deterministic campaign-level
  evaluator identity from official current scores: the identity covering the
  most result IDs wins, then lexical identity breaks ties. Campaign and mode
  aggregates use that same cohort; a mode with current noncanonical scores is
  marked `evaluator_metadata_mismatch` and is not comparable.
- Quality sample counts are per-result classifications. A result is exactly one
  of valid, evaluating, failed, or missing for each metric. A terminal RAGAS
  target failure takes precedence over a score only while that target is not
  official; a durably official target retains its current compatible score if
  a later shared scope fails. Failed non-official targets contribute to
  `failed_samples` and are never included in `missing_samples`.
- Latency `p50` and `p95` use the nearest-rank method: sort observed successful
  run latencies, then select `ceil(percentile * sample_count)`. Values are
  observed samples, not interpolated estimates.
- The response exposes independent quality, token-accounting, pricing, and
  phase-attribution statuses. `partial`, `unknown`, and nullable values are
  expected states for incomplete work; a mode is comparable only when all
  token-accounting, quality, evaluator-identity, and schema checks pass.
  Monetary pricing is optional and never blocks token-only comparability.
- Pricing is read from the audited snapshot configured by
  `EVALUATION_PRICE_SNAPSHOT_PATH`. If a model or its price is unavailable,
  price totals remain `null` and the pricing status explains why.
- Pricing is optional for token-focused evaluations: leaving
  `EVALUATION_PRICE_SNAPSHOT_PATH` unset intentionally keeps monetary totals
  unknown and does not invalidate token accounting or mode comparison.
- The default RAGAS evaluator is the stable `gemini-3.1-flash-lite`. The old
  `gemini-3.1-flash-lite-preview` model was shut down and should not be selected
  in deployment configuration. Existing score rows retain their exact
  historical evaluator signature and remain readable only when durable
  work/job metadata can verify the persisted model/configuration; new work
  uses the stable model.
- `GET /api/evaluation/campaigns/{campaign_id}/research-summary` is an
  authenticated, campaign-owned dashboard contract. It returns strict
  campaign/mode totals, official RAGAS observations, observed latency,
  execution accounting, RAGAS overhead, and explicit warning/comparability
  states without deriving proxy metrics from legacy overview data.

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

### Condition-Level Ablation Analytics

- `GET /api/evaluation/campaigns/{campaign_id}/ablation` adds `summaries.condition_comparison` only when at least two persisted condition IDs are present. The projection groups completed and failed executions by immutable `condition_id`, label, and `ablation_flags` from each run snapshot; it never reads current environment flags.
- Each condition row exposes `execution_count`, `completed_count`, `failed_count`, finite RAGAS quality summaries for `answer_correctness`, `faithfulness`, and `answer_relevancy`, `mean_tokens`, and `mean_latency_ms`. Missing or non-finite values remain nullable and are counted as missing.
- The paired section matches `(question_id, repeat_number)` for the selected baseline/guided condition order. It reports completed-pair and metric-pair counts, guided-minus-baseline deltas, and numeric exclusion reasons such as `run_not_completed`, `missing_opposite_condition`, and `missing_metric`.
- `availability.ragas_rows_found` and `availability.warning` make an unscored campaign explicit. Failed, unpaired, or non-finite observations are never converted to zero.

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
    - `include_run_observability` (default `false`)
    - `include_raw_trace_payloads`
    - `include_prompt_previews`
    - `include_full_prompts`
    - `include_answers`
    - `include_retrieved_excerpts`
    - `format` (`json` only)
- Export Schema v2 behavior:
  - `EvaluationExportService.export_campaign` is the only composer; the route no longer has a legacy response path
  - required panel sections are composed from their canonical services, match the active HTTP panel objects, and fail all-or-error
  - summary exports do not load detailed observability; full exports use one campaign snapshot and require exact result/run-ID equality
  - a failed run that terminated before creating an observability container remains in a full export as a typed empty projection with `availability=partial` and reason `run_failed_before_observability`; a completed run missing its container remains a hard export error
  - full export and selected-run detail share the interactive safe projector before export-only content flags are applied, so provenance, availability, empty safe payloads, and structural identifiers cannot drift
  - campaign observability and accounting snapshot load counts are constant for one or fifty runs; no per-run observability/accounting loaders are used
  - every `runs[]` row contains the fixed result projection, official finite-only RAGAS metrics, accounting, latency, and a fixed observability envelope
  - trace event `payload` is blank unless `include_raw_trace_payloads=true`; included payloads recursively exclude provider bodies/errors/stack traces, redact credentials, and apply answer/excerpt policy to equivalent nested keys
  - every exported free-text field is credential-redacted and bounded; typed identity and locator fields remain present
  - named analytics/diagnostics and agentic-v9 comparison use strict export-owned shapes; unrestricted JSON exists only at the sanitized trace payload boundary
  - prompt previews follow `include_prompt_previews`; full prompts require `include_full_prompts=true` and execution-time capture
  - `include_answers=false` clears result/preview/claim/fact/final-claim content, including the human-evaluation queue preview
  - `include_retrieved_excerpts=false` clears excerpts and evidence statements while retaining identity and locator fields
  - provider bodies, non-trace arbitrary payloads, unrestricted errors, stack traces, credentials, and authorization headers are permanently excluded
- The HTTP parity release test uses authenticated temporary durable storage and compares complete serialized panel/export objects for two modes, official RAGAS/accounting, v9 evidence, routing, ablation, human evaluation, errors, and stage warnings. It also preserves the release endpoint's nested `not_applicable` shape when no benchmark is configured.
- Sanitized errors are stored and exported instead of raw provider dumps. Multiline stack traces and obvious secrets are redacted.

### Durable Evaluation Recovery

- Dataset work units are question/mode/run/condition identities. RAGAS work units are result/metric identities with an exact per-result evaluation signature and a separate compatible batch key.
- Creating new RAGAS work commits the pending job items and the owned campaign's `evaluating` transition together before workers are notified. `evaluation_total_units` counts distinct target results, not metric-item rows.
- Attempts are append-only. A failed, cancelled, or interrupted attempt never replaces the official result; only a compatible successful attempt is promoted atomically.
- Restart recovery reclaims interrupted work from SQLite. Default worker limits are four execution claims and two RAGAS provider calls; RAGAS batches are capped at four.
- Missing or non-finite metrics remain missing and are excluded from means, deltas, ECR, and exports. Warning payloads expose missing and failed-work counts.
- Rerun endpoints are `POST /api/evaluation/campaigns/{campaign_id}/reruns`, `GET /api/evaluation/campaigns/{campaign_id}/jobs`, `GET /api/evaluation/jobs/{job_id}`, `GET /api/evaluation/jobs/{job_id}/items`, and `POST /api/evaluation/jobs/{job_id}/cancel`.
