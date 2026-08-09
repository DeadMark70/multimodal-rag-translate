# Agentic v9 Durable Observability Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist measured v9 provider calls and rerank provenance when a campaign runs through the durable execution worker.

**Architecture:** The worker will create the same run-scoped `EvaluationRunRecorder` used by the in-process campaign engine and bind it for the entire provider invocation. Rerank persistence will use the already-propagated `(doc_id, source_chunk_id)` identity before falling back to an unambiguous raw-content hash, so context normalization cannot discard valid telemetry.

**Tech Stack:** Python 3.11, asyncio, pytest/pytest-asyncio, SQLite-backed evaluation observability.

## Global Constraints

- Do not alter Agentic v9 retrieval, reranking, authorization, routing, or answer-generation behavior.
- Do not synthesize missing telemetry; unavailable or ambiguous identities remain `not_instrumented`.
- Keep v9 official token accounting provider-event based; do not restore the legacy aggregate fallback row.
- Preserve existing fail-soft observability behavior and expose persistence problems as partial status.

---

### Task 1: Bind the durable worker to the v9 LLM observer

**Files:**
- Modify: `evaluation/execution_worker.py:12-16, 99-151`
- Modify: `tests/test_evaluation_execution_worker.py`

**Interfaces:**
- Consumes: `EvaluationRunRecorder`, `llm_call_observer_scope`, `current_llm_call_observer()`.
- Produces: one provider-attempt record per measured v9 provider invocation, with the existing phase/reservation metadata.

- [x] **Step 1: Write the failing regression test**

Add a durable v9 worker test whose runner obtains `current_llm_call_observer()`, invokes `invoke_budgeted_llm(...)` with `phase="final_answer"`, and returns a v9 payload containing a budget reservation. After `worker.execute(claim)`, assert the promoted result has exactly one LLM call with phase `final_answer` and the measured provider total.

- [x] **Step 2: Run the focused test to verify it fails**

Run: `pytest tests/test_evaluation_execution_worker.py::test_v9_durable_worker_persists_measured_provider_phase -q`

Expected: FAIL because the durable worker has no run-scoped LLM observer.

- [x] **Step 3: Write the minimal implementation**

Create an `EvaluationRunRecorder` after the durable `run_id` is allocated. Populate its provider/model/prompt-capture metadata from the immutable model snapshot. Nest `llm_call_observer_scope(recorder)` inside `llm_accounting_scope(scope.context)` around the runner call, and copy any recorder partial state into `ExecutedCampaignUnit`.

- [x] **Step 4: Run the focused test to verify it passes**

Run: `pytest tests/test_evaluation_execution_worker.py::test_v9_durable_worker_persists_measured_provider_phase -q`

Expected: PASS with one measured phase record and no legacy `campaign_generation` row.

- [ ] **Step 5: Commit**

```bash
git add evaluation/execution_worker.py tests/test_evaluation_execution_worker.py
git commit -m "fix(evaluation): observe durable v9 provider calls"
```

### Task 2: Join rerank diagnostics by source chunk identity

**Files:**
- Modify: `evaluation/campaign_engine.py:525-606`
- Modify: `tests/test_campaign_engine.py`

**Interfaces:**
- Consumes: v9 diagnostic rows containing `doc_id`, `chunk_id`, raw `content_hash`, ranks, and score; result contexts containing `source_chunk_ids`.
- Produces: per-retrieval-chunk rerank metadata when `(doc_id, source_chunk_id)` is unique, otherwise only an unambiguous raw content-hash fallback.

- [x] **Step 1: Write the failing regression test**

Add a campaign result test with a diagnostic row whose raw content is `"A  B"` and a result context normalized to `"A B"`, while both share `doc_id="doc-a"` and `source_chunk_id="chunk-a"`. Assert ranks, score, and status are persisted as `executed`.

- [x] **Step 2: Run the focused test to verify it fails**

Run: `pytest tests/test_campaign_engine.py::test_campaign_result_joins_v9_rerank_diagnostics_by_source_chunk_after_context_normalization -q`

Expected: FAIL because the existing content-hash-first lookup misses after normalization.

- [x] **Step 3: Write the minimal implementation**

Index each selected diagnostic in both a raw `(doc_id, content_hash)` bucket and a `(doc_id, chunk_id)` bucket. Consume the source-chunk bucket first when a source chunk id is present, consuming the same row once; only use the raw hash bucket when no chunk id is available and the hash match is unambiguous.

- [x] **Step 4: Run the focused test to verify it passes**

Run: `pytest tests/test_campaign_engine.py::test_campaign_result_joins_v9_rerank_diagnostics_by_source_chunk_after_context_normalization -q`

Expected: PASS with `reranker_status="executed"` and the original diagnostic rank/score values.

- [ ] **Step 5: Commit**

```bash
git add evaluation/campaign_engine.py tests/test_campaign_engine.py
git commit -m "fix(evaluation): preserve v9 rerank provenance"
```

### Task 3: Verify the combined repair

**Files:**
- Verify only: `evaluation/execution_worker.py`, `evaluation/campaign_engine.py`, and their focused tests.

- [x] **Step 1: Run focused regression coverage**

Run: `pytest tests/test_evaluation_execution_worker.py tests/test_campaign_engine.py -q`

Expected: PASS.

- [x] **Step 2: Run static checks for modified modules**

Run: `ruff check evaluation/execution_worker.py evaluation/campaign_engine.py tests/test_evaluation_execution_worker.py tests/test_campaign_engine.py`

Expected: PASS.

- [x] **Step 3: Inspect the staged diff and working tree**

Run: `git diff --check; git status --short`

Expected: no whitespace errors and only intended tracked changes.

## Self-Review

- **Spec coverage:** Task 1 covers the empty durable-worker `llm_calls` and phase-accounting defect. Task 2 covers the all-`not_instrumented` rerank telemetry defect. Task 3 verifies both together.
- **Placeholder scan:** No implementation or testing step is deferred or described generically.
- **Type consistency:** Task 1 uses existing `EvaluationRunRecorder` and observer context interfaces. Task 2 retains the existing diagnostic row shape while adding only an index keyed by the existing `source_chunk_id` field.
