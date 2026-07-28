# Agentic v9 Observability and Capability Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to execute this plan task by task.

**Goal:** Make the current Agentic v9 run explainable and token-accountable without changing its successful retrieval or answer behavior, then prevent unavailable Graph/Visual capabilities from falsely downgrading otherwise sufficient text answers.

**Architecture:** Wave A is behavior-neutral: it carries existing rerank, source-identity, and provider-attempt data from runtime through durable observability and export. Wave B changes only the final capability-status decision: a capability that has no eligible input is reported as unavailable but does not invalidate sufficient text evidence; a capability that was actually attempted and failed still remains fail-closed. Corrective retrieval and claim-extraction redesign are explicitly deferred.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, SQLite/aiosqlite, pytest, Ruff.

## Global Constraints

- Preserve the current Agentic v9 retrieval behavior: Hybrid candidates `8` → rerank candidates `8` → selected chunks `4` per retrieval task.
- Preserve reranker fail-soft behavior and the current final answer prompts.
- Do not add LLM calls, per-slot calls, retrieval rounds, schema migrations, or frontend work.
- `expected_sources` is evaluation metadata only. It must never become a runtime retrieval or authorization filter.
- Do not fabricate rerank scores, ranks, source matches, token phases, or capability success.
- Keep official runtime token totals authoritative. Phase attribution is complete only when measured provider attempts reconcile exactly with that total.
- Execute Wave A first and checkpoint it before any Wave B behavior change.
- After Wave B, run only the five-question smoke set first: Q2, Q9, Q14, Q15, Q16; batch size `1`; same model preset and corpus. Do not run the full 16-question evaluation until the smoke gate passes.

---

## Wave A — Behavior-Neutral Observability

### Task 1: Preserve Complete Rerank Diagnostics in the v9 Runtime Trace

**Files:**

- Modify: `evaluation/agentic_v9_campaign_runtime.py`
- Modify: `evaluation/retrieval_profiles.py`
- Test: `tests/test_agentic_v9_campaign_runtime.py`
- Test: `tests/test_evaluation_retrieval_profiles.py`

**Step 1: Write failing runtime projection tests**

Extend the existing rerank tests to assert that each `agentic_v9.retrieval_diagnostics` entry records:

```python
{
    "task_id": "task:...",
    "status": "executed",  # or "fallback"
    "fallback_reason": None,
    "candidate_count": 8,
    "selected_count": 4,
    "selected": [
        {
            "doc_id": "uuid",
            "chunk_id": "chunk-id",
            "content_hash": "...",
            "pre_rerank_rank": 7,
            "post_rerank_rank": 1,
            "rerank_score": 0.91,
        }
    ],
}
```

Add a fallback case that proves `rerank_score` stays `None`, `status` is `fallback`, and `fallback_reason` is retained.

**Step 2: Run the focused tests and confirm failure**

Run:

```powershell
pytest -q tests/test_agentic_v9_campaign_runtime.py tests/test_evaluation_retrieval_profiles.py
```

Expected: failure because the current projection omits `doc_id` and `content_hash`, and the execution profile still says reranking is off.

**Step 3: Implement the minimal projection**

Update `_retrieval_diagnostic_projection()` to copy stable identity and ranking fields from each selected document:

- `doc_id`, using the same document-ID helper as the retrieval runtime.
- `chunk_id`.
- `content_hash`, computed from the exact selected `page_content`.
- `pre_rerank_rank`.
- `post_rerank_rank`.
- `rerank_score`.

Keep `status`, `fallback_reason`, `candidate_count`, and `selected_count` at task level. Do not alter `_retrieve_documents()`, selected document order, or selected document count.

Update the value of `AGENTIC_V9_OPEN_CORPUS_PROFILE` to describe the actual behavior, for example:

```text
agentic_eval_v9_open_corpus_hybrid8_rerank8_top4_failsoft
```

Keep the Python constant name unchanged to avoid unnecessary call-site edits.

**Step 4: Run the focused tests**

Run:

```powershell
pytest -q tests/test_agentic_v9_campaign_runtime.py tests/test_evaluation_retrieval_profiles.py
ruff check evaluation/agentic_v9_campaign_runtime.py evaluation/retrieval_profiles.py tests/test_agentic_v9_campaign_runtime.py tests/test_evaluation_retrieval_profiles.py
```

Expected: pass.

**Step 5: Commit**

```powershell
git add evaluation/agentic_v9_campaign_runtime.py evaluation/retrieval_profiles.py tests/test_agentic_v9_campaign_runtime.py tests/test_evaluation_retrieval_profiles.py
git commit -m "fix(evaluation): expose v9 rerank diagnostics"
```

### Task 2: Join Runtime Rerank Diagnostics to Durable Retrieval Chunks

**Files:**

- Modify: `evaluation/campaign_engine.py`
- Test: `tests/test_campaign_engine.py`
- Test: `tests/test_evaluation_export_redaction.py`

**Step 1: Write failing persistence tests**

Add a v9 campaign-result fixture with two retrieval tasks and duplicate-looking excerpts from different documents. Assert that durable `EvaluationRetrievalChunk` rows contain:

- Actual `rank_before_rerank`.
- Actual `rank_after_rerank`.
- Actual `rerank_score` for executed reranking.
- `payload.reranker_status`.
- `payload.reranker_fallback_reason`.
- `payload.retrieval_task_id`.
- `payload.rerank_candidate_count`.
- `payload.rerank_selected_count`.

Add an unmatched-result case. Assert that ranks and score remain `None`, and the payload says `reranker_status: "not_instrumented"` rather than inventing result-level ranks.

Add an export assertion proving the same safe diagnostic fields survive redacted export while excerpts still obey the existing redaction option.

**Step 2: Run the focused tests and confirm failure**

Run:

```powershell
pytest -q tests/test_campaign_engine.py tests/test_evaluation_export_redaction.py
```

Expected: failure because `_record_unit_research_observability()` currently writes `index` into both rank fields and omits the runtime rerank metadata.

**Step 3: Implement a deterministic diagnostic join**

In `evaluation/campaign_engine.py`, add a private helper that flattens:

```python
trace_payload["agentic_v9"]["retrieval_diagnostics"]
```

into stable diagnostic rows keyed by:

```python
(doc_id, content_hash)
```

Store multiple rows per key in order so identical text does not overwrite another task's row. Join each final context to at most one diagnostic row. Use `chunk_id` as a secondary check when it is available.

When a join succeeds, populate the durable fields and payload listed above. When it does not succeed, leave the rerank fields `None` and mark the instrumentation as unavailable. Do not change `used_in_context`, `used_in_answer`, context order, or retrieval event hit-rate logic.

**Step 4: Run the focused tests**

Run:

```powershell
pytest -q tests/test_campaign_engine.py tests/test_evaluation_export_redaction.py
ruff check evaluation/campaign_engine.py tests/test_campaign_engine.py tests/test_evaluation_export_redaction.py
```

Expected: pass.

**Step 5: Commit**

```powershell
git add evaluation/campaign_engine.py tests/test_campaign_engine.py tests/test_evaluation_export_redaction.py
git commit -m "fix(evaluation): persist v9 rerank provenance"
```

### Task 3: Resolve Evaluation Source Filenames to Canonical Document IDs

**Files:**

- Modify: `evaluation/campaign_engine.py`
- Test: `tests/test_campaign_engine.py`
- Test: `tests/test_evaluation_export_redaction.py`

**Step 1: Write failing source-identity tests**

Cover these cases:

1. Test-case source filename resolves to the UUID found on a retrieved chunk:

```python
expected_evidence_match is True
payload["expected_evidence_match_status"] == "matched"
```

2. Resolution succeeds but the retrieved UUID differs:

```python
expected_evidence_match is False
payload["expected_evidence_match_status"] == "not_matched"
```

3. A filename is missing, ambiguous, or the resolver raises:

```python
expected_evidence_match is False
payload["expected_evidence_match_status"] == "identity_unresolved"
```

The third case must not fail the run or alter runtime contexts.

**Step 2: Run the focused tests and confirm failure**

Run:

```powershell
pytest -q tests/test_campaign_engine.py tests/test_evaluation_export_redaction.py
```

Expected: failure because expected filenames are currently compared directly with retrieved UUIDs.

**Step 3: Implement evaluation-only source resolution**

Reuse `data_base.repository.resolve_document_references()` inside the observability path. Resolve `execution.payload.expected_sources` or the test-case `source_docs` for the current `user_id`.

Return both:

- A set of canonical authorized document IDs for comparison.
- A resolution status: `resolved` or `identity_unresolved`.

Use that set only to compute evaluation observability fields. Never pass it back into `AgenticV9CampaignRuntime`, source scope, retrieval filters, or evidence admission.

Persist the tri-state result in retrieval-chunk payloads. Keep the existing boolean field conservative for compatibility.

**Step 4: Run the focused tests**

Run:

```powershell
pytest -q tests/test_campaign_engine.py tests/test_evaluation_export_redaction.py
ruff check evaluation/campaign_engine.py tests/test_campaign_engine.py tests/test_evaluation_export_redaction.py
```

Expected: pass.

**Step 5: Commit**

```powershell
git add evaluation/campaign_engine.py tests/test_campaign_engine.py tests/test_evaluation_export_redaction.py
git commit -m "fix(evaluation): resolve expected source identity"
```

### Task 4: Make v9 Provider-Phase Accounting Provable

**Files:**

- Modify: `evaluation/campaign_engine.py`
- Modify only if the integration test proves the observer payload is incomplete: `data_base/agentic_v9/budgeted_llm.py`
- Test: `tests/test_campaign_engine.py`
- Test: `tests/test_agentic_v9_budgeted_llm.py`
- Test: `tests/test_evaluation_analytics_context.py`
- Test: `tests/test_evaluation_research_analytics.py`

**Step 1: Write a failing end-to-end accounting test**

Execute a minimal v9 run through `CampaignEngine._execute_unit()` and persistence with a fake provider that reports measured usage. Assert:

- Every provider attempt has `phase`, `reservation_id`, and `provider_attempt`.
- Every measured attempt payload has `official_total_tokens`.
- The sum of measured provider-attempt totals equals `campaign_results.total_tokens`.
- `reconcile_official_tokens()` returns `status == "complete"`.
- `by_phase` contains the actual phase names and no `unknown`.

Add a mismatch case. Assert the official runtime total remains available, but phase attribution is `partial` with `provider_runtime_total_mismatch`; no synthetic phase allocation is created.

**Step 2: Run the focused tests and identify the broken boundary**

Run:

```powershell
pytest -q tests/test_campaign_engine.py tests/test_agentic_v9_budgeted_llm.py tests/test_evaluation_analytics_context.py tests/test_evaluation_research_analytics.py
```

Expected: the integration assertion fails at the exact boundary responsible for the raw export's `By phase: N/A`.

**Step 3: Apply the smallest fix at the proven boundary**

Preferred path:

- Keep `BudgetedLlmInvoker` as the sole source of provider-attempt rows.
- Ensure the campaign's `llm_call_observer_scope(recorder)` reaches every v9 `BudgetedLlmInvoker`.
- Preserve the invoker's real `phase`, `reservation_id`, `provider_attempt`, measured components, and `official_total_tokens`.
- Keep `_record_unit_llm_usage()` only as the legacy aggregate row for modes without provider-attempt instrumentation.
- For a v9 payload with `budget_reservations`, do not create an additional `campaign_generation` fallback row that can be mistaken for phase evidence.

If the test proves `BudgetedLlmInvoker` receives usage but drops a required normalized field, modify only its observation payload; do not estimate or distribute tokens across phases.

If any attempt reports estimated/missing usage or totals do not reconcile, retain the official aggregate total and mark phase attribution partial with the exact reason.

**Step 4: Run accounting tests**

Run:

```powershell
pytest -q tests/test_campaign_engine.py tests/test_agentic_v9_budgeted_llm.py tests/test_evaluation_analytics_context.py tests/test_evaluation_research_analytics.py
ruff check evaluation/campaign_engine.py data_base/agentic_v9/budgeted_llm.py tests/test_campaign_engine.py tests/test_agentic_v9_budgeted_llm.py tests/test_evaluation_analytics_context.py tests/test_evaluation_research_analytics.py
```

Expected: complete for fully measured/reconciled usage; partial with an explicit reason for every fail-closed case.

**Step 5: Commit**

```powershell
git add evaluation/campaign_engine.py data_base/agentic_v9/budgeted_llm.py tests/test_campaign_engine.py tests/test_agentic_v9_budgeted_llm.py tests/test_evaluation_analytics_context.py tests/test_evaluation_research_analytics.py
git commit -m "fix(evaluation): reconcile v9 provider phases"
```

### Task 5: Wave A Regression Checkpoint

**Files:**

- No production changes expected.

**Step 1: Run the Wave A suite**

Run:

```powershell
pytest -q tests/test_agentic_v9_campaign_runtime.py tests/test_evaluation_retrieval_profiles.py tests/test_campaign_engine.py tests/test_agentic_v9_budgeted_llm.py tests/test_evaluation_analytics_context.py tests/test_evaluation_research_analytics.py tests/test_evaluation_export_redaction.py tests/test_agentic_v9_smoke_runner.py
ruff check evaluation data_base/agentic_v9 tests/test_agentic_v9_campaign_runtime.py tests/test_evaluation_retrieval_profiles.py tests/test_campaign_engine.py tests/test_agentic_v9_budgeted_llm.py tests/test_evaluation_analytics_context.py tests/test_evaluation_research_analytics.py tests/test_evaluation_export_redaction.py
```

Expected: pass.

**Step 2: Inspect the diff**

Run:

```powershell
git diff HEAD~4 -- evaluation data_base/agentic_v9 tests
git status --short
```

Confirm that no retrieval count, prompt, answer parser, source-authorization input, or final-response policy changed in Wave A.

---

## Wave B — Capability Calibration

### Task 6: Distinguish Capability Unavailable from Capability Failure

**Files:**

- Modify: `evaluation/agentic_v9_campaign_runtime.py`
- Modify: `evaluation/campaign_engine.py`
- Test: `tests/test_agentic_v9_campaign_runtime.py`
- Test: `tests/test_campaign_engine.py`

**Step 1: Write failing response-status tests**

Add these cases:

1. Sufficient text evidence + visual required by route + `no_eligible_visual_evidence`:
   - Final response remains `complete`.
   - Visual execution is `capability_unavailable`.
   - Text contexts and evidence packets remain intact.

2. Sufficient text evidence + graph required by route + `no_eligible_graph_source_evidence`:
   - Final response remains `complete`.
   - Graph execution is `capability_unavailable`.

3. Provider/extractor/graph-stage timeout or exception after an eligible input was selected:
   - Final response becomes `qualified_partial`.
   - Stage remains a real partial/failure warning.

4. Insufficient text evidence:
   - Existing sufficiency result remains authoritative.
   - Capability calibration cannot upgrade it to `complete`.

5. Successfully executed Graph/Visual stage:
   - Existing complete behavior is unchanged.

Add campaign-observability assertions that unavailable capability stages are recorded as warnings/capability gaps, not sanitized execution errors, while true attempted failures remain partial.

**Step 2: Run the focused tests and confirm failure**

Run:

```powershell
pytest -q tests/test_agentic_v9_campaign_runtime.py tests/test_campaign_engine.py
```

Expected: failure because the current final-status block downgrades every required-but-not-executed capability.

**Step 3: Implement explicit capability outcome classification**

Add a small private classifier in `evaluation/agentic_v9_campaign_runtime.py` with outcomes:

```text
not_requested
executed
capability_unavailable
execution_failed
```

Classify these as unavailable:

- `no_eligible_graph_source_evidence`
- `no_eligible_visual_evidence`
- A required stage that has no eligible input and was not actually invoked

Classify timeouts, provider exceptions, extraction exceptions, and graph runtime errors after an eligible attempt as execution failures.

Change only the final response-status guard:

- `execution_failed` may downgrade `complete` to `qualified_partial`.
- `capability_unavailable` may not downgrade an otherwise complete, sufficient text answer.
- Existing non-complete sufficiency status stays unchanged.

Project the explicit outcome in `graph_execution` and `visual_execution`. In `evaluation/campaign_engine.py`, record unavailable outcomes as capability warnings without attaching a trace error; preserve partial trace status for actual execution failures.

**Step 4: Run focused tests**

Run:

```powershell
pytest -q tests/test_agentic_v9_campaign_runtime.py tests/test_campaign_engine.py
ruff check evaluation/agentic_v9_campaign_runtime.py evaluation/campaign_engine.py tests/test_agentic_v9_campaign_runtime.py tests/test_campaign_engine.py
```

Expected: pass.

**Step 5: Commit**

```powershell
git add evaluation/agentic_v9_campaign_runtime.py evaluation/campaign_engine.py tests/test_agentic_v9_campaign_runtime.py tests/test_campaign_engine.py
git commit -m "fix(agentic-v9): fail soft on unavailable capabilities"
```

### Task 7: Local Verification and Remote Five-Question Smoke Gate

**Files:**

- No production changes expected.

**Step 1: Run local verification**

Run:

```powershell
pytest -q tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_budgeted_llm.py tests/test_evaluation_retrieval_profiles.py tests/test_campaign_engine.py tests/test_evaluation_analytics_context.py tests/test_evaluation_research_analytics.py tests/test_evaluation_export_redaction.py tests/test_agentic_v9_smoke_runner.py
ruff check evaluation data_base/agentic_v9
git status --short
```

Expected: all pass; only intentional commits and unrelated pre-existing user files remain.

**Step 2: Deploy and run the bounded smoke**

On the deployment server, use:

```text
Questions: Q2, Q9, Q14, Q15, Q16
Modes: agentic-v9 only
Repeat: 1
Batch size: 1
Model preset: identical to the latest successful 16-question run
Corpus/index: unchanged
```

**Step 3: Apply the smoke acceptance gate**

Pass only if:

- `5 / 5` runs complete with no execution failure.
- Each retrieval task reports candidate/selected counts and an honest rerank status.
- Executed reranking has real pre/post ranks and score; fallback has a reason and no fake score.
- Expected-source status is `matched`, `not_matched`, or `identity_unresolved`, never a silent filename/UUID false negative.
- Official total tokens remain available.
- Phase attribution is complete when all attempts are measured and reconciled; otherwise it is partial with a concrete reason.
- Missing eligible Graph/Visual assets do not erase text contexts or by themselves downgrade a sufficient answer.
- Actual Graph/Visual execution errors still downgrade and remain traceable.
- Retrieval/answer content for non-capability questions is behaviorally unchanged apart from ordinary provider nondeterminism.

If any gate fails, stop and fix only the failed boundary before running another smoke.

**Step 4: Full evaluation remains user-gated**

Only after the five-question smoke passes should the user run the full 16-question paired evaluation. Compare:

- Correctness.
- Faithfulness.
- Relevancy.
- Total and per-phase tokens.
- Mean/P95 latency.
- `complete` versus `qualified_partial`.
- Reranker executed/fallback rate.
- Graph/Visual unavailable versus execution-failed rate.

Do not add corrective retrieval in this implementation cycle.
