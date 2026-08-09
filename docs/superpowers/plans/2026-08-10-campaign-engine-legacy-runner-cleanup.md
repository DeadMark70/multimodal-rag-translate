# CampaignEngine Legacy Runner Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the obsolete process-local campaign runner, make `CampaignEngine` a durable facade, and preserve unique execution behavior through durable-worker tests.

**Architecture:** Extract the execution contracts and deterministic result/observability projections from `evaluation/campaign_engine.py` into one `evaluation/campaign_execution.py` owner. Keep job construction and pre-ledger recovery in `CampaignEngine`; keep claimed work execution in `DatasetExecutionWorker`; migrate direct `_run_campaign()` tests to durable claims before deleting the old path.

**Tech Stack:** Python 3.13, asyncio, FastAPI, Pydantic, aiosqlite/SQLite, pytest, pytest-asyncio, Ruff, GitHub Actions maintenance gates.

## Global Constraints

- Work in `D:\flutterserver\pdftopng`; do not modify the frontend repository.
- Preserve every public campaign API signature, request/response schema, route, and database schema.
- Preserve `CampaignEngine._prepare_legacy_recovery()` and pre-ledger database compatibility.
- Do not modify durable job-state semantics, retry classification, RAGAS behavior, provider execution, Docker, or dependency versions.
- Use one new production module only: `evaluation/campaign_execution.py`.
- Do not introduce a service layer, dependency-injection framework, or runtime class hierarchy.
- Do not split or simplify `CampaignEngine.create_rerun()`.
- Delete only `tests/test_agents_static_analysis.py`; leave `tests/test_evaluator_audit.py` and unrelated source-boundary tests unchanged.
- Preserve cancellation, accounting, safe error projection, observability partial-state, official-result promotion, and idempotency behavior.
- Keep implementation commits mechanical and independently reviewable.
- The implementation must be net-deleting and must not increase C901 finding count or score.

## File Structure

- Create `evaluation/campaign_execution.py`: execution dataclasses, runner protocol alias, snapshots, metrics, failure projection, trace enrichment, and campaign-specific observability persistence.
- Modify `evaluation/campaign_engine.py`: import execution contracts from their new owner; retain facade/job/recovery operations; remove process-local execution methods.
- Modify `evaluation/execution_worker.py`: import shared contracts and helpers from `campaign_execution.py`.
- Modify `quality/ruff-complexity-baseline.json`: move the unchanged `_record_unit_research_observability` owner key.
- Modify `tests/test_evaluation_execution_worker.py`: add a runtime ownership regression test.
- Create `tests/test_evaluation_execution_observability.py`: durable-claim coverage migrated from the old private runner.
- Modify `tests/test_campaign_engine.py`: retain facade/recovery tests and remove direct private-runner tests.
- Delete `tests/test_agents_static_analysis.py`: remove low-value source/shape heuristics.
- Modify `docs/design-docs/evaluation-runtime.md`: document the facade/execution/worker ownership after extraction.

---

### Task 1: Remove the obsolete agent static-analysis test

**Files:**
- Delete: `tests/test_agents_static_analysis.py`
- Verify only: `tests/test_evaluator.py`
- Verify only: `tests/test_evaluator_audit.py`
- Verify only: `tests/test_planner_integration.py`

**Interfaces:**
- Consumes: maintained imports and behavior tests for `TaskPlanner`, `RAGEvaluator`, and `ResultSynthesizer`.
- Produces: no replacement interface; Ruff and behavior tests remain authoritative.

- [ ] **Step 1: Record the current heuristic test behavior**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_agents_static_analysis.py tests/test_evaluator.py tests/test_evaluator_audit.py tests/test_planner_integration.py
```

Expected: exit 0. Confirm the static file contributes exactly five collected tests and does not expose a unique runtime behavior failure.

- [ ] **Step 2: Delete the static test file in full**

Delete `tests/test_agents_static_analysis.py`. Do not move or recreate these assertions:

```python
assert hasattr(module, "symbol")
assert "try:" in inspect.getsource(module)
assert "except" in inspect.getsource(module)
assert parameter.annotation != inspect.Parameter.empty
```

- [ ] **Step 3: Verify maintained behavior coverage**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_evaluator.py tests/test_evaluator_audit.py tests/test_planner_integration.py
.venv\Scripts\python.exe -m ruff check tests/test_evaluator.py tests/test_evaluator_audit.py tests/test_planner_integration.py --select E9,F63,F7,F82,F401,F841
$matches = rg -n "test_agents_static_analysis" tests scripts .github -g '*.py' -g '*.yml' -g '*.yaml'; $rgExit = $LASTEXITCODE
if ($rgExit -eq 0) { $matches; throw "obsolete static-test references remain" }
if ($rgExit -ne 1) { exit $rgExit }
Write-Output "No obsolete static-test references"
```

Expected: pytest and Ruff exit 0; `rg` returns no references.

- [ ] **Step 4: Commit the isolated deletion**

```powershell
git add -- tests/test_agents_static_analysis.py
git commit -m "test: remove obsolete agent static analysis"
```

Expected: the commit contains only the deleted test file.

---

### Task 2: Extract the campaign execution owner without changing behavior

**Files:**
- Create: `evaluation/campaign_execution.py`
- Modify: `evaluation/campaign_engine.py:1-1263`
- Modify: `evaluation/execution_worker.py:1-40`
- Modify: `quality/ruff-complexity-baseline.json`
- Modify: `tests/test_evaluation_execution_worker.py`
- Test: `tests/test_campaign_engine.py`
- Test: `tests/test_evaluation_execution_worker.py`
- Test: `tests/test_evaluation_export_redaction.py`

**Interfaces:**
- Consumes: `BenchmarkExecutionResult`, `TestCase`, trace schemas, evidence helpers, `EvaluationObservabilityRepository`, and document ownership lookup.
- Produces:
  - `CampaignRunner = Callable[..., Awaitable[BenchmarkExecutionResult]]`
  - `CampaignUnit`
  - `ExecutedCampaignUnit`
  - `_duration_ms(started_at, completed_at) -> float`
  - `_build_question_snapshot(test_case) -> dict[str, Any]`
  - `_build_system_version_snapshot(*, unit, payload) -> dict[str, Any]`
  - `_build_derived_metrics(*, unit, payload) -> dict[str, Any]`
  - `_final_answer_hash(answer) -> str | None`
  - `_enrich_agent_trace_payload(...) -> dict[str, Any]`
  - `_record_unit_root_span(...) -> str | None`
  - `_record_unit_llm_usage(...) -> None`
  - `_record_unit_research_observability(...) -> None`

- [ ] **Step 1: Add a failing runtime ownership test**

In `tests/test_evaluation_execution_worker.py`, add the module import and test:

```python
import evaluation.execution_worker as execution_worker


def test_execution_worker_uses_campaign_execution_contract_owner() -> None:
    assert execution_worker.CampaignUnit.__module__ == "evaluation.campaign_execution"
    assert (
        execution_worker.ExecutedCampaignUnit.__module__
        == "evaluation.campaign_execution"
    )
```

- [ ] **Step 2: Run the ownership test and verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_evaluation_execution_worker.py::test_execution_worker_uses_campaign_execution_contract_owner
```

Expected: FAIL because both dataclasses are still owned by `evaluation.campaign_engine`.

- [ ] **Step 3: Create `evaluation/campaign_execution.py` with the existing contracts and projections**

Move, without behavioral edits, these definitions from `campaign_engine.py`:

```text
CampaignRunner
_SAFE_FAILURE_CODES
CampaignUnit
ExecutedCampaignUnit
_utc_now
_duration_ms
_extract_total_tokens
_build_question_snapshot
_build_system_version_snapshot
_build_derived_metrics
_safe_failure_message
_failure_diagnostics
_final_answer_hash
_trace_payload
_trace_event_status
_claim_support_status
_claim_text
_enrich_agent_trace_payload
_record_unit_root_span
_record_unit_llm_usage
_v9_rerank_diagnostics_by_context
_safe_candidate_stage_projection
_consume_v9_rerank_diagnostic
_resolve_expected_source_document_ids
_graph_trace_outcome
_record_unit_research_observability
```

The module docstring must be:

```python
"""Campaign execution contracts and durable result projections."""
```

Copy only imports used by those definitions. Keep `logger = logging.getLogger(__name__)`. Do not move `_TERMINAL_STATUSES`, `_LEGACY_RAGAS_METRIC`, `_unit_key`, `_cancel_and_drain_tasks`, or any `CampaignEngine` method.

- [ ] **Step 4: Repoint production imports**

In `evaluation/execution_worker.py`, replace the import from `evaluation.campaign_engine` with:

```python
from evaluation.campaign_execution import (
    CampaignRunner,
    CampaignUnit,
    ExecutedCampaignUnit,
    _build_derived_metrics,
    _build_question_snapshot,
    _build_system_version_snapshot,
    _duration_ms,
    _enrich_agent_trace_payload,
    _final_answer_hash,
    _record_unit_llm_usage,
    _record_unit_research_observability,
    _record_unit_root_span,
)
```

In `evaluation/campaign_engine.py`, import the moved `CampaignRunner`,
`CampaignUnit`, `ExecutedCampaignUnit`, and every moved helper still used by the
temporarily retained process-local methods. Remove imports that became exclusive
to `campaign_execution.py` only after Ruff identifies them as unused.

- [ ] **Step 5: Move the complexity baseline owner key**

In `quality/ruff-complexity-baseline.json`, replace exactly:

```json
"evaluation/campaign_engine.py::_record_unit_research_observability": 19
```

with:

```json
"evaluation/campaign_execution.py::_record_unit_research_observability": 19
```

Do not change the score for `evaluation/campaign_engine.py::create_rerun`.

- [ ] **Step 6: Verify the extraction and ownership test GREEN**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_evaluation_execution_worker.py::test_execution_worker_uses_campaign_execution_contract_owner tests/test_evaluation_execution_worker.py tests/test_campaign_engine.py tests/test_evaluation_export_redaction.py
.venv\Scripts\python.exe -m ruff check evaluation/campaign_execution.py evaluation/campaign_engine.py evaluation/execution_worker.py tests/test_evaluation_execution_worker.py --select E9,F63,F7,F82,F401,F841
.venv\Scripts\python.exe scripts/check_complexity_ratchet.py --check
git diff --check
```

Expected: tests pass; the ownership test reports `evaluation.campaign_execution`; Ruff correctness and the ratchet pass with no regression.

- [ ] **Step 7: Verify that this commit has not removed the legacy runner yet**

Run:

```powershell
rg -n "def _run_campaign|_run_campaign\(" evaluation/campaign_engine.py tests/test_campaign_engine.py
git diff --stat
```

Expected: the definition and existing direct tests still exist. This task changes ownership only.

- [ ] **Step 8: Commit the mechanical ownership move**

```powershell
git add -- evaluation/campaign_execution.py evaluation/campaign_engine.py evaluation/execution_worker.py quality/ruff-complexity-baseline.json tests/test_evaluation_execution_worker.py
git commit -m "refactor(evaluation): extract campaign execution support"
```

Expected: one new production module, import changes, the baseline-key move, and one ownership test.

---

### Task 3: Transfer result and provider observability tests to durable claims

**Files:**
- Create: `tests/test_evaluation_execution_observability.py`
- Modify: `tests/test_campaign_engine.py`
- Test: `tests/test_evaluation_execution_worker.py`

**Interfaces:**
- Consumes: `DatasetExecutionWorker.execute(claim)`, `EvaluationJobStore`, `WorkItemSpec`, `ClaimedEvaluationWork`, campaign repositories, and the execution contracts from Task 2.
- Produces: focused durable-claim fixtures and five tests for result snapshots, provider attempts, trace projection, and actual route persistence.

- [ ] **Step 1: Create a focused durable observability fixture**

Start `tests/test_evaluation_execution_observability.py` with the same temporary SQLite isolation pattern used by `tests/test_evaluation_execution_worker.py`. Define this local fixture:

```python
@pytest_asyncio.fixture
async def store(monkeypatch: pytest.MonkeyPatch) -> EvaluationJobStore:
    database_path = (
        Path(os.environ["EVALUATION_TEST_TMPDIR"])
        / f"dataset-observability-{uuid4().hex}"
        / "worker.db"
    )
    database_path.parent.mkdir(parents=True)
    monkeypatch.setattr(evaluation_db, "EVALUATION_DB_PATH", database_path)
    await evaluation_db.force_init_db()
    async with evaluation_db.connect_db() as connection:
        now = "2026-08-10T00:00:00+00:00"
        config = json.dumps(
            {
                "test_case_ids": ["Q1"],
                "modes": ["agentic"],
                "model_config": {
                    "id": "cfg-1",
                    "name": "test",
                    "model_name": "test-model",
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": 1,
                    "max_input_tokens": 8192,
                    "max_output_tokens": 2048,
                    "thinking_mode": False,
                    "thinking_budget": 0,
                },
            }
        )
        await connection.execute(
            """
            INSERT INTO campaigns
              (id, user_id, name, status, config_json, created_at, updated_at)
            VALUES ('cmp-1', 'user-a', NULL, 'pending', ?, ?, ?)
            """,
            (config, now, now),
        )
        await connection.commit()
    try:
        yield EvaluationJobStore()
    finally:
        for path in (
            database_path,
            database_path.with_suffix(".db-shm"),
            database_path.with_suffix(".db-wal"),
        ):
            path.unlink(missing_ok=True)
        rmtree(database_path.parent, ignore_errors=True)
```

Add a local claim builder with the exact interface:

```python
async def _claim_execution(
    store: EvaluationJobStore,
    *,
    mode: str = "agentic",
    agentic_execution_version: str = "v9",
    source_docs: list[str] | None = None,
    model_config: dict[str, object] | None = None,
) -> ClaimedEvaluationWork:
    await store.create_job_with_items(
        user_id="user-a",
        campaign_id="cmp-1",
        job_type="initial",
        selection={},
        config_snapshot={},
        items=[
            WorkItemSpec(
                work_type=EvaluationWorkType.DATASET_EXECUTION,
                logical_key=f"execution:Q1:{mode}:1:none",
                input_snapshot={
                    "user_id": "user-a",
                    "campaign_id": "cmp-1",
                    "test_case": {
                        "id": "Q1",
                        "question": "What is the answer?",
                        "ground_truth": "42",
                        "source_docs": source_docs or [],
                        "requires_multi_doc_reasoning": False,
                    },
                    "mode": mode,
                    "run_number": 1,
                    "repeat_number": 1,
                    "condition_id": None,
                    "condition_label": None,
                    "ablation_flags": None,
                    "budget": None,
                    "model_config": model_config or {},
                    "agentic_execution_version": agentic_execution_version,
                    "shadow_evaluation_policy": None,
                },
            )
        ],
    )
    claims = await store.claim_ready_items(
        limit=1,
        now=datetime.now(timezone.utc),
    )
    assert len(claims) == 1
    return claims[0]
```

This helper creates one immutable dataset-execution claim and asserts that the store returns exactly one ready item.

- [ ] **Step 2: Transfer the snapshot and partial-observability tests**

Move these test bodies from `tests/test_campaign_engine.py` and keep their assertions:

```text
test_completed_run_persists_snapshots_and_root_observability_span
test_llm_observer_write_failure_preserves_answer_and_marks_run_partial
```

Replace `CampaignEngine._run_campaign(...)` with:

```python
claim = await _claim_execution(store, mode=mode)
worker = DatasetExecutionWorker(store=store, runner=runner)
await worker.execute(claim)
```

Read the promoted result, trace events, and derived metrics from the same repositories used by the existing assertions. The partial-observability test must still assert that the answer is official and the partial reason is stored.

- [ ] **Step 3: Transfer provider-attempt and trace tests**

Move these test bodies and drive them through `DatasetExecutionWorker.execute()`:

```text
test_v9_campaign_persists_default_visual_and_final_provider_attempts
test_agentic_trace_persists_routing_and_tool_observability
test_v9_actual_route_is_persisted_separately_from_retrospective_route
```

Preserve these assertions:

- visual and final-answer provider attempts retain their measured phases;
- routing/tool rows are materialized for the promoted run id; and
- the actual execution route is stored separately from retrospective analysis.

- [ ] **Step 4: Run the transferred tests before deleting their originals**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_evaluation_execution_observability.py -k "snapshots or observer_write_failure or visual_and_final or routing_and_tool or actual_route"
```

Expected: all five tests pass on the current durable worker. If a test fails, stop and identify whether the production worker lacks the behavior; do not add a compatibility call to `_run_campaign()`.

- [ ] **Step 5: Remove the five original private-runner tests**

Delete the five transferred test definitions from `tests/test_campaign_engine.py`. Remove only imports that become unused after those exact deletions.

- [ ] **Step 6: Verify the focused suites and direct-call count**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_evaluation_execution_observability.py tests/test_evaluation_execution_worker.py tests/test_campaign_engine.py
.venv\Scripts\python.exe -m ruff check tests/test_evaluation_execution_observability.py tests/test_evaluation_execution_worker.py tests/test_campaign_engine.py --select E9,F63,F7,F82,F401,F841
rg -n "_run_campaign\(" tests/test_campaign_engine.py
git diff --check
```

Expected: tests and Ruff pass; exactly twelve direct legacy test calls remain.

- [ ] **Step 7: Commit the first behavior-transfer batch**

```powershell
git add -- tests/test_evaluation_execution_observability.py tests/test_campaign_engine.py
git commit -m "test(evaluation): cover durable execution observability"
```

---

### Task 4: Transfer evidence and claim observability tests to durable claims

**Files:**
- Modify: `tests/test_evaluation_execution_observability.py`
- Modify: `tests/test_campaign_engine.py`
- Test: `tests/test_evaluation_execution_worker.py`

**Interfaces:**
- Consumes: the `store` fixture and `_claim_execution(...)` helper introduced by Task 3.
- Produces: durable-worker tests for evidence flow, rerank diagnostics, expected-source resolution, and claims.

- [ ] **Step 1: Transfer retrieval and rerank projection tests**

Move these tests from `tests/test_campaign_engine.py`:

```text
test_campaign_result_records_retrieval_context_and_evidence_flow
test_campaign_result_joins_v9_rerank_diagnostics_to_retrieval_chunks
```

Replace the private runner call with a claimed item and `DatasetExecutionWorker.execute()`. Preserve assertions for retrieval event, selected chunks, context pack, evidence flow, before/after rerank rank, score, candidate-stage projection, and durable source-chunk identity.

- [ ] **Step 2: Transfer expected-source identity tests**

Move these three tests:

```text
test_campaign_result_resolves_expected_source_filenames_for_chunk_statuses
test_campaign_result_marks_unresolved_expected_source_identity_without_mutating_result
test_campaign_result_marks_resolver_exception_as_unresolved_expected_source_identity
```

Keep their document-repository patches and assertions. Drive persistence through the claimed worker item. Both unresolved cases must leave the official answer unchanged and record a bounded unresolved diagnostic rather than raw exception text.

- [ ] **Step 3: Transfer claim-row and derived-metric coverage**

Move:

```text
test_campaign_result_persists_claim_rows_and_derived_claim_metrics
```

Use the durable claim path and preserve assertions for normalized claim rows, support status, source linkage, and derived supported/unsupported counts.

- [ ] **Step 4: Run the six transferred tests before deleting their originals**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_evaluation_execution_observability.py -k "retrieval_context or rerank_diagnostics or expected_source or unresolved_expected or resolver_exception or claim_rows"
```

Expected: all six tests pass without calling `CampaignEngine._run_campaign()`.

- [ ] **Step 5: Remove the six originals and clean test imports**

Delete only the six transferred test definitions from `tests/test_campaign_engine.py`. Use Ruff F401 to remove imports that are no longer used in that file; keep imports required by facade, recovery, retry, and API integration tests.

- [ ] **Step 6: Verify both test owners**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_evaluation_execution_observability.py tests/test_evaluation_execution_worker.py tests/test_campaign_engine.py tests/test_evaluation_export_redaction.py
.venv\Scripts\python.exe -m ruff check tests/test_evaluation_execution_observability.py tests/test_campaign_engine.py --select E9,F63,F7,F82,F401,F841
rg -n "_run_campaign\(" tests/test_campaign_engine.py
git diff --check
```

Expected: tests and Ruff pass; exactly six direct legacy calls remain, all in tests whose durable equivalents already exist.

- [ ] **Step 7: Commit the evidence-transfer batch**

```powershell
git add -- tests/test_evaluation_execution_observability.py tests/test_campaign_engine.py
git commit -m "test(evaluation): move evidence checks to durable worker"
```

---

### Task 5: Remove duplicate legacy tests and the process-local runner

**Files:**
- Modify: `tests/test_campaign_engine.py`
- Modify: `evaluation/campaign_engine.py`
- Verify: `evaluation/campaign_execution.py`
- Verify: `evaluation/execution_worker.py`

**Interfaces:**
- Consumes: durable-worker coverage in `tests/test_evaluation_execution_worker.py` and `tests/test_evaluation_execution_observability.py`.
- Produces: a `CampaignEngine` facade with no process-local execution entrypoint.

- [ ] **Step 1: Confirm the six remaining tests have named durable equivalents**

Use this mapping:

```text
test_v9_campaign_persists_measured_provider_phase_without_legacy_fallback
  -> test_v9_durable_worker_persists_measured_provider_phase

test_v9_campaign_keeps_runtime_total_when_provider_phase_total_mismatches
  -> test_execution_worker_promotes_ledger_total_not_payload_total

test_run_campaign_persists_failure_execution_profiles_directly
  -> test_failed_durable_execution_persists_current_evaluation_profile
     and test_failed_durable_execution_prefers_captured_trace_profile

test_run_campaign_persists_safe_failure_diagnostics
  -> test_failed_result_projects_classified_timeout_diagnostics
     and test_failed_result_does_not_persist_raw_exception_text

test_run_campaign_rejects_oversized_answer_without_ragas_evaluation
  -> test_execution_worker_marks_oversized_answer_failed_without_scheduling_ragas

test_campaign_failure_cancels_and_drains_pending_batch_tasks
  -> test_worker_completion_after_campaign_cancellation_exits_cleanly
     and test_stop_cancels_active_handler_and_recovers_its_attempt
```

Run the mapped tests explicitly:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_evaluation_execution_worker.py::test_v9_durable_worker_persists_measured_provider_phase tests/test_evaluation_execution_worker.py::test_execution_worker_promotes_ledger_total_not_payload_total tests/test_evaluation_execution_worker.py::test_failed_durable_execution_persists_current_evaluation_profile tests/test_evaluation_execution_worker.py::test_failed_durable_execution_prefers_captured_trace_profile tests/test_evaluation_execution_worker.py::test_failed_result_projects_classified_timeout_diagnostics tests/test_evaluation_execution_worker.py::test_failed_result_does_not_persist_raw_exception_text tests/test_evaluation_execution_worker.py::test_execution_worker_marks_oversized_answer_failed_without_scheduling_ragas tests/test_evaluation_execution_worker.py::test_worker_completion_after_campaign_cancellation_exits_cleanly tests/test_evaluation_job_worker.py::test_stop_cancels_active_handler_and_recovers_its_attempt
```

Expected: all mapped durable tests pass.

- [ ] **Step 2: Delete the six duplicate private-runner tests**

Remove exactly the six source test definitions listed in Step 1 from `tests/test_campaign_engine.py`. Do not remove the independent `test_run_with_retry_retries_resource_exhausted` test or any facade/recovery test.

- [ ] **Step 3: Verify the legacy symbol is now production-only**

Run:

```powershell
rg -n "_run_campaign" evaluation tests
```

Expected: the definition and internal calls in `evaluation/campaign_engine.py` remain; no test calls remain.

- [ ] **Step 4: Delete the process-local method chain**

From `CampaignEngine`, delete:

```text
_run_campaign
_run_evaluation_only
_run_ragas_evaluation
_evaluate_campaign_results
_execute_unit
_persist_unit_result
```

Delete module-level `_cancel_and_drain_tasks` and `_unit_key`, which become unreferenced. Do not alter:

```text
create_and_start
create_rerun
evaluate_campaign
recover_inflight_campaigns
ensure_campaign_task
_start_worker_if_available
_prepare_legacy_recovery
_resolve_test_cases
_build_units
_work_item_spec
get_campaign_engine
```

- [ ] **Step 5: Clean imports with Ruff as the authority**

Run:

```powershell
.venv\Scripts\python.exe -m ruff check evaluation/campaign_engine.py tests/test_campaign_engine.py --select F401,F841
```

Remove only imports and locals reported unused after the deleted code. Re-run until Ruff exits 0. Do not remove `run_with_retry` from the test file while its independent retry test remains.

- [ ] **Step 6: Prove the dead runner and reversed dependency are gone**

Run:

```powershell
$runnerMatches = rg -n "_run_campaign" evaluation tests; $runnerExit = $LASTEXITCODE
if ($runnerExit -eq 0) { $runnerMatches; throw "legacy runner references remain" }
if ($runnerExit -ne 1) { exit $runnerExit }
$importMatches = rg -n "from evaluation\.campaign_engine import" evaluation/execution_worker.py; $importExit = $LASTEXITCODE
if ($importExit -eq 0) { $importMatches; throw "execution worker still imports the facade" }
if ($importExit -ne 1) { exit $importExit }
Write-Output "Legacy runner and reversed dependency are absent"
```

Expected: both commands return no matches.

- [ ] **Step 7: Run the complete focused evaluation runtime suite**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_campaign_engine.py tests/test_evaluation_execution_worker.py tests/test_evaluation_execution_observability.py tests/test_evaluation_job_worker.py tests/test_evaluation_job_store.py tests/test_evaluation_ragas_worker.py tests/test_evaluation_api.py tests/test_evaluation_export_redaction.py tests/test_rag_startup.py
.venv\Scripts\python.exe -m ruff check evaluation/campaign_engine.py evaluation/campaign_execution.py evaluation/execution_worker.py tests/test_campaign_engine.py tests/test_evaluation_execution_worker.py tests/test_evaluation_execution_observability.py --select E9,F63,F7,F82,F401,F841
.venv\Scripts\python.exe scripts/check_complexity_ratchet.py --check
git diff --check
```

Expected: focused tests pass; Ruff correctness passes; the ratchet reports no new or increased finding; whitespace is clean.

- [ ] **Step 8: Confirm the scope and net deletion**

Run:

```powershell
git diff --stat
git status --short
```

Before committing, confirm no database, schema, router, frontend, Docker, or dependency file is modified and the cumulative implementation remains net-deleting.

- [ ] **Step 9: Commit the dead-code removal**

```powershell
git add -- evaluation/campaign_engine.py tests/test_campaign_engine.py
git commit -m "refactor(evaluation): remove process-local campaign runner"
```

---

### Task 6: Update runtime ownership documentation and run all gates

**Files:**
- Modify: `docs/design-docs/evaluation-runtime.md`
- Verify: all files changed since design commit `c6d2890`

**Interfaces:**
- Consumes: the final module ownership established by Tasks 2 and 5.
- Produces: accurate runtime documentation and complete verification evidence.

- [ ] **Step 1: Update the evaluation runtime ownership list**

In `docs/design-docs/evaluation-runtime.md`, replace the single engine ownership line with these three lines:

```markdown
- Campaign facade and recovery: `evaluation/campaign_engine.py`
- Campaign execution contracts and projections: `evaluation/campaign_execution.py`
- Durable dataset execution adapter: `evaluation/execution_worker.py`
```

Keep router, storage, analytics, generic observability, observability repository, trace-schema, and campaign-schema ownership lines unchanged.

- [ ] **Step 2: Verify generated contracts and documentation**

Run:

```powershell
.venv\Scripts\python.exe scripts/sync_openapi_artifacts.py --check
.venv\Scripts\python.exe scripts/check_markdown_links.py
```

Expected: OpenAPI artifacts are current and Markdown links are valid. No generated artifact should change because public APIs are unchanged.

- [ ] **Step 3: Commit the ownership documentation**

```powershell
git add -- docs/design-docs/evaluation-runtime.md
git commit -m "docs: document campaign execution ownership"
```

- [ ] **Step 4: Run the complete backend test suite with the warning budget**

Run:

```powershell
.venv\Scripts\python.exe scripts/run_pytest_with_warning_budget.py --max-warnings 56 -- -q
```

Expected: all backend tests pass, existing intentional skips remain skips, and the warning count is at or below 56.

- [ ] **Step 5: Run final maintenance gates**

Run:

```powershell
.venv\Scripts\python.exe -m ruff check . --select E9,F63,F7,F82,F401,F841
.venv\Scripts\python.exe scripts/check_complexity_ratchet.py --check
.venv\Scripts\python.exe scripts/sync_openapi_artifacts.py --check
.venv\Scripts\python.exe scripts/check_markdown_links.py
git diff --check
```

Expected: every command exits 0; complexity finding count and score do not increase.

- [ ] **Step 6: Run final structural and scope checks**

Run:

```powershell
$runnerMatches = rg -n "_run_campaign" evaluation tests; $runnerExit = $LASTEXITCODE
if ($runnerExit -eq 0) { $runnerMatches; throw "legacy runner references remain" }
if ($runnerExit -ne 1) { exit $runnerExit }
$importMatches = rg -n "from evaluation\.campaign_engine import" evaluation/execution_worker.py; $importExit = $LASTEXITCODE
if ($importExit -eq 0) { $importMatches; throw "execution worker still imports the facade" }
if ($importExit -ne 1) { exit $importExit }
git diff --name-status c6d2890..HEAD
git diff --shortstat c6d2890..HEAD -- evaluation tests quality
git status --short
```

Expected:

- both `rg` commands return no matches;
- the change list contains only the files named by this plan;
- the implementation is net-deleting; and
- the worktree is clean after all planned commits.

- [ ] **Step 7: Record the final handoff facts**

Report:

```text
- deleted legacy methods and obsolete tests
- final campaign_engine.py physical line count
- implementation insertions/deletions
- focused and full pytest results
- warning-budget result
- Ruff/complexity/OpenAPI/Markdown results
- exact commit list
- branch and push status
```

Do not claim completion unless every command in Steps 4-6 has fresh successful output from the final committed tree.
