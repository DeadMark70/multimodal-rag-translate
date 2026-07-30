# Evaluation Wave A Accounting and Failure Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist Agentic v9 comparison-planner usage under its real phase and project safe, non-empty diagnostics for failed evaluation runs without changing RAG behavior.

**Architecture:** Extend the existing typed LLM-call phase contract with the already-emitted `comparison_plan` value, leaving the generic SQLite phase storage and reconciliation algorithm unchanged. Reuse the worker's existing `ErrorDecision` as the single sanitized failure source for both durable attempts and failed campaign-result projections.

**Tech Stack:** Python 3.11, Pydantic v2, asyncio, aiosqlite, pytest, pytest-asyncio, Ruff

## Global Constraints

- Agentic v9 retrieval, reranking, comparison planning, prompts, synthesis, scoring, timeouts, and retry behavior must not change.
- `comparison_plan` must remain distinct from `contract_planning`.
- No SQLite migration or historical campaign backfill.
- Failed runs without measured usage must remain unavailable for token comparison; never manufacture zero-token completeness.
- Persist only classified safe failure messages, never raw provider exception text.
- Preserve the special `EVALUATION_ANSWER_TOO_LARGE` failure code.
- Keep patches minimal, typed, and independently reviewable.

---

## File Structure

- `evaluation/trace_schemas.py`: owns the typed persisted observability contract; add the supported planner phase here.
- `evaluation/execution_worker.py`: owns durable execution failure projection; pass the existing classified decision into the projection.
- `tests/test_evaluation_observability_schema.py`: validates the public Pydantic phase contract.
- `tests/test_evaluation_token_cost.py`: validates recorder conversion from terminal planner observation to persisted LLM-call data.
- `tests/test_evaluation_analytics_context.py`: validates successful-run token reconciliation and phase attribution.
- `tests/test_evaluation_execution_worker.py`: validates sanitized failed-result persistence and fail-closed token semantics.
- `tests/test_evaluation_export_redaction.py`: unchanged regression suite proving export sanitization remains intact.

### Task 1: Persist comparison-planner LLM usage

**Files:**
- Modify: `evaluation/trace_schemas.py:20-27`
- Modify: `tests/test_evaluation_observability_schema.py:272-294`
- Modify: `tests/test_evaluation_token_cost.py:1-17,279-330`
- Modify: `tests/test_evaluation_analytics_context.py:149-185`

**Interfaces:**
- Consumes: `LlmAttemptObservation.phase: str` emitted by `BudgetedLlmInvoker`.
- Produces: `EvaluationLlmCall.phase == "comparison_plan"` accepted by Pydantic, persisted by `EvaluationRunRecorder.on_terminal_attempt`, and grouped by `reconcile_official_tokens`.

- [ ] **Step 1: Add a failing schema-contract test**

Add this test beside
`test_llm_call_schema_exposes_attempt_and_capture_provenance`:

```python
def test_llm_call_schema_accepts_comparison_plan_phase() -> None:
    call = EvaluationLlmCall(
        llm_call_id="llm-comparison-plan",
        run_id="run-1",
        campaign_id="campaign-1",
        phase="comparison_plan",
        purpose="agentic_v9_comparison_plan",
        prompt_tokens=8,
        completion_tokens=3,
        total_tokens=11,
        created_at=datetime.now(timezone.utc),
    )

    assert call.phase == "comparison_plan"
    assert call.total_tokens == 11
```

- [ ] **Step 2: Run the schema test and verify the current contract rejects the phase**

Run:

```powershell
python -m pytest tests/test_evaluation_observability_schema.py::test_llm_call_schema_accepts_comparison_plan_phase -q
```

Expected: FAIL with a Pydantic literal-validation error stating that
`comparison_plan` is not an accepted `phase`.

- [ ] **Step 3: Extend the typed phase contract minimally**

Change only `LlmCallPhase`:

```python
LlmCallPhase = Literal[
    "unknown",
    "contract_planning",
    "comparison_plan",
    "evidence_extract",
    "retrieval_judge",
    "visual_extract",
    "final_answer",
]
```

Do not alter `_canonical_observation_phase`, database initialization, or
aggregation code.

- [ ] **Step 4: Run the schema test and verify it passes**

Run:

```powershell
python -m pytest tests/test_evaluation_observability_schema.py::test_llm_call_schema_accepts_comparison_plan_phase -q
```

Expected: PASS.

- [ ] **Step 5: Add a recorder-level regression test**

In `tests/test_evaluation_token_cost.py`, import
`LlmAttemptObservation`:

```python
from data_base.agentic_v9.budgeted_llm import LlmAttemptObservation
```

Make the fake repository report a successful write:

```python
async def record_llm_call(self, call):
    self.calls.append(call)
    return True
```

Add:

```python
@pytest.mark.asyncio
async def test_recorder_persists_comparison_plan_terminal_attempt() -> None:
    repository = FakeLlmCallRepository()
    recorder = EvaluationRunRecorder(
        run_id="run-1",
        campaign_id="campaign-1",
        user_id="user-a",
        llm_call_repository=repository,
    )
    observation = LlmAttemptObservation(
        phase="comparison_plan",
        purpose="agentic_v9_comparison_plan",
        reservation_id="reservation-plan",
        provider_attempt=1,
        provider="google",
        model_name="gemini-test",
        prompt_hash="prompt-hash",
        prompt_preview="plan the comparison",
        full_prompt=None,
        prompt_capture_status="captured",
        full_prompt_capture_status="not_captured_at_execution",
        response_hash="response-hash",
        latency_ms=25.0,
        status="success",
        error={},
        usage={
            "input_tokens": 8,
            "output_tokens": 3,
            "reasoning_tokens": 0,
            "other_tokens": 0,
            "total_tokens": 11,
            "usage_status": "measured",
            "official_total_tokens": 11,
        },
    )

    recorded = await recorder.on_terminal_attempt(observation)

    assert recorded is True
    assert recorder.observability_partial_reasons == []
    assert len(repository.calls) == 1
    call = repository.calls[0]
    assert call.phase == "comparison_plan"
    assert call.reservation_id == "reservation-plan"
    assert call.provider_attempt == 1
    assert (call.prompt_tokens, call.completion_tokens, call.total_tokens) == (
        8,
        3,
        11,
    )
    assert call.payload["usage_status"] == "measured"
    assert call.payload["official_total_tokens"] == 11
```

- [ ] **Step 6: Run the recorder regression**

Run:

```powershell
python -m pytest tests/test_evaluation_token_cost.py::test_recorder_persists_comparison_plan_terminal_attempt -q
```

Expected: PASS and no `llm_call_observer_failed` partial reason.

- [ ] **Step 7: Add an accounting-reconciliation regression**

Add a test beside
`test_official_token_reconciliation_counts_retries_and_token_components`:

```python
def test_official_token_reconciliation_attributes_comparison_plan() -> None:
    calls = [
        SimpleNamespace(
            llm_call_id="comparison-plan",
            phase="comparison_plan",
            reservation_id="reservation-plan",
            provider_attempt=1,
            total_tokens=11,
            prompt_tokens=8,
            completion_tokens=3,
            reasoning_tokens=0,
            other_tokens=0,
            payload={"usage_status": "measured", "official_total_tokens": 11},
        ),
        SimpleNamespace(
            llm_call_id="final-answer",
            phase="final_answer",
            reservation_id="reservation-final",
            provider_attempt=1,
            total_tokens=19,
            prompt_tokens=14,
            completion_tokens=5,
            reasoning_tokens=0,
            other_tokens=0,
            payload={"usage_status": "measured", "official_total_tokens": 19},
        ),
    ]

    result = reconcile_official_tokens(runtime_total_tokens=30, calls=calls)

    assert result.status == "complete"
    assert result.provider_total_tokens == 30
    assert result.by_phase == {"comparison_plan": 11, "final_answer": 19}
    assert result.reasons == ()
```

- [ ] **Step 8: Run all Task 1 focused tests**

Run:

```powershell
python -m pytest -q `
  tests/test_evaluation_observability_schema.py `
  tests/test_evaluation_token_cost.py `
  tests/test_evaluation_analytics_context.py
```

Expected: all tests PASS.

- [ ] **Step 9: Lint Task 1 files**

Run:

```powershell
python -m ruff check `
  evaluation/trace_schemas.py `
  tests/test_evaluation_observability_schema.py `
  tests/test_evaluation_token_cost.py `
  tests/test_evaluation_analytics_context.py
```

Expected: exit code 0.

- [ ] **Step 10: Commit Task 1**

```powershell
git add -- `
  evaluation/trace_schemas.py `
  tests/test_evaluation_observability_schema.py `
  tests/test_evaluation_token_cost.py `
  tests/test_evaluation_analytics_context.py
git diff --cached --check
git commit -m "fix(evaluation): persist comparison planner usage"
```

### Task 2: Project classified failure diagnostics safely

**Files:**
- Modify: `evaluation/execution_worker.py:42,202-218,346-405`
- Modify: `tests/test_evaluation_execution_worker.py:405-520`
- Test unchanged: `tests/test_evaluation_error_policy.py`
- Test unchanged: `tests/test_evaluation_export_redaction.py`

**Interfaces:**
- Consumes: `ErrorDecision` returned once by
  `classify_evaluation_error(exc: BaseException)`.
- Produces: failed `CampaignResult` with a safe non-empty `error_message`,
  `derived_metrics.error_type`, an empty non-answer, and no fabricated token
  usage.

- [ ] **Step 1: Add a failing timeout-projection test**

Import the classifier in `tests/test_evaluation_execution_worker.py`:

```python
from evaluation.error_policy import classify_evaluation_error
```

Add:

```python
@pytest.mark.asyncio
async def test_failed_result_projects_classified_timeout_diagnostics(
    store: EvaluationJobStore,
) -> None:
    worker = DatasetExecutionWorker(store=store)
    claim = await _claim_seeded_execution(store, mode="agentic")
    exc = TimeoutError()
    decision = classify_evaluation_error(exc)

    await worker._persist_failed_result(
        claim,
        exc,
        decision=decision,
    )

    result = (
        await evaluation_db.CampaignResultRepository().list_for_campaign(
            user_id="user-a",
            campaign_id="cmp-1",
        )
    )[0]
    assert result.status.value == "failed"
    assert result.answer == ""
    assert result.error_message == "The evaluation provider request timed out."
    assert result.derived_metrics["response_status"] == "failed"
    assert result.derived_metrics["error_type"] == "timeout"
    assert result.total_tokens is None
    assert result.token_usage == {}
    assert result.final_answer_hash is None
```

- [ ] **Step 2: Add a failing raw-error-redaction test**

Add:

```python
@pytest.mark.asyncio
async def test_failed_result_does_not_persist_raw_exception_text(
    store: EvaluationJobStore,
) -> None:
    worker = DatasetExecutionWorker(store=store)
    claim = await _claim_seeded_execution(store, mode="agentic")
    exc = RuntimeError("apiKey=secret-provider-detail")
    decision = classify_evaluation_error(exc)

    await worker._persist_failed_result(
        claim,
        exc,
        decision=decision,
    )

    result = (
        await evaluation_db.CampaignResultRepository().list_for_campaign(
            user_id="user-a",
            campaign_id="cmp-1",
        )
    )[0]
    serialized = json.dumps(result.model_dump(mode="json"))
    assert result.error_message == "An unexpected evaluation error occurred."
    assert result.derived_metrics["error_type"] == "unknown"
    assert "secret-provider-detail" not in serialized
```

- [ ] **Step 3: Run the two tests and verify the current projection fails**

Run:

```powershell
python -m pytest -q `
  tests/test_evaluation_execution_worker.py::test_failed_result_projects_classified_timeout_diagnostics `
  tests/test_evaluation_execution_worker.py::test_failed_result_does_not_persist_raw_exception_text
```

Expected: FAIL because `_persist_failed_result` does not accept `decision` and
still uses `str(exc)`.

- [ ] **Step 4: Make `ErrorDecision` part of the private worker contract**

Change the import:

```python
from evaluation.error_policy import ErrorDecision, classify_evaluation_error
```

Pass the already-classified decision from the exception handler:

```python
await self._persist_failed_result(
    claim,
    exc,
    decision=decision,
    payload=payload,
)
```

Change the private method signature:

```python
async def _persist_failed_result(
    self,
    claim: ClaimedEvaluationWork,
    exc: Exception,
    *,
    decision: ErrorDecision,
    payload: BenchmarkExecutionResult | None = None,
) -> None:
```

Do not call `classify_evaluation_error` a second time inside the method.

- [ ] **Step 5: Replace raw exception projection with safe classified fields**

Immediately after `oversized_answer`, derive stable values:

```python
safe_error_message = (
    EVALUATION_ANSWER_TOO_LARGE
    if oversized_answer
    else decision.safe_message
)
error_type = (
    EVALUATION_ANSWER_TOO_LARGE
    if oversized_answer
    else decision.error_type
)
```

Use these values in `CampaignResultRepository.create`:

```python
answer="",
token_usage={},
total_tokens=None,
error_message=safe_error_message,
derived_metrics={
    "agentic_execution_version": unit.agentic_execution_version,
    "response_status": "failed",
    "error_type": error_type,
},
final_answer_hash=None,
```

Remove both `f"ERROR: {exc}"` expressions. Preserve the existing execution
profile selection and all identity/snapshot fields.

- [ ] **Step 6: Run the focused diagnostics and oversized-answer tests**

Run:

```powershell
python -m pytest -q `
  tests/test_evaluation_execution_worker.py::test_failed_result_projects_classified_timeout_diagnostics `
  tests/test_evaluation_execution_worker.py::test_failed_result_does_not_persist_raw_exception_text `
  tests/test_evaluation_execution_worker.py::test_execution_worker_marks_oversized_answer_failed_without_scheduling_ragas
```

Expected: all three PASS.

- [ ] **Step 7: Run failure-policy and export regressions**

Run:

```powershell
python -m pytest -q `
  tests/test_evaluation_error_policy.py `
  tests/test_evaluation_execution_worker.py `
  tests/test_evaluation_export_redaction.py
```

Expected: all tests PASS; redacted export contains no raw provider details.

- [ ] **Step 8: Lint Task 2 files**

Run:

```powershell
python -m ruff check `
  evaluation/execution_worker.py `
  tests/test_evaluation_execution_worker.py
```

Expected: exit code 0.

- [ ] **Step 9: Commit Task 2**

```powershell
git add -- `
  evaluation/execution_worker.py `
  tests/test_evaluation_execution_worker.py
git diff --cached --check
git commit -m "fix(evaluation): persist classified run failures"
```

### Task 3: Verify Wave A as an integrated observability repair

**Files:**
- Verify: `evaluation/trace_schemas.py`
- Verify: `evaluation/execution_worker.py`
- Verify: `evaluation/analytics.py`
- Verify: `evaluation/analytics.py`
- Verify: all Task 1 and Task 2 tests

**Interfaces:**
- Consumes: the two committed changes from Tasks 1 and 2.
- Produces: verification evidence that successful planner runs reconcile and
  failed runs remain diagnosable and fail-closed.

- [ ] **Step 1: Run the complete focused evaluation suite**

Run:

```powershell
python -m pytest -q `
  tests/test_evaluation_observability_schema.py `
  tests/test_evaluation_observability_repository.py `
  tests/test_evaluation_token_cost.py `
  tests/test_evaluation_analytics_context.py `
  tests/test_evaluation_execution_worker.py `
  tests/test_evaluation_error_policy.py `
  tests/test_evaluation_export_redaction.py `
  tests/test_evaluation_research_end_to_end.py
```

Expected: all tests PASS.

- [ ] **Step 2: Run final lint and formatting checks**

Run:

```powershell
python -m ruff check `
  evaluation/trace_schemas.py `
  evaluation/execution_worker.py `
  tests/test_evaluation_observability_schema.py `
  tests/test_evaluation_token_cost.py `
  tests/test_evaluation_analytics_context.py `
  tests/test_evaluation_execution_worker.py
python -m ruff format --check `
  evaluation/trace_schemas.py `
  evaluation/execution_worker.py `
  tests/test_evaluation_observability_schema.py `
  tests/test_evaluation_token_cost.py `
  tests/test_evaluation_analytics_context.py `
  tests/test_evaluation_execution_worker.py
```

Expected: both commands exit 0.

- [ ] **Step 3: Inspect the final diff and commit boundaries**

Run:

```powershell
git status --short
git diff --check HEAD~2..HEAD
git log -3 --oneline
git show --stat --oneline HEAD~1
git show --stat --oneline HEAD
```

Expected:

- one Task 1 commit containing only the phase contract and its tests;
- one Task 2 commit containing only classified failure projection and its
  tests;
- no unrelated workspace files staged or committed.

- [ ] **Step 4: Run a production smoke after deployment**

Create a small Agentic v9 evaluation containing:

- one comparison question that invokes the comparison planner; and
- one controlled timeout or naturally timed-out run, if available.

Verify the exported data:

```text
successful comparison run:
  llm_calls contains phase=comparison_plan
  planner call usage_status=measured
  observability_partial_reasons excludes llm_call_observer_failed
  token accounting=complete
  phase attribution=complete

failed timeout run:
  status=failed
  error_type=timeout
  error_message=The evaluation provider request timed out.
  answer is empty
  no fabricated provider usage
```

The smoke is deployment verification, not a third production-code commit.

## Self-review

- Spec coverage: phase persistence, safe failure projection, fail-closed token
  behavior, compatibility, redaction, and regression verification are each
  mapped to a task.
- Placeholder scan: no deferred implementation markers or unspecified error
  handling remain.
- Type consistency: Task 2 introduces one explicit
  `decision: ErrorDecision` keyword argument and uses that exact signature in
  both the worker call site and tests.
- Scope check: no RAG behavior, retry, timeout, migration, frontend, or
  historical-data work is included.
