# Evaluation Wave A: Accounting and Failure Diagnostics Design

**Date:** 2026-07-31
**Status:** Approved
**Scope:** Agentic v9 evaluation observability only

## Context

The 16-question, four-repeat Agentic v9 campaign completed 62 of 64
execution units. The successful runs reported 153,138 tokens, but campaign
token accounting and phase attribution remained partial.

Inspection of the exported run data and runtime code established two separate
causes:

1. Comparison-planner usage is included in the runtime total but its LLM-call
   observation is rejected before persistence. The invoker emits the
   `comparison_plan` phase, while `EvaluationLlmCall.phase` does not permit
   that value.
2. Failed execution attempts are correctly classified by the worker, but the
   failed campaign-result projection discards the classification and persists
   `str(exc)`. Exceptions such as a bare `TimeoutError()` therefore produce an
   empty answer and error message.

This wave repairs those data contracts without changing retrieval, planning,
prompting, synthesis, or evaluation behavior.

## Goals

- Persist every admitted comparison-planner LLM terminal attempt under a
  distinct `comparison_plan` phase.
- Remove `llm_call_observer_failed` from successful runs caused by the missing
  phase contract.
- Make successful-run token accounting and phase attribution complete when all
  provider usage is measured.
- Persist safe, actionable failure diagnostics for failed execution units.
- Preserve fail-closed accounting: a failed run with unavailable usage must
  not be treated as a measured zero-token run.
- Preserve compatibility with existing campaigns and the current SQLite
  schema.

## Non-goals

- No planner routing or subject-extraction changes.
- No retrieval, reranking, context-packing, prompt, or synthesis changes.
- No retry-policy, timeout, batch-size, or provider configuration changes.
- No historical campaign backfill.
- No frontend redesign.
- No generic telemetry outbox or arbitrary unvalidated phase strings.

## Design

### 1. Comparison-planner phase contract

`comparison_plan` becomes an explicit supported `LlmCallPhase`.

The phase remains distinct from `contract_planning`. Aliasing it to an existing
phase would make the write succeed but would erase the research distinction
between route/contract planning and comparison-subject planning.

The database needs no migration because the persisted phase column is already
plain text. Existing phase values and existing campaign rows remain valid.

The data path becomes:

```text
BudgetedLlmInvoker(phase="comparison_plan")
  -> LlmAttemptObservation
  -> EvaluationRunRecorder.on_terminal_attempt()
  -> EvaluationLlmCall(phase="comparison_plan")
  -> evaluation_llm_calls
  -> campaign phase aggregation
```

An observer persistence failure remains best-effort for runtime availability,
but the supported phase must no longer trigger it.

### 2. Failed-result diagnostic projection

The execution worker continues to classify an exception once through
`classify_evaluation_error`. The resulting `ErrorDecision` becomes the
authoritative source for both:

- the durable attempt failure; and
- the visible failed campaign-result projection.

The projection stores:

- `error_message`: the decision's safe message;
- `derived_metrics.error_type`: the decision's stable error type;
- `derived_metrics.response_status`: `failed`; and
- the existing execution-version metadata.

The projection must not persist raw exception text. Its answer must not present
an empty `ERROR: ` string as if it were a generated answer. Oversized-answer
handling retains its existing stable code.

The durable job attempt remains the source of truth for retry and promotion.
A later successful attempt continues to replace the failed projection through
the existing completion transaction.

### 3. Accounting semantics

Successful runs are complete only when their measured runtime total can be
reconciled with persisted LLM calls and attributed phases.

Failed runs with no provider usage remain failed and unavailable for token
comparison. The implementation must not manufacture zero usage to make the
campaign look complete. Consequently, a mixed campaign may still be partial
because it contains genuine failed runs; it must no longer be partial because
successful comparison-planner calls disappeared from telemetry.

### 4. Compatibility

- No database migration is required.
- Old campaigns are read unchanged.
- Existing phase consumers continue to accept all prior phase values.
- The new phase is additive and must appear in exports and phase aggregation
  through the existing generic phase handling.
- Failure messages remain sanitized and suitable for redacted exports.

## Verification

Focused tests must establish:

1. `EvaluationLlmCall` accepts `comparison_plan`.
2. A terminal comparison-planner observation is persisted with its measured
   input, output, and total tokens.
3. The supported planner phase does not mark
   `llm_call_observer_failed`.
4. Phase aggregation includes `comparison_plan` and reconciles the successful
   run total without unclassified usage.
5. A bare `TimeoutError()` produces:
   - `status=failed`;
   - `error_message="The evaluation provider request timed out."`; and
   - `derived_metrics.error_type="timeout"`.
6. Raw exception text is not written to the failed result.
7. Failed runs without usage are not reclassified as complete zero-token runs.
8. Existing final-answer and oversized-answer failure tests remain green.

After focused tests, run the evaluation observability, execution-worker,
accounting, and export regression suites. A production smoke campaign should
then verify that new successful comparison-planner runs have complete phase
attribution and that an induced timeout is diagnosable.

## Acceptance criteria

- Every attempted comparison-planner provider call has a corresponding
  `comparison_plan` LLM-call row when persistence is available.
- New successful runs no longer carry `llm_call_observer_failed` because of the
  planner phase.
- Planner token totals are visible in phase breakdowns and no longer disappear
  into partial accounting.
- Timeout failures expose a stable type and safe non-empty message.
- No retrieval or answer-quality behavior changes in this wave.
