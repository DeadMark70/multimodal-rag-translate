# Agentic v9 Checkpoint 1 — Task 1 Report

Date: 2026-08-17  
Baseline: `7783bf7dbed4047fb72e30a305e9c2720dfe6a69`  
Artifact: `D:\flutterserver\evl_json\c7267013-0bfe-4a86-8e23-719cfcec21a8-observability-custom-v2.json`

## Scope and constraints

The current main worktree was used. No live provider call was made. The
`source_docs` scope, evidence qualification hard-gate/soft-gate policy,
configured model, and pipeline topology were not changed. Claim verification
remains one batched provider call at most. This report covers Checkpoint 1
Task 1 only; the hard-gate architecture checkpoint was not started.

## Root-cause reconfirmation

| Brief item | Current-HEAD status before patch | Root cause and result |
|---|---|---|
| Structured claim-verifier boundary | Present | The runtime selected the generic synthesizer path without a strict claim-verifier response schema, and claim verification only handled string-like content. It now uses the shared boundary, configured synthesizer, strict schema, and common string/content-block normalization. |
| Post-final budget admission | Present | Planning reserved the final-answer envelope before final synthesis and reserved it again for the post-final verifier. The runtime now marks the final envelope as already consumed while retaining route/phase/total/token ceilings and one-call verifier admission. |
| Verifier diagnostics | Present | Call count was derived from pending claims and all provider/response/claim failures collapsed to `claim_verifier_unavailable_or_invalid`. Call count now records actual invocation and bounded diagnostics distinguish budget rejection, provider failure, invalid response, and claim rejection; provider output and exception text are not exported. |
| Bounded final evidence references | Present | Provider context exposed long canonical IDs and accepted provider-selected IDs without a provider-facing allow-list. Packed evidence now receives deterministic `E1`, `E2`, ... aliases; only aliases from the current packed set map back to canonical IDs. Unknown/stale/unpacked IDs fail closed. |
| Repair observability | Present | Repair records were always emitted with `resulting_evidence_ids=[]`. The latest repair record now receives stable newly produced evidence IDs; a round with no new evidence remains an explicit empty list. |
| Raw retrieval observability | Present | Runtime retrieval trace had candidate/qualified/packed/used evidence information, but export reconstruction relied on empty `RAGResult.contexts` and lost stage independence when claims were rejected. The canonical normalized export path now projects trace-level packets and stage flags without changing answer contexts/RAGAS semantics. |
| Planner diagnostics | Present | Unexpected semantic planner failures were returned as a generic degraded result with no bounded stage/code. Real semantic failures now report `semantic_validation` / `planner_semantic_rejection`; deterministic planning behavior is unchanged. |
| Provider token accounting | Present | Compatible Google component usage (`input + output + reasoning`) could exceed the provider-reported total because reasoning was counted twice. The known total now reconciles visible output once while preserving estimated/partial behavior when official usage is unavailable. |

## TDD evidence

### RED

Before production edits, the following exact focused command was run:

```text
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_provider_boundary.py::test_claim_verifier_provider_uses_synthesizer_with_strict_schema tests/test_agentic_v9_claim_verifier.py::test_verifier_accepts_gemini_text_content_block_response tests/test_agentic_v9_claim_verifier.py::test_verifier_reports_bounded_provider_failure_without_exception_text tests/test_agentic_v9_final_answer.py::test_final_answer_maps_bounded_provider_alias_to_canonical_evidence_id tests/test_agentic_v9_budget_feasibility.py::test_post_final_claim_verifier_does_not_reserve_final_answer_again tests/test_agentic_v9_budgeted_llm.py::test_google_component_usage_does_not_double_count_reasoning_tokens tests/test_agentic_v9_contract_planner.py::test_planner_diagnostic_bounds_unexpected_semantic_failure tests/test_evaluation_execution_observability.py::test_v9_raw_retrieval_stages_export_when_final_claim_is_rejected -q --disable-warnings
```

Expected genuine RED result before implementation: `7 failed, 1 passed, 23
warnings in 4.91s`. The failures were the missing claim-verifier schema,
unsupported content-block list, unbounded/generic provider diagnostic,
unmapped alias payload, unsupported post-final admission argument, double
counted Google usage, and missing planner stage/code. The raw retrieval test
was the one pre-existing passing test in that initial set.

### GREEN and focused verification

After the minimal production changes, the focused Agentic v9 command was:

```text
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_claim_verifier.py tests/test_agentic_v9_final_answer.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_repair.py tests/test_agentic_v9_comparison_planner.py tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_budgeted_llm.py tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_execution_core.py -q --disable-warnings
```

Result: `256 passed, 23 warnings in 5.48s`.

Additional focused regressions:

```text
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_campaign_runtime.py::test_v9_comparison_repairs_a_missing_subject_once_and_caps_status -q --disable-warnings
```

Result: `2 passed` (success and no-new-evidence repair paths).

```text
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_execution_observability.py::test_v9_raw_retrieval_stages_export_when_final_claim_is_rejected -q --disable-warnings
```

Result: `1 passed`.

The full campaign runtime file also passed: `59 passed, 23 warnings`.

Scoped lint and syntax checks passed:

```text
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9 evaluation/agentic_v9_campaign_runtime.py evaluation/campaign_execution.py tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_claim_verifier.py tests/test_agentic_v9_final_answer.py tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_budgeted_llm.py tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_campaign_runtime.py tests/test_evaluation_execution_observability.py
```

Result: `All checks passed!`

```text
.\.venv\Scripts\python.exe -m compileall -q data_base\agentic_v9 evaluation\agentic_v9_campaign_runtime.py evaluation\campaign_execution.py
```

Result: passed with no output.

## Broader verification

```text
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_export_v2_schemas.py tests/test_evaluation_export_redaction.py tests/test_evaluation_analytics_context.py tests/test_evaluation_execution_observability.py -q --disable-warnings
```

Result: `118 passed, 1 failed, 25 warnings in 10.49s`. The remaining failure
is the pre-existing fixture `test_v9_campaign_persists_default_visual_and_final_provider_attempts`:
its fixture expects visual/final attempts, while the current runtime records
only the evidence-extract failure for that fixture. It is unrelated to the
Checkpoint 1 changes and was not expanded into this scope.

The attempt-persistence baseline was reconfirmed:

```text
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_v9_attempt_persistence.py -q --disable-warnings
```

Result: `2 failed, 7 passed, 23 warnings in 4.99s`. Both failures are the
same pre-existing expectation that `evidence_slot_ids` is absent from a
persisted subject payload; the baseline targeted run had the same failures.

OpenAPI was checked because internal Pydantic trace/metrics schemas changed:

```text
.\.venv\Scripts\python.exe scripts/sync_openapi_artifacts.py --check
```

Result: stale generated artifacts (`contracts/openapi-contract.json` and
`openapi.json`). No public route/schema contract changed, so generated
OpenAPI artifacts and frontend code were intentionally not modified. There
was no frontend contract change to verify.

`git diff --check` completed without whitespace errors.

## Changed files and design rationale

- `data_base/agentic_v9/provider_boundary.py`: shared response normalization,
  strict claim-verifier schema, and configured synthesizer binding.
- `data_base/agentic_v9/claim_verifier.py`: bounded response/error handling,
  truthful invocation count, diagnostic code, and safe claim rejection reason.
- `data_base/agentic_v9/final_answer.py` and
  `data_base/agentic_v9/final_synthesis_context.py`: bounded `E#` provider
  aliases with deterministic canonical-ID persistence and fail-closed unknown
  selection.
- `data_base/agentic_v9/budget_feasibility.py`,
  `evaluation/agentic_v9_campaign_runtime.py`: post-final admission without
  duplicate final reservation while retaining all ceilings and one verifier
  call.
- `data_base/agentic_v9/budgeted_llm.py`: reconcile known provider component
  usage without double counting reasoning tokens.
- `data_base/agentic_v9/contract_planner.py`: bounded semantic failure stage
  and code.
- `data_base/agentic_v9/schemas.py`: typed verifier diagnostic field.
- `evaluation/agentic_v9_campaign_runtime.py`: stable repair evidence IDs and
  trace-level candidate/qualified/packed/used stage data.
- `evaluation/campaign_execution.py`: canonical export projection for raw
  retrieval stages, explicitly separate from `RAGResult.contexts` and answer
  contexts.
- The corresponding `tests/` files add RED-to-GREEN regressions for every
  production behavior above and update affected runtime expectations.

## Commit and final worktree

Implementation commit at report finalization: `5ecae9d`.

The report metadata is included in the same single implementation commit via
the final amend. Final `git status --short` is clean after that amend.

## Residual risks

1. Generated OpenAPI files are stale at baseline/check time. They were not
   regenerated because this task did not alter a public API contract.
2. One broader observability fixture and two attempt-persistence assertions
   remain failing as described above; both were reproduced outside the new
   regressions and are not caused by this patch.
3. Provider usage remains estimated/partial when an upstream provider does
   not expose official usage; only the confirmed Google component-overlap
   path was repaired.
