# Pytest Failure Reconciliation Design

## Goal

Eliminate the 17 reproducible backend pytest failures without deleting useful
regression coverage, weakening architectural boundaries, reverting intentional
production contracts, or reading and rewriting local secret values.

## Evidence Summary

The baseline is `1601 passed, 17 failed, 1 skipped`. Every failure was rerun
individually and compared with the current implementation, adjacent passing
tests, and the commits that introduced the relevant contract or refactor.
There is no evidence of an application-level production regression.

The failures divide into four root-cause groups:

1. Eleven stale tests or test doubles still assert an earlier contract or patch
   a symbol at its pre-refactor location.
2. Four architecture scans incorrectly traverse registered worktrees under
   `.worktrees/` and report copies of valid production files as violations.
3. One golden-dataset integrity test hashes checkout-specific CRLF bytes instead
   of canonical repository content.
4. One dependency test treats the ignored local secrets file as a mandatory
   default pytest input even though the missing setting has a safe runtime
   default.

## Selected Strategy

Repair the tests and the shared production-scope scanner. Do not delete the
tests and do not restore older production behavior.

This is the narrowest complete strategy because the current production behavior
is supported by newer focused tests and explicit commits. Deleting the failing
tests would discard useful regression coverage, while reverting production
would undo intended security, fail-closed budgeting, bounded analytics, and
module-isolation changes.

## Failure-by-Failure Treatment

### Current Contract Assertions

- `test_execution_policy_has_the_initial_runtime_bounds`: retain the exact
  policy contract and add the intentional `comparison_plan: 68.0` timeout.
- `test_missing_visual_capability_preserves_authorized_text_answer`: update the
  expectation to `complete` when the visual extractor returns no eligible
  evidence but does not fail. Preserve the distinct extractor-error coverage
  that expects `qualified_partial`.
- `test_campaign_integration_keeps_running_when_one_mode_fails`: assert the
  redacted public failure message and structured error code. Assert that the
  original exception text is absent.
- `test_research_analytics_endpoints_return_owned_run_details`: treat an
  unspecified new campaign as v9, matching the schema default.
- The two v9 campaign preflight failures: use a five-call setup budget for the
  fail-closed pre-route reservation. Keep the resolved route's three-call
  contract assertions separate.
- `test_question_comparison_is_measured_and_fail_closed`: update the test double
  to expose the bounded result DTO's direct `required_modalities` field.

### Refactor-Aware Test Boundaries

- Move the evaluation phase ownership assertion from
  `data_base/RAG_QA_service.py` to `data_base/rag_generation.py`, where the
  provider call remains wrapped by `llm_accounting_phase("answer_generation")`.
- Retarget Graph locator patches from `data_base.RAG_QA_service` to
  `data_base.rag_graph_locator`.
- Retarget answer-generation metadata and usage patches to
  `data_base.rag_generation`.
- Prefer the public `locate_graph_sources` boundary for planning-only and scope
  filtering checks. Tests must prove that out-of-scope chunks are removed before
  scoring and must not pass accidentally because a merge ratio rounds to zero.

### Hermetic Production Scanning

Add `.worktrees` to `core.production_scope.NON_PRODUCTION_ROOTS`. The scanner
must continue to include ordinary production Python files while excluding
registered worktree copies. Add a focused test that creates representative
paths and proves both sides of the boundary.

Do not expand Gemini or router allowlists. The main checkout already satisfies
the existing architecture rules.

### Cross-Platform Golden Integrity

Canonicalize CRLF to LF inside the golden SHA-256 helper before hashing. This
allows Windows `core.autocrlf=true` checkouts while continuing to fail on every
semantic byte change, missing file, route change, manifest change, or unexpected
line-ending sequence.

The expected hashes and golden JSON files remain unchanged.

### Local Environment Validation

Default pytest runs must not inspect `.env` or `config.env`. The strict key
comparison remains available only when `VALIDATE_LOCAL_ENV=1` is explicitly set.
When enabled, it retains the current behavior and reports missing key names only;
it never prints values.

This keeps ordinary and CI test runs hermetic while preserving an opt-in
deployment preflight for developers and operators.

## Testing Strategy

- Use each existing failing test as the red reproducer for stale-contract and
  fixture repairs.
- Before changing `core.production_scope`, add a focused test proving that
  `.worktrees/example/core/app_factory.py` is excluded while
  `core/app_factory.py` remains included.
- Add coverage for default local-env skipping and opt-in strict validation
  without reading or emitting real secret values.
- Re-run each repaired subsystem independently: execution/golden/env, Agentic
  and analytics, architecture boundaries, and Graph locator/generation.
- Run the complete backend pytest suite. Completion requires zero failures; any
  newly exposed failure returns to root-cause analysis instead of being deleted
  or broadly skipped.

## Commit Structure

1. Make production-scope scanning hermetic and add its regression test.
2. Repair cross-platform and opt-in repository contract tests.
3. Synchronize Agentic v9 and evaluation analytics tests with current contracts.
4. Retarget Graph and generation tests to their current module boundaries.
5. Run subsystem and full-suite verification.

## Compatibility and Non-Goals

- No API, response schema, provider budget, routing, retrieval, or answer behavior
  changes are intended.
- No secret file is read, printed, committed, moved, or rewritten by the repair.
- No worktree is deleted or modified.
- No architecture allowlist is broadened.
- Existing unrelated untracked documentation remains untouched.
