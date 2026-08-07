# Cross-Stack Maintainability Baseline Design

**Date:** 2026-08-08

**Status:** Approved for implementation planning

**Canonical repository:** `pdftopng`

**Frontend counterpart:** `D:\flutterserver\Multimodal_RAG_System`

## Objective

Create a low-risk maintainability baseline across the backend and frontend that:

1. runs the complete backend verification surface in CI;
2. prevents warning and cyclomatic-complexity debt from increasing;
3. detects OpenAPI, contract-pin, and generated-document drift;
4. removes two confirmed-unmounted legacy evaluation components; and
5. separates current documentation from completed plans and decision history.

This work establishes guardrails. It does not attempt to eliminate all current warnings, refactor every complex function, or replace handwritten frontend API types.

## Scope

### Backend CI and quality ratchets

- Keep Python 3.11 deployment-source compilation.
- Run the complete pytest suite under Python 3.13 with fake providers and external network access blocked.
- Set the initial pytest warning ceiling to `56`, matching the approved baseline.
- Fail CI when the warning count rises above the baseline; allow the count to fall without requiring a baseline edit.
- Run Ruff correctness checks in CI.
- Store a C901 baseline keyed by source path and function name, including the observed complexity score.
- Fail CI when:
  - a new production function exceeds the configured complexity threshold; or
  - an existing baseline function's complexity score increases.
- Allow baseline entries to disappear or decrease without failing.

The ratchet applies to production Python sources only. Tests, generated files, virtual environments, registered worktrees, experiments, and archived utilities are excluded.

### OpenAPI contract and generated documentation

- Keep handwritten TypeScript API types in this phase.
- Add a deterministic backend command that derives the runtime OpenAPI schema and synchronizes:
  - `openapi.json`;
  - a committed SHA-256 manifest; and
  - a generated route-inventory section inside `docs/generated/api-surface.md`.
- Provide a read-only `--check` mode that fails when any committed artifact differs from the deterministic output.
- Preserve human-maintained evaluation/runtime notes outside generated marker blocks.
- Add a frontend drift command that compares the committed backend OpenAPI hash with `AGENTIC_V9_API_CONTRACT.openapi_sha256`.
- Local frontend checks use the sibling `pdftopng` checkout.
- Frontend CI checks out `DeadMark70/multimodal-rag-translate` and runs the same comparison.
- Advance the frontend pin only when the backend commit and OpenAPI hash are intentionally reviewed together.

### Frontend UI surface documentation

- Add a deterministic script that reads `src/App.tsx` and `vite.config.ts`.
- Generate only bounded marker sections:
  - the route/page/access inventory; and
  - the build/chunking facts.
- Preserve detailed human-maintained feature notes outside marker blocks.
- The generated build facts must reflect whether `manualChunks` is actually configured; they must not retain historical chunk names that are absent from `vite.config.ts`.
- Add `--check` behavior for CI drift detection.

### Legacy evaluation component removal

Remove the following confirmed-unmounted production components:

- `src/components/evaluation/EvaluationResults.tsx`
- `src/components/evaluation/AgentTraceViewer.tsx`

Remove their component-specific tests when present. Do not remove shared services, shared types, or shared components merely because these files used them. Additional deletion requires independent proof from production-import search, TypeScript checking, and the frontend test suite.

### Documentation lifecycle

- Move `docs/exec-plans/active/2026-07-evaluation-chat-loading-performance.md` to `docs/exec-plans/completed/`.
- Update the backend and frontend execution-plan indexes to point at the completed location.
- Move `docs/exec-plans/active/google-genai-stage2-langchain-paths.md` to backend references because it is a decision memo, not active executable work.
- Keep `docs/exec-plans/active/genai-langchain-layering-plan.md` active.
- Correct `docs/FRONTEND.md` and generated UI documentation so their chunking statements match the current Vite configuration.
- Add a deterministic local Markdown-link checker to CI for both repositories.

## Architecture

### Backend quality runner

Quality behavior is implemented in small scripts rather than embedded shell parsing:

- a pytest runner launches pytest, streams output, extracts the final warning count, preserves pytest's exit code, and enforces the approved ceiling;
- a Ruff complexity checker consumes Ruff JSON, normalizes findings to stable path/function identifiers, and compares scores with a committed JSON baseline; and
- focused unit tests validate parsing, lower-count acceptance, new-function rejection, score-increase rejection, and subprocess exit-code propagation.

CI composes these tools but does not reimplement their logic in YAML.

### Artifact synchronization

Synchronization commands have two modes:

- update mode writes deterministic artifacts; and
- check mode calculates the same output in memory and exits non-zero without writing when drift exists.

Generated Markdown is bounded by explicit begin/end markers. Missing, duplicated, or reversed markers are errors rather than reasons to overwrite a whole document.

### Cross-repository contract pin

The frontend remains the consumer-side pin owner. It stores the reviewed backend commit and OpenAPI hash in the existing fixture. A Node script reads the backend hash manifest from a configurable backend checkout path and compares both values. The default path is the local sibling repository; CI supplies the checkout path explicitly.

## Error Handling

- Quality scripts return the underlying tool's non-zero exit code before evaluating debt budgets.
- Missing or malformed baseline/manifest files produce concise non-zero errors without stack-trace noise.
- OpenAPI generation runs with test/fake-provider environment settings and must not require or print local secrets.
- Cross-repo checks fail clearly when the backend checkout or manifest is missing.
- Document generators refuse to rewrite files whose marker topology is invalid.
- Link checks ignore HTTP(S), mail, and anchor-only targets and understand Windows absolute paths.

## Testing Strategy

### Backend

- Unit-test warning-summary parsing and warning-ceiling decisions.
- Unit-test Ruff C901 parsing and baseline comparisons with synthetic findings.
- Unit-test generated-marker replacement, invalid markers, deterministic route ordering, and check-mode drift detection.
- Run the complete pytest suite through the warning-budget runner.
- Run Ruff correctness checks and the complexity ratchet.

### Frontend

- Unit-test OpenAPI manifest parsing, matching pins, hash mismatch, commit mismatch, and missing backend checkout behavior.
- Unit-test route/build document generation against fixture source strings.
- Verify removed legacy components have no production imports.
- Run lint, TypeScript checking, Vitest, and the production build.
- Run OpenAPI and UI-document drift checks.

### Documentation

- Run local Markdown-link checks in both repositories.
- Assert active/completed/reference indexes point to existing files.
- Verify generated-document check mode succeeds after update mode.

## CI Layout

### Backend workflow

1. Python 3.11 compile check.
2. Python 3.13 dependency installation.
3. Ruff correctness check.
4. C901 complexity ratchet.
5. OpenAPI and generated API document drift check.
6. Markdown-link check.
7. Complete pytest suite through the warning-budget runner.

### Frontend workflow

1. Install Node dependencies.
2. Check out the backend contract repository.
3. Run lint and TypeScript checking.
4. Run Vitest.
5. Run the production build.
6. Run backend OpenAPI-pin drift check.
7. Run generated UI-document and Markdown-link drift checks.

## Non-Goals

- Replacing handwritten frontend API types.
- Adding `openapi-typescript`, Orval, or another client generator.
- Reducing the current warning count below `56` in this change.
- Refactoring the current 71 backend or 56 frontend high-complexity functions.
- Deleting shared evaluation APIs or types beyond the two named legacy components.
- Redesigning Vite chunk strategy.
- Reorganizing all historical Superpowers plans/specifications.

## Acceptance Criteria

- Backend CI runs the full suite and fails for warning count `>56`.
- A synthetic new C901 finding or increased existing score fails the ratchet; removals and reductions pass.
- Runtime OpenAPI, its hash manifest, and generated backend route inventory are deterministic and clean in check mode.
- Frontend CI compares its pin to a checked-out backend manifest without generating TypeScript types.
- Generated UI route/build sections match `App.tsx` and `vite.config.ts`.
- Both named legacy components and their component-specific tests are gone, with frontend lint, type checking, tests, and build green.
- Completed and reference documents no longer appear in the backend active-plan index.
- Both repositories pass local Markdown-link checks.
- No local secret file is read, printed, copied, or committed by the new tooling.
