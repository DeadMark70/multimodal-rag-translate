# Evaluation Campaign v9 Default Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every newly created evaluation campaign default to Agentic v9 without changing daily Chat, explicit v8 requests, or historical-data fallbacks.

**Architecture:** Set the creation default independently at the backend contract boundary and the Evaluation Setup UI boundary. Preserve explicit execution identity and all read-time compatibility behavior.

**Tech Stack:** Python 3.13, Pydantic v2, pytest, React, TypeScript, Vitest, Testing Library.

## Global Constraints

- Existing campaign snapshots and stored results are immutable.
- Historical parsing fallbacks remain `v8`.
- Daily Agentic Chat defaults remain unchanged.
- Explicit `v8` requests remain valid.

---

### Task 1: Backend evaluation creation defaults

**Files:**
- Modify: `evaluation/campaign_schemas.py`
- Test: `tests/test_campaign_schemas.py`

**Interfaces:**
- Consumes: `CampaignConfig`, `CampaignCreateRequest`, and `ModelConfig`.
- Produces: omitted `agentic_execution_version` resolves to `"v9"` only for new evaluation campaign contracts.

- [ ] **Step 1: Write the failing tests**

Add tests which instantiate `CampaignConfig` and `CampaignCreateRequest` without
`agentic_execution_version` and assert `"v9"`. Add an explicit
`agentic_execution_version="v8"` case and assert it remains `"v8"`.

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_campaign_schemas.py -q
```

Expected: the omitted-version assertions fail with `v8`.

- [ ] **Step 3: Implement the minimum backend change**

Change only the creation/config defaults in `CampaignConfig` and
`CampaignCreateRequest` from `"v8"` to `"v9"`. Do not change result, trace,
database, analytics, worker, or Chat fallbacks.

- [ ] **Step 4: Run tests to verify GREEN**

Run the focused schema test and the campaign identity regression tests:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_campaign_schemas.py tests/test_evaluation_task14.py tests/test_rag_modes_agentic.py -q
```

- [ ] **Step 5: Commit**

Commit only `evaluation/campaign_schemas.py` and
`tests/test_campaign_schemas.py`.

### Task 2: Evaluation Setup UI defaults

**Files:**
- Modify: `src/components/evaluation/CampaignRunner.tsx`
- Test: `src/components/evaluation/CampaignRunner.test.tsx`

**Interfaces:**
- Consumes: existing `AgenticExecutionVersion` state and campaign preflight/create APIs.
- Produces: untouched Agentic selection displays and submits `v9`.

- [ ] **Step 1: Update the existing test to require v9 by default**

In the v9 preflight test, remove the manual version change, assert the selector
initially equals `v9`, start the evaluation, and retain the assertions that v9
preflight runs and the request records `agentic_execution_version: "v9"`.

- [ ] **Step 2: Run test to verify RED**

Run:

```powershell
npm test -- --run src/components/evaluation/CampaignRunner.test.tsx
```

Expected: selector assertion receives `v8`.

- [ ] **Step 3: Implement the minimum frontend change**

Initialize `agenticExecutionVersion` with `"v9"` instead of `"v8"`. Do not
change stored campaign rendering, shadow behavior, or the daily Chat UI.

- [ ] **Step 4: Run tests and lint**

Run:

```powershell
npm test -- --run src/components/evaluation/CampaignRunner.test.tsx
npm run lint
```

- [ ] **Step 5: Commit**

Commit only `CampaignRunner.tsx` and `CampaignRunner.test.tsx`.

### Task 3: Cross-repository verification

**Files:**
- No production changes expected.

**Interfaces:**
- Consumes: Task 1 and Task 2 commits.
- Produces: evidence that new campaigns default to v9 while compatibility remains intact.

- [ ] **Step 1: Run backend targeted regression suite**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_campaign_schemas.py tests/test_evaluation_task14.py tests/test_rag_modes_agentic.py tests/test_campaign_engine.py -q
```

- [ ] **Step 2: Run frontend targeted suite and lint**

```powershell
npm test -- --run src/components/evaluation/CampaignRunner.test.tsx src/types/evaluation.contract.test.ts
npm run lint
```

- [ ] **Step 3: Verify diffs and worktrees**

Confirm only intended tracked files changed and pre-existing untracked files
remain untouched.

