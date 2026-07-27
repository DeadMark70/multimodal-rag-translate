# Agentic v9 Full Behavior Rollback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the production Agentic v9 execution behavior that was working at commit `710474b`, while retaining the current Evaluation Setup, LLM accounting observer, and evaluation persistence contracts.

**Architecture:** The rollback is behavioral, not a repository reset. The production runtime and its direct behavior tests return to the `710474b` orchestration model; a small compatibility layer passes the current setup snapshot and observer into the existing budgeted provider boundary. Evaluation database schemas, analytics, export, and frontend contracts remain unchanged.

**Tech Stack:** Python 3.11, pytest, Pydantic, LangChain documents, current evaluation observability and token accounting.

## Global Constraints

- Work directly on `main` as explicitly approved by the user.
- Preserve Git history; do not use `git reset --hard`.
- Do not modify or commit existing untracked files under `.pytest-tmp/`, `data/`, or unrelated documentation.
- The source-authorization boundary remains fail-closed.
- Graph and visual capability failures may downgrade status, but must not erase authorized text evidence.
- Final generation accepts the legacy provider response envelope used by the working V9 runtime.
- Evaluation Setup remains authoritative for provider/model/output limits.

---

### Task 1: Lock the working V9 behavior with a regression test

**Files:**
- Create: `tests/test_agentic_v9_full_rollback.py`

**Interfaces:**
- Consumes: `AgenticV9CampaignRuntime.execute(...) -> RAGResult`
- Produces: A regression contract proving that missing visual capability does not erase authorized text evidence or a valid legacy provider answer.

- [ ] **Step 1: Write the failing test**

Create a visual-required contract, return one authorized text document, make visual extraction return no packets, and return a legacy text response from the provider. Assert that the result retains documents, uses the provider answer, and is no worse than `qualified_partial`.

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
pytest -q tests/test_agentic_v9_full_rollback.py
```

Expected: FAIL because the current strict runtime either blocks on the asset-manifest path or rejects the legacy final response.

### Task 2: Restore the working runtime and add only required compatibility

**Files:**
- Restore then modify: `evaluation/agentic_v9_campaign_runtime.py`
- Restore: `tests/test_agentic_v9_campaign_runtime.py`

**Interfaces:**
- Consumes: current `build_v9_admission_contract`, `BudgetedLlmInvoker`, `current_llm_call_observer`, and current `V9ExecutionCore` final-stage signature.
- Produces: the existing `AgenticV9CampaignRuntime` class and `RAGResult.agent_trace["agentic_v9"]` shape.

- [ ] **Step 1: Restore the two files from `710474b`**

Use Git restore with `710474b` as the source so the change is visible as a normal working-tree rollback.

- [ ] **Step 2: Add the minimal compatibility layer**

Add optional `llm_call_observer` constructor injection, fall back to `current_llm_call_observer()`, forward provider/model/capture metadata to `BudgetedLlmInvoker`, pass `setup_policy=setup_snapshot` to admission, and accept the current sufficiency-report final-stage argument without changing legacy synthesis.

- [ ] **Step 3: Run the rollback regression test**

Run:

```powershell
pytest -q tests/test_agentic_v9_full_rollback.py
```

Expected: PASS.

- [ ] **Step 4: Run the restored runtime suite**

Run:

```powershell
pytest -q tests/test_agentic_v9_campaign_runtime.py
```

Expected: PASS.

### Task 3: Verify and commit the behavioral rollback

**Files:**
- Verify: `evaluation/agentic_v9_campaign_runtime.py`
- Verify: `tests/test_agentic_v9_campaign_runtime.py`
- Verify: `tests/test_agentic_v9_full_rollback.py`

**Interfaces:**
- Consumes: all Agentic v9 tests and Ruff configuration.
- Produces: one auditable rollback commit on `main`.

- [ ] **Step 1: Run the complete Agentic v9 test selection**

Run:

```powershell
pytest -q -k agentic_v9
```

Expected: all relevant tests pass except the already documented Windows CRLF golden-hash exclusion if selected.

- [ ] **Step 2: Run Ruff**

Run:

```powershell
ruff check evaluation/agentic_v9_campaign_runtime.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_full_rollback.py
```

Expected: PASS.

- [ ] **Step 3: Inspect the rollback diff**

Confirm that no evaluation database, frontend, or unrelated untracked file is included.

- [ ] **Step 4: Commit**

```powershell
git add evaluation/agentic_v9_campaign_runtime.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_full_rollback.py docs/superpowers/plans/2026-07-28-agentic-v9-full-behavior-rollback.md
git commit -m "revert(agentic-v9): restore working v9 runtime behavior"
```
