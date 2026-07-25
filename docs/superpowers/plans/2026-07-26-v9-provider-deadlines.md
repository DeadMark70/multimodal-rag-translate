# Agentic v9 Provider Deadline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise Agentic v9 provider-facing request limits from the observed 8-second default to 32 seconds, while capping visual extraction independently at 16 seconds.

**Architecture:** `ExecutionPolicy` is the single source of truth for an attempt-wide deadline and per-phase caps. The runtime clamps every phase to the remaining attempt-wide deadline, so both the phase values and total deadline must change together. No evaluator, RAGAS, job-worker, or legacy v8 timeout is changed.

**Tech Stack:** Python 3.11, Pydantic, asyncio, pytest.

## Global Constraints

- Agentic v9 LLM phases use a 32.0-second cap: `route_plan`, `retrieval_judge`, `evidence_extract`, and `final_answer`.
- The attempt-wide v9 deadline is 32.0 seconds.
- `visual_extract` remains isolated and uses a 16.0-second cap.
- The runtime must continue to clamp every per-phase value to the remaining attempt-wide time.
- Legacy v8, RAGAS, and external provider configuration remain unchanged.

---

### Task 1: Update the v9 timeout contract

**Files:**
- Modify: `data_base/agentic_v9/schemas.py:343-359`
- Test: `tests/test_agentic_v9_execution_policy.py:17-34`

**Interfaces:**
- Consumes: `ExecutionPolicy` defaults.
- Produces: the default timing contract used by `V9ExecutionPolicyRuntime`.

- [ ] **Step 1: Write the failing test**

Change `test_execution_policy_has_the_initial_runtime_bounds` to assert:

```python
assert policy.total_deadline_s == 32.0
assert policy.phase_timeouts_s == {
    "route_plan": 32.0,
    "retrieval_judge": 32.0,
    "evidence_extract": 32.0,
    "visual_extract": 16.0,
    "final_answer": 32.0,
}
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `pytest tests/test_agentic_v9_execution_policy.py::test_execution_policy_has_the_initial_runtime_bounds -q`

Expected: FAIL because the current policy uses `24.0`, `2.0`, `8.0`, `8.0`, and `15.0` defaults.

- [ ] **Step 3: Implement the minimal policy change**

Set `ExecutionPolicy.total_deadline_s` to `32.0` and replace its default factory with:

```python
{
    "route_plan": 32.0,
    "retrieval_judge": 32.0,
    "evidence_extract": 32.0,
    "visual_extract": 16.0,
    "final_answer": 32.0,
}
```

- [ ] **Step 4: Run the focused test and verify it passes**

Run: `pytest tests/test_agentic_v9_execution_policy.py::test_execution_policy_has_the_initial_runtime_bounds -q`

Expected: PASS.

### Task 2: Verify runtime clamping retains the 32-second contract

**Files:**
- Modify: `tests/test_agentic_v9_execution_policy.py:37-50`
- Test: `tests/test_agentic_v9_execution_policy.py:37-50`

**Interfaces:**
- Consumes: `ExecutionDeadline(32.0)` and `V9ExecutionPolicyRuntime(ExecutionPolicy())`.
- Produces: regression coverage proving request caps are 32 seconds before the overall deadline, then clamp to the actual remaining time.

- [ ] **Step 1: Write the failing test expectations**

Update the deadline test to use `ExecutionDeadline(32.0, ...)`, assert `32.0` for both `evidence_extract` and `final_answer` at start, then advance the monotonic clock to `130.5` and assert `1.5` for both. Add an assertion that `visual_extract` is `16.0` at the start.

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `pytest tests/test_agentic_v9_execution_policy.py::test_deadline_clamps_every_phase_timeout_without_resetting -q`

Expected: FAIL while defaults still contain the old phase caps.

- [ ] **Step 3: Keep the implementation minimal**

Do not alter `V9ExecutionPolicyRuntime.timeout_for`; the policy values from Task 1 must make the existing `min(phase_cap, remaining_deadline)` implementation satisfy the test.

- [ ] **Step 4: Run the focused test and verify it passes**

Run: `pytest tests/test_agentic_v9_execution_policy.py::test_deadline_clamps_every_phase_timeout_without_resetting -q`

Expected: PASS.

### Task 3: Verify and commit the bounded change

**Files:**
- Modify: `data_base/agentic_v9/schemas.py`
- Modify: `tests/test_agentic_v9_execution_policy.py`

- [ ] **Step 1: Run the v9 policy and core test suites**

Run: `pytest tests/test_agentic_v9_execution_policy.py tests/test_agentic_v9_execution_core.py -q`

Expected: PASS with no failures.

- [ ] **Step 2: Inspect the diff for scope**

Run: `git diff -- data_base/agentic_v9/schemas.py tests/test_agentic_v9_execution_policy.py`

Expected: only policy defaults and their regression tests changed.

- [ ] **Step 3: Commit the implementation**

Run:

```bash
git add data_base/agentic_v9/schemas.py tests/test_agentic_v9_execution_policy.py docs/superpowers/plans/2026-07-26-v9-provider-deadlines.md
git commit -m "fix(agentic-v9): extend provider phase deadlines"
```
