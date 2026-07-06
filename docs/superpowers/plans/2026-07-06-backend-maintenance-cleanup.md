# Backend Maintenance Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Converge GraphRAG maintenance code out of routers, tighten backend boundary guards, and establish a Python 3.13 verification baseline.

**Architecture:** Routers should own HTTP request/response/auth only. Long-running GraphRAG maintenance jobs and shared graph file operations should live in `graph_rag/maintenance.py`, while static tests enforce router boundaries.

**Tech Stack:** Python 3.13, FastAPI, Pytest, Ruff, AST-based static tests.

---

### Task 1: Baseline and Current Diff

**Files:**
- Inspect: `graph_rag/router.py`
- Inspect: `graph_rag/maintenance.py`
- Inspect: `tests/test_graph_router_rebuild_full.py`
- Inspect: `tests/test_graph_router_copy.py`

- [ ] Review current diff and existing GraphRAG maintenance tests.
- [ ] Run focused baseline tests before expanding the patch.

### Task 2: Converge GraphRAG Maintenance Helpers

**Files:**
- Modify: `graph_rag/maintenance.py`
- Modify: `graph_rag/router.py`
- Modify tests as needed.

- [ ] Move shared graph maintenance helpers into `graph_rag/maintenance.py` as the single implementation.
- [ ] Keep router-level aliases only where existing tests or endpoint code need stable patch seams.
- [ ] Preserve endpoint paths, response models, and auth dependencies.
- [ ] Run GraphRAG focused tests.

### Task 3: Add Boundary Guards

**Files:**
- Modify: `tests/test_router_boundaries.py`

- [ ] Keep non-router production modules from importing routers.
- [ ] Add a guard that router modules do not define long-running background task functions.
- [ ] Keep allowlists explicit for app assembly and package exports.

### Task 4: Pytest Cache Warning

**Files:**
- Modify: `pytest.ini` only if needed.

- [ ] Investigate `.pytest_cache` permission warnings.
- [ ] Prefer a workspace-local writable cache path or disable cache provider if the warning is environment-specific and harmless.
- [ ] Verify focused pytest output no longer includes cache write warnings.

### Task 5: Verification Baseline

**Files:**
- No source change unless failures reveal a scoped issue.

- [ ] Run focused backend tests for touched scope.
- [ ] Run FastAPI app import smoke with `TEST_MODE=true` and `USE_FAKE_PROVIDERS=true`.
- [ ] Run a broader backend pytest baseline on Python 3.13 and report pass/fail counts.
- [ ] Run read-only subagent review after implementation.
