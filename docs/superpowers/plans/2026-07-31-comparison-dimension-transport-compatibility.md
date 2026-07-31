# Comparison Dimension Transport Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow safe Gemini object-shaped comparison dimensions without weakening the trusted planner contract.

**Architecture:** Add one provider-boundary `mode="before"` Pydantic validator to `_PlannerPayload`. It converts narrowly allowlisted object shapes into strings; all downstream models and guards remain unchanged.

**Tech Stack:** Python 3.13, Pydantic v2, pytest, Ruff

## Global Constraints

- Do not modify retrieval, reranking, synthesis, subject validation, or numeric guards.
- Do not persist raw provider dimension values.
- Preserve fail-soft fallback for malformed dimensions.
- Work directly on `main` as explicitly authorized by the user.

---

### Task 1: Normalize provider dimension objects

**Files:**
- Modify: `data_base/agentic_v9/comparison_planner.py`
- Test: `tests/test_agentic_v9_comparison_planner.py`

**Interfaces:**
- Consumes: provider JSON field `dimensions: list[unknown]`
- Produces: `_PlannerPayload.dimensions: list[str]`

- [ ] **Step 1: Write failing parameterized tests**

Add tests proving that `{"name": "FLOPs"}` and an object with one preferred
label plus descriptive metadata become trusted string dimensions, while
conflicting supported labels and unsupported scalar/object values remain
`transport_schema` fallbacks.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_agentic_v9_comparison_planner.py -k "dimension_object" -q
```

Expected: the valid object-shaped response falls back with
`dimensions.0/string_type`.

- [ ] **Step 3: Implement minimal boundary normalization**

Add a `field_validator("dimensions", mode="before")` and a focused helper.
The helper returns strings, selects only the allowlisted label keys, and raises
`ValueError` for unsupported or conflicting inputs.

- [ ] **Step 4: Verify GREEN and regressions**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_agentic_v9_comparison_planner.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_agentic_v9_campaign_runtime.py tests\test_evaluation_v9_attempt_persistence.py tests\test_evaluation_export_redaction.py tests\test_agentic_v9_smoke_runner.py -q
.\.venv\Scripts\python.exe -m ruff check data_base\agentic_v9\comparison_planner.py tests\test_agentic_v9_comparison_planner.py
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 5: Commit**

```powershell
git add data_base/agentic_v9/comparison_planner.py tests/test_agentic_v9_comparison_planner.py docs/superpowers/specs/2026-07-31-comparison-dimension-transport-compatibility-design.md docs/superpowers/plans/2026-07-31-comparison-dimension-transport-compatibility.md
git commit -m "fix(agentic-v9): normalize planner dimensions"
```
