# Comparison Planner Safe Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist safe, stage-specific comparison-planner validation diagnostics so a three-question production smoke identifies the exact Gemini incompatibility without changing planner behavior.

**Architecture:** Extend the typed planner outcome with a bounded diagnostic stage and safe validation issue records. Populate them only at existing fallback boundaries, propagate them through the v9 runtime comparison projection, and allowlist them in durable observability/export projection.

**Tech Stack:** Python 3.11, Pydantic v2, pytest, existing Agentic v9 runtime and evaluation observability storage.

## Global Constraints

- Do not change prompts, planner acceptance/rejection decisions, retrieval, reranking, repair, synthesis, provider calls, or timeouts.
- Never persist raw response text, invalid input values, exception messages, source names, document IDs, Pydantic URLs, or validation context.
- Preserve the existing `status` and `fallback_reason` contract for all consumers.
- Bound diagnostics to eight deduplicated, sorted `{path, type}` records.

---

### Task 1: Typed planner-boundary diagnostics

**Files:**
- Modify: `data_base/agentic_v9/schemas.py:48-200`
- Modify: `data_base/agentic_v9/comparison_planner.py:113-370`
- Test: `tests/test_agentic_v9_comparison_planner.py`

**Interfaces:**
- Produces: `ComparisonPlannerDiagnosticStage`, `ComparisonPlannerValidationIssue`, and optional `fallback_stage` / `validation_issues` fields on `ComparisonPlannerOutcome`.
- Produces: `_validation_issues(error: ValidationError) -> list[ComparisonPlannerValidationIssue]` that serializes only normalized locations and stable Pydantic error types.
- Consumes: existing `ComparisonPlannerFallbackReason`, `_PlannerPayload`, `ComparisonPlan`, and `ComparisonPlannerOutcome`.

- [ ] **Step 1: Write failing outcome and transport-schema tests**

Add tests that pass a payload whose first subject lacks `subject_role` and contains a secret invalid value. Assert the outcome remains `fallback/schema_violation`, has `fallback_stage == "transport_schema"`, serializes `[{"path": "subjects.0.subject_role", "type": "missing"}]`, and does not contain the secret value, Pydantic message, URL, or context.

Also construct a planned `ComparisonPlannerOutcome` with diagnostics and assert Pydantic rejects it, preserving success/fallback consistency.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_agentic_v9_comparison_planner.py -q
```

Expected: failures because diagnostic types and outcome fields do not exist.

- [ ] **Step 3: Implement typed bounded diagnostics**

Add these exact public types in `schemas.py`:

```python
ComparisonPlannerDiagnosticStage = Literal[
    "response_decode",
    "transport_schema",
    "subject_validation",
    "trusted_plan_validation",
    "numeric_guard",
]

class ComparisonPlannerValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, max_length=160)
    type: str = Field(min_length=1, max_length=80)
```

Extend `ComparisonPlannerOutcome` with optional `fallback_stage` and a list of at most eight issues. Require planned outcomes to have neither field and permit diagnostics only on fallback outcomes.

In `comparison_planner.py`, convert `ValidationError.errors(include_url=False, include_context=False, include_input=False)` into safe issues. Join integer/string locations with dots, replace unsupported path/type characters with `_`, deduplicate, sort, and cap at eight.

Populate stages without changing decisions:

- JSON/type decode failure → `response_decode`, no issues.
- `_PlannerPayload.model_validate` failure → `transport_schema`, safe issues.
- subject-count/validation failure → `subject_validation`, no issues.
- `ComparisonPlan` Pydantic failure → `trusted_plan_validation`, safe issues.
- `_reject_invented_numbers` failure → `numeric_guard`, no issues.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same focused command. Expected: all planner tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add data_base/agentic_v9/schemas.py data_base/agentic_v9/comparison_planner.py tests/test_agentic_v9_comparison_planner.py
git commit -m "feat(agentic-v9): classify planner validation failures"
```

---

### Task 2: Runtime and redacted export propagation

**Files:**
- Modify: `evaluation/agentic_v9_campaign_runtime.py:220-320,1035-1090`
- Modify: `evaluation/observability_storage.py:180-375`
- Test: `tests/test_agentic_v9_campaign_runtime.py`
- Test: `tests/test_evaluation_v9_attempt_persistence.py`

**Interfaces:**
- Consumes: `ComparisonPlannerOutcome.fallback_stage` and `.validation_issues` from Task 1.
- Produces: `comparison.fallback_stage` and `comparison.validation_issues` in the sanitized durable trace/export projection.

- [ ] **Step 1: Write failing runtime and allowlist tests**

In the runtime test, return a transport-invalid planner payload and assert `agent_trace.agentic_v9.comparison_planner` plus the final `comparison` projection contain `fallback_stage` and the safe issue list.

In the observability test, call `safe_comparison_projection` with valid diagnostics plus extra keys, raw values, and secret text. Assert only the allowlisted stage and `{path, type}` survive, entries are capped at eight, and serialized output contains none of the secret fields or values.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_agentic_v9_campaign_runtime.py tests\test_evaluation_v9_attempt_persistence.py -q
```

Expected: failures because runtime/projection omit diagnostic fields.

- [ ] **Step 3: Propagate and sanitize diagnostics**

Initialize runtime planner state with `fallback_stage: None` and `validation_issues: []`. Copy the typed outcome fields after planner completion and include them in `_comparison_projection`.

In `safe_comparison_projection`, allowlist only the five documented stages. Accept at most eight mapping entries; keep only `path` and `type` strings bounded to 160 and 80 characters. Reject unknown stages as `unknown` and discard every other diagnostic key.

- [ ] **Step 4: Run focused and regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_agentic_v9_campaign_runtime.py tests\test_evaluation_v9_attempt_persistence.py tests\test_evaluation_export_redaction.py tests\test_agentic_v9_smoke_runner.py -q
```

Expected: all tests pass and provider-call-count assertions remain unchanged.

- [ ] **Step 5: Run lint and full related verification**

```powershell
.\.venv\Scripts\python.exe -m ruff check data_base\agentic_v9\schemas.py data_base\agentic_v9\comparison_planner.py evaluation\agentic_v9_campaign_runtime.py evaluation\observability_storage.py tests\test_agentic_v9_comparison_planner.py tests\test_agentic_v9_campaign_runtime.py tests\test_evaluation_v9_attempt_persistence.py
.\.venv\Scripts\python.exe -m pytest tests\test_agentic_v9_comparison_planner.py tests\test_agentic_v9_campaign_runtime.py tests\test_evaluation_v9_attempt_persistence.py tests\test_evaluation_export_redaction.py tests\test_agentic_v9_smoke_runner.py -q
git diff --check
```

- [ ] **Step 6: Commit Task 2**

```powershell
git add evaluation/agentic_v9_campaign_runtime.py evaluation/observability_storage.py tests/test_agentic_v9_campaign_runtime.py tests/test_evaluation_v9_attempt_persistence.py
git commit -m "feat(evaluation): export safe planner diagnostics"
```

---

### Task 3: Production evidence checkpoint

**Files:**
- No code changes.

**Interfaces:**
- Consumes: a newly exported Q3/Q4/Q14 campaign after deploying Tasks 1 and 2.
- Produces: the exact failing boundary and safe validation paths used to design the separate compatibility fix.

- [ ] **Step 1: Deploy the committed backend changes**

Rebuild/restart the backend so its Git HEAD contains both diagnostics commits.

- [ ] **Step 2: Run one smoke campaign**

Run Q3, Q4, and Q14 with `agentic-v9`, repeat 1, and batch size 1.

- [ ] **Step 3: Export redacted JSON and inspect diagnostics**

Verify every planner fallback has `fallback_stage` and, for Pydantic failures, bounded `validation_issues`. Stop here: do not alter planner acceptance until this production evidence is available.
