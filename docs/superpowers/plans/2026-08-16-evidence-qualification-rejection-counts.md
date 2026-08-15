# Evidence Qualification Rejection Counts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist three safe rejection counts that distinguish unknown evidence IDs, unauthorized source-slot bindings, and non-verbatim statements.

**Architecture:** The extractor classifies row rejection at its existing validation boundary and returns counts in `EvidenceQualificationOutcome`. Campaign runtime aggregates counts across qualification rounds into canonical `V9ExecutionMetrics`; Export v2 reuses that model. The frontend strict decoder mirrors the resulting OpenAPI contract.

**Tech Stack:** Python 3.13, Pydantic v2, pytest, React/TypeScript, Zod, Vitest.

## Global Constraints

- Work directly on the current backend and frontend main worktrees as explicitly requested.
- Do not change model selection, prompts, Structured Outputs schemas, budgets, evidence authorization, quote validation, sufficiency, or final generation.
- Do not persist provider response text, prompts, questions, answers, statements, source contents, or exception messages.
- Reuse existing test modules; create no new test module.

---

### Task 1: Backend rejection classification and persistence

**Files:**
- Modify: `data_base/agentic_v9/evidence_extractor.py`
- Modify: `data_base/agentic_v9/schemas.py`
- Modify: `evaluation/agentic_v9_campaign_runtime.py`
- Modify: `tests/test_agentic_v9_evidence_extractor.py`
- Modify: `tests/test_agentic_v9_campaign_runtime.py`
- Modify: `tests/test_evaluation_export_v2_schemas.py`
- Generated: `openapi.json`
- Generated: `contracts/openapi-contract.json`

**Interfaces:**
- Produces three `int >= 0` fields named exactly `qualification_unknown_source_id_count`, `qualification_unauthorized_source_slot_count`, and `qualification_statement_not_verbatim_count` on `EvidenceQualificationOutcome` and `V9ExecutionMetrics`.
- Export v2 receives the fields through its existing `V9ExecutionMetrics` owner.

- [ ] **Step 1: Write focused failing extractor and runtime assertions**

Extend the existing mixed-row extractor test so one row in each rejection category and one valid row produce literal counts `1, 1, 1`. Extend the existing two-round campaign fixture so the final run metrics prove round aggregation.

- [ ] **Step 2: Run RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_agentic_v9_evidence_extractor.py tests\test_agentic_v9_campaign_runtime.py -k "rejection or qualification" -q
```

Expected: failures because the three fields do not exist.

- [ ] **Step 3: Implement minimal classification and aggregation**

Add immutable integer fields with zero defaults to `EvidenceQualificationOutcome`. Increment only at the existing dynamic validation branches and when `validate_prose_packet(...).reason == "statement_is_not_a_verbatim_source_span"`. Initialize three runtime counters to zero, add each outcome's values per round, and project them into `V9ExecutionMetrics`.

- [ ] **Step 4: Lock canonical/export shape and regenerate OpenAPI**

Add the same zero-default fields to `V9ExecutionMetrics`, update existing export fixture assertions, then run:

```powershell
.\.venv\Scripts\python.exe scripts\sync_openapi_artifacts.py --write
```

- [ ] **Step 5: Run backend GREEN and commit**

Run the affected extractor/runtime/export/OpenAPI tests and scoped Ruff. Commit only backend production, tests, and generated artifacts as `feat(agentic-v9): expose qualification rejection counts`.

### Task 2: Frontend strict contract synchronization

**Files:**
- Modify: `src/types/evaluation.ts`
- Modify: `src/services/evaluationExportSchema.ts`
- Modify: `src/services/evaluationExportSchema.test.ts`
- Modify: `src/types/evaluation.contract.test.ts`
- Generated: `src/test/fixtures/agenticV9ApiContract.ts`

**Interfaces:**
- Consumes the three exact backend field names from Task 1.
- Export v2 requires non-negative integers; interactive/agent behavior projections remain nullable or optional according to their backend OpenAPI schema.

- [ ] **Step 1: Pin the new backend contract and write failing strict-decoder assertions**

Run `npm run contract:pin`, then update existing valid export fixtures to include literal counts and add one negative mutation for a negative count.

- [ ] **Step 2: Run RED**

Run:

```powershell
npm test -- --run src/services/evaluationExportSchema.test.ts src/types/evaluation.contract.test.ts
```

Expected: strict-object/type failures because production TypeScript/Zod owners lack the new fields.

- [ ] **Step 3: Synchronize TypeScript and Zod owners**

Add the exact three properties to `V9ExecutionMetrics`, `AgentBehaviorRow`, `v9ExecutionMetricsSchema`, and `behaviorV9MetricsSchema`, preserving each owner's existing required/nullable semantics.

- [ ] **Step 4: Run frontend GREEN and commit**

Run focused Vitest, `npm run contract:check`, `npm run lint:ci`, and `npm run build`. Commit only frontend contract, decoder, tests, and pinned fixture as `fix(evaluation-ui): decode qualification rejection counts`.
