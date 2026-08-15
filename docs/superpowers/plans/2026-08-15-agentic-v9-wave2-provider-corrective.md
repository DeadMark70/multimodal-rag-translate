# Agentic RAG v9 Wave 2 Provider Corrective Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore real provider-backed evidence qualification, preserve fail-closed sufficiency, and make provider failures visible before rerunning Q1-Q32.

**Architecture:** Keep the existing execution core and batch qualification loop. Repair only the provider request boundary, return a small typed diagnostic outcome from the extractor to the campaign adapter, then synchronize the changed metrics enum through OpenAPI and the frontend decoder/UI.

**Tech Stack:** Python 3.13, Pydantic v2, LangChain Google GenAI, pytest, FastAPI/OpenAPI, TypeScript, Zod, React, Vitest.

## Global Constraints

- Work directly in the existing backend and frontend worktrees.
- Keep Wave 2 deployed but keep campaign test traffic paused until the checkpoint passes.
- Do not start Wave 3 or change final synthesis.
- Do not relax `is_qualified_evidence()` or restore blanket `deterministic_valid` promotion.
- Do not add per-slot or per-chunk provider calls.
- Do not change route, retrieval, repair, final-answer, token, or provider-call limits.
- Do not change Atomic Planner behavior without a separately reproduced server-side cause.
- Every implementation task gets one focused commit in its owning repository.
- Follow TDD: record the focused RED before production edits, then run focused and impacted GREEN gates.

---

### Task 1: Repair the evidence qualification provider and failure propagation

**Files:**
- Modify: `data_base/agentic_v9/provider_boundary.py`
- Modify: `data_base/agentic_v9/evidence_extractor.py`
- Modify: `data_base/agentic_v9/schemas.py`
- Modify: `evaluation/agentic_v9_campaign_runtime.py`
- Test: `tests/test_agentic_v9_evidence_extractor.py`
- Test: `tests/test_agentic_v9_campaign_runtime.py`
- Test: `tests/test_agentic_v9_provider_boundary.py`

**Interfaces:**
- Produces: `evidence_qualification_response_schema() -> dict[str, Any]`
- Produces: `build_evidence_qualification_provider(*, response_schema: Mapping[str, Any]) -> Any`
- Produces: immutable `EvidenceQualificationOutcome` with `packets`, `status`, `failure_code`, `provider_call_attempted`, and `provider_response_received`
- Preserves: `EvidenceExtractor.extract(...) -> list[EvidencePacket]` as a compatibility wrapper
- Adds: `EvidenceExtractor.extract_with_outcome(...) -> EvidenceQualificationOutcome` for campaign execution
- Preserves: execution-core `ProseCuratorStage` return type and all budget ceilings

- [ ] **Step 1: Add provider-boundary and request-shape RED tests**

Add tests that require the new factory to call `bind_json_schema(get_llm("synthesizer"), schema=...)`, reject additional response fields, and build at least one Gemini `contents` entry from the extractor message. The adapter-shape test must construct the request locally and perform no network call.

```python
request = model._prepare_request(messages)
assert len(request["contents"]) == 1
assert request["contents"][0].parts
```

- [ ] **Step 2: Add extractor outcome RED tests**

Cover these exact outcomes:

```text
provider raises                 -> provider_failed / provider_attempt_failed
provider returns malformed JSON -> invalid_response / invalid_provider_response
provider returns {packets: []}  -> no_match / null failure
budget rejects invocation       -> not_attempted / budget_not_admitted
deterministic packet covers all -> deterministic / null failure
provider returns valid packet   -> provider_qualified / null failure
```

Also assert the legacy `extract()` wrapper returns only `outcome.packets`.

- [ ] **Step 3: Add campaign propagation RED tests**

Assert a failed provider attempt produces:

```python
assert metrics.semantic_qualification == "provider_failed"
assert metrics.qualification_failure_code == "provider_attempt_failed"
assert metrics.qualified_packet_count == 0
assert result.sufficiency.response_status == "insufficient"
```

Add Q5 and Q23 positive fixtures with valid provider packets and assert each is answerable with at least one qualified packet.

- [ ] **Step 4: Run the focused RED gate**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_v9_campaign_runtime.py -k "evidence_qualification or qualification_outcome or provider_failed or q5 or q23" -q
```

Expected: failures for the missing schema-bound factory, system-only request, missing outcome API, and obsolete metrics enum.

- [ ] **Step 5: Implement the fixed provider boundary**

Add a strict schema factory and bind it through the existing shared boundary. Update `_provider_for_purpose()` so only `evidence_extraction` selects it. Send the existing formatted qualification prompt as one non-empty user message.

```python
messages = [{"role": "user", "content": formatted_prompt}]
```

- [ ] **Step 6: Implement the minimal typed outcome**

Use one frozen dataclass or Pydantic model local to `evidence_extractor.py`. Do not change the execution core. `extract_with_outcome()` owns exception and parser classification; `extract()` delegates and returns the packet list. Never persist exception text.

- [ ] **Step 7: Propagate outcome metrics in the campaign adapter**

Have `prose_curate()` call `extract_with_outcome()`, write the latest status/failure code to run state, and return only selected packets. Replace the metrics schema enum with:

```python
Literal[
    "not_attempted",
    "deterministic",
    "provider_qualified",
    "no_match",
    "provider_failed",
    "invalid_response",
    "not_enabled",  # legacy persisted runs only
    "not_instrumented",
]
```

Historical exports remain readable; runtime and current-profile smoke tests
must ensure new runs do not emit `not_enabled`.

- [ ] **Step 8: Run focused and impacted GREEN gates**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_v9_evidence_validator.py tests/test_agentic_v9_sufficiency_gate.py tests/test_agentic_v9_execution_core.py tests/test_agentic_v9_campaign_runtime.py -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9/provider_boundary.py data_base/agentic_v9/evidence_extractor.py data_base/agentic_v9/schemas.py evaluation/agentic_v9_campaign_runtime.py tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_v9_campaign_runtime.py
```

Expected: all pass; only pre-existing dependency warnings may remain.

- [ ] **Step 9: Commit Task 1**

```powershell
git add data_base/agentic_v9/provider_boundary.py data_base/agentic_v9/evidence_extractor.py data_base/agentic_v9/schemas.py evaluation/agentic_v9_campaign_runtime.py tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_v9_campaign_runtime.py
git commit -m "fix(agentic-v9): repair evidence qualification provider"
```

---

### Task 2: Add the evidence canary and publish the corrected backend contract

**Files:**
- Create: `scripts/agentic_v9_evidence_qualification_canary.py`
- Create: `tests/test_agentic_v9_evidence_qualification_canary.py`
- Modify: `evaluation/smoke_verification.py`
- Modify: `tests/test_agentic_v9_smoke_runner.py`
- Modify: `tests/test_evaluation_export_v2_schemas.py`
- Modify: `tests/test_evaluation_export_redaction.py`
- Modify: `docs/agentic-v9-smoke-verification.md`
- Modify: `docs/BACKEND.md`
- Regenerate: `openapi.json`
- Regenerate: `contracts/openapi-contract.json`

**Interfaces:**
- Consumes: Task 1 evidence provider factory, schema, and outcome statuses
- Produces: CLI canary with construction-only/default-safe mode and explicit one-call real-server mode
- Produces: OpenAPI enum consumed by frontend Task 3

- [ ] **Step 1: Add canary RED tests**

Require config validation before provider import, no live call in construction mode, `max_retries=0` in one-call mode, the shared evidence provider factory, one non-empty user message, local schema validation, sanitized failures, and nonzero qualified packets for the fixed canary fixture.

- [ ] **Step 2: Add smoke/export RED tests**

Require new runs to reject `semantic_qualification="not_enabled"`, enforce the status/failure invariants, preserve safe failure codes in full observability export, and reject a forged run where a persisted qualification provider failure is represented by a null failure code.

- [ ] **Step 3: Run the focused RED gate**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_evidence_qualification_canary.py tests/test_agentic_v9_smoke_runner.py tests/test_evaluation_export_v2_schemas.py tests/test_evaluation_export_redaction.py -k "qualification or canary or semantic" -q
```

Expected: failures for the absent canary and old smoke/OpenAPI contract.

- [ ] **Step 4: Implement the smallest production-equivalent canary**

Reuse Task 1 code; do not duplicate provider construction, schema, prompt formatting, or parsing. The one-call mode must print only a structured pass/fail summary and safe failure code.

- [ ] **Step 5: Tighten smoke and export invariants**

Validate only current Wave 2 profiles with the new status semantics. Continue tolerating historical profiles and exports containing legacy `not_enabled`.

- [ ] **Step 6: Update backend docs and OpenAPI artifacts**

Document the exact two canary commands and Wave 2 checkpoint order, then run:

```powershell
.\.venv\Scripts\python.exe scripts/sync_openapi_artifacts.py --write
.\.venv\Scripts\python.exe scripts/sync_openapi_artifacts.py --check
```

- [ ] **Step 7: Run the backend GREEN gate**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_evidence_qualification_canary.py tests/test_agentic_v9_smoke_runner.py tests/test_evaluation_retrieval_profiles.py tests/test_evaluation_export_v2_schemas.py tests/test_evaluation_export_redaction.py tests/test_openapi_artifacts.py -q
.\.venv\Scripts\python.exe -m ruff check scripts/agentic_v9_evidence_qualification_canary.py evaluation/smoke_verification.py tests/test_agentic_v9_evidence_qualification_canary.py tests/test_agentic_v9_smoke_runner.py
git diff --check
```

- [ ] **Step 8: Commit Task 2**

```powershell
git add scripts/agentic_v9_evidence_qualification_canary.py tests/test_agentic_v9_evidence_qualification_canary.py evaluation/smoke_verification.py tests/test_agentic_v9_smoke_runner.py tests/test_evaluation_export_v2_schemas.py tests/test_evaluation_export_redaction.py docs/agentic-v9-smoke-verification.md docs/BACKEND.md openapi.json contracts/openapi-contract.json
git commit -m "test(agentic-v9): gate real evidence qualification"
```

---

### Task 3: Synchronize the frontend decoder and trace display

**Files:**
- Modify: `src/types/evaluation.ts`
- Modify: `src/types/evaluation.contract.test.ts`
- Modify: `src/services/evaluationExportSchema.ts`
- Modify: `src/services/evaluationExportSchema.test.ts`
- Modify: `src/components/evaluation/AgenticV9Trace.tsx`
- Create: `src/components/evaluation/AgenticV9Trace.test.tsx`
- Regenerate: `src/test/fixtures/agenticV9ApiContract.ts`

**Interfaces:**
- Consumes: Task 2 OpenAPI enum and exact safe failure-code semantics
- Preserves: export v2 decoder strictness and existing historical display compatibility

- [ ] **Step 1: Pin the new backend contract**

Run:

```powershell
npm run contract:pin
```

Verify the fixture identifies the Task 2 backend commit before editing types.

- [ ] **Step 2: Add decoder and UI RED tests**

Add one valid populated export for each new status, retain `not_enabled` only
for historical export compatibility, and render:

```text
Semantic qualification: provider_failed
Qualified packets: 0 / 8
Qualification calls: 1
Qualification failure: provider_attempt_failed
```

- [ ] **Step 3: Run the frontend RED gate**

Run:

```powershell
npm test -- --run src/services/evaluationExportSchema.test.ts src/types/evaluation.contract.test.ts src/components/evaluation/AgenticV9Trace.test.tsx
```

Expected: enum and missing UI rows fail.

- [ ] **Step 4: Implement exact type/Zod/UI synchronization**

Replace the active metrics enum with the backend values plus legacy
`not_enabled`, and add four simple `MetricLine` rows. Do not add cross-field
decoder machinery, new panels, or state management; current-profile smoke is
the authority that prevents new runs from emitting the legacy value.

- [ ] **Step 5: Run the frontend GREEN gate**

Run:

```powershell
npm test -- --run src/services/evaluationExportSchema.test.ts src/types/evaluation.contract.test.ts src/components/evaluation/AgenticV9Trace.test.tsx src/components/evaluation/AblationDashboardTab.test.tsx
npm run contract:check
npm run lint:ci
npm run build
```

Expected: all tests/checks/build pass; the existing Vite chunk-size advisory is non-blocking.

- [ ] **Step 6: Commit Task 3**

```powershell
git add src/types/evaluation.ts src/types/evaluation.contract.test.ts src/services/evaluationExportSchema.ts src/services/evaluationExportSchema.test.ts src/components/evaluation/AgenticV9Trace.tsx src/components/evaluation/AgenticV9Trace.test.tsx src/test/fixtures/agenticV9ApiContract.ts
git commit -m "fix(evaluation-ui): show qualification provider failures"
```

---

## Final Local Verification

- [ ] Run the combined impacted backend suites from Tasks 1 and 2.
- [ ] Run OpenAPI `--check`, scoped Ruff, and `git diff --check`.
- [ ] Run the combined frontend Vitest set, `contract:check`, `lint:ci`, and production build.
- [ ] Confirm both worktrees are clean and each task has exactly one focused commit.
- [ ] Do not run a real provider call locally unless the user explicitly supplies/authorizes the deployed model configuration.
- [ ] Keep Wave 3 blocked until the deployed planner canary, evidence canary, Q5/Q23 campaign, and Q1-Q32 checkpoint all pass.
