# Agentic RAG v9 Grounded Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover Agentic RAG v9 correctness, faithfulness, and relevancy by fixing the three verified breaks in order: degraded Atomic Contract retrieval, disconnected evidence qualification, and disconnected structured final synthesis.

**Architecture:** Preserve the active Query Contract v2, deterministic route, authorized source scope, retrieval/reranking stack, and execution budgets. Wave 1 makes contract-planner failures diagnosable and makes degraded retrieval question-specific. Wave 2 reconnects the existing batch `EvidenceExtractor`/validator before sufficiency. Wave 3 reconnects the existing typed final-answer path through a compact synthesis projection and derives claims/status/used evidence from accepted findings. Each Wave is independently deployable and ends at a hard checkpoint for a real Q1-Q32 campaign.

**Tech Stack:** Python 3.13, Pydantic v2, FastAPI, LangChain Google GenAI, pytest/pytest-asyncio, Ruff, OpenAPI artifact generation, React 18, TypeScript, Zod, Vitest.

## Source of Truth and Supersession

- Retain the domain decisions from `docs/superpowers/specs/2026-08-14-agentic-v9-active-atomic-contract-design.md` and the completed Active Atomic Contract implementation.
- This plan supersedes the execution order in `docs/superpowers/plans/2026-08-14-agentic-v9-faithfulness-recovery.md` wherever that plan reconnects qualification and final synthesis in one deployment.
- Do not restore the reverted Wave 2 as one atomic change. Its observed failure mode is now a required regression test.
- Runtime truth comes from production code and exported observability, not the local SQLite database.

## Verified Baseline

The reference observability export is:

```text
D:\flutterserver\evl_json\314ab70d-e045-433f-b99a-4f9ab7ff7344-observability-custom-v2.json
SHA256: 1012B05FC081DB594D3A93A85DD59B765CD8DE19F02EA65C781EE704882063AA
```

It contains 64 runs: 32 Agentic and 32 Naive.

| Metric | Agentic | Naive |
| --- | ---: | ---: |
| Correctness | 0.3603 | 0.4848 |
| Faithfulness | 0.5729 | 0.8115 |
| Relevancy | 0.3800 | 0.5982 |
| Accounting | 14 complete / 18 partial | complete |

Verified causal evidence:

1. All 32 Agentic contracts are v2, but planner provenance is 11 deterministic, 21 safe fallback, and 0 successful LLM planner outcomes.
2. The 21 safe fallbacks compile to the same query: `Resolve the complete source-bound requirement in the original question.`
3. Expected-source packet match is 2/84 for safe fallback and 50/76 for deterministic planning.
4. Fourteen `contract_planning` provider calls failed while 28 `final_answer` and one graph provider call succeeded. This is a planner-stage failure, not evidence of a global provider outage.
5. The exported error is currently collapsed to `provider_attempt_failed`; the exact provider/schema failure is not observable.
6. `validate_post_contract_feasibility` reserves an evidence-extraction provider call from the contract flag even though the current campaign `prose_curate` adapter is deterministic/no-op. Seven runs were therefore rejected as `semantic_planning_not_admitted` for work that was not actually called.
7. All 160 packets are marked `deterministic_valid`, but none has `source_span_hash`, `extractor_version`, or `prompt_version`. Slot ownership is inherited from retrieval tasks, not semantically proven.
8. The current final provider prompt only contains `Question` and concatenated `Evidence`; successful Agentic runs create one whole-answer claim with no slot ID and mark every packed evidence ID as used.
9. The reverted Wave 2 campaign produced 64/64 Agentic `insufficient`, empty contexts, correctness 0.1947, faithfulness 0.8906, and relevancy 0.0. Its 116 evidence-extraction calls had missing usage/provider failures. Fail-closed qualification must therefore have a provider canary and positive-control gate before deployment.
10. Existing reusable components already implement the intended ordering and typed behavior: `EvidenceExtractor`, `evidence_validator`, `V9ExecutionCore`, and `FinalAnswerRenderer`. The campaign adapter is the disconnected boundary.

Required regression questions:

- Positive controls: Q5 and Q23.
- Retrieval-fallback proof: Q13 and Q24.
- Relevant-evidence/unsupported-claim proof: Q18.
- Sufficiency false-negative proof: Q20 and Q29.
- Q24 must cite the actual SegVol Table 3 values if qualified evidence is retrieved; otherwise it must be partial, never invent a replacement value.

## Global Constraints

- Work in the existing main worktrees. Do not create a Git worktree.
- Follow TDD for every behavior change: add a focused test, capture the expected RED result, implement the minimum change, then rerun the same test GREEN.
- Each Task ends with exactly one commit in the repository it changes. Do not combine backend and frontend changes in one commit.
- Stop after every Wave. The user pushes and runs a real-system Q1-Q32 campaign before the next Wave begins.
- Active Atomic Contract v2 is mandatory. Do not add a feature flag, v1 runtime fallback, corpus membership gate, or golden/expected-route dependency.
- Preserve `RoutePlanner`, `RouteDecision`, authorized documents, graph/visual policy, deadline, repair cap, retrieval/reranking behavior, and final-answer reserve.
- Do not change Native RAG, v8, chat, ingestion, RAGAS scoring, or ground-truth handling.
- No per-chunk or per-slot LLM calls. Evidence qualification is one batch per qualification round.
- Wave 2 may perform one initial batch qualification and, only after actual repair retrieval, one further batch qualification through the existing bounded repair loop.
- Final generation remains at most one call. The high-risk claim verifier remains at most one call per run.
- Do not add an automatic token-ratio gate. The existing Agentic/Naive runtime-token ratio target `<= 3.0` is measured manually at each real-system checkpoint.
- A provider failure may reduce a run to `qualified_partial` or `insufficient`; it must never promote raw evidence to qualified evidence.
- Do not put raw provider exception text, credentials, prompts, answers, or source excerpts into non-redacted diagnostics.
- Any backend export/OpenAPI change must be consumed by the strict frontend TypeScript/Zod contract in the same Wave before that Wave is complete.

## Wave Map

| Wave | Behavioral change | Deliberately unchanged | Stop condition |
| --- | --- | --- | --- |
| 1 | Diagnose planner failure; make safe fallback query-specific; reserve only real provider calls | Evidence remains current behavior; final prompt remains current behavior | Planner failure classified, Q13/Q24 no longer share generic query, Q5/Q23 unaffected |
| 2 | Reconnect batch evidence extraction/validation before sufficiency | Current simple final prompt/claim behavior | Positive controls remain answerable; raw/unqualified packets cannot satisfy slots; not all runs collapse to insufficient |
| 3 | Reconnect compact structured final synthesis and exact claim/evidence accounting | Route, retrieval, qualification, repair policy | Natural-language answers, exact used IDs, honest complete/partial status, Q18 unsupported rationale blocked |

---

# Wave 1 — Planner Diagnosis and Retrieval Stabilization

## Task 1: Add behavior-neutral planner and retrieval diagnostics

**Repository:** `D:\flutterserver\pdftopng`

**Files:**
- Modify: `data_base/agentic_v9/schemas.py`
- Modify: `data_base/agentic_v9/contract_planner.py`
- Modify: `data_base/agentic_v9/budgeted_llm.py`
- Modify: `evaluation/agentic_v9_campaign_runtime.py`
- Modify: `evaluation/campaign_schemas.py`
- Modify: `evaluation/research_analytics.py`
- Modify: `evaluation/export_schemas.py`
- Modify: `evaluation/export_service.py`
- Modify: `tests/test_agentic_v9_contract_planner.py`
- Modify: `tests/test_agentic_v9_budgeted_llm.py`
- Modify: `tests/test_agentic_v9_campaign_runtime.py`
- Modify: `tests/test_evaluation_analytics_context.py`
- Modify: `tests/test_evaluation_export_v2_schemas.py`
- Modify: `tests/test_evaluation_export_redaction.py`

**Interfaces:**

Add an exact nested diagnostic model owned by `data_base/agentic_v9/schemas.py`:

```python
AtomicPlannerOutcome = Literal["deterministic", "planned", "degraded"]
AtomicPlannerFailureStage = Literal[
    "budget_rejected",
    "provider_invocation",
    "provider_empty_response",
    "response_decode",
    "schema_validation",
    "semantic_validation",
]

class AtomicPlannerDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    outcome: AtomicPlannerOutcome
    failure_stage: AtomicPlannerFailureStage | None = None
    failure_code: str | None = Field(default=None, max_length=96)
    provider_response_received: bool
    retrieval_query_strategy: Literal[
        "atomic_slots", "safe_fallback_original_question"
    ]
    compiled_retrieval_task_count: int = Field(ge=0)
```

`failure_code` is a stable application code such as `provider_attempt_failed`, `invalid_json`, `pydantic_validation_failed`, or `planner_semantic_rejection`; it is never the raw exception string.

- [ ] **Step 1: Add RED tests for exact failure classification**

Cover provider exception, empty content, malformed JSON, valid JSON with Pydantic mismatch, semantically invalid slots, budget rejection, deterministic success, and planner success. Assert that each produces exactly one outcome/stage/code combination and that raw secrets in an exception are absent from model dumps and Export v2.

- [ ] **Step 2: Run RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_budgeted_llm.py tests/test_agentic_v9_campaign_runtime.py tests/test_evaluation_analytics_context.py tests/test_evaluation_export_v2_schemas.py tests/test_evaluation_export_redaction.py -k "planner_diagnostic or planner_failure or retrieval_query_strategy" -q
```

Expected: failures because the diagnostic type and stage-specific mapping do not exist.

- [ ] **Step 3: Preserve the exception boundary while classifying it**

Introduce internal typed exceptions or results at the narrow boundaries:

```python
class PlannerProviderInvocationError(RuntimeError): ...
class PlannerResponseDecodeError(ValueError): ...
class PlannerSchemaValidationError(ValueError): ...
class PlannerSemanticValidationError(ValueError): ...
```

`BudgetedLlmInvoker` continues to sanitize external errors, but records a stable failure code. `QuestionContractPlanner` maps only these known boundaries to `AtomicPlannerDiagnostics`; it does not expose provider messages.

- [ ] **Step 4: Persist and export the typed diagnostic**

Add the diagnostic to the canonical `V9ExecutionObservability` in `evaluation/campaign_schemas.py`, assemble it through `evaluation/research_analytics.py`, and add it to the export-owned fixed projection. Keep it additive/nullable for historical rows. Project it through the sole redaction/export owner; do not use `dict[str, Any]`.

- [ ] **Step 5: Run GREEN and Ruff**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_budgeted_llm.py tests/test_agentic_v9_campaign_runtime.py tests/test_evaluation_analytics_context.py tests/test_evaluation_export_v2_schemas.py tests/test_evaluation_export_redaction.py -k "planner_diagnostic or planner_failure or retrieval_query_strategy" -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9/schemas.py data_base/agentic_v9/contract_planner.py data_base/agentic_v9/budgeted_llm.py evaluation/agentic_v9_campaign_runtime.py evaluation/campaign_schemas.py evaluation/research_analytics.py evaluation/export_schemas.py evaluation/export_service.py tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_budgeted_llm.py tests/test_evaluation_analytics_context.py
```

- [ ] **Step 6: Commit Task 1**

```powershell
git add data_base/agentic_v9/schemas.py data_base/agentic_v9/contract_planner.py data_base/agentic_v9/budgeted_llm.py evaluation/agentic_v9_campaign_runtime.py evaluation/campaign_schemas.py evaluation/research_analytics.py evaluation/export_schemas.py evaluation/export_service.py tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_budgeted_llm.py tests/test_agentic_v9_campaign_runtime.py tests/test_evaluation_analytics_context.py tests/test_evaluation_export_v2_schemas.py tests/test_evaluation_export_redaction.py
git commit -m "fix(agentic-v9): classify atomic planner failures"
```

## Task 2: Make safe fallback retrieval question-specific

**Repository:** `D:\flutterserver\pdftopng`

**Files:**
- Modify: `data_base/agentic_v9/contract_planner.py`
- Modify: `data_base/agentic_v9/retrieval_tasks.py`
- Modify: `tests/test_agentic_v9_contract_planner.py`
- Modify: `tests/test_agentic_v9_retrieval_tasks.py`
- Modify: `tests/test_agentic_v9_campaign_runtime.py`

**Design:** The fallback contract stays v2/degraded and keeps the route, scope, budget, graph/visual policy, and repair limits. Its single `S1.description` becomes the normalized original question. The existing `_atomic_query()` then compiles that description normally. Do not add a fallback special case inside the compiler.

- [ ] **Step 1: Add RED tests**

```python
outcome = await planner.plan(
    question="What exact values does SegVol Table 3 report?",
    base_contract=base_contract,
    preparation=low_confidence,
    allow_semantic_planning=False,
)
assert outcome.contract.required_slots[0].slot_id == "S1"
assert outcome.contract.required_slots[0].description == (
    "What exact values does SegVol Table 3 report?"
)
tasks = compile_retrieval_tasks(outcome.contract)
assert tasks[0].query == "What exact values does SegVol Table 3 report?"
```

Add a second unrelated question and assert its query differs. Replay Q13 and Q24 and assert neither compiles the old generic sentence. Assert deterministic Q5/Q23 compiler output is unchanged.

- [ ] **Step 2: Run RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_campaign_runtime.py -k "safe_fallback or q5 or q13 or q23 or q24" -q
```

- [ ] **Step 3: Implement the minimal fallback change**

Normalize only whitespace and length-bound the question using the same contract field constraint. Do not summarize, translate, inject source names, or infer expected answers. Set diagnostic strategy to `safe_fallback_original_question`.

- [ ] **Step 4: Run GREEN and Ruff**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_campaign_runtime.py -k "safe_fallback or q5 or q13 or q23 or q24" -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9/contract_planner.py data_base/agentic_v9/retrieval_tasks.py tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_retrieval_tasks.py
```

- [ ] **Step 5: Commit Task 2**

```powershell
git add data_base/agentic_v9/contract_planner.py data_base/agentic_v9/retrieval_tasks.py tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_campaign_runtime.py
git commit -m "fix(agentic-v9): preserve the question in fallback retrieval"
```

## Task 3: Reserve only real post-contract provider work

**Repository:** `D:\flutterserver\pdftopng`

**Files:**
- Modify: `data_base/agentic_v9/budget_feasibility.py`
- Modify: `evaluation/agentic_v9_campaign_runtime.py`
- Modify: `tests/test_agentic_v9_budget_feasibility.py`
- Modify: `tests/test_agentic_v9_campaign_runtime.py`

**Interface:** Replace inference from `contract.evidence_extraction_required` with an explicit count:

```python
def validate_post_contract_feasibility(
    ...,
    evidence_qualification_provider_calls: int = 0,
) -> FeasibilityResult:
```

Wave 1 passes `0` because its campaign qualification adapter performs no provider call. Wave 2 will pass `1` for the initial batch. Optional repair qualification is admitted dynamically by the existing controller and must retain the final-answer reserve.

- [ ] **Step 1: Add RED feasibility tests**

Assert the same contract is feasible with `evidence_qualification_provider_calls=0`, reserves exactly one `evidence_extract` call with `1`, rejects negative or greater-than-one initial values, and still protects final generation. Assert the contract flag alone no longer reserves a provider call.

- [ ] **Step 2: Run RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_campaign_runtime.py -k "qualification_provider_calls or semantic_planning_not_admitted" -q
```

- [ ] **Step 3: Implement the explicit parameter and Wave 1 wiring**

Remove only the false inference. Do not change phase token limits, route budgets, or the controller's repair/final reserve logic.

- [ ] **Step 4: Run GREEN and Ruff**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_campaign_runtime.py -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9/budget_feasibility.py evaluation/agentic_v9_campaign_runtime.py tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_campaign_runtime.py
```

- [ ] **Step 5: Commit Task 3**

```powershell
git add data_base/agentic_v9/budget_feasibility.py evaluation/agentic_v9_campaign_runtime.py tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_campaign_runtime.py
git commit -m "fix(agentic-v9): budget actual qualification calls"
```

## Task 4: Add a production-equivalent contract-planner canary

**Repository:** `D:\flutterserver\pdftopng`

**Files:**
- Modify: `core/llm_factory.py`
- Create: `data_base/agentic_v9/provider_boundary.py`
- Modify: `evaluation/agentic_v9_campaign_runtime.py`
- Create: `scripts/agentic_v9_contract_planner_canary.py`
- Create: `tests/test_agentic_v9_contract_planner_canary.py`
- Modify: `tests/test_agentic_v9_provider_boundary.py`
- Modify: `tests/test_llm_factory_override.py`
- Modify: `docs/agentic-v9-smoke-verification.md`
- Modify: `docs/superpowers/plans/2026-08-15-agentic-v9-grounded-recovery.md`

**Shared provider boundary:** Create `build_contract_planning_provider(*, response_schema: Mapping[str, Any]) -> Any` in `data_base/agentic_v9/provider_boundary.py`. It is the sole owner of `get_llm("synthesizer")` plus `bind_json_schema(...)`. The campaign `_provider_for_purpose("atomic_contract_planning")` and both canary schema modes must call this helper. No second provider builder is allowed.

**Canary contract:** One command supports `--schema current` and `--schema minimal`, requires `--model-config-json <path>`, makes exactly one wire-level provider attempt per invocation, and prints only JSON containing success, failure stage/code, package versions, model identifier, and response-received boolean. The JSON file must validate through the canonical `evaluation.schemas.ModelConfig` before importing provider-dependent modules; the canary then applies `normalize_model_config_for_runtime(...)`, `llm_runtime_override(..., max_retries=0)`, and the resolved `contract_planning` phase policy before calling the shared provider boundary. Add `max_retries` as an optional task-local runtime override in `core/llm_factory.py`; normal campaign execution does not set it and retains the existing provider default. It never prints the config body, prompts, response bodies, keys, raw exceptions, or import tracebacks.

- [ ] **Step 1: Add RED canary tests with fake providers**

Test exit 0 for valid structured content, nonzero for each classified failure, one wrapper invocation, `max_retries=0` at actual provider construction, sanitized output, invalid/missing model-config rejection before importing the provider stack, sanitized import/setup failure, and parity proving runtime/current/minimal modes all use the shared provider boundary.

- [ ] **Step 2: Implement the canary and run local unit GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_contract_planner_canary.py tests/test_agentic_v9_provider_boundary.py -q
.\.venv\Scripts\python.exe -m ruff check core/llm_factory.py data_base/agentic_v9/provider_boundary.py evaluation/agentic_v9_campaign_runtime.py scripts/agentic_v9_contract_planner_canary.py tests/test_agentic_v9_contract_planner_canary.py tests/test_agentic_v9_provider_boundary.py tests/test_llm_factory_override.py
```

- [ ] **Step 3: Document the two-call staging checkpoint without running it locally**

```powershell
.\.venv\Scripts\python.exe scripts/agentic_v9_contract_planner_canary.py --schema current --model-config-json <real-server-model-config.json>
.\.venv\Scripts\python.exe scripts/agentic_v9_contract_planner_canary.py --schema minimal --model-config-json <real-server-model-config.json>
```

The user runs these two commands in the real server environment after Wave 1 is pushed. Record both sanitized JSON results in the checkpoint report and select exactly one follow-up branch:

| Current schema | Minimal schema | Required correction |
| --- | --- | --- |
| fails | succeeds | Replace only unsupported schema constructs with a Google-supported reduced response schema; retain strict local Pydantic validation of the full domain result |
| fails | fails | Correct provider/model/config/deployment wiring; do not weaken the schema |
| succeeds | succeeds | Treat as deployment/version drift; pin the server package/config combination and add a parity test |
| succeeds | fails | Stop: canary is invalid or the minimal fixture is malformed; repair the canary before production changes |

The installed local SDK accepting `t_schema(None, deepcopy(production_schema))` is not enough to select a branch. This Task must not change `bind(response_mime_type=..., response_schema=...)` merely from suspicion; official LangChain support and the actual runtime call must be reconciled with the real-server canary result.

- [ ] **Step 4: Commit Task 4**

```powershell
git add core/llm_factory.py data_base/agentic_v9/provider_boundary.py evaluation/agentic_v9_campaign_runtime.py scripts/agentic_v9_contract_planner_canary.py tests/test_agentic_v9_contract_planner_canary.py tests/test_agentic_v9_provider_boundary.py tests/test_llm_factory_override.py docs/agentic-v9-smoke-verification.md docs/superpowers/plans/2026-08-15-agentic-v9-grounded-recovery.md
git commit -m "feat(agentic-v9): add planner provider canary"
```

## Task 5: Publish the Wave 1 backend contract and execution profile

**Backend files:**
- Modify: `evaluation/retrieval_profiles.py`
- Modify: `evaluation/smoke_verification.py`
- Modify: `tests/test_evaluation_retrieval_profiles.py`
- Modify: `tests/test_agentic_v9_smoke_runner.py`
- Modify: `docs/BACKEND.md`
- Modify: `docs/agentic-v9-smoke-verification.md`
- Generate: `openapi.json`
- Generate: `contracts/openapi-contract.json`

- [ ] **Step 1: Version backend profile as `finalpack_r1_active_atomic_contract_v2_retrieval_safe` and add smoke assertions**

Smoke must verify diagnostic presence, question-specific safe fallback strategy, planner calls `<=1`, and no independent comparison planner call. It must not require planner success when staging canary did not prove it.

- [ ] **Step 2: Generate and verify backend artifacts**

```powershell
.\.venv\Scripts\python.exe scripts/sync_openapi_artifacts.py --write
.\.venv\Scripts\python.exe scripts/sync_openapi_artifacts.py --check
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_smoke_runner.py tests/test_evaluation_retrieval_profiles.py tests/test_evaluation_export_v2_schemas.py tests/test_evaluation_export_redaction.py tests/test_openapi_artifacts.py -q
```

- [ ] **Step 3: Commit backend publication**

```powershell
git add evaluation/retrieval_profiles.py evaluation/smoke_verification.py tests/test_evaluation_retrieval_profiles.py tests/test_agentic_v9_smoke_runner.py docs/BACKEND.md docs/agentic-v9-smoke-verification.md openapi.json contracts/openapi-contract.json
git commit -m "docs(agentic-v9): publish retrieval-safe diagnostics"
```

## Task 6: Synchronize the Wave 1 frontend export contract

**Repository:** `D:\flutterserver\Multimodal_RAG_System`

**Files:**
- Modify: `src/types/evaluation.ts`
- Modify: `src/services/evaluationExportSchema.ts`
- Modify: `src/services/evaluationExportSchema.test.ts`
- Modify: `src/types/evaluation.contract.test.ts`
- Modify: `src/components/evaluation/AblationDashboardTab.test.tsx`
- Generate: `src/test/fixtures/agenticV9ApiContract.ts`

- [ ] **Step 1: Pin frontend contract and capture decoder RED**

```powershell
cd D:\flutterserver\Multimodal_RAG_System
npm run contract:pin
npm test -- --run src/services/evaluationExportSchema.test.ts src/types/evaluation.contract.test.ts
```

Add strict populated/null fixtures for `AtomicPlannerDiagnostics`. Add a full-observability download regression so valid non-null diagnostics do not produce `Invalid export response.`

- [ ] **Step 2: Implement exact TypeScript/Zod parity and run GREEN**

```powershell
npm test -- --run src/services/evaluationExportSchema.test.ts src/types/evaluation.contract.test.ts src/components/evaluation/AblationDashboardTab.test.tsx
npm run contract:check
npm run lint:ci
npm run build
```

- [ ] **Step 3: Commit Task 6**

```powershell
git add src/types/evaluation.ts src/services/evaluationExportSchema.ts src/services/evaluationExportSchema.test.ts src/types/evaluation.contract.test.ts src/components/evaluation/AblationDashboardTab.test.tsx src/test/fixtures/agenticV9ApiContract.ts
git commit -m "fix(evaluation-ui): decode planner diagnostics"
```

## Wave 1 Checkpoint — Stop for Deployment

Run a fresh Q1-Q32 Agentic campaign and export full observability. Do not begin Wave 2 until:

- Q13 and Q24 no longer compile the same generic fallback query;
- deterministic Q5/Q23 remain correct positive controls;
- every planner failure has a stable failure stage/code;
- staging canary result is recorded; planner success is claimed only if the production-equivalent current-schema canary passes;
- `semantic_planning_not_admitted` is no longer caused by a provider call that Wave 1 does not make;
- no route, authorized-source, repair-cap, or final-prompt drift occurred;
- backend/frontend worktrees are clean and task commit ranges are recorded.

---

# Wave 2 — Qualified Evidence Before Sufficiency

**Entry condition:** The production-equivalent current-schema planner canary from Wave 1 must pass before enabling fail-closed provider-backed qualification. If it does not pass, stop at Wave 1; do not reconnect the qualification adapter on an unproven provider boundary.

## Task 7: Reconnect the existing batch evidence qualification stage

**Repository:** `D:\flutterserver\pdftopng`

**Files:**
- Modify: `evaluation/agentic_v9_campaign_runtime.py`
- Modify: `data_base/agentic_v9/evidence_extractor.py`
- Modify: `data_base/agentic_v9/evidence_validator.py`
- Modify: `data_base/agentic_v9/budget_feasibility.py`
- Modify: `tests/test_agentic_v9_campaign_runtime.py`
- Modify: `tests/test_agentic_v9_evidence_extractor.py`
- Modify: `tests/test_agentic_v9_evidence_validator.py`
- Modify: `tests/test_agentic_v9_budget_feasibility.py`
- Modify: `tests/test_agentic_v9_execution_core.py`

**Design:** `V9ExecutionCore` already separates `candidate_packets` and `qualified_packets` and calls qualification before sufficiency. Replace the campaign adapter's deterministic/no-op `prose_curate` with one `EvidenceExtractor` batch. Keep the core ordering and repair loop unchanged.

- [x] **Step 1: Add RED candidate-vs-qualified tests**

Assert:

- raw retrieval packets enter the candidate pool with the existing `validation_status="invalid"`, never `deterministic_valid`;
- exact numeric/formula/table/quote spans produce direct packets only after validator success;
- every usable direct packet has `source_span_hash` and `extractor_version`; LLM-curated packets also have `prompt_version`;
- an unrelated verbatim quote is rejected for a slot even though it exists in the source text;
- one batch call handles all unresolved slots; no per-slot/per-chunk calls occur;
- provider failure returns zero newly qualified packets;
- Q5/Q23 positive fixtures remain qualified; Q24 without Table 3 evidence remains unresolved.

- [x] **Step 2: Run RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_v9_evidence_validator.py tests/test_agentic_v9_execution_core.py tests/test_agentic_v9_campaign_runtime.py -k "qualification or candidate or q5 or q23 or q24 or unrelated" -q
```

- [x] **Step 3: Stop stamping raw chunks as `deterministic_valid`**

At the retrieval-to-packet adapter, construct candidates without usable status/provenance. Preserve document ID, chunk locator, raw statement/excerpt, task-derived candidate slot IDs, and ranking metadata. Candidate slot IDs authorize which slot may be proposed; they do not prove support.

- [x] **Step 4: Wire one batch `EvidenceExtractor.extract()` call**

Create one `BudgetedLlmInvoker` for `phase="evidence_extract"`, `purpose="evidence_extraction"`. Pass the whole candidate pool and all unresolved slots to the existing extractor. Deterministic exact extraction runs first; the optional prose curator receives all remaining eligible candidates in one request.

Use the existing extractor/validator allow-list rules:

- proposed evidence ID must exist in the candidate pool;
- proposed slot ID must be valid and eligible for that candidate;
- statement must be an exact source-bound span;
- validator creates `source_span_hash`;
- unsupported, contradictory, malformed, or unbound proposals are dropped.

- [x] **Step 5: Budget the initial batch honestly**

Pass `evidence_qualification_provider_calls=1` only when the active adapter can actually invoke the prose curator. The repair loop may qualify again only after it retrieves new candidates and only while the existing controller preserves final-answer reserve. Do not raise the route's call/token limit.

- [x] **Step 6: Run GREEN and focused no-regression tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_v9_evidence_validator.py tests/test_agentic_v9_execution_core.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_repair.py -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9/evidence_extractor.py data_base/agentic_v9/evidence_validator.py data_base/agentic_v9/budget_feasibility.py evaluation/agentic_v9_campaign_runtime.py tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_v9_evidence_validator.py
```

- [x] **Step 7: Commit Task 7**

```powershell
git add evaluation/agentic_v9_campaign_runtime.py data_base/agentic_v9/evidence_extractor.py data_base/agentic_v9/evidence_validator.py data_base/agentic_v9/budget_feasibility.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_v9_evidence_validator.py tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_execution_core.py
git commit -m "feat(agentic-v9): qualify evidence before sufficiency"
```

## Task 8: Make sufficiency and observability qualification-authoritative

**Repository:** `D:\flutterserver\pdftopng`

**Files:**
- Modify: `data_base/agentic_v9/sufficiency_gate.py`
- Modify: `data_base/agentic_v9/conflict_gate.py`
- Modify: `data_base/agentic_v9/claim_verifier.py`
- Modify: `data_base/agentic_v9/schemas.py`
- Modify: `evaluation/agentic_v9_campaign_runtime.py`
- Modify: `evaluation/export_schemas.py`
- Modify: `evaluation/export_service.py`
- Modify: `tests/test_agentic_v9_sufficiency_gate.py`
- Modify: `tests/test_agentic_v9_campaign_runtime.py`
- Modify: `tests/test_evaluation_export_v2_schemas.py`
- Modify: `tests/test_evaluation_export_redaction.py`

- [x] **Step 1: Add RED gate tests**

Assert that a packet cannot satisfy a slot when it has only task-inherited slot IDs, lacks `source_span_hash`, lacks extractor provenance, is contradictory, or contains a non-matching span. Assert `quote_bound` and validated deterministic structured packets can satisfy slots. Assert calculated packets are usable only when every premise is qualified direct evidence.

- [x] **Step 2: Replace status-name trust with one shared predicate**

Create one canonical function in `evidence_validator.py`:

```python
def is_qualified_evidence(packet: EvidencePacket) -> bool:
    ...
```

Use it from sufficiency, conflict handling, context packing eligibility, and claim verification. Remove duplicated local sets that currently treat any `deterministic_valid` label as sufficient.

- [x] **Step 3: Add exact qualification diagnostics**

Extend metrics/export with fixed fields:

```python
candidate_packet_count: int
qualified_packet_count: int
qualification_round_count: int
qualification_provider_call_count: int
qualification_failure_code: str | None
```

Historical rows may omit the nested/new fields; new profile rows must populate them. Keep failure code sanitized.

- [x] **Step 4: Run GREEN and export tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_sufficiency_gate.py tests/test_agentic_v9_campaign_runtime.py tests/test_evaluation_export_v2_schemas.py tests/test_evaluation_export_redaction.py -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9/sufficiency_gate.py data_base/agentic_v9/conflict_gate.py data_base/agentic_v9/claim_verifier.py data_base/agentic_v9/evidence_validator.py evaluation/agentic_v9_campaign_runtime.py evaluation/export_schemas.py evaluation/export_service.py
```

- [x] **Step 5: Commit Task 8**

```powershell
git add data_base/agentic_v9/sufficiency_gate.py data_base/agentic_v9/conflict_gate.py data_base/agentic_v9/claim_verifier.py data_base/agentic_v9/evidence_validator.py data_base/agentic_v9/schemas.py evaluation/agentic_v9_campaign_runtime.py evaluation/export_schemas.py evaluation/export_service.py tests/test_agentic_v9_sufficiency_gate.py tests/test_agentic_v9_campaign_runtime.py tests/test_evaluation_export_v2_schemas.py tests/test_evaluation_export_redaction.py
git commit -m "fix(agentic-v9): make sufficiency qualification-authoritative"
```

## Task 9: Publish the Wave 2 backend contract and qualification profile

**Backend files:**
- Modify: `evaluation/retrieval_profiles.py`
- Modify: `evaluation/smoke_verification.py`
- Modify: `tests/test_agentic_v9_smoke_runner.py`
- Modify: `tests/test_evaluation_retrieval_profiles.py`
- Modify: `docs/BACKEND.md`
- Modify: `docs/agentic-v9-smoke-verification.md`
- Generate: `openapi.json`
- Generate: `contracts/openapi-contract.json`

- [x] **Step 1: Version the backend profile as `finalpack_r1_active_atomic_contract_v2_quote_qualified_v1`**

Smoke requirements for new runs:

- every sufficiency-supported slot has at least one `is_qualified_evidence()` packet;
- raw candidate count may exceed qualified count;
- provider failure cannot increase qualified count;
- qualification calls equal the actual persisted LLM calls;
- Q5/Q23 positive-control fixtures are answerable in local integration tests;
- the old 64/64 `insufficient` pattern fails the smoke fixture.

- [x] **Step 2: Generate backend artifacts, run backend gate, and commit**

```powershell
.\.venv\Scripts\python.exe scripts/sync_openapi_artifacts.py --write
.\.venv\Scripts\python.exe scripts/sync_openapi_artifacts.py --check
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_v9_evidence_validator.py tests/test_agentic_v9_sufficiency_gate.py tests/test_agentic_v9_execution_core.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_smoke_runner.py tests/test_evaluation_retrieval_profiles.py tests/test_evaluation_export_v2_schemas.py tests/test_evaluation_export_redaction.py tests/test_openapi_artifacts.py -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9 evaluation/agentic_v9_campaign_runtime.py evaluation/smoke_verification.py evaluation/retrieval_profiles.py
git diff --check
git add evaluation/retrieval_profiles.py evaluation/smoke_verification.py tests/test_agentic_v9_smoke_runner.py tests/test_evaluation_retrieval_profiles.py docs/BACKEND.md docs/agentic-v9-smoke-verification.md openapi.json contracts/openapi-contract.json
git commit -m "docs(agentic-v9): publish qualified evidence profile"
```

## Task 10: Synchronize the Wave 2 frontend qualification contract

**Repository:** `D:\flutterserver\Multimodal_RAG_System`

**Files:**
- Modify: `src/types/evaluation.ts`
- Modify: `src/services/evaluationExportSchema.ts`
- Modify: `src/services/evaluationExportSchema.test.ts`
- Modify: `src/types/evaluation.contract.test.ts`
- Modify: `src/components/evaluation/AblationDashboardTab.test.tsx`
- Generate: `src/test/fixtures/agenticV9ApiContract.ts`

- [x] **Step 1: Pin and synchronize the frontend strict decoder**

Add valid/invalid full-observability fixtures for the five qualification fields. Include one download regression with non-null planner, comparison, and qualification data.

```powershell
cd D:\flutterserver\Multimodal_RAG_System
npm run contract:pin
npm test -- --run src/services/evaluationExportSchema.test.ts src/types/evaluation.contract.test.ts src/components/evaluation/AblationDashboardTab.test.tsx
npm run contract:check
npm run lint:ci
npm run build
```

- [x] **Step 2: Commit Task 10**

```powershell
git add src/types/evaluation.ts src/services/evaluationExportSchema.ts src/services/evaluationExportSchema.test.ts src/types/evaluation.contract.test.ts src/components/evaluation/AblationDashboardTab.test.tsx src/test/fixtures/agenticV9ApiContract.ts
git commit -m "fix(evaluation-ui): decode qualification diagnostics"
```

## Wave 2 Checkpoint — Stop for Deployment

Before Wave 3, deploy and run Q1-Q32 with full observability. Required evidence:

- Q5 and Q23 remain answerable and contain at least one qualified packet;
- Q24 is either supported by qualified SegVol Table 3 evidence or is honestly partial;
- provider failure produces no raw-packet promotion;
- candidate and qualified counts are visible and consistent;
- at least one Agentic run is answerable; a repeat of 32/32 or 64/64 `insufficient` blocks Wave 3;
- current simple final prompt remains unchanged, isolating quality change to qualification;
- record correctness, faithfulness, relevancy, latency, provider calls, runtime tokens, and manual Agentic/Naive ratio.

---

# Wave 3 — Compact Structured Final Synthesis

## Task 11: Define a compact final-synthesis projection

**Repository:** `D:\flutterserver\pdftopng`

**Files:**
- Create: `data_base/agentic_v9/final_synthesis_context.py`
- Modify: `data_base/agentic_v9/final_answer.py`
- Modify: `data_base/agentic_v9/schemas.py`
- Modify: `tests/test_agentic_v9_final_answer.py`
- Create: `tests/test_agentic_v9_final_synthesis_context.py`

**Design:** Do not serialize the complete Query Contract and packet payload. The prior full payload has a median size of 14,823 characters and a maximum of 29,356. The compact projection target observed on the same export is a median of 3,752 and maximum of 7,454 characters.

**Exact models:**

```python
class FinalSynthesisSlot(BaseModel):
    slot_id: str
    description: str
    expected_answer_type: ExpectedAnswerType

class FinalSynthesisEvidence(BaseModel):
    evidence_id: str
    slot_ids: list[str]
    statement: str
    source_doc_id: str
    locator: EvidenceLocator
    support_type: EvidenceSupportType
    premise_evidence_ids: list[str]

class FinalSynthesisContext(BaseModel):
    question: str
    slots: list[FinalSynthesisSlot]
    synthesis_obligations: list[SynthesisObligation]
    response_constraints: list[ResponseConstraint]
    slot_resolutions: list[SlotResolution]
    unresolved_requirements: list[UnresolvedRequirement]
    packed_evidence: list[FinalSynthesisEvidence]
    arbitration: list[ConflictCandidate]
```

Use the existing canonical `RequiredSlot.expected_answer_type` spelling directly; do not add an alias.

- [ ] **Step 1: Add RED exact-shape and size tests**

Assert the projection includes every required synthesis input but excludes route budgets, graph policies, authorized-document duplicates, trace payloads, rankings, raw excerpts outside the accepted statement, and unrelated packet metadata. Assert stable ordering by contract slot order and packed evidence order.

- [ ] **Step 2: Run RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_final_synthesis_context.py tests/test_agentic_v9_final_answer.py -k "compact or payload or projection" -q
```

- [ ] **Step 3: Implement the pure projector and use it in `_final_payload()`**

The projector validates that every evidence ID is unique, every evidence slot exists, every unresolved slot exists, and all packets satisfy `is_qualified_evidence()`. Serialize only `FinalSynthesisContext.model_dump(mode="json")`.

- [ ] **Step 4: Run GREEN and Ruff**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_final_synthesis_context.py tests/test_agentic_v9_final_answer.py -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9/final_synthesis_context.py data_base/agentic_v9/final_answer.py tests/test_agentic_v9_final_synthesis_context.py tests/test_agentic_v9_final_answer.py
```

- [ ] **Step 5: Commit Task 11**

```powershell
git add data_base/agentic_v9/final_synthesis_context.py data_base/agentic_v9/final_answer.py data_base/agentic_v9/schemas.py tests/test_agentic_v9_final_synthesis_context.py tests/test_agentic_v9_final_answer.py
git commit -m "feat(agentic-v9): define compact final synthesis context"
```

## Task 12: Reconnect typed final generation in the campaign runtime

**Repository:** `D:\flutterserver\pdftopng`

**Files:**
- Modify: `evaluation/agentic_v9_campaign_runtime.py`
- Modify: `data_base/agentic_v9/final_answer.py`
- Modify: `data_base/agentic_v9/citation_renderer.py`
- Modify: `data_base/agentic_v9/claim_verifier.py`
- Modify: `tests/test_agentic_v9_campaign_runtime.py`
- Modify: `tests/test_agentic_v9_final_answer.py`
- Modify: `tests/test_agentic_v9_provider_boundary.py`

- [ ] **Step 1: Add RED campaign regressions**

Cover:

1. two required slots produce two slot-bound findings/claims;
2. `used_evidence_ids` is the ordered union of evidence and premise IDs from accepted claims only;
3. rejected or unreferenced packed evidence is not marked used;
4. `complete` requires an accepted claim and supported resolution for every required slot;
5. missing one slot yields `qualified_partial`, not `complete`;
6. no accepted claims yields `insufficient`;
7. final provider output is typed JSON internally, while the returned answer is natural language;
8. ordinary complete answers do not expose JSON or forced `Confirmed/Unresolved` headings;
9. partial answers visibly distinguish supported findings from unresolved requirements;
10. Q18-style unsupported rationale is rejected even if its entities appear in relevant evidence;
11. high-risk comparative/causal/SOTA/best/outperform claims trigger at most one verifier call for the whole run;
12. low-risk direct claims do not trigger the verifier.

- [ ] **Step 2: Run RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_final_answer.py tests/test_agentic_v9_provider_boundary.py -k "structured_final or used_evidence or response_status or q18 or verifier" -q
```

- [ ] **Step 3: Replace the campaign's simplified final adapter**

Call the existing `FinalAnswerRenderer.render()` with:

- original question;
- active Query Contract v2;
- packed qualified packets;
- current slot resolutions;
- sufficiency report;
- arbitration results.

Remove campaign code that wraps the whole answer into one `support_type="direct"` claim, assigns all packed IDs as used, or unconditionally sets `response_status="complete"`.

- [ ] **Step 4: Keep generation and verification cardinality bounded**

Use one final provider call. Parse `FinalAnswerDraft` strictly. Filter findings against real slots and packed evidence IDs. Run the existing verifier once at most over the batch of high-risk claims; do not invoke it per claim. On malformed output or provider failure, return the existing claim-free terminal partial/insufficient result without regenerating.

- [ ] **Step 5: Render natural language from accepted typed claims**

The backend renderer owns user-facing prose and citations. Complete responses should read as normal answers. Partial responses may add a concise unresolved section. Never return the internal JSON payload directly.

- [ ] **Step 6: Run GREEN and Ruff**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_final_answer.py tests/test_agentic_v9_final_synthesis_context.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_execution_core.py tests/test_agentic_v9_claim_verifier.py -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9/final_answer.py data_base/agentic_v9/citation_renderer.py data_base/agentic_v9/claim_verifier.py evaluation/agentic_v9_campaign_runtime.py tests/test_agentic_v9_final_answer.py tests/test_agentic_v9_campaign_runtime.py
```

- [ ] **Step 7: Commit Task 12**

```powershell
git add evaluation/agentic_v9_campaign_runtime.py data_base/agentic_v9/final_answer.py data_base/agentic_v9/citation_renderer.py data_base/agentic_v9/claim_verifier.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_final_answer.py tests/test_agentic_v9_provider_boundary.py
git commit -m "feat(agentic-v9): activate structured final synthesis"
```

## Task 13: Publish Wave 3 backend observability and the final profile

**Backend files:**
- Modify: `data_base/agentic_v9/schemas.py`
- Modify: `evaluation/export_schemas.py`
- Modify: `evaluation/export_service.py`
- Modify: `evaluation/retrieval_profiles.py`
- Modify: `evaluation/smoke_verification.py`
- Modify: `tests/test_evaluation_export_v2_schemas.py`
- Modify: `tests/test_evaluation_export_redaction.py`
- Modify: `tests/test_agentic_v9_smoke_runner.py`
- Modify: `tests/test_evaluation_retrieval_profiles.py`
- Modify: `docs/BACKEND.md`
- Modify: `docs/agentic-v9-smoke-verification.md`
- Generate: `openapi.json`
- Generate: `contracts/openapi-contract.json`

- [ ] **Step 1: Add exact final-synthesis metrics**

Reuse the existing canonical `final_claim_count` and `final_generation_count` fields, and add only the missing fixed metrics:

```python
used_evidence_count: int
unresolved_requirement_count: int
claim_verifier_call_count: int
```

Counts must be derived from the returned typed result, not inferred from trace text. Historical rows remain nullable/absent according to the existing historical compatibility convention.

- [ ] **Step 2: Version profile as `finalpack_r1_active_atomic_contract_v2_structured_synthesis_v1`**

Smoke asserts final generation `<=1`, verifier `<=1`, used IDs are a subset of packed qualified IDs, every accepted claim names a valid slot, and `complete` covers every required slot.

- [ ] **Step 3: Generate artifacts, run the full backend Wave 1-3 gate, and commit**

```powershell
.\.venv\Scripts\python.exe scripts/sync_openapi_artifacts.py --write
.\.venv\Scripts\python.exe scripts/sync_openapi_artifacts.py --check
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_budgeted_llm.py tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_v9_evidence_validator.py tests/test_agentic_v9_sufficiency_gate.py tests/test_agentic_v9_execution_core.py tests/test_agentic_v9_final_synthesis_context.py tests/test_agentic_v9_final_answer.py tests/test_agentic_v9_claim_verifier.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_smoke_runner.py tests/test_evaluation_retrieval_profiles.py tests/test_evaluation_export_v2_schemas.py tests/test_evaluation_export_redaction.py tests/test_openapi_artifacts.py -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9 evaluation/agentic_v9_campaign_runtime.py evaluation/smoke_verification.py evaluation/retrieval_profiles.py scripts/agentic_v9_contract_planner_canary.py
git diff --check
git add data_base/agentic_v9/schemas.py evaluation/export_schemas.py evaluation/export_service.py evaluation/retrieval_profiles.py evaluation/smoke_verification.py tests/test_evaluation_export_v2_schemas.py tests/test_evaluation_export_redaction.py tests/test_agentic_v9_smoke_runner.py tests/test_evaluation_retrieval_profiles.py docs/BACKEND.md docs/agentic-v9-smoke-verification.md openapi.json contracts/openapi-contract.json
git commit -m "docs(agentic-v9): publish structured synthesis profile"
```

## Task 14: Synchronize the Wave 3 frontend final-synthesis contract

**Repository:** `D:\flutterserver\Multimodal_RAG_System`

**Files:**
- Modify: `src/types/evaluation.ts`
- Modify: `src/services/evaluationExportSchema.ts`
- Modify: `src/services/evaluationExportSchema.test.ts`
- Modify: `src/types/evaluation.contract.test.ts`
- Modify: `src/components/evaluation/AblationDashboardTab.test.tsx`
- Generate: `src/test/fixtures/agenticV9ApiContract.ts`

- [ ] **Step 1: Pin and synchronize frontend strict contracts**

Run `npm run contract:pin`, add a populated fixture containing the new final-synthesis metrics, and first capture RED from the strict decoder/type contract before modifying production TypeScript or Zod code.

```powershell
cd D:\flutterserver\Multimodal_RAG_System
npm run contract:pin
npm test -- --run src/services/evaluationExportSchema.test.ts src/types/evaluation.contract.test.ts src/components/evaluation/AblationDashboardTab.test.tsx
npm run contract:check
npm run lint:ci
npm run build
```

The populated fixture must include non-null comparison, full observability, qualification diagnostics, and final-synthesis counts. A successful response must create one download; malformed counts/IDs must be rejected before download.

- [ ] **Step 2: Commit Task 14**

```powershell
git add src/types/evaluation.ts src/services/evaluationExportSchema.ts src/services/evaluationExportSchema.test.ts src/types/evaluation.contract.test.ts src/components/evaluation/AblationDashboardTab.test.tsx src/test/fixtures/agenticV9ApiContract.ts
git commit -m "fix(evaluation-ui): decode final synthesis diagnostics"
```

## Wave 3 Checkpoint — Final Deployment Study

Stop implementation and run a paired Q1-Q32 Agentic/Naive campaign with summary plus all-run observability export.

Required analysis:

- correctness, faithfulness, and relevancy by mode and by question;
- required-slot count versus each quality delta;
- deterministic/planned/degraded planner groups;
- candidate/qualified/packed/used evidence counts;
- complete/qualified_partial/insufficient distribution;
- accepted claims per required slot;
- qualification, repair, final-generation, and verifier provider calls;
- runtime tokens and latency by phase;
- manual Agentic/Naive runtime-token ratio, target `<=3.0`;
- Q5/Q23 positive controls, Q13/Q24 retrieval regressions, Q18 unsupported-rationale regression, and Q20/Q29 sufficiency regressions.

Release interpretation:

- Improved faithfulness with collapsed correctness/relevancy and widespread insufficient output is not success.
- Improved correctness with raw/unqualified evidence promotion is not success.
- `complete` is valid only when every required slot has an accepted, evidence-linked claim.
- If a Wave fails its checkpoint, revert that Wave's complete commit range. Do not partially restore the old adapter, add a runtime flag, or mix incompatible profiles.

## Final Verification and Handoff

- [ ] Confirm each Task produced exactly one scoped commit in the repository it changed.
- [ ] Confirm both tracked worktrees are clean.
- [ ] Confirm backend OpenAPI artifacts are current and frontend `contract:check` points to the final backend commit/hash.
- [ ] Record all real-system campaign IDs, export SHA256 values, package versions, and task commit ranges in the implementation ledger.
- [ ] Request one consolidated code review after Wave 3 implementation and local gates, then correct only validated findings in separate focused commits.
- [ ] Do not start later router/classifier/context-replacement architecture work until the Wave 3 study identifies the remaining bottleneck.
