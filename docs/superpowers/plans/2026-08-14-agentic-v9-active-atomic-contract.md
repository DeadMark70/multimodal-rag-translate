# Agentic RAG v9 Active Atomic Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the active Agentic v9 generic/shadow slot path with one route-preserving Atomic Query Contract v2 that uses deterministic decomposition first, at most one consolidated semantic-planning call, explicit evidence-slot ownership for comparisons, and honest observability.

**Architecture:** Keep `RoutePlanner` and source admission authoritative, then apply a narrow atomic overlay that may change only slots, synthesis obligations, response constraints, comparison metadata, and slot-plan provenance. Compile retrieval and repair only from `S1..S8` evidence slots, remove the active independent Comparison Planner call, and preserve current evidence qualification and final synthesis unchanged for this checkpoint.

**Tech Stack:** Python 3.13, Pydantic v2, pytest/pytest-asyncio, Ruff, FastAPI/OpenAPI artifact generation, React 18, TypeScript, Zod, Vitest, Chakra UI.

## Global Constraints

- Execute in the existing main worktrees: backend `D:\flutterserver\pdftopng`, frontend `D:\flutterserver\Multimodal_RAG_System`; do not create a Git worktree.
- Follow TDD for every behavior change: demonstrate the intended RED failure before modifying production code, then run the same test GREEN.
- End every task with exactly one focused commit in the repository that task changes. Do not combine backend and frontend commits.
- Atomic Contract is active for every new Agentic v9 evaluation run. Do not add a feature flag, shadow mode, campaign option, or runtime v1 fallback.
- Preserve the existing deterministic route, `RouteDecision`, authorized source scope, graph/visual policies, route budgets, deadline, and `evidence_extraction_required` exactly.
- Only direct evidence requirements become `RequiredSlot` values and participate in retrieval, sufficiency, repair, and required-first context packing.
- Synthesis obligations and response constraints are typed contract data but do not create retrieval tasks or participate in sufficiency.
- Use `S1` through `S8` for every active evidence slot. Comparison subjects bind explicitly to those IDs through `evidence_slot_ids`.
- Use deterministic decomposition whenever acceptable. Low-confidence or ambiguous decomposition may invoke at most one `contract_planning` provider call per run, with zero retries and no per-slot/per-chunk planning calls.
- Remove the active independent `comparison_plan` provider call. Historical `comparison_plan` accounting/schema values remain readable.
- If optional semantic planning cannot be admitted without the final-answer reserve, skip it and emit a degraded v2 contract with one safe `S1`; do not return `configuration_incompatible` for that optional failure.
- Do not add span/quote extraction, semantic qualification, claim verification, final-prompt fields, or structured final synthesis in this checkpoint.
- Keep the current final prompt and final-answer behavior byte-for-byte unchanged unless a test proves an unavoidable serialization-only adjustment.
- Do not add a new automatic Agentic/Naive ratio gate. The existing runtime-token ratio target `<= 3.0` remains a manual campaign check.
- Do not use ground truth, golden answers, expected routes, Q1-Q16 membership, or answer values in contract planning.
- Update strict backend/OpenAPI/frontend contracts atomically before declaring the checkpoint complete.

## File and Ownership Map

### Backend domain and planning

- `data_base/agentic_v9/schemas.py` — canonical typed Query Contract, comparison subject-to-slot binding, and execution metrics.
- `data_base/agentic_v9/requirement_decomposition.py` — deterministic evidence/synthesis/constraint classification and semantic-planning decision.
- `data_base/agentic_v9/contract_planner.py` — preparation, one-call strict semantic planning, safe fallback, answer/source validation, and route-preserving atomic overlay.
- `prompts/agentic_v9_contract_planner.json` — one answer-free combined atomic/comparison planning prompt.
- `data_base/agentic_v9/retrieval_tasks.py` — atomic and comparison retrieval compilation from explicit evidence slot IDs.
- `data_base/agentic_v9/repair.py` — missing-slot-only atomic/comparison repair.
- `data_base/agentic_v9/budget_feasibility.py` — optional `contract_planning` feasibility without converting planner rejection into a run failure.
- `data_base/agentic_v9/execution_core.py` — correct planning-stage timeout selection.
- `evaluation/agentic_v9_campaign_runtime.py` — Active runtime switch, budgeted planner wiring, removal of active shadow guidance and independent comparison calls, and trace projection.
- `evaluation/retrieval_profiles.py` — execution-profile version identifying the active atomic behavior.
- `evaluation/smoke_verification.py` — post-deployment invariants for v2, planner calls, slot mappings, and comparison evidence.

### Backend API and documentation

- `evaluation/campaign_schemas.py` and `evaluation/export_schemas.py` consume `QueryContract` and `V9ExecutionMetrics` directly; this plan intentionally leaves them unchanged and locks that reuse with export/OpenAPI tests.
- `openapi.json` and `contracts/openapi-contract.json` — generated API artifacts after the backend schema is final.
- `docs/BACKEND.md` — active runtime ownership and provider-call behavior.
- `docs/agentic-v9-smoke-verification.md` — checkpoint inspection and failure interpretation.

### Frontend strict consumer

- `src/types/evaluation.ts` — exact additive Query Contract, comparison, and metrics fields.
- `src/services/evaluationExportSchema.ts` — strict Export v2 decoder synchronized to the backend contract.
- `src/components/evaluation/AgenticV9Trace.tsx` — compact planner provenance, obligations, constraints, and honest binding/qualification display.
- `src/components/evaluation/RunTraceTab.test.tsx` — visible Atomic Contract trace assertions.
- `src/services/evaluationExportSchema.test.ts` and `src/types/evaluation.contract.test.ts` — strict valid/invalid contract fixtures.
- `src/test/fixtures/agenticV9ApiContract.ts` — generated backend OpenAPI pin.

---

### Task 1: Define the Active Atomic Contract domain model

**Files:**
- Modify: `data_base/agentic_v9/schemas.py:18-275,615-627`
- Modify: `tests/test_agentic_v9_schemas.py:259-330`
- Modify: `tests/test_agentic_v9_comparison_planner.py:250-360`

**Interfaces:**
- Consumes: existing `RequiredSlot`, `ComparisonPlan`, `ComparisonSubject`, `QueryContract`, and `V9ExecutionMetrics`.
- Produces: `SynthesisObligation`, `ResponseConstraint`, `SlotPlanSource`, `SynthesisObligationKind`, `ResponseConstraintKind`; `ComparisonSubject.evidence_slot_ids`; additive `QueryContract` planning fields; additive `V9ExecutionMetrics` instrumentation.

- [ ] **Step 1: Add RED schema tests for fixed atomic structures and explicit comparison binding**

Add tests that construct a non-empty v2 contract and assert exact serialization:

```python
contract = QueryContract(
    contract_version="2",
    route="bounded_compare",
    intent="Compare two models from authorized sources",
    required_slots=[
        RequiredSlot(slot_id="S1", description="Retrieve A latency."),
        RequiredSlot(slot_id="S2", description="Retrieve B latency."),
    ],
    synthesis_obligations=[
        SynthesisObligation(
            obligation_id="O1",
            kind="comparison",
            description="Compare the two reported latencies.",
            depends_on_slot_ids=["S1", "S2"],
        )
    ],
    response_constraints=[
        ResponseConstraint(
            constraint_id="C1",
            kind="prohibition",
            description="Do not claim a universal ranking.",
        )
    ],
    comparison_plan=ComparisonPlan(
        subjects=[
            ComparisonSubject(
                subject_id="model_a",
                display_name="Model A",
                retrieval_query="Model A reported latency",
                evidence_slot_ids=["S1"],
            ),
            ComparisonSubject(
                subject_id="model_b",
                display_name="Model B",
                retrieval_query="Model B reported latency",
                evidence_slot_ids=["S2"],
            ),
        ]
    ),
    slot_plan_status="complete",
    slot_plan_source="deterministic",
    slot_plan_confidence="high",
    slot_plan_fallback_reason=None,
    truncated_requirement_count=0,
)
assert contract.route == "bounded_compare"
assert contract.comparison_plan.subjects[0].evidence_slot_ids == ["S1"]
assert contract.model_dump(mode="json")["synthesis_obligations"][0]["obligation_id"] == "O1"
```

Also assert rejection of duplicate `S1`, obligation references to unknown slots, active comparison references to unknown slots, more than eight active slots, and non-sequential IDs in the active overlay validation helper. Add a historical compatibility test proving a deserialized comparison subject without `evidence_slot_ids` remains readable with `[]`.

- [ ] **Step 2: Run the schema tests and capture RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_schemas.py tests/test_agentic_v9_comparison_planner.py -k "atomic or comparison_plan or historical" -q
```

Expected: failures because the new classes, contract fields, comparison binding, and metrics fields do not exist.

- [ ] **Step 3: Add the minimal strict Pydantic models and validators**

Implement the canonical shapes in `schemas.py`:

```python
SlotPlanSource = Literal["deterministic", "llm_planner", "safe_fallback"]
SynthesisObligationKind = Literal[
    "comparison", "selection", "causal", "aggregation", "qualification"
]
ResponseConstraintKind = Literal[
    "conditional_scope", "output_format", "prohibition", "allowed_labels"
]

class SynthesisObligation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    obligation_id: str = Field(pattern=r"^O[1-9][0-9]*$")
    kind: SynthesisObligationKind
    description: str = Field(min_length=1, max_length=512)
    depends_on_slot_ids: list[str] = Field(min_length=1, max_length=8)

class ResponseConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    constraint_id: str = Field(pattern=r"^C[1-9][0-9]*$")
    kind: ResponseConstraintKind
    description: str = Field(min_length=1, max_length=512)
```

Add `ComparisonSubject.evidence_slot_ids` as `Field(default_factory=list, max_length=8, exclude_if=lambda value: not value)`. Add these `QueryContract` fields with historical-safe defaults/omission:

```python
synthesis_obligations: list[SynthesisObligation] = Field(
    default_factory=list, max_length=8, exclude_if=lambda value: not value
)
response_constraints: list[ResponseConstraint] = Field(
    default_factory=list, max_length=8, exclude_if=lambda value: not value
)
slot_plan_source: SlotPlanSource | None = Field(
    default=None, exclude_if=lambda value: value is None
)
slot_plan_confidence: Literal["high", "medium", "low"] | None = Field(
    default=None, exclude_if=lambda value: value is None
)
slot_plan_fallback_reason: str | None = Field(
    default=None, max_length=160, exclude_if=lambda value: value is None
)
truncated_requirement_count: int | None = Field(
    default=None, ge=0, exclude_if=lambda value: value is None
)
```

Validate unique slot IDs and valid dependency references without rejecting historical `comparison-subject:*` contracts. Put the stricter `S1..S8` and non-empty subject binding rule in an exported `validate_active_atomic_contract(contract: QueryContract) -> QueryContract` helper so only new Active contracts receive those invariants.

Extend metrics exactly:

```python
atomic_planner_call_count: int = Field(default=0, ge=0, le=1)
comparison_planner_call_count: Literal[0] = 0
slot_binding_method: Literal["task_target_inherited", "not_instrumented"] = "not_instrumented"
semantic_qualification: Literal["not_enabled", "not_instrumented"] = "not_instrumented"
```

- [ ] **Step 4: Run focused schema GREEN and Ruff**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_schemas.py tests/test_agentic_v9_comparison_planner.py -k "atomic or comparison_plan or historical" -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9/schemas.py tests/test_agentic_v9_schemas.py tests/test_agentic_v9_comparison_planner.py
```

Expected: all selected tests pass and Ruff reports `All checks passed!`.

- [ ] **Step 5: Commit Task 1**

```powershell
git add data_base/agentic_v9/schemas.py tests/test_agentic_v9_schemas.py tests/test_agentic_v9_comparison_planner.py
git commit -m "feat(agentic-v9): define active atomic contract schema"
```

---

### Task 2: Classify deterministic evidence, synthesis, and constraints

**Files:**
- Modify: `data_base/agentic_v9/requirement_decomposition.py:12-397`
- Modify: `tests/test_agentic_v9_requirement_decomposition.py:1-220`
- Create: `tests/fixtures/agentic_v9_atomic_questions_v1.json`

**Interfaces:**
- Consumes: question text only; the source fixture `D:\flutterserver\q1-q32_v9_schema_fixed.json` supplies only `id`, `question`, and `source_docs`.
- Produces: `QuestionDecomposition.requirements`, `.synthesis_obligations`, `.response_constraints`, `.comparison_subjects`, `.semantic_planning_reasons`, `.confidence`, and `.requires_semantic_planning`.

- [ ] **Step 1: Create a question-only Q1-Q32 regression fixture**

Create `tests/fixtures/agentic_v9_atomic_questions_v1.json` with this exact shape and all 32 rows copied from the source file:

```json
{
  "schema_version": "atomic_questions_v1",
  "questions": [
    {
      "id": "Q1",
      "question": "在 BraTS 類 3D 腦腫瘤分割場景，若訓練資料偏少且 GPU 資源受限，SwinUNETR、MedNeXt、nnMamba 在「長距離依賴建模方式」與「計算成本/穩定性」上應如何取捨？請以「首選 / 次選 / 不優先」格式給出選型裁決，並註明此結論屬於跨文獻相對建議，不是同配置 benchmark 排名。",
      "source_docs": [
        "2201.01266v1SwinUNETR.pdf",
        "2303.09975v5MedNeXt.pdf",
        "2402.03526v2nnMamba.pdf"
      ]
    }
  ]
}
```

Do not copy `ground_truth`, `ground_truth_short`, `key_points`, `expected_evidence`, `atomic_facts`, `test_objective`, or any expected route.

- [ ] **Step 2: Add RED deterministic-classification tests**

Add focused tests proving direct evidence and derived reasoning are distinct:

```python
result = decompose_question(
    "分別找出 Model-A 與 Model-B 的 latency，然後比較哪個較低；"
    "不要宣稱為通用排名。"
)
assert [item.entity_ids for item in result.requirements] == [
    ("Model-A",),
    ("Model-B",),
]
assert [(item.kind, item.depends_on_requirement_indexes) for item in result.synthesis_obligations] == [
    ("comparison", (0, 1)),
]
assert [item.kind for item in result.response_constraints] == ["prohibition"]
assert result.comparison_subjects == ("Model-A", "Model-B")
```

Add cases for numbered Chinese/English requirements, causal/selection/aggregation obligations, output-format constraints, complex unpunctuated Chinese, truncation above eight, and one vague compound requirement. For the 32-row fixture assert only structural invariants: non-empty normalized question, no answer fields in the fixture, bounded output, and either an acceptable deterministic plan or explicit semantic-planning reasons.

- [ ] **Step 3: Run RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_requirement_decomposition.py -q
```

Expected: failures because synthesis drafts, entity bindings, comparison subjects, confidence aggregation, and semantic-planning reasons are absent.

- [ ] **Step 4: Extend the deterministic decomposition without corpus templates**

Add immutable draft types:

```python
@dataclass(frozen=True, slots=True)
class SynthesisObligationDraft:
    kind: SynthesisObligationKind
    text: str
    depends_on_requirement_indexes: tuple[int, ...]

@dataclass(frozen=True, slots=True)
class DecomposedRequirement:
    text: str
    method: DecompositionMethod
    confidence: DecompositionConfidence
    entity_ids: tuple[str, ...] = ()
```

Extend `QuestionDecomposition` with typed tuples for synthesis obligations, comparison subjects, semantic-planning reasons, and aggregate confidence. Define `requires_semantic_planning` as `bool(semantic_planning_reasons)`.

Use general structural cues only. Classify an explicit compare/select/trade-off conclusion as an obligation over previously emitted subject evidence requirements. Preserve source-explanation questions as evidence requirements unless the parser can prove they are derived; add `evidence_vs_synthesis_ambiguous` when it cannot. Emit reasons such as `low_confidence`, `compound_collapsed`, `comparison_subjects_unclear`, `dependency_unclear`, `truncated_requirements`, and `complex_unpunctuated_chinese`.

Remove the current behavior that appends “整合各主體的比較、選擇或 trade-off 結論” to `requirements`; emit it as `SynthesisObligationDraft` instead.

- [ ] **Step 5: Run GREEN, corpus invariants, and Ruff**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_requirement_decomposition.py -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9/requirement_decomposition.py tests/test_agentic_v9_requirement_decomposition.py
```

Expected: all tests pass, every Q1-Q32 row is handled without golden-route knowledge, and Ruff passes.

- [ ] **Step 6: Commit Task 2**

```powershell
git add data_base/agentic_v9/requirement_decomposition.py tests/test_agentic_v9_requirement_decomposition.py tests/fixtures/agentic_v9_atomic_questions_v1.json
git commit -m "feat(agentic-v9): classify atomic question requirements"
```

---

### Task 3: Build the route-preserving one-call Atomic Contract planner

**Files:**
- Modify: `data_base/agentic_v9/contract_planner.py:1-692`
- Modify: `prompts/agentic_v9_contract_planner.json`
- Modify: `tests/test_agentic_v9_contract_planner.py:1-590`
- Modify: `tests/test_prompt_loader.py:225-260`

**Interfaces:**
- Consumes: an immutable base `QueryContract` from `RoutePlanner`, a `QuestionDecomposition`, and an optional `LlmInvoker` already wrapped by `BudgetedLlmInvoker`.
- Produces: `AtomicContractPreparation`, `AtomicContractPlanningOutcome`, `QuestionContractPlanner.prepare(...)`, `QuestionContractPlanner.plan(...)`, `apply_atomic_contract_overlay(...)`, and `atomic_contract_planner_response_schema()`.

- [ ] **Step 1: Add RED tests for preparation, overlay immutability, one-call output, and degraded v2**

Lock the public interface:

```python
preparation = QuestionContractPlanner.prepare(
    question=question,
    base_contract=base_contract,
)
outcome = await QuestionContractPlanner(llm_invoker=invoker).plan(
    question=question,
    base_contract=base_contract,
    preparation=preparation,
    allow_semantic_planning=True,
)
assert outcome.contract.contract_version == "2"
assert outcome.contract.route == base_contract.route
assert outcome.contract.route_decision == base_contract.route_decision
assert outcome.contract.max_llm_calls == base_contract.max_llm_calls
assert outcome.contract.evidence_extraction_required == base_contract.evidence_extraction_required
assert outcome.planner_call_count in {0, 1}
```

Add exact tests for:

- accepted deterministic preparation: zero calls and `slot_plan_source="deterministic"`;
- low-confidence preparation: exactly one call;
- combined LLM response containing evidence requirements, obligations, constraints, and comparison metadata;
- planner response cannot contain route or document IDs;
- code assigns `S*`, `O*`, and `C*` IDs;
- numeric answer leakage, unknown source names, invalid indexes, malformed JSON, timeout, `BudgetExceededError`, unavailable invoker, and `allow_semantic_planning=False` each yield one safe `S1`, `slot_plan_status="degraded"`, and the exact bounded fallback reason;
- no retry after invalid output;
- overlay changes only the allow-listed fields.

- [ ] **Step 2: Run planner RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_contract_planner.py tests/test_prompt_loader.py -q
```

Expected: failures because the current planner selects its own route, accepts provider document IDs, lacks obligations/constraints/comparison binding, and returns `QueryContract` directly.

- [ ] **Step 3: Replace the planner transport with one combined strict schema**

Use provider-facing indexes rather than provider-assigned IDs:

```python
class _PlannerEvidenceRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str
    source_name_hints: list[str]
    locator_hints: list[str]
    expected_answer_type: ExpectedAnswerType
    depends_on_requirement_indexes: list[int]
    visual_policy: VisualPolicy

class _PlannerDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_requirements: list[_PlannerEvidenceRequirement]
    synthesis_obligations: list[_PlannerSynthesisObligation]
    response_constraints: list[_PlannerResponseConstraint]
    comparison: _PlannerComparison | None
    confidence: float
```

The comparison subject transport includes `evidence_requirement_indexes`, not slot IDs. It contains no route, document ID, answer, score, or golden field. Export `atomic_contract_planner_response_schema()` from `_PlannerDecision.model_json_schema()`.

- [ ] **Step 4: Implement preparation, safe promotion, and narrow overlay**

Define these exact public shapes:

```python
@dataclass(frozen=True, slots=True)
class AtomicContractPreparation:
    decomposition: QuestionDecomposition
    semantic_planning_requested: bool
    comparison_candidate: bool

@dataclass(frozen=True, slots=True)
class AtomicContractPlanningOutcome:
    contract: QueryContract
    planner_call_count: Literal[0, 1]
    latency_ms: float
```

`prepare()` is pure and deterministic. `plan()` invokes the provider only when both `semantic_planning_requested` and `allow_semantic_planning` are true. Promote source-name hints through `base_contract.resolved_source_scope.source_name_to_doc_ids`; never trust provider IDs.

For accepted deterministic comparisons, create a `ComparisonPlan` only when there are two to four distinct subjects and every subject maps to at least one evidence requirement. Build each subject's `retrieval_query` from its display name plus the descriptions of its mapped evidence slots, then set `evidence_slot_ids` after code assigns `S1..S8`. If that complete mapping cannot be made deterministically, preparation must request semantic planning instead of emitting a partial comparison plan.

The safe fallback uses exactly one slot with description `Resolve the complete source-bound requirement in the original question.` and the base contract's full authorized scope. Set confidence to `low`. Use only these bounded fallback reasons: `deterministic_unusable`, `semantic_planning_not_admitted`, `planner_unavailable`, `planner_timeout`, `planner_budget_rejected`, `unauthorized_source_expansion`, and `invalid_planner_output`.

Implement `apply_atomic_contract_overlay()` with `base_contract.model_copy(update=...)` and only these keys:

```python
{
    "contract_version": "2",
    "required_slots": slots,
    "synthesis_obligations": obligations,
    "response_constraints": constraints,
    "comparison_plan": comparison_plan,
    "slot_plan_status": status,
    "slot_plan_source": source,
    "slot_plan_confidence": confidence,
    "slot_plan_fallback_reason": fallback_reason,
    "truncated_requirement_count": truncated_count,
}
```

Pass the result through `validate_active_atomic_contract`. Do not update `route`, `intent`, `entities`, `locator_hints`, scope, policies, rounds, LLM calls, token budget, strategy tier, or `route_decision`.

- [ ] **Step 5: Replace the prompt with answer-free combined planning instructions**

Keep the existing prompt registry key `atomic_contract_planning`. Require one JSON object matching the transport schema, authorized source names only, direct evidence facts separated from derived reasoning, comparison subject-to-requirement indexes, and no route selection. Explicitly forbid answers, numbers not present in the question, document IDs, external sources, markdown fences, and extra keys.

- [ ] **Step 6: Run planner GREEN and Ruff**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_contract_planner.py tests/test_prompt_loader.py tests/test_agentic_v9_schemas.py -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9/contract_planner.py prompts/agentic_v9_contract_planner.json tests/test_agentic_v9_contract_planner.py tests/test_prompt_loader.py
```

Expected: all tests pass; deterministic cases make zero calls, low-confidence cases make one, and every failure yields degraded v2 without changing the route.

- [ ] **Step 7: Commit Task 3**

```powershell
git add data_base/agentic_v9/contract_planner.py prompts/agentic_v9_contract_planner.json tests/test_agentic_v9_contract_planner.py tests/test_prompt_loader.py
git commit -m "feat(agentic-v9): consolidate atomic contract planning"
```

---

### Task 4: Compile comparison retrieval and repair from atomic slot IDs

**Files:**
- Modify: `data_base/agentic_v9/retrieval_tasks.py:28-450`
- Modify: `data_base/agentic_v9/repair.py:50-285`
- Modify: `tests/test_agentic_v9_retrieval_tasks.py:100-350`
- Modify: `tests/test_agentic_v9_repair.py:130-470`
- Modify: `tests/test_agentic_v9_sufficiency_gate.py`
- Modify: `tests/test_agentic_v9_context_packer.py`

**Interfaces:**
- Consumes: active `QueryContract.required_slots` using `S1..S8`, `ComparisonSubject.evidence_slot_ids`, synthesis obligations, and response constraints.
- Produces: subject-aware retrieval/repair tasks that target only real evidence slots; historical comparison contracts remain readable through an isolated fallback mapping.

- [ ] **Step 1: Add RED tests for explicit subject-slot mapping**

Create a comparison contract where subject A owns `S1,S2`, subject B owns `S3,S4`, and `O1` depends on all four. Assert:

```python
plan = compile_retrieval_tasks(
    question="Compare A and B.", query_id="q-1", contract=contract
)
assert [task.target_slot_ids for task in plan.tasks] == [["S1", "S2"], ["S3", "S4"]]
assert all("O1" not in task.target_slot_ids for task in plan.tasks)
assert plan.tasks[0].source_scope.authorized_doc_ids == ["doc-a"]
assert plan.tasks[1].source_scope.authorized_doc_ids == ["doc-b"]
```

Add repair tests where only `S4` is `not_found`; exactly one repair query targets `S4` and subject B. Add tests proving a synthesis obligation never appears in sufficiency, repairable IDs, or context packing. Preserve a historical fixture with `comparison-subject:a` and empty `evidence_slot_ids` to lock read compatibility only.

- [ ] **Step 2: Run retrieval/repair RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_repair.py tests/test_agentic_v9_sufficiency_gate.py tests/test_agentic_v9_context_packer.py -q
```

Expected: active comparison tests fail because compiler, coverage, and repair infer `comparison-subject:*` IDs.

- [ ] **Step 3: Update retrieval compiler to use explicit evidence slots**

Replace `_compile_comparison_subjects` slot-name inference with:

```python
target_slot_ids = _comparison_subject_slot_ids(subject, contract)
slots = [slot for slot in contract.required_slots if slot.slot_id in target_slot_ids]
authorized_doc_ids = _unique(
    doc_id
    for slot in slots
    for doc_id in authorized_doc_ids_for_slot(slot, scope)
)
```

The active path must raise when a subject has no valid mapped slots or no authorized source intersection. The historical helper may use `comparison-subject:{subject_id}` only when `evidence_slot_ids` is empty and that exact historical slot exists.

- [ ] **Step 4: Update comparison repair to target missing mapped slots**

Build `subjects_by_slot_id` from each subject's explicit `evidence_slot_ids`. For the first missing subject in planner order, include only its currently `not_found` slots in `target_slot_ids`, narrow the authorized scope to those slots, and retain the existing single bounded comparison repair query.

- [ ] **Step 5: Run GREEN, then all atomic consumer tests and Ruff**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_repair.py tests/test_agentic_v9_sufficiency_gate.py tests/test_agentic_v9_context_packer.py -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9/retrieval_tasks.py data_base/agentic_v9/repair.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_repair.py
```

Expected: all pass; only evidence slots enter downstream work.

- [ ] **Step 6: Commit Task 4**

```powershell
git add data_base/agentic_v9/retrieval_tasks.py data_base/agentic_v9/repair.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_repair.py tests/test_agentic_v9_sufficiency_gate.py tests/test_agentic_v9_context_packer.py
git commit -m "fix(agentic-v9): bind comparison retrieval to atomic slots"
```

---

### Task 5: Admit one optional contract-planning call without sacrificing final reserve

**Files:**
- Modify: `data_base/agentic_v9/schemas.py:588-614`
- Modify: `data_base/agentic_v9/execution_core.py:145-175`
- Modify: `data_base/agentic_v9/budget_feasibility.py:196-280`
- Modify: `tests/test_agentic_v9_execution_core.py`
- Modify: `tests/test_agentic_v9_budget_feasibility.py:78-230`
- Modify: `tests/test_agentic_v9_execution_policy.py`

**Interfaces:**
- Consumes: `AtomicContractPreparation.semantic_planning_requested`, existing phase policies, base route budget, and final-answer reserve.
- Produces: `V9ExecutionRequest.contract_plan_requested`; post-contract feasibility for an optional `contract_planning` call; execution-core planning timeout selection.

- [ ] **Step 1: Add RED budget and core tests**

Replace the transient request flag with the accurate name and assert the core selects the right outer phase:

```python
request = V9ExecutionRequest(
    question="ambiguous question",
    trace_id="trace-1",
    contract_plan_requested=True,
)
await core.execute(request, runtime_context=context)
assert stage_calls[1].phase == "contract_planning"
```

Add feasibility tests for:

```python
with_planner = validate_post_contract_feasibility(
    contract=contract,
    setup_snapshot=setup,
    remaining_token_budget=contract.runtime_token_budget,
    remaining_llm_calls=contract.max_llm_calls,
    contract_plan_requested=True,
)
without_planner = validate_post_contract_feasibility(
    contract=contract,
    setup_snapshot=setup,
    remaining_token_budget=contract.runtime_token_budget,
    remaining_llm_calls=contract.max_llm_calls,
    contract_plan_requested=False,
)
assert with_planner.reason == "required_provider_calls_exceed_call_budget"
assert without_planner.status is FeasibilityStatus.FEASIBLE
```

Also assert `contract_planning` is reserved once when affordable, no `comparison_plan` reservation is introduced, and the contract's `max_llm_calls` is never incremented.

- [ ] **Step 2: Run budget/core RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_execution_core.py tests/test_agentic_v9_execution_policy.py -q
```

Expected: failures because the request/core still use `comparison_plan_requested` and feasibility reserves `comparison_plan`.

- [ ] **Step 3: Replace the transient request flag and feasibility argument**

Change `V9ExecutionRequest` to:

```python
contract_plan_requested: bool = False
```

In `execution_core.py`, use `contract_planning` only when this flag is true; otherwise preserve `route_plan`. In `validate_post_contract_feasibility`, replace `comparison_plan_requested` with `contract_plan_requested` and add one pending `contract_planning` call. Keep `route_plan_used` for callers that already consumed route planning; do not count both for the Active deterministic route path.

Do not remove the historical `comparison_plan` phase from phase policy, LLM-call schema, or accounting enums; historical campaigns must remain readable.

- [ ] **Step 4: Run GREEN and Ruff**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_execution_core.py tests/test_agentic_v9_execution_policy.py -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9/schemas.py data_base/agentic_v9/execution_core.py data_base/agentic_v9/budget_feasibility.py tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_execution_core.py
```

Expected: optional planning is represented by `contract_planning`, downstream feasibility remains independently testable, and final reserve is still mandatory.

- [ ] **Step 5: Commit Task 5**

```powershell
git add data_base/agentic_v9/schemas.py data_base/agentic_v9/execution_core.py data_base/agentic_v9/budget_feasibility.py tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_execution_core.py tests/test_agentic_v9_execution_policy.py
git commit -m "fix(agentic-v9): admit optional atomic contract planning"
```

---

### Task 6: Activate Atomic Contract in the campaign runtime

**Files:**
- Modify: `evaluation/agentic_v9_campaign_runtime.py:30-80,161-260,340-480,800-950,1120-1280,1540-1555`
- Modify: `tests/test_agentic_v9_campaign_runtime.py`
- Modify: `tests/test_agentic_v9_provider_boundary.py`
- Modify: `tests/test_agentic_v9_full_rollback.py`
- Modify: `tests/test_agentic_v9_smoke_runner.py`

**Interfaces:**
- Consumes: deterministic admission `runtime_contract`, `QuestionContractPlanner.prepare/plan`, `BudgetedLlmInvoker`, optional planner feasibility, explicit comparison slot bindings, and current execution stages.
- Produces: Active v2 runtime for every Agentic v9 run, one consolidated planner call at most, zero active independent comparison calls, and honest trace/metrics.

- [ ] **Step 1: Add RED Active-runtime tests before changing production**

Add integration tests proving:

```python
v9 = result.agent_trace["agentic_v9"]
contract = v9["query_contract"]
assert contract["contract_version"] == "2"
assert [slot["slot_id"] for slot in contract["required_slots"]] == ["S1", "S2"]
assert v9["metrics"]["atomic_planner_call_count"] <= 1
assert v9["metrics"]["comparison_planner_call_count"] == 0
assert v9["metrics"]["slot_binding_method"] == "task_target_inherited"
assert v9["metrics"]["semantic_qualification"] == "not_enabled"
assert not any(call["phase"] == "comparison_plan" for call in recorded_calls)
```

Cover four runtime cases:

1. high-confidence deterministic question: zero contract-planning calls;
2. low-confidence comparison: exactly one `contract_planning` call supplies both slots and comparison metadata;
3. planner budget rejection: degraded v2 executes retrieval/final answer rather than returning `configuration_incompatible`;
4. malformed/unauthorized planner response: degraded v2, no retry, route/scope/budget unchanged.

Add a test proving the actual final provider message remains exactly the current `Question: ...\n\nEvidence:\n...` form. Add a test proving new runs no longer emit behavior-affecting `requirement_guidance` and do not call `ComparisonPlanner`.

- [ ] **Step 2: Run campaign RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_full_rollback.py -q
```

Expected: failures because admission's v1 contract is still active, comparison uses a separate provider call, and shadow guidance still alters retrieval queries.

- [ ] **Step 3: Prepare atomic decomposition before creating the execution request**

Immediately after `build_v9_admission_contract`, call:

```python
preparation = QuestionContractPlanner.prepare(
    question=question,
    base_contract=runtime_contract,
)
request = V9ExecutionRequest(
    question=question,
    requested_doc_ids=list(source_scope.authorized_doc_ids),
    setup_snapshot=dict(setup_snapshot),
    trace_id=trace_id,
    contract_plan_requested=preparation.semantic_planning_requested,
)
```

Do not use `is_suspected_comparison` to request a separate call.

- [ ] **Step 4: Implement optional planner admission and degraded fallback**

Inside `plan_contract`:

1. evaluate feasibility with `contract_plan_requested=True` when preparation requests it;
2. if that optional reservation is incompatible, evaluate downstream feasibility again with `False`;
3. raise `_ConfigurationIncompatible` only if the downstream-only check also fails;
4. create the existing `RunBudgetController` from the unchanged base route budget;
5. create one `BudgetedLlmInvoker` for phase `contract_planning`, purpose `atomic_contract_planning`;
6. call `QuestionContractPlanner.plan(... allow_semantic_planning=planner_admitted)`;
7. store its v2 contract, exact call count, and latency in `state`.

Do not add one to `runtime_contract.max_llm_calls`.

- [ ] **Step 5: Bind strict structured output for the consolidated purpose**

Change `_provider_for_purpose` so `atomic_contract_planning` receives `bind_json_schema(... atomic_contract_planner_response_schema())`. Retain the old `agentic_v9_comparison_plan` binding only for non-active historical/unit compatibility; no campaign runtime path may request it.

- [ ] **Step 6: Remove active shadow guidance and independent comparison wiring**

Delete the runtime imports and invocation of `ComparisonPlanner` and `apply_comparison_overlay`. Stop calling `_initial_requirement_guidance` and `_requirement_guided_query`; retrieval uses `task.query` directly. New traces do not emit `requirement_guidance` or `requirement_shadow` as an alternative plan.

Do not delete historical parser/accounting support for stored `comparison_plan` LLM-call rows.

- [ ] **Step 7: Update comparison coverage to explicit slot mappings and emit honest metrics**

Replace every `comparison-subject:{subject_id}` coverage check with membership in `subject.evidence_slot_ids`. Derive final evidence subject IDs the same way.

Set execution metrics from actual state populated inside `plan_contract`:

```python
{
    "atomic_planner_call_count": state["atomic_planner_call_count"],
    "comparison_planner_call_count": 0,
    "slot_binding_method": "task_target_inherited",
    "semantic_qualification": "not_enabled",
}
```

For new runs, emit the existing `comparison` diagnostic projection only when a valid `comparison_plan` exists. Its legacy-named `planner_status` is `planned`; latency comes from the consolidated outcome and is zero for deterministic comparison plans. A degraded contract without a valid comparison plan relies on the Query Contract's atomic fallback fields and omits the comparison projection, avoiding false comparison-planner diagnostics. Historical projections remain readable unchanged.

- [ ] **Step 8: Run campaign GREEN, full focused v9 runtime suite, and Ruff**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_full_rollback.py tests/test_agentic_v9_smoke_runner.py tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_repair.py tests/test_agentic_v9_sufficiency_gate.py tests/test_agentic_v9_context_packer.py -q
.\.venv\Scripts\python.exe -m ruff check evaluation/agentic_v9_campaign_runtime.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_provider_boundary.py
```

Expected: all pass; active runtime is v2, optional planner failure is fail-soft, independent comparison provider work is zero, and final prompt remains unchanged.

- [ ] **Step 9: Commit Task 6**

```powershell
git add evaluation/agentic_v9_campaign_runtime.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_full_rollback.py tests/test_agentic_v9_smoke_runner.py
git commit -m "feat(agentic-v9): activate atomic query contracts"
```

---

### Task 7: Synchronize backend observability, smoke checks, profile, docs, and OpenAPI

**Files:**
- Modify: `evaluation/smoke_verification.py:500-700`
- Modify: `evaluation/retrieval_profiles.py:15-30`
- Modify: `tests/test_agentic_v9_smoke_runner.py`
- Modify: `tests/test_evaluation_retrieval_profiles.py`
- Modify: `tests/test_evaluation_export_redaction.py`
- Modify: `tests/test_evaluation_export_v2_schemas.py`
- Modify: `tests/test_openapi_artifacts.py`
- Modify: `docs/BACKEND.md`
- Modify: `docs/agentic-v9-smoke-verification.md`
- Generate: `openapi.json`
- Generate: `contracts/openapi-contract.json`

**Interfaces:**
- Consumes: final backend `QueryContract`, `ComparisonPlan`, and `V9ExecutionMetrics` from Tasks 1-6.
- Produces: strict API/export contract, smoke-verification rules, versioned execution profile, and operator documentation consumed by the frontend task.

- [ ] **Step 1: Add RED smoke and export tests for active atomic invariants**

Assert the verifier fails when:

- a successful new Agentic v9 run has contract version 1;
- slot IDs are not sequential `S1..S8`;
- a comparison subject references an unknown slot;
- `atomic_planner_call_count > 1`;
- `comparison_planner_call_count != 0`;
- any active LLM call uses phase `comparison_plan`;
- binding/qualification instrumentation is missing or falsely claims semantic validation.

Assert Export v2 round-trips a contract with non-empty obligations, constraints, comparison `evidence_slot_ids`, planner provenance, and all four new metrics. Also assert historical absent additive contract fields remain serializable.

- [ ] **Step 2: Run smoke/export RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_smoke_runner.py tests/test_evaluation_export_redaction.py tests/test_evaluation_export_v2_schemas.py tests/test_evaluation_retrieval_profiles.py -q
```

Expected: smoke assertions/profile expectations and exact export fixtures fail on the new contract.

- [ ] **Step 3: Update smoke verification and version the execution profile**

Validate provider calls by purpose/phase:

```python
atomic_calls = [
    call for call in _calls_for_run(llm_calls, run)
    if call.get("phase") == "contract_planning"
    and call.get("purpose") == "atomic_contract_planning"
]
comparison_calls = [
    call for call in _calls_for_run(llm_calls, run)
    if call.get("phase") == "comparison_plan"
]
```

Require `len(atomic_calls) <= 1` and `comparison_calls == []` for the new execution profile. Use `evidence_slot_ids` for comparison coverage verification. Keep historical profile behavior readable by branching on execution-profile version, not on Q IDs.

Change both Agentic v9 profile constants from `comparison_structured_v2` to the exact suffix `active_atomic_contract_v1`; update exact profile tests.

- [ ] **Step 4: Update operator documentation with actual semantics**

In `docs/BACKEND.md` and `docs/agentic-v9-smoke-verification.md`, document:

- deterministic route plus atomic overlay ownership;
- evidence slots versus obligations/constraints;
- zero/one `contract_planning` calls and zero active `comparison_plan` calls;
- degraded v2 behavior;
- `task_target_inherited` and `semantic_qualification=not_enabled` limitations;
- why slot coverage can improve without proving faithfulness;
- the exact fields to inspect before the next evidence-qualification checkpoint.

- [ ] **Step 5: Generate and verify OpenAPI artifacts**

```powershell
.\.venv\Scripts\python.exe scripts/sync_openapi_artifacts.py --write
.\.venv\Scripts\python.exe scripts/sync_openapi_artifacts.py --check
.\.venv\Scripts\python.exe -m pytest tests/test_openapi_artifacts.py -q
```

Expected: generated artifacts contain the new optional historical-safe contract fields, `ComparisonSubject.evidence_slot_ids`, and exact metrics.

- [ ] **Step 6: Run the backend checkpoint gate and Ruff**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_schemas.py tests/test_agentic_v9_requirement_decomposition.py tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_repair.py tests/test_agentic_v9_sufficiency_gate.py tests/test_agentic_v9_context_packer.py tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_execution_core.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_smoke_runner.py tests/test_evaluation_export_redaction.py tests/test_evaluation_export_v2_schemas.py tests/test_evaluation_retrieval_profiles.py tests/test_openapi_artifacts.py -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9 evaluation/agentic_v9_campaign_runtime.py evaluation/smoke_verification.py evaluation/retrieval_profiles.py tests/test_agentic_v9_schemas.py tests/test_agentic_v9_requirement_decomposition.py tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_repair.py tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_campaign_runtime.py
git diff --check
```

Expected: all tests and Ruff pass; OpenAPI is current; diff check is clean.

- [ ] **Step 7: Commit Task 7**

```powershell
git add evaluation/smoke_verification.py evaluation/retrieval_profiles.py tests/test_agentic_v9_smoke_runner.py tests/test_evaluation_retrieval_profiles.py tests/test_evaluation_export_redaction.py tests/test_evaluation_export_v2_schemas.py tests/test_openapi_artifacts.py docs/BACKEND.md docs/agentic-v9-smoke-verification.md openapi.json contracts/openapi-contract.json
git commit -m "docs(agentic-v9): publish active atomic observability"
```

---

### Task 8: Synchronize the strict frontend contract and expose Atomic planning state

**Repository:** `D:\flutterserver\Multimodal_RAG_System`

**Files:**
- Modify: `src/types/evaluation.ts:700-765,930-960,1000-1020`
- Modify: `src/services/evaluationExportSchema.ts:350-410,541-560`
- Modify: `src/services/evaluationExportSchema.test.ts`
- Modify: `src/types/evaluation.contract.test.ts`
- Modify: `src/components/evaluation/AgenticV9Trace.tsx:90-220`
- Modify: `src/components/evaluation/RunTraceTab.test.tsx:250-330`
- Modify: `docs/FRONTEND.md`
- Generate: `src/test/fixtures/agenticV9ApiContract.ts`

**Interfaces:**
- Consumes: backend OpenAPI artifacts and exact final shapes from Task 7.
- Produces: strict TypeScript/Zod parity and visible run-level Atomic Contract diagnostics without changing campaign execution.

- [ ] **Step 1: Pin the final backend OpenAPI and demonstrate compile/decoder RED**

Run:

```powershell
npm run contract:pin
```

Add a fully populated Export v2 fixture with:

```typescript
contract: {
  contract_version: '2',
  route: 'bounded_compare',
  intent: 'Compare A and B',
  required_slots: [slotS1, slotS2],
  synthesis_obligations: [{
    obligation_id: 'O1', kind: 'comparison',
    description: 'Compare the reported values.',
    depends_on_slot_ids: ['S1', 'S2'],
  }],
  response_constraints: [{
    constraint_id: 'C1', kind: 'prohibition',
    description: 'Do not claim a universal ranking.',
  }],
  comparison_plan: {
    subjects: [
      { subject_id: 'a', display_name: 'A', aliases: [], retrieval_query: 'A value', evidence_slot_ids: ['S1'] },
      { subject_id: 'b', display_name: 'B', aliases: [], retrieval_query: 'B value', evidence_slot_ids: ['S2'] },
    ],
    dimensions: ['reported value'], qualification: null,
  },
  slot_plan_status: 'complete',
  slot_plan_source: 'llm_planner',
  slot_plan_confidence: 'medium',
  slot_plan_fallback_reason: null,
  truncated_requirement_count: 0,
},
metrics: {
  atomic_planner_call_count: 1,
  comparison_planner_call_count: 0,
  slot_binding_method: 'task_target_inherited',
  semantic_qualification: 'not_enabled',
}
```

Run:

```powershell
npm test -- --run src/services/evaluationExportSchema.test.ts src/types/evaluation.contract.test.ts
npm run build
```

Expected: strict decoder and/or TypeScript compilation fails because the new fields are absent from frontend types/schemas.

- [ ] **Step 2: Add exact frontend types and strict Zod schemas**

Define exact unions and interfaces for obligations, constraints, plan source/confidence, and instrumentation. Add `evidence_slot_ids?: string[]` to `V9ComparisonSubject` so historical contracts remain readable while active v2 fixtures require it.

For historical compatibility, make new contract fields optional in `V9QueryContract` and `.optional()` in `v9ContractSchema`; when present they must be strict and exact. New metric fields are required in the Export v2 decoder because backend export normalization always emits them. Use `z.literal(0)` for `comparison_planner_call_count` and exact literals for binding/qualification.

Add invalid fixtures that reject unknown obligation kinds, unknown constraint kinds, planner count 2, comparison planner count 1, and unsupported semantic-qualification labels.

- [ ] **Step 3: Add RED UI assertions for planner provenance and honest limitations**

In `RunTraceTab.test.tsx`, render the populated v2 evidence and assert visible text:

```typescript
expect(screen.getByText('Atomic planning')).toBeInTheDocument();
expect(screen.getByText(/Source: llm_planner/)).toBeInTheDocument();
expect(screen.getByText(/Status: complete/)).toBeInTheDocument();
expect(screen.getByText(/Evidence requirements: 2/)).toBeInTheDocument();
expect(screen.getByText(/Synthesis obligations: 1/)).toBeInTheDocument();
expect(screen.getByText(/Response constraints: 1/)).toBeInTheDocument();
expect(screen.getByText(/task_target_inherited/)).toBeInTheDocument();
expect(screen.getByText(/Semantic qualification: not_enabled/)).toBeInTheDocument();
```

Also render a historical v1 record and assert all new values remain `N/A`, not zero or complete.

- [ ] **Step 4: Implement the minimal trace presentation**

Add one `Atomic planning` `TraceSection` to `AgenticV9Trace.tsx`. Display status, source, confidence, fallback reason, evidence/obligation/constraint counts, truncated count, atomic planner calls, independent comparison calls, binding method, and semantic qualification.

Below the counts, preview obligation and constraint descriptions using the existing bounded `PreviewList`. Do not add a new page, chart, filter, or editable control. Keep `ClaimEvidenceTab` limited to evidence slots.

- [ ] **Step 5: Update frontend documentation and run focused GREEN**

Document that new runs use Active Atomic Contract v2 and that `task_target_inherited` is not semantic support.

Run:

```powershell
npm test -- --run src/services/evaluationExportSchema.test.ts src/types/evaluation.contract.test.ts src/components/evaluation/RunTraceTab.test.tsx
npm run contract:check
npm run lint:ci
npm run build
```

Expected: all commands exit 0; build may retain the existing Vite large-chunk advisory but no errors.

- [ ] **Step 6: Run frontend docs/scripts gates**

```powershell
npm run docs:sync
npm run docs:check
npm run docs:links
npm run test:scripts
git diff --check
```

Expected: all pass and generated documentation is current.

- [ ] **Step 7: Commit Task 8**

```powershell
git add src/types/evaluation.ts src/services/evaluationExportSchema.ts src/services/evaluationExportSchema.test.ts src/types/evaluation.contract.test.ts src/components/evaluation/AgenticV9Trace.tsx src/components/evaluation/RunTraceTab.test.tsx src/test/fixtures/agenticV9ApiContract.ts docs/FRONTEND.md
git commit -m "feat(evaluation-ui): show active atomic contract state"
```

---

## Atomic Deployment Checkpoint

After Task 8, stop implementation. Do not begin semantic evidence qualification, final-prompt work, claim verification, or another review/fix wave until the user pushes both repositories and validates the real system.

### Final local verification

Backend:

```powershell
cd D:\flutterserver\pdftopng
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_schemas.py tests/test_agentic_v9_requirement_decomposition.py tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_repair.py tests/test_agentic_v9_sufficiency_gate.py tests/test_agentic_v9_context_packer.py tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_execution_core.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_smoke_runner.py tests/test_evaluation_export_redaction.py tests/test_evaluation_export_v2_schemas.py tests/test_evaluation_retrieval_profiles.py tests/test_openapi_artifacts.py -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9 evaluation/agentic_v9_campaign_runtime.py evaluation/smoke_verification.py evaluation/retrieval_profiles.py
.\.venv\Scripts\python.exe scripts/sync_openapi_artifacts.py --check
git status --short
```

Frontend:

```powershell
cd D:\flutterserver\Multimodal_RAG_System
npm test -- --run src/services/evaluationExportSchema.test.ts src/types/evaluation.contract.test.ts src/components/evaluation/RunTraceTab.test.tsx
npm run contract:check
npm run lint:ci
npm run build
npm run docs:check
npm run docs:links
git status --short
```

Expected: every command passes and both tracked worktrees are clean.

### Real-system validation package

Run the same Agentic v9 campaign corpus used for the current baseline. Export summary plus all-run observability and verify:

- every successful Agentic run has `contract_version=2`;
- active slots are sequential `S1..S8`;
- every comparison subject maps to existing evidence slots;
- deterministic questions report zero atomic planner calls;
- low-confidence questions report at most one atomic planner call;
- independent comparison planner calls are zero;
- planner fallback runs remain executable degraded v2 rather than `configuration_incompatible`;
- `slot_binding_method=task_target_inherited` and `semantic_qualification=not_enabled` are visible;
- retrieval query count, tokens, latency, correctness, faithfulness, and relevancy are compared against the same pre-checkpoint corpus;
- no conclusion is drawn from slot coverage alone about semantic faithfulness.

Record the backend and frontend task commit ranges. If the checkpoint must be removed, revert the complete Atomic task ranges rather than introducing a runtime flag or restoring only part of the old comparison/shadow path.
