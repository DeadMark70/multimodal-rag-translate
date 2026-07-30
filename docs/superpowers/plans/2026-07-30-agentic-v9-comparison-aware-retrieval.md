# Agentic v9 Comparison-Aware Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded semantic comparison planner to Agentic v9 so explicit comparison subjects receive independent retrieval, balanced final evidence, one corrective retrieval when needed, and auditable fail-soft behavior.

**Architecture:** Keep the admitted v9 contract and source scope authoritative, then apply an optional comparison overlay before retrieval-task compilation. A single budgeted planner identifies 2–4 subjects and dimensions; subject tasks run through the existing Hybrid 8 → rerank 8 boundary, subject-aware selection retains 4–6 packets, and existing sufficiency/repair/final synthesis completes the run. Planner failures discard only the overlay and execute the current whole-question v9 path.

**Tech Stack:** Python 3.11, asyncio, Pydantic, existing Agentic v9 execution core and budget controller, LangChain `Document`, pytest, Ruff.

## Global Constraints

- Runtime must never consume evaluation `expected_sources`, `ground_truth`, expected evidence, or benchmark atomic facts.
- Only suspected comparison or judgment questions may invoke the planner.
- Planner provider calls: at most one, 64-second timeout or remaining overall deadline, no retry.
- Overall Agentic v9 deadline remains 128 seconds.
- Planner/provider/parse failures must return to the current whole-question v9 path without failing the run or clearing contexts.
- Source authorization remains authoritative; the comparison overlay cannot add documents.
- Each subject uses Hybrid 8 → rerank 8 → subject top 2.
- Two-subject final evidence limit is 4; three- or four-subject limit is 6.
- At most one corrective retrieval round; it does not call the planner.
- Missing post-repair subjects produce `qualified_partial`, not a failed run.
- Native RAG, Graph/Visual behavior, existing final synthesis prompt, and non-comparison v9 behavior remain unchanged.
- Every provider call must be admitted by `RunBudgetController` and attributed to a real phase.
- Implementation is backend-only. Do not add Evaluation Center UI work in this plan.

---

## File Structure

### New files

- `data_base/agentic_v9/comparison_planner.py` — intent pre-filter, strict planner parsing, timeout classification, and comparison-overlay construction.
- `data_base/agentic_v9/comparison_context.py` — deterministic subject-balanced evidence selection.
- `prompts/agentic_v9_comparison_planner.json` — answer-free planner prompt.
- `tests/test_agentic_v9_comparison_planner.py` — planner schema, parsing, intent, timeout, and fallback tests.
- `tests/test_agentic_v9_comparison_context.py` — subject task and final-selection tests.

### Modified files

- `data_base/agentic_v9/schemas.py` — typed comparison subjects, plan, outcome, and optional task/contract subject identity.
- `data_base/agentic_v9/phase_policy.py` — `comparison_plan` provider phase.
- `data_base/agentic_v9/budget_feasibility.py` — reserve the optional comparison-planning call before provider execution.
- `data_base/agentic_v9/retrieval_tasks.py` — compile one task per planned comparison subject.
- `data_base/agentic_v9/repair.py` — one deterministic missing-subject repair.
- `evaluation/agentic_v9_campaign_runtime.py` — overlay integration, fail-soft execution, subject mapping, balanced packing, and trace fields.
- `evaluation/campaign_engine.py` — allowlisted persistence of comparison diagnostics.
- `evaluation/smoke_verification.py` — verify planner/coverage invariants in redacted exports.
- `tests/test_agentic_v9_phase_policy.py`
- `tests/test_agentic_v9_budget_feasibility.py`
- `tests/test_agentic_v9_retrieval_tasks.py`
- `tests/test_agentic_v9_repair.py`
- `tests/test_agentic_v9_campaign_runtime.py`
- `tests/test_campaign_engine.py`
- `tests/test_evaluation_export_redaction.py`
- `tests/test_agentic_v9_smoke_runner.py`

---

### Task 1: Typed Comparison Planner and Budget Phase

**Files:**
- Create: `data_base/agentic_v9/comparison_planner.py`
- Create: `prompts/agentic_v9_comparison_planner.json`
- Create: `tests/test_agentic_v9_comparison_planner.py`
- Modify: `data_base/agentic_v9/schemas.py:63-166`
- Modify: `data_base/agentic_v9/phase_policy.py:24-48`
- Modify: `tests/test_agentic_v9_phase_policy.py:1-80`
- Test: `tests/test_agentic_v9_schemas.py`

**Interfaces:**
- Produces:
  - `ComparisonPlannerFallbackReason = Literal["timeout", "provider_error", "invalid_response", "schema_violation", "not_comparison"]`
  - `ComparisonSubject(subject_id, display_name, aliases, retrieval_query)`
  - `ComparisonPlan(subjects, dimensions, qualification)`
  - `ComparisonPlannerOutcome(status, plan, fallback_reason, latency_ms)`
  - `is_suspected_comparison(question: str) -> bool`
  - `ComparisonPlanner.plan(...) -> ComparisonPlannerOutcome`
  - `apply_comparison_overlay(contract: QueryContract, plan: ComparisonPlan) -> QueryContract`
- Consumes: existing `LlmInvoker.invoke(phase, purpose, messages)` interface.

- [ ] **Step 1: Write failing schema and intent tests**

Add tests proving:

```python
def test_q4_is_suspected_comparison() -> None:
    assert is_suspected_comparison(
        "結合 Params 與 FLOPs，Mamba 是否具當前 3D 醫療分割最高計算效率？"
    )


def test_metric_only_lookup_is_not_suspected_comparison() -> None:
    assert not is_suspected_comparison("請找出 nnMamba 的 Params 與 FLOPs。")


def test_comparison_plan_rejects_duplicate_subjects() -> None:
    with pytest.raises(ValueError):
        ComparisonPlan(
            subjects=[
                _subject("nnmamba", "nnMamba"),
                _subject("nnmamba", "nnMamba"),
            ],
            dimensions=["parameters"],
        )
```

Also cover 2–4 subjects, a fifth subject, unknown fields, duplicate aliases,
bounded lengths, and a retrieval query that omits its subject name/alias.

- [ ] **Step 2: Run the new tests and verify failure**

Run:

```powershell
pytest tests/test_agentic_v9_comparison_planner.py tests/test_agentic_v9_schemas.py -q
```

Expected: collection or import failure because comparison types do not exist.

- [ ] **Step 3: Add the typed comparison schema**

In `schemas.py`, add strict Pydantic models with `extra="forbid"`:

```python
class ComparisonSubject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject_id: str = Field(min_length=1, max_length=80)
    display_name: str = Field(min_length=1, max_length=160)
    aliases: list[str] = Field(default_factory=list, max_length=8)
    retrieval_query: str = Field(min_length=1, max_length=512)


class ComparisonPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subjects: list[ComparisonSubject] = Field(min_length=2, max_length=4)
    dimensions: list[str] = Field(default_factory=list, max_length=12)
    qualification: str | None = Field(default=None, max_length=512)


class ComparisonPlannerOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["planned", "fallback"]
    plan: ComparisonPlan | None = None
    fallback_reason: ComparisonPlannerFallbackReason | None = None
    latency_ms: float = Field(ge=0)
```

Validators must normalize IDs and aliases, enforce unique subject IDs, remove
duplicate aliases, require the retrieval query to contain the display name or
an alias, and reject filenames/document IDs. Numeric tokens copied from the
question remain valid; newly invented numeric tokens are rejected by the
planner parser, which receives the original question.

Extend:

```python
class QueryContract(BaseModel):
    comparison_plan: ComparisonPlan | None = None


class RetrievalTask(BaseModel):
    subject_id: str | None = None
```

The optional defaults preserve existing serialized contracts and tasks.

- [ ] **Step 4: Add the comparison prompt and strict parser**

Create `prompts/agentic_v9_comparison_planner.json` with a system instruction
that:

- identifies explicit subjects and dimensions only;
- never answers the question or selects a winner;
- never invents filenames, document IDs, or numeric results;
- returns one JSON object matching the schema;
- returns `{"is_comparison": false}` when no explicit comparison exists.

Implement in `comparison_planner.py`:

```python
class ComparisonPlanner:
    def __init__(self, *, llm_invoker: LlmInvoker) -> None: ...

    async def plan(
        self,
        *,
        question: str,
        authorized_source_names: Sequence[str],
        timeout_seconds: float,
    ) -> ComparisonPlannerOutcome: ...
```

Use `asyncio.timeout(timeout_seconds)`. Parse both direct JSON and a single
fenced JSON block. Convert timeout, provider exception, malformed JSON, and
schema failure into outcomes; do not raise those errors to the run.

Implement `is_suspected_comparison()` with bounded multilingual markers for
comparison, relative judgment, contradiction, selection, and superlatives.
Metrics alone must not trigger it.

Implement `apply_comparison_overlay()` so it:

- keeps `route`, `resolved_source_scope`, graph/visual policy, and setup limits;
- creates one required `RequiredSlot` per subject;
- gives each slot only that subject's aliases/name as `entity_ids`;
- stores `comparison_plan`;
- sets `max_repair_rounds` to at least one;
- does not alter source authorization.

- [ ] **Step 5: Add the provider phase**

In `phase_policy.py`, add:

```python
"comparison_plan": PhasePolicy(0.10, 0.80, 20, 768),
```

Update the phase-policy parameterized test so the phase is setup-capped and
has exactly one provider attempt.

- [ ] **Step 6: Add parser/fallback tests**

Use an injected fake invoker to prove:

- valid Q4 JSON returns subjects `nnMamba` and `EfficientMedNeXt-L`;
- `Params` and `FLOPs` are dimensions, not subjects;
- fenced JSON parses;
- timeout maps to `fallback/timeout`;
- provider error maps to `fallback/provider_error`;
- malformed JSON maps to `fallback/invalid_response`;
- schema errors map to `fallback/schema_violation`;
- `is_comparison=false` maps to `fallback/not_comparison`;
- the invoker is called once at most.

Add generalization fixtures for:

- `SwinUNETR` versus `MedNeXt`;
- `Model A / Model B / Model C` latency comparison;
- mixed Chinese/English subject names and aliases;
- a non-comparison technical summary;
- a five-subject response that must be grouped by the prompt or rejected by
  schema validation and handled through fallback.

- [ ] **Step 7: Run focused tests**

Run:

```powershell
pytest tests/test_agentic_v9_comparison_planner.py tests/test_agentic_v9_schemas.py tests/test_agentic_v9_phase_policy.py -q
ruff check data_base/agentic_v9/comparison_planner.py data_base/agentic_v9/schemas.py data_base/agentic_v9/phase_policy.py tests/test_agentic_v9_comparison_planner.py
```

Expected: all tests pass and Ruff reports no errors.

- [ ] **Step 8: Commit Task 1**

```powershell
git add data_base/agentic_v9/comparison_planner.py data_base/agentic_v9/schemas.py data_base/agentic_v9/phase_policy.py prompts/agentic_v9_comparison_planner.json tests/test_agentic_v9_comparison_planner.py tests/test_agentic_v9_schemas.py tests/test_agentic_v9_phase_policy.py
git commit -m "feat(agentic-v9): add bounded comparison planner"
```

---

### Task 2: Subject Retrieval Tasks and Balanced Evidence Selection

**Files:**
- Create: `data_base/agentic_v9/comparison_context.py`
- Create: `tests/test_agentic_v9_comparison_context.py`
- Modify: `data_base/agentic_v9/retrieval_tasks.py:28-215`
- Modify: `tests/test_agentic_v9_retrieval_tasks.py`

**Interfaces:**
- Consumes: `QueryContract.comparison_plan`, `RetrievalTask.subject_id`.
- Produces:
  - `RetrievalTaskCompiler.compile()` emits one task per comparison subject.
  - `select_balanced_comparison_packets(...) -> tuple[EvidencePacket, ...]`.
  - `comparison_final_limit(subject_count: int) -> int`.

- [ ] **Step 1: Write failing subject-task tests**

Add a Q4 contract with two comparison subjects and assert:

```python
plan = RetrievalTaskCompiler().compile(
    question=Q4,
    query_id="q4",
    contract=comparison_contract,
)

assert [(task.subject_id, task.target_slot_ids) for task in plan.tasks] == [
    ("nnmamba", ["comparison-subject:nnmamba"]),
    ("efficientmednext_l", ["comparison-subject:efficientmednext_l"]),
]
assert all(len(task.target_slot_ids) == 1 for task in plan.tasks)
```

Assert each query contains the correct subject, all tasks retain the same
authorized scope, no qualification task is added, and non-comparison contracts
produce the existing task plan byte-for-byte.

- [ ] **Step 2: Run retrieval-task tests and verify failure**

Run:

```powershell
pytest tests/test_agentic_v9_retrieval_tasks.py -q
```

Expected: comparison contracts still use the old bounded-compare compiler.

- [ ] **Step 3: Compile one subject task per planned subject**

At the top of `RetrievalTaskCompiler.compile()`, after source-scope validation,
branch on `contract.comparison_plan is not None`. Add
`_compile_comparison_subjects()` that creates only round-one tasks:

```python
RetrievalTask(
    task_id=f"{query_id}:round-1:comparison:{subject.subject_id}",
    round_id="round-1",
    query_id=query_id,
    query=subject.retrieval_query,
    target_slot_ids=[f"comparison-subject:{subject.subject_id}"],
    source_scope=scope,
    source_group_id=f"comparison:{subject.subject_id}",
    subject_id=subject.subject_id,
    locator_hints=[],
    graph_policy=contract.graph_policy or "never",
    visual_required=False,
)
```

Do not modify `_compile_bounded_compare()` for contracts without the overlay.

- [ ] **Step 4: Write failing balanced-selection tests**

Create packets tied to subject slots and prove:

- two subjects select at most two packets each and four total;
- three subjects select at least one available packet each and six total;
- four subjects select at least one available packet each and six total;
- an exact duplicate is emitted once;
- remaining positions use descending quality;
- a low-scoring subject's only packet is retained;
- packets without a declared subject slot are excluded from specialized
  selection;
- the input packet sequence is not mutated.

- [ ] **Step 5: Implement deterministic balanced selection**

In `comparison_context.py`, implement:

```python
def comparison_final_limit(subject_count: int) -> int:
    if subject_count == 2:
        return 4
    if 3 <= subject_count <= 4:
        return 6
    raise ValueError("comparison subject count must be between 2 and 4")


def select_balanced_comparison_packets(
    packets: Sequence[EvidencePacket],
    *,
    plan: ComparisonPlan,
    quality_by_evidence_id: Mapping[str, float],
) -> tuple[EvidencePacket, ...]:
    ...
```

Selection order:

1. deduplicate by stable evidence/source identity;
2. group only by the subject slot declared in the plan;
3. sort each group by finite quality descending, then original order;
4. for two subjects, take up to two per subject;
5. for three/four subjects, take the best available packet per subject, then
   fill remaining positions by quality;
6. never exceed the final limit.

Do not add a minimum score threshold.

- [ ] **Step 6: Run focused tests**

Run:

```powershell
pytest tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_comparison_context.py -q
ruff check data_base/agentic_v9/retrieval_tasks.py data_base/agentic_v9/comparison_context.py tests/test_agentic_v9_comparison_context.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 2**

```powershell
git add data_base/agentic_v9/retrieval_tasks.py data_base/agentic_v9/comparison_context.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_comparison_context.py
git commit -m "feat(agentic-v9): retrieve comparison subjects independently"
```

---

### Task 3: Runtime Integration, Budget Admission, and Planner Fallback

**Files:**
- Modify: `data_base/agentic_v9/budget_feasibility.py:191-285`
- Modify: `evaluation/agentic_v9_campaign_runtime.py:116-245`
- Modify: `tests/test_agentic_v9_budget_feasibility.py`
- Modify: `tests/test_agentic_v9_campaign_runtime.py`

**Interfaces:**
- Consumes: Task 1 `ComparisonPlanner`, `is_suspected_comparison()`,
  `apply_comparison_overlay()`.
- Produces: an admitted comparison overlay or a durable
  `comparison_planner_fallback` state while preserving the base contract.

- [ ] **Step 1: Write failing budget-feasibility tests**

Extend `validate_post_contract_feasibility()` tests with:

```python
result = validate_post_contract_feasibility(
    contract=contract,
    setup_snapshot=setup,
    remaining_token_budget=contract.runtime_token_budget,
    remaining_llm_calls=contract.max_llm_calls + 1,
    comparison_plan_requested=True,
)

assert result.required_provider_calls["comparison_plan"] == 1
```

Also prove insufficient call or token budget returns
`configuration_incompatible` before a planner provider is invoked, and the
default `comparison_plan_requested=False` response remains unchanged.

- [ ] **Step 2: Run budget tests and verify failure**

Run:

```powershell
pytest tests/test_agentic_v9_budget_feasibility.py -q
```

Expected: the function does not accept `comparison_plan_requested`.

- [ ] **Step 3: Admit the optional planner phase**

Add `comparison_plan_requested: bool = False`. When true:

- add `comparison_plan: 1` to pending provider calls;
- include its phase reservation in token feasibility;
- include one additional call in effective contract capacity;
- preserve every existing default result when false.

Do not represent the comparison planner as `route_plan` or
`contract_planning`; accounting must use its real phase.

- [ ] **Step 4: Write failing runtime integration tests**

Use injected planner/fake provider and retrieval adapters to prove:

- a suspected Q4 question invokes the planner once;
- the budget controller exists before the invocation;
- planned output returns an overlaid contract with subject slots;
- a non-comparison question does not invoke the planner;
- planner timeout, provider error, invalid response, and `not_comparison`
  compile the original base contract;
- fallback retrieval still returns contexts and a usable answer;
- a forced planner timeout never clears contexts;
- `comparison_specialization_enabled=False` restores the old path;
- overall execution uses the existing 128-second deadline.

- [ ] **Step 5: Integrate the overlay in `plan_contract`**

Add constructor injection:

```python
comparison_specialization_enabled: bool = True
```

Within `execute()`:

1. compute `comparison_plan_requested` from the feature flag and
   `is_suspected_comparison(question)`;
2. create an effective base contract with one additional allowed LLM call only
   when requested;
3. run post-contract feasibility with `comparison_plan_requested`;
4. create `RunBudgetController` before planner invocation;
5. call `ComparisonPlanner` through `BudgetedLlmInvoker` using phase
   `comparison_plan` and purpose `agentic_v9_comparison_plan`;
6. cap planner wait at `min(64, deadline.remaining_seconds())`;
7. apply a valid overlay or retain the base contract on fallback;
8. store a structured planner state in runtime `state`.

The base `route`, source scope, setup snapshot, graph policy, and visual policy
must remain unchanged by the overlay.

- [ ] **Step 6: Preserve per-subject retrieval identity**

In `retrieve()`:

- record `task.subject_id` beside `task_slot_ids`;
- after the existing retrieval/rerank boundary, keep at most the first two
  documents for a comparison subject task;
- retain the pre-selection candidate diagnostics;
- add `subject_id` and selected count to the task diagnostic;
- do not slice non-comparison tasks.

The runtime must continue to request the existing Hybrid 8 → rerank 8 profile.

- [ ] **Step 7: Run focused runtime tests**

Run:

```powershell
pytest tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_budget_controller.py tests/test_agentic_v9_budgeted_llm.py -q
ruff check data_base/agentic_v9/budget_feasibility.py evaluation/agentic_v9_campaign_runtime.py tests/test_agentic_v9_campaign_runtime.py
```

Expected: all tests pass and existing non-comparison runtime tests remain
unchanged.

- [ ] **Step 8: Commit Task 3**

```powershell
git add data_base/agentic_v9/budget_feasibility.py evaluation/agentic_v9_campaign_runtime.py tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_campaign_runtime.py
git commit -m "feat(agentic-v9): integrate comparison planning fail-soft"
```

---

### Task 4: Subject Coverage, Balanced Packing, and One-Shot Repair

**Files:**
- Modify: `data_base/agentic_v9/repair.py:48-150`
- Modify: `evaluation/agentic_v9_campaign_runtime.py:248-480,1144-1225`
- Modify: `tests/test_agentic_v9_repair.py`
- Modify: `tests/test_agentic_v9_campaign_runtime.py`
- Test: `tests/test_agentic_v9_sufficiency_gate.py`
- Test: `tests/test_agentic_v9_context_packer.py`

**Interfaces:**
- Consumes: comparison subject slots/tasks and
  `select_balanced_comparison_packets()`.
- Produces: subject-bound evidence, one missing-subject repair, final coverage,
  and `qualified_partial` when coverage remains incomplete.

- [ ] **Step 1: Write failing evidence-binding tests**

Add a runtime projection test with:

- nnMamba task targeting only `comparison-subject:nnmamba`;
- EfficientMedNeXt-L task targeting only
  `comparison-subject:efficientmednext_l`;
- two nnMamba chunks and no EfficientMedNeXt-L chunks.

Assert nnMamba packets contain only the nnMamba slot and that sufficiency leaves
the EfficientMedNeXt-L slot `not_found`. Evidence from A must not support B.

- [ ] **Step 2: Run the evidence tests and verify failure**

Run:

```powershell
pytest tests/test_agentic_v9_campaign_runtime.py -q
```

Expected: the current projection or runtime state does not expose complete
subject-bound coverage behavior.

- [ ] **Step 3: Bind evidence to originating subject tasks**

Extend runtime state with `task_subject_ids`. Ensure
`_evidence_packets_for_results()` receives only the target slot IDs recorded
for the originating task. Add `subject_id` to its quality/diagnostic projection
without changing `EvidencePacket` validity semantics.

Do not infer subject membership from chunk text, filenames, or reranker score.

- [ ] **Step 4: Write failing repair tests**

Add tests proving:

```python
repair = build_repair_plan(
    contract=comparison_contract,
    sufficiency=missing_efficient,
    query_id="q4",
    repair_round_index=1,
    final_budget_available=True,
)

assert len(repair.tasks) == 1
assert repair.tasks[0].subject_id == "efficientmednext_l"
assert "EfficientMedNeXt-L" in repair.tasks[0].query
assert "FLOPs" in repair.tasks[0].query
```

Also assert:

- the query reuses dimensions from the plan;
- source scope remains unchanged;
- a second repair round returns `repair_round_cap_reached`;
- no planner call is involved;
- no repair is emitted when all subjects are covered.

- [ ] **Step 5: Implement one deterministic subject repair**

When `contract.comparison_plan` exists:

- use a comparison-specific cap of exactly one repair round regardless of the
  base route's legacy cap;
- find missing subject slots from `SufficiencyEvaluation`;
- build at most one repair task for the missing subject group;
- query from subject display name, aliases, dimensions, and original slot
  description;
- set the task's `subject_id` and target only the missing subject slot;
- preserve the authorized source scope;
- never call the planner.

Keep legacy repair behavior unchanged when no comparison plan exists.

- [ ] **Step 6: Apply balanced selection before existing packing**

In runtime `pack()`:

```python
packets_for_final = (
    select_balanced_comparison_packets(
        packets,
        plan=contract.comparison_plan,
        quality_by_evidence_id=state["quality_by_evidence_id"],
    )
    if contract.comparison_plan is not None
    else packets
)
```

Pass `packets_for_final` into the existing `EvidenceContextPacker` with the
existing `soft_final_pack_r1` policy and unchanged final instruction.

After repair, recompute balanced selection from all available packets. The
limit remains four for two subjects and six for three/four subjects.

- [ ] **Step 7: Prove final status cannot upgrade**

Add end-to-end runtime tests:

- both subjects covered → `complete`;
- one subject missing before repair, found during repair → `complete`;
- one subject missing after repair → successful `qualified_partial`;
- partial result retains supported evidence;
- final generation never reports `complete` over a partial sufficiency report;
- synthesis system/user prompt text is unchanged.

- [ ] **Step 8: Run focused coverage tests**

Run:

```powershell
pytest tests/test_agentic_v9_repair.py tests/test_agentic_v9_sufficiency_gate.py tests/test_agentic_v9_context_packer.py tests/test_agentic_v9_comparison_context.py tests/test_agentic_v9_campaign_runtime.py -q
ruff check data_base/agentic_v9/repair.py evaluation/agentic_v9_campaign_runtime.py tests/test_agentic_v9_repair.py
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 4**

```powershell
git add data_base/agentic_v9/repair.py evaluation/agentic_v9_campaign_runtime.py tests/test_agentic_v9_repair.py tests/test_agentic_v9_campaign_runtime.py
git commit -m "feat(agentic-v9): enforce comparison subject coverage"
```

---

### Task 5: Durable Diagnostics, Redacted Export, and Smoke Verification

**Files:**
- Modify: `evaluation/agentic_v9_campaign_runtime.py:500-620,752-825`
- Modify: `evaluation/campaign_engine.py:500-620`
- Modify: `evaluation/smoke_verification.py:500-620`
- Modify: `tests/test_agentic_v9_campaign_runtime.py`
- Modify: `tests/test_campaign_engine.py`
- Modify: `tests/test_evaluation_export_redaction.py`
- Modify: `tests/test_agentic_v9_smoke_runner.py`

**Interfaces:**
- Consumes: runtime comparison-planner state, subject task diagnostics, repair
  plans, final packet selection.
- Produces: allowlisted `agent_trace.agentic_v9.comparison` diagnostics and
  smoke-verification requirements.

- [ ] **Step 1: Write failing trace-projection tests**

Assert the runtime trace contains:

```python
comparison = trace["agentic_v9"]["comparison"]
assert comparison["planner_status"] == "planned"
assert comparison["planner_latency_ms"] >= 0
assert comparison["subjects"][0]["subject_id"] == "nnmamba"
assert comparison["coverage_before_repair"] == ["nnmamba"]
assert comparison["missing_before_repair"] == ["efficientmednext_l"]
assert comparison["repair_executed"] is True
assert comparison["coverage_after_repair"] == [
    "nnmamba",
    "efficientmednext_l",
]
```

Also assert every final selected chunk/evidence item has its `subject_id`,
planner usage is attributed to `comparison_plan`, and fallback traces contain
only a safe enumerated reason.

- [ ] **Step 2: Run persistence/export tests and verify failure**

Run:

```powershell
pytest tests/test_campaign_engine.py tests/test_evaluation_export_redaction.py tests/test_agentic_v9_smoke_runner.py -q
```

Expected: comparison diagnostics are not allowlisted or verified.

- [ ] **Step 3: Add one bounded runtime projection**

Persist under:

```python
"comparison": {
    "planner_status": ...,
    "planner_latency_ms": ...,
    "planner_fallback_reason": ...,
    "is_comparison": ...,
    "subjects": ...,
    "dimensions": ...,
    "task_diagnostics": ...,
    "coverage_before_repair": ...,
    "missing_before_repair": ...,
    "repair_executed": ...,
    "coverage_after_repair": ...,
    "missing_after_repair": ...,
    "final_status": ...,
    "final_evidence_subjects": ...,
}
```

Subject task diagnostics include a bounded query preview or SHA-256 query hash,
candidate/selected counts, reranker state, and final document/chunk IDs. Do not
store full prompt text in this projection.

- [ ] **Step 4: Allowlist safe persistence and redacted export**

Update `campaign_engine.py` projection to retain only:

- enumerated status/reason values;
- subject IDs, bounded names/aliases, and dimensions;
- query hash and bounded preview;
- counts, coverage arrays, repair boolean;
- document/chunk IDs under the existing export policy.

Reject or omit unknown nested keys. Ensure `analytics.export_campaign()` uses
the stored allowlisted projection and does not reintroduce expected source
metadata.

Add export tests proving:

- default redaction exports comparison structure;
- secrets and full prompts remain absent;
- benchmark `ground_truth`, `expected_sources`, and expected evidence do not
  appear as planner inputs;
- explicit existing full-prompt controls behave exactly as before.

- [ ] **Step 5: Extend smoke verification**

For Agentic v9 comparison runs, add requirements that:

- planned subjects are between 2 and 4;
- planner call count is at most one;
- planner fallback still has contexts;
- final evidence subject IDs are declared subjects;
- complete comparison status covers all required subjects;
- qualified partial lists at least one missing subject;
- final count is within 4/6;
- comparison-plan token usage has complete accounting.

Non-comparison exports must remain valid without a `comparison` block.

- [ ] **Step 6: Run the full scoped suite**

Run:

```powershell
pytest tests/test_agentic_v9_comparison_planner.py tests/test_agentic_v9_comparison_context.py tests/test_agentic_v9_phase_policy.py tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_v9_repair.py tests/test_agentic_v9_sufficiency_gate.py tests/test_agentic_v9_context_packer.py tests/test_agentic_v9_campaign_runtime.py tests/test_campaign_engine.py tests/test_evaluation_export_redaction.py tests/test_agentic_v9_smoke_runner.py -q
ruff check data_base/agentic_v9 evaluation/agentic_v9_campaign_runtime.py evaluation/campaign_engine.py evaluation/smoke_verification.py tests/test_agentic_v9_comparison_planner.py tests/test_agentic_v9_comparison_context.py
```

Expected: all tests pass, no Ruff failures, and existing non-comparison/Native
tests remain unchanged.

- [ ] **Step 7: Commit Task 5**

```powershell
git add evaluation/agentic_v9_campaign_runtime.py evaluation/campaign_engine.py evaluation/smoke_verification.py tests/test_agentic_v9_campaign_runtime.py tests/test_campaign_engine.py tests/test_evaluation_export_redaction.py tests/test_agentic_v9_smoke_runner.py
git commit -m "feat(evaluation): persist v9 comparison coverage"
```

---

## Post-implementation Experimental Gate

This gate is intentionally not an implementation task or automatic quality
claim.

1. Run Q4 three times with the same model preset, thinking configuration,
   source corpus, reranker, and batch size as the current baseline.
2. Confirm:
   - subjects are nnMamba and EfficientMedNeXt-L;
   - final contexts contain both subject groups when both are retrievable;
   - planner call count is at most one;
   - final context count is at most four;
   - planner timeout injection returns a usable fallback answer;
   - token and phase accounting are complete.
3. Only after Q4 passes, run the fixed 16-question paired evaluation.
4. Compare per-question correctness, faithfulness, relevancy, latency, total
   tokens, planner tokens, fallback frequency, and subject coverage.
5. Roll back the comparison overlay if non-comparison questions change.
6. If evidence from both subjects reaches final synthesis but the answer still
   mixes subjects or declares an unsupported winner, create a separate
   evidence-aware synthesis design; do not patch the prompt inside this wave.

## Estimated Change Size

- Five reviewable backend tasks.
- Two focused new modules plus one prompt.
- Approximately 7–10 production files touched, mainly with optional fields and
  isolated branches.
- No database migration is expected because comparison diagnostics live in the
  existing JSON trace projection.
- No frontend or Native RAG changes.

This is a medium change with a bounded blast radius. The planner itself is
small; runtime budgeting, subject evidence identity, and fail-soft fallback are
the parts requiring the strongest review.
