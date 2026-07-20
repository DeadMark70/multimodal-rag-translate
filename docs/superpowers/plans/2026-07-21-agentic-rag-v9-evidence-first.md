# Agentic RAG v9 Evidence-First Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace full-answer-per-subtask Agentic v8 execution with a budgeted v9 flow that performs retrieval-only subtasks, builds provenance-safe evidence packets, checks collection-level sufficiency, and generates at most one formal final answer.

**Architecture:** Add a versioned `data_base.agentic_v9` core while retaining v8 as the rollback baseline. Split generic retrieval from answer generation, enforce phase policy and run budgets under Setup model/thinking authority, then project the shared core through Evaluation and streaming adapters.

**Tech Stack:** Python 3.13, FastAPI, Pydantic, LangChain Google Gemini, FAISS, BM25, Jina reranker, NetworkX GraphRAG, SQLite, pytest, React 18, TypeScript, Chakra UI, Vitest.

## Global Constraints

- Evaluation Setup controls `model_name`, thinking, and input/output ceilings.
- v9 phase policy controls `temperature`, `top_p`, and `top_k`.
- Effective output is `min(setup_max_output_tokens, phase_output_cap)`.
- Context packing is bounded by Setup max input and remaining run budget.
- Do not change model, embedding, FAISS, BM25, reranker, or add multi-model routing.
- v9 produces zero full subtask answers and at most one formal final answer.
- Graph is locator-only and is independent of complexity.
- Persist a versioned `execution_profile`; default remains v8 until release gates pass.
- New persisted/transport fields are optional or have defaults for historical compatibility.
- Prompts stay in `prompts/*.json` with version, description, required variables, and template.
- Use TDD, focused tests before broad tests, and one reviewable commit per task.

## File Map

New package `data_base/agentic_v9/` contains `schemas.py`, `phase_policy.py`, `route_planner.py`, `retrieval_tasks.py`, `evidence_pool.py`, `evidence_extractor.py`, `sufficiency_gate.py`, `conflict_gate.py`, `budget_controller.py`, and `execution_core.py`. Generic RAG boundaries live in new `data_base/rag_pipeline_schemas.py`, `rag_retrieval.py`, and `rag_generation.py`. Existing Evaluation, chat, LLM, prompt, persistence, and frontend modules remain adapters/consumers.

---

### Task 1: Define Versioned v9 Contracts

**Files:**
- Create: `data_base/agentic_v9/__init__.py`
- Create: `data_base/agentic_v9/schemas.py`
- Create: `tests/test_agentic_v9_schemas.py`
- Modify: `evaluation/trace_schemas.py`

**Interfaces:**
- Produces: `QueryContract`, `RequiredSlot`, `RetrievalTask`, `RetrievalResult`, `EvidenceItem`, `EvidencePacket`, `SufficiencyReport`, `FinalClaim`, `FinalAnswerResult`, `V9ExecutionResult`.
- Compatibility: v9 fields added to trace models use `None`/empty defaults.

- [ ] **Step 1: Write failing schema tests**

```python
def test_exact_structured_contract_rejects_graph() -> None:
    with pytest.raises(ValidationError):
        QueryContract(
            route="exact_structured",
            intent="benchmark_data",
            required_slots=[],
            entities=["Polyp-SAM"],
            locator_hints=["Table 1"],
            requires_graph=True,
            max_retrieval_rounds=1,
            max_llm_calls=4,
            runtime_token_budget=7500,
        )

def test_packet_rejects_missing_own_source_identity() -> None:
    with pytest.raises(ValidationError):
        EvidencePacket(
            evidence_id="E1", target_slot="value", statement="Dice is 0.877",
            support_type="direct", source={"doc_id": "", "chunk_id": "chunk-1"},
            verbatim_span="0.877", modality="text", confidence=0.9,
        )
```

- [ ] **Step 2: Run `.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_schemas.py -q`**

Expected: collection FAIL because the v9 module does not exist.

- [ ] **Step 3: Implement exact types**

```python
AgenticV9Route = Literal[
    "single_lookup", "bounded_compare", "exact_structured",
    "multi_document_exact", "multi_hop", "graph_relational",
]
SupportType = Literal[
    "direct", "calculated", "comparative_inference",
    "scope_constraint", "contradictory", "missing",
]
SlotStatus = Literal["supported", "missing", "conflicted", "insufficient"]
StopReason = Literal[
    "sufficient", "repair_limit_reached", "llm_call_budget_exhausted",
    "runtime_token_budget_exhausted", "input_budget_exhausted",
    "retrieval_exhausted",
]
```

`QueryContract` validates route budgets and permits `requires_graph=True` only for `multi_hop`/`graph_relational`. `EvidencePacket` requires canonical `doc_id` or an explicit `provenance_status="missing"`; it never borrows a source ID.

- [ ] **Step 4: Run `.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_schemas.py tests/test_evaluation_observability_schema.py -q`**

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add data_base/agentic_v9/__init__.py data_base/agentic_v9/schemas.py evaluation/trace_schemas.py tests/test_agentic_v9_schemas.py
git commit -m "feat(agentic-v9): define evidence-first contracts"
```

---

### Task 2: Implement A-Type Phase Policy and Setup Ceilings

**Files:**
- Create: `data_base/agentic_v9/phase_policy.py`
- Create: `tests/test_agentic_v9_phase_policy.py`
- Modify: `core/llm_factory.py:87-327`
- Modify: `evaluation/model_capabilities.py:118-146`
- Modify: `evaluation/rag_modes.py:344-365`
- Test: `tests/test_llm_factory_override.py`
- Test: `tests/test_model_capabilities.py`

**Interfaces:**
- Produces: `resolve_phase_policy(...) -> EffectivePhasePolicy` and `agentic_phase_policy_scope(...)`.
- Preserves: v8 still uses Setup sampling/max output.

- [ ] **Step 1: Write failing policy tests**

```python
def test_final_answer_uses_phase_sampling_and_setup_ceiling() -> None:
    policy = resolve_phase_policy(
        "final_answer", setup_output_ceiling=800,
        setup_input_ceiling=8192, remaining_input_budget=5000,
    )
    assert (policy.temperature, policy.top_p, policy.top_k) == (0.25, 0.9, 40)
    assert policy.max_output_tokens == 800
    assert policy.max_input_tokens == 5000

def test_phase_scope_preserves_setup_model_and_thinking() -> None:
    with llm_runtime_override(model_name="gemini-2.5-flash-lite", thinking_enabled=False,
                              setup_max_output_tokens=8192, setup_max_input_tokens=16384):
        policy = resolve_phase_policy(
            "route_plan", setup_output_ceiling=8192,
            setup_input_ceiling=16384, remaining_input_budget=16384,
        )
        with agentic_phase_policy_scope(policy):
            config = current_llm_runtime_overrides()
            assert config["model_name"] == "gemini-2.5-flash-lite"
            assert config["thinking_enabled"] is False
            assert config["max_output_tokens"] == 384
```

- [ ] **Step 2: Run `.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_phase_policy.py tests/test_llm_factory_override.py -q`**

Expected: new tests FAIL; existing Setup-authority tests PASS.

- [ ] **Step 3: Implement fixed policy values**

```python
PHASE_POLICIES = {
    "route_plan": PhasePolicy(0.10, 0.80, 20, 384),
    "retrieval_judge": PhasePolicy(0.10, 0.70, 10, 96),
    "evidence_extract": PhasePolicy(0.10, 0.80, 20, 768),
    "conflict_arbitration": PhasePolicy(0.10, 0.80, 20, 256),
    "final_answer": PhasePolicy(0.25, 0.90, 40, 1536),
    "visual_extract": PhasePolicy(0.10, 0.80, 20, 768),
}
```

Carry `setup_max_output_tokens` and `setup_max_input_tokens` in runtime metadata. A nested v9 phase scope overrides sampling only and computes both ceilings.

- [ ] **Step 4: Run `.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_phase_policy.py tests/test_llm_factory_override.py tests/test_model_capabilities.py tests/test_rag_modes_agentic.py -q`**

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add data_base/agentic_v9/phase_policy.py core/llm_factory.py evaluation/model_capabilities.py evaluation/rag_modes.py tests/test_agentic_v9_phase_policy.py tests/test_llm_factory_override.py tests/test_model_capabilities.py
git commit -m "feat(agentic-v9): enforce setup ceilings and phase policies"
```

---

### Task 3: Add Run Budget Controller and Single Token Ledger

**Files:**
- Create: `data_base/agentic_v9/budget_controller.py`
- Create: `tests/test_agentic_v9_budget_controller.py`
- Modify: `core/llm_usage_context.py:29-135`
- Modify: `core/llm_usage_callback.py:29-105`
- Modify: `core/llm_factory.py:247-297`
- Modify: `evaluation/agentic_evaluation_service.py:1640-1735`

**Interfaces:**
- Produces: `RunBudgetController.reserve_call()`, `reconcile_usage()`, `snapshot()`, `agentic_budget_scope()`.
- Records exactly one ledger entry per provider call with effective config and actual usage.

- [ ] **Step 1: Write failing budget tests**

```python
def test_controller_stops_before_call_limit() -> None:
    controller = RunBudgetController(max_llm_calls=1, runtime_token_budget=1000)
    controller.reserve_call("route_plan", "planner", 100, {})
    with pytest.raises(BudgetExceededError, match="llm_call_budget_exhausted"):
        controller.reserve_call("final_answer", "synthesizer", 100, {})

def test_reconciliation_is_idempotent() -> None:
    controller = RunBudgetController(max_llm_calls=2, runtime_token_budget=1000)
    reservation = controller.reserve_call("route_plan", "planner", 100, {})
    usage = {"input_tokens": 80, "output_tokens": 20, "total_tokens": 100}
    controller.reconcile_usage(reservation.id, usage)
    controller.reconcile_usage(reservation.id, usage)
    assert controller.snapshot().total_tokens == 100
```

- [ ] **Step 2: Run `.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_budget_controller.py tests/test_llm_usage_callback.py -q`**

Expected: new tests FAIL.

- [ ] **Step 3: Implement a task-local budget observer**

The LLM callback reserves at start using estimated prompt tokens and reconciles at terminal callback using provider usage. Outside v9 scope it is inert. Record phase, purpose, effective model/thinking/sampling, ceilings, token categories, latency, status, and stop reason.

```python
@contextmanager
def agentic_budget_scope(controller: RunBudgetController):
    token = _AGENTIC_BUDGET_CONTROLLER.set(controller)
    try:
        yield controller
    finally:
        _AGENTIC_BUDGET_CONTROLLER.reset(token)
```

- [ ] **Step 4: Remove synthetic synthesis token duplication**

In the v8 trace, replace `token_usage={"total_tokens": total_tokens}` on `report_synthesis` with `token_usage={}` and metadata `token_accounting_source="provider_ledger"`. Provider `agent_synthesis` rows remain authoritative.

- [ ] **Step 5: Run `.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_budget_controller.py tests/test_llm_usage_callback.py tests/test_agentic_evaluation_service.py tests/test_evaluation_accounting.py -q`**

Expected: PASS; no duplicated synthesis total.

- [ ] **Step 6: Commit**

```powershell
git add data_base/agentic_v9/budget_controller.py core/llm_usage_context.py core/llm_usage_callback.py core/llm_factory.py evaluation/agentic_evaluation_service.py tests/test_agentic_v9_budget_controller.py tests/test_llm_usage_callback.py tests/test_agentic_evaluation_service.py
git commit -m "fix(agentic): make provider ledger the token source of truth"
```

---

### Task 4: Split Retrieval from Generation Without Behavior Drift

**Files:**
- Create: `data_base/rag_pipeline_schemas.py`
- Create: `data_base/rag_retrieval.py`
- Create: `data_base/rag_generation.py`
- Create: `tests/test_rag_retrieval_generation_split.py`
- Modify: `data_base/RAG_QA_service.py:1806-2545`

**Interfaces:**
- Produces: `retrieve_rag_evidence(...) -> RetrievalResult` with documents/metadata and no answer.
- Produces: `generate_answer_from_evidence(...) -> GeneratedRagAnswer`.
- Preserves: `rag_answer_question(...) -> RAGResult` and all v8 modes.

- [ ] **Step 1: Write wrapper parity tests**

```python
@pytest.mark.asyncio
async def test_wrapper_calls_retrieval_then_generation_once(monkeypatch) -> None:
    retrieval = RetrievalResult(documents=[document("doc-a", "chunk-a")], source_doc_ids=["doc-a"])
    retrieve = AsyncMock(return_value=retrieval)
    generate = AsyncMock(return_value=GeneratedRagAnswer(answer="supported", usage={"total_tokens": 12}))
    monkeypatch.setattr(rag_service, "retrieve_rag_evidence", retrieve)
    monkeypatch.setattr(rag_service, "generate_answer_from_evidence", generate)
    result = await rag_service.rag_answer_question(question="q", user_id="u", return_docs=True)
    retrieve.assert_awaited_once()
    generate.assert_awaited_once()
    assert result.answer == "supported"
```

Add parity fixtures for Naive, Advanced Multi-Query/CRAG, Graph locator-to-chunk, empty retrieval, and visual verification.

- [ ] **Step 2: Run `.\.venv\Scripts\python.exe -m pytest tests/test_rag_retrieval_generation_split.py -q`**

Expected: FAIL before extraction.

- [ ] **Step 3: Extract retrieval phases**

Move query expansion, hybrid retrieval, RRF, filtering, reranking, CRAG, Graph locator expansion, context enrichment, source resolution, and image collection into `retrieve_rag_evidence`. It must not call `get_llm("rag_qa")`.

- [ ] **Step 4: Extract generation and aggregate visual usage**

Move prompt construction, initial answer, and visual loop into `generate_answer_from_evidence`. Return `sum_normalized_usage([initial_usage, *visual_iteration_usage])`.

- [ ] **Step 5: Run `.\.venv\Scripts\python.exe -m pytest tests/test_rag_retrieval_generation_split.py tests/test_rag_graph_evidence_docs.py tests/test_rag_modes_agentic.py tests/test_rag_qa_prompts.py -q`**

Expected: PASS with wrapper parity.

- [ ] **Step 6: Commit**

```powershell
git add data_base/rag_pipeline_schemas.py data_base/rag_retrieval.py data_base/rag_generation.py data_base/RAG_QA_service.py tests/test_rag_retrieval_generation_split.py
git commit -m "refactor(rag): split retrieval from answer generation"
```

---

### Task 5: Build Shared Evidence Pool and Fix Provenance

**Files:**
- Create: `data_base/agentic_v9/evidence_pool.py`
- Create: `tests/test_agentic_v9_evidence_pool.py`
- Modify: `evaluation/agentic_evaluation_service.py:910-942`

**Interfaces:**
- Produces: `EvidencePool.add_documents(task, documents)` and `items_for_slots(slot_ids)`.
- Consumes each `Document.metadata`; never a run-level `primary_source`.

- [ ] **Step 1: Write failing provenance/dedup tests**

```python
def test_pool_preserves_each_documents_source() -> None:
    pool = EvidencePool("run-Q16-1")
    pool.add_documents(task("t1", ["a"]), [document("doc-a", "chunk-1", "A")])
    pool.add_documents(task("t2", ["b"]), [document("doc-b", "chunk-2", "B")])
    assert [(x.source.doc_id, x.source.chunk_id) for x in pool.items] == [
        ("doc-a", "chunk-1"), ("doc-b", "chunk-2")]

def test_content_hash_is_fallback_only() -> None:
    pool = EvidencePool("run-1")
    pool.add_documents(task("t1", ["a"]), [
        document("doc-a", None, " repeated text "),
        document("doc-a", None, "repeated text"),
    ])
    assert len(pool.items) == 1
```

- [ ] **Step 2: Run `.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_evidence_pool.py tests/test_agentic_evaluation_service.py -q`**

Expected: new tests FAIL and current first-source behavior is exposed.

- [ ] **Step 3: Implement identity and score merge**

Primary key is `(doc_id, chunk_id, parent_id, page, asset_id)`; fallback is `(doc_id, sha256(normalized_text))`. Merge target slots and highest present scores without converting missing scores to zero.

- [ ] **Step 4: Change v8 `_build_evidence_units` to accept `documents: list[Document]`**

Derive every evidence unit from its own metadata. Missing provenance remains `None`/explicitly missing and never borrows another document ID.

- [ ] **Step 5: Run `.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_evidence_pool.py tests/test_agentic_evaluation_service.py tests/test_evaluation_analytics_context.py -q`**

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add data_base/agentic_v9/evidence_pool.py evaluation/agentic_evaluation_service.py tests/test_agentic_v9_evidence_pool.py tests/test_agentic_evaluation_service.py
git commit -m "fix(agentic): preserve per-chunk evidence provenance"
```

---

### Task 6: Implement Query Contract Routing and Retrieval Tasks

**Files:**
- Create: `data_base/agentic_v9/route_planner.py`
- Create: `data_base/agentic_v9/retrieval_tasks.py`
- Create: `tests/test_agentic_v9_route_planner.py`
- Create: `tests/test_agentic_v9_retrieval_tasks.py`
- Modify: `agents/planner.py:246-290, 491-520`
- Modify: `prompts/agentic_rag_prompts.json`

**Interfaces:**
- Produces: `build_query_contract(question, *, doc_ids, budget) -> QueryContract`.
- Produces: `build_retrieval_tasks(contract, question, doc_ids) -> list[RetrievalTask]`.
- Allows exactly one `route_plan` call when deterministic confidence is insufficient.

- [ ] **Step 1: Write failing Q9/Q15/Q16 route tests**

```python
@pytest.mark.parametrize(
    ("question", "route", "graph"),
    [
        ("SwinUNETR 和 nnU-Net 哪個表現更好？", "bounded_compare", False),
        ("根據 Figure 1、Table 1 計算 0.877 與 0.837 差值。", "exact_structured", False),
        ("查 GEPAR3D Appendix D、ODES 公式、U-KAN Theorem 1。", "multi_document_exact", False),
        ("找出未知文獻間的方法演進關係路徑。", "graph_relational", True),
    ],
)
def test_deterministic_routes(question: str, route: str, graph: bool) -> None:
    result = deterministic_route(question, doc_ids=None)
    assert result.contract.route == route
    assert result.contract.requires_graph is graph
    assert result.requires_llm is False
```

Add ambiguous mixed-query tests: one RoutePlan call, strict JSON parse, and fallback to capped `multi_hop` on timeout/invalid output.

- [ ] **Step 2: Run `.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_route_planner.py tests/test_agentic_v9_retrieval_tasks.py -q`**

Expected: FAIL before implementation.

- [ ] **Step 3: Implement deterministic precedence**

```text
multi-document locator groups
> explicit Figure/Table/Appendix/Theorem/formula/numeric locator
> explicit named A-vs-B comparison
> single lookup/definition/enumeration
> unknown-scope relation/lineage/path
> ambiguous mixed/multi-hop fallback
```

Graph must not be inferred from complexity, number of models, or metric words alone.

- [ ] **Step 4: Add one RoutePlan prompt and strict parser**

The prompt returns Query Contract JSON only and runs under `route_plan`. Persist `contract_source=deterministic|llm|fallback`.

- [ ] **Step 5: Build bounded retrieval tasks**

Each task contains target slot IDs, locator hints, doc scope, Graph/visual flags, and round index. Q9 produces two bounded queries; Q15 produces explicit locator tasks; Q16 produces three parallel source/locator groups.

- [ ] **Step 6: Run `.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_route_planner.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_rag_prompts.py tests/test_agentic_evaluation_service.py -q`**

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add data_base/agentic_v9/route_planner.py data_base/agentic_v9/retrieval_tasks.py agents/planner.py prompts/agentic_rag_prompts.json tests/test_agentic_v9_route_planner.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_rag_prompts.py
git commit -m "feat(agentic-v9): add query contracts and bounded retrieval tasks"
```

---

### Task 7: Add Evidence Packet Extraction

**Files:**
- Create: `data_base/agentic_v9/evidence_extractor.py`
- Create: `tests/test_agentic_v9_evidence_extractor.py`
- Modify: `prompts/agentic_rag_prompts.json`

**Interfaces:**
- Produces: `extract_evidence_packets(contract, pool, budget) -> list[EvidencePacket]`.
- Deterministic first: numbers, calculations, formulas, locator spans, theorem ranges, table rows, enumerations.
- LLM fallback: at most one batched `evidence_extract` call for unresolved prose slots.

- [ ] **Step 1: Write failing deterministic extraction tests**

```python
def test_numeric_extraction_preserves_span_and_source() -> None:
    packets = extract_numeric_packets(
        slot=slot("dice_values"),
        items=[evidence_item("doc-polyp", "chunk-table-1", "Seen 0.877; Unseen 0.837")],
    )
    assert [packet.exact_value for packet in packets] == ["0.877", "0.837"]
    assert all(packet.source.doc_id == "doc-polyp" for packet in packets)

def test_calculation_references_direct_premises() -> None:
    packet = calculate_difference(slot("dice_gap"), direct("E1", "0.877"), direct("E2", "0.837"))
    assert packet.exact_value == "0.040"
    assert packet.support_type == "calculated"
    assert packet.premise_evidence_ids == ["E1", "E2"]
```

Also cover normalized formulas and extraction of the Theorem 1 `m` range.

- [ ] **Step 2: Run `.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_evidence_extractor.py -q`**

Expected: FAIL.

- [ ] **Step 3: Implement deterministic extractors**

Preserve the surrounding sentence/row/block, modality, locator, scores, target slot, and source. A calculated packet stores the operation and direct premise IDs.

- [ ] **Step 4: Implement one batched prose curator**

Only unresolved non-inference slots enter the LLM. Reject returned evidence IDs or verbatim spans absent from the pool. Comparative verdicts remain for final generation.

- [ ] **Step 5: Run `.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_rag_prompts.py -q`**

Expected: PASS and at most one mocked prose-curation call.

- [ ] **Step 6: Commit**

```powershell
git add data_base/agentic_v9/evidence_extractor.py prompts/agentic_rag_prompts.json tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_rag_prompts.py
git commit -m "feat(agentic-v9): extract structured evidence packets"
```

---

### Task 8: Implement Sufficiency, Targeted Repair, and Selective CRAG

**Files:**
- Create: `data_base/agentic_v9/sufficiency_gate.py`
- Create: `tests/test_agentic_v9_sufficiency_gate.py`
- Modify: `data_base/agentic_v9/retrieval_tasks.py`
- Modify: `agents/evaluator.py`

**Interfaces:**
- Produces: `evaluate_sufficiency(contract, packets) -> SufficiencyReport`.
- Produces: `build_targeted_repair_tasks(contract, report, round_index) -> list[RetrievalTask]`.
- Produces: `classify_retrieval_action(...) -> pass|correct|llm_judge`.

- [ ] **Step 1: Write failing slot/repair tests**

```python
def test_missing_theorem_range_creates_only_targeted_repair() -> None:
    report = evaluate_sufficiency(q16_contract(), packets_without_m_range())
    tasks = build_targeted_repair_tasks(q16_contract(), report, round_index=1)
    assert report.missing_slots == ["theorem_m_range"]
    assert len(tasks) == 1
    assert tasks[0].target_slot_ids == ["theorem_m_range"]
    assert "Theorem 1" in tasks[0].query
    assert tasks[0].requires_graph is False

def test_supported_or_explicitly_insufficient_slots_stop() -> None:
    report = evaluate_sufficiency(contract(), supported_and_insufficient_packets())
    assert report.sufficient is True
    assert report.stop_reason == "sufficient"
```

Cover route repair limits and every budget stop reason.

- [ ] **Step 2: Write selective CRAG branch tests**

Exact locator hit -> `pass`; no documents/wrong scope -> `correct`; semantically relevant but partial/conflicting -> `llm_judge`; grader failure -> conservative deterministic correction rather than fail-open relevance.

- [ ] **Step 3: Run `.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_sufficiency_gate.py tests/test_agentic_v9_retrieval_tasks.py -q`**

Expected: FAIL.

- [ ] **Step 4: Implement collection-level sufficiency**

Do not use answer length. `comparative_inference` is supported only when all required premises exist. `missing` closes a slot only as explicit insufficiency, never positive support.

- [ ] **Step 5: Implement targeted rewrite**

Use `{missing_entity} {slot_description} {locator_hint}`, one query by default and maximum two. LLM grading runs only for `llm_judge` under the 96-token output cap.

- [ ] **Step 6: Run `.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_sufficiency_gate.py tests/test_agentic_v9_retrieval_tasks.py tests/test_agentic_evaluation_service.py -q`**

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add data_base/agentic_v9/sufficiency_gate.py data_base/agentic_v9/retrieval_tasks.py agents/evaluator.py tests/test_agentic_v9_sufficiency_gate.py tests/test_agentic_v9_retrieval_tasks.py
git commit -m "feat(agentic-v9): add slot sufficiency and targeted repair"
```

---

### Task 9: Add Scope-Aware Conflict and One Final Answer

**Files:**
- Create: `data_base/agentic_v9/conflict_gate.py`
- Create: `tests/test_agentic_v9_conflict_gate.py`
- Create: `tests/test_agentic_v9_final_answer.py`
- Modify: `agents/synthesizer.py:130-365`
- Modify: `prompts/agentic_rag_prompts.json`

**Interfaces:**
- Produces: `detect_conflict_candidates(packets)`.
- Produces: `generate_final_answer(...) -> FinalAnswerResult`.
- Produces: `render_answer_with_citations(...)` and `verify_high_risk_claims(...)`.

- [ ] **Step 1: Write failing conflict tests**

```python
def test_different_dataset_is_scope_difference() -> None:
    packets = [metric_packet("E1", "BraTS", "0.90"), metric_packet("E2", "AMOS", "0.82")]
    assert detect_conflict_candidates(packets) == []

def test_same_scope_different_value_is_conflict() -> None:
    packets = [metric_packet("E1", "BraTS", "0.90"), metric_packet("E2", "BraTS", "0.82")]
    assert detect_conflict_candidates(packets)[0].evidence_ids == ["E1", "E2"]
```

- [ ] **Step 2: Write failing one-generation/verifier tests**

```python
@pytest.mark.asyncio
async def test_final_answer_calls_llm_once_and_uses_known_evidence_ids() -> None:
    invoke = AsyncMock(return_value=structured_answer([claim("C1", ["E1"])]))
    result = await generate_final_answer("q", contract(), [packet("E1")], sufficient(), None, budget(), invoke)
    invoke.assert_awaited_once()
    assert "[E1]" in render_answer_with_citations(result, [packet("E1")])

def test_bad_numeric_claim_is_removed_without_regeneration() -> None:
    verified = verify_high_risk_claims(answer_with_numeric("0.999", ["E1"]), [numeric_packet("E1", "0.877")])
    assert verified.claims[0].status == "unsupported"
    assert verified.requires_regeneration is False
```

- [ ] **Step 3: Run `.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_conflict_gate.py tests/test_agentic_v9_final_answer.py -q`**

Expected: FAIL.

- [ ] **Step 4: Implement conditional arbitration**

Only unresolved identical-scope candidates invoke an arbitration call under `conflict_arbitration`. Input is evidence packets/provenance, never subtask answers.

- [ ] **Step 5: Implement structured final generation and renderer**

Accept only `verdict`, `claims`, and `unsupported_slots`. Every claim references existing evidence IDs. Comparative claims reference all required premises. Invalid IDs are rejected before deterministic citation rendering.

- [ ] **Step 6: Implement selective high-risk verification**

Numbers, calculations, formulas, source identity, and comparative premises are deterministic. Permit one batched verifier only for unresolved causal claims. Failed claims are removed/qualified without another final-generation call.

- [ ] **Step 7: Run `.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_conflict_gate.py tests/test_agentic_v9_final_answer.py tests/test_agentic_rag_prompts.py tests/test_agentic_evaluation_service.py -q`**

Expected: PASS; v8 synthesizer tests remain PASS.

- [ ] **Step 8: Commit**

```powershell
git add data_base/agentic_v9/conflict_gate.py agents/synthesizer.py prompts/agentic_rag_prompts.json tests/test_agentic_v9_conflict_gate.py tests/test_agentic_v9_final_answer.py tests/test_agentic_rag_prompts.py
git commit -m "feat(agentic-v9): generate one evidence-linked final answer"
```

---

### Task 10: Implement Bounded v9 Execution Core

**Files:**
- Create: `data_base/agentic_v9/execution_core.py`
- Create: `tests/test_agentic_v9_execution_core.py`
- Modify: `data_base/agentic_v9/__init__.py`

**Interfaces:**
- Produces: `V9ExecutionCore.execute(request: V9ExecutionRequest) -> V9ExecutionResult`.
- Emits structured state events shared by Evaluation and SSE adapters.

- [ ] **Step 1: Write failing state-machine tests**

```python
@pytest.mark.asyncio
async def test_single_lookup_has_no_planner_graph_repair_or_subtask_answer() -> None:
    result = await core_with_fakes("single_lookup").execute(request("simple"))
    assert result.metrics.planner_calls == 0
    assert result.metrics.graph_calls == 0
    assert result.metrics.repair_rounds == 0
    assert result.metrics.full_subtask_answer_count == 0
    assert result.metrics.final_answer_generation_count == 1

@pytest.mark.asyncio
async def test_q16_repairs_only_m_range_then_stops() -> None:
    result = await q16_core_fixture().execute(request("Q16"))
    assert result.repair_tasks[0].target_slot_ids == ["theorem_m_range"]
    assert result.metrics.graph_calls == 0
    assert result.metrics.final_answer_generation_count == 1
```

Add budget exhaustion, retrieval exhaustion, true conflict, no conflict, and partial-answer cases.

- [ ] **Step 2: Run `.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_execution_core.py -q`**

Expected: FAIL.

- [ ] **Step 3: Implement explicit transitions**

```text
resolve_setup -> route -> retrieve -> pool -> extract -> sufficiency
-> bounded repair? -> conflict gate -> arbitration? -> final generation
-> verification -> render -> complete
```

Every loop edge checks call, token, input, and route-repair budgets. Budget exhaustion returns a qualified partial answer and typed stop reason.

- [ ] **Step 4: Add architectural guard**

Patch `rag_answer_question` to raise and prove v9 calls only `retrieve_rag_evidence` plus final generation.

- [ ] **Step 5: Run `.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_*.py -q`**

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add data_base/agentic_v9/execution_core.py data_base/agentic_v9/__init__.py tests/test_agentic_v9_execution_core.py
git commit -m "feat(agentic-v9): add bounded evidence-first execution core"
```

---

### Task 11: Add Evaluation Adapter, Versioning, and Observability

**Files:**
- Modify: `evaluation/agentic_evaluation_service.py:650-1795`
- Modify: `evaluation/rag_modes.py:27-380`
- Modify: `evaluation/campaign_engine.py:140-240, 560-660`
- Modify: `evaluation/trace_schemas.py`
- Create: `tests/test_agentic_v9_evaluation_adapter.py`
- Test: `tests/test_campaign_engine.py`
- Test: `tests/test_rag_modes_agentic.py`

**Interfaces:**
- Produces profiles `agentic_eval_v8_*` and `agentic_eval_v9_evidence_first_*`.
- Persists contract, slots, evidence/provenance counts, repairs, Graph/conflict decisions, budget ledger, stop reason, and final-generation count.
- Preserves v8 default and historical projections.

- [ ] **Step 1: Write failing version-selection tests**

```python
@pytest.mark.asyncio
async def test_v9_flag_uses_shared_core_and_v9_profile(monkeypatch) -> None:
    monkeypatch.setenv("AGENTIC_EXECUTION_VERSION", "v9")
    execute = AsyncMock(return_value=v9_result())
    monkeypatch.setattr(v9_core, "execute", execute)
    result = await run_campaign_case(...)
    execute.assert_awaited_once()
    assert result.execution_profile.startswith("agentic_eval_v9_evidence_first_")

def test_default_remains_v8(monkeypatch) -> None:
    monkeypatch.delenv("AGENTIC_EXECUTION_VERSION", raising=False)
    assert resolve_agentic_execution_version() == "v8"
```

- [ ] **Step 2: Write trace/accounting assertions**

Assert zero full subtask answers, at most one final generation, every repair has target slots, ledger total equals persisted runtime total, and synthesis trace does not duplicate tokens.

- [ ] **Step 3: Run `.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_evaluation_adapter.py tests/test_rag_modes_agentic.py -q`**

Expected: FAIL before adapter wiring.

- [ ] **Step 4: Implement the thin Evaluation adapter**

Translate campaign input to `V9ExecutionRequest`, project `V9ExecutionResult` to `RAGResult`, and create trace metadata. Do not retain planning/retrieval/repair/synthesis logic in the adapter.

- [ ] **Step 5: Add opt-in shadow comparison**

`AGENTIC_V9_SHADOW=true` runs only for explicitly marked research campaigns while v8 stays authoritative. Store the shadow as a distinct run/profile; never silently double normal campaign cost or replace the v8 answer.

- [ ] **Step 6: Run `.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_evaluation_adapter.py tests/test_rag_modes_agentic.py tests/test_campaign_engine.py tests/test_evaluation_analytics_api.py -q`**

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add evaluation/agentic_evaluation_service.py evaluation/rag_modes.py evaluation/campaign_engine.py evaluation/trace_schemas.py tests/test_agentic_v9_evaluation_adapter.py tests/test_rag_modes_agentic.py tests/test_campaign_engine.py
git commit -m "feat(evaluation): add versioned agentic v9 adapter"
```

---

### Task 12: Project Shared v9 Core Through Streaming Chat

**Files:**
- Modify: `data_base/agentic_chat_service.py:58-630`
- Modify: `data_base/schemas_agentic_chat.py`
- Modify: `data_base/sse_events.py`
- Create: `tests/test_agentic_v9_chat_adapter.py`
- Test: `tests/test_agentic_chat_service.py`

**Interfaces:**
- Consumes the same core/state events as Evaluation.
- Produces existing SSE names with optional v9 fields; terminal `complete` matches synchronous result.
- Preserves legacy v7 when v9 is not selected for chat.

- [ ] **Step 1: Write failing event-projection test**

```python
@pytest.mark.asyncio
async def test_v9_stream_projects_core_events_in_order() -> None:
    events = await collect_stream(fake_core_events([
        "route_resolved", "retrieval_started", "evidence_ready",
        "sufficiency_checked", "final_generation_started", "complete",
    ]))
    assert event_types(events) == [
        "plan_complete", "task_start", "task_done",
        "evaluation_update", "synthesis_start", "complete",
    ]
    assert complete_payload(events)["answer"] == synchronous_result().answer
```

Add cancellation/error coverage and assert the adapter contains no independent drilldown loop.

- [ ] **Step 2: Run `.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_chat_adapter.py tests/test_agentic_chat_service.py -q`**

Expected: new tests FAIL before conversion.

- [ ] **Step 3: Implement v9 event projection**

Keep legacy v7 for rollback. The v9 path subscribes to core events and maps SSE only; it does not override `_execute_tasks`, `_drill_down_loop`, or `_synthesize_execution_results`.

- [ ] **Step 4: Run `.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_chat_adapter.py tests/test_agentic_chat_service.py tests/test_deep_research_persistence.py -q`**

Expected: PASS and the legacy-v7 policy test remains PASS.

- [ ] **Step 5: Commit**

```powershell
git add data_base/agentic_chat_service.py data_base/schemas_agentic_chat.py data_base/sse_events.py tests/test_agentic_v9_chat_adapter.py tests/test_agentic_chat_service.py
git commit -m "feat(agentic-chat): stream shared v9 execution events"
```

---

### Task 13: Expose v9 Research Observability in Evaluation Center

**Files:**
- Modify: `Multimodal_RAG_System/src/services/evaluationApi.ts`
- Modify: `Multimodal_RAG_System/src/pages/EvaluationCenter.mappers.ts`
- Modify: `Multimodal_RAG_System/src/components/evaluation/AgentTraceViewer.tsx`
- Modify: `Multimodal_RAG_System/src/components/evaluation/AgentTraceViewer.test.tsx`
- Modify: `Multimodal_RAG_System/src/pages/EvaluationCenter.mappers.test.ts`

**Interfaces:**
- Consumes optional v9 trace fields; old campaigns render without synthetic zeroes.
- Displays route/contract source, required slots, evidence/provenance, repairs, budget/stop reason, Graph/conflict decisions, and final-generation count.

- [ ] **Step 1: Write failing UI tests**

```tsx
it('renders v9 states without inventing zeroes', () => {
  render(<AgentTraceViewer detail={v9Detail({ missingProvenance: null })} />);
  expect(screen.getByText('multi_document_exact')).toBeInTheDocument();
  expect(screen.getByText('theorem_m_range')).toBeInTheDocument();
  expect(screen.getByText('runtime_token_budget_exhausted')).toBeInTheDocument();
  expect(screen.getByText('Missing provenance: N/A')).toBeInTheDocument();
});
```

Add a legacy fixture where the v9 panel is absent rather than populated with zeroes.

- [ ] **Step 2: Run `npm test -- --run src/components/evaluation/AgentTraceViewer.test.tsx src/pages/EvaluationCenter.mappers.test.ts` from `D:\flutterserver\Multimodal_RAG_System`**

Expected: FAIL before type/projection changes.

- [ ] **Step 3: Add optional API types and null-safe mapping**

Measured values use `number | null`; absent evidence counts, budgets, and decisions never default to zero/success.

- [ ] **Step 4: Render compact Contract, Evidence, Repair, Budget, and Decisions sections**

Use existing Chakra components and token-only language. Do not add monetary fields.

- [ ] **Step 5: Run frontend verification**

Run from `D:\flutterserver\Multimodal_RAG_System`:

```powershell
npm test -- --run src/components/evaluation/AgentTraceViewer.test.tsx src/pages/EvaluationCenter.mappers.test.ts src/pages/EvaluationCenter.integration.test.tsx
npm run lint:ci
npm run build
```

Expected: every command exits 0.

- [ ] **Step 6: Commit in the frontend repository**

```powershell
git add src/services/evaluationApi.ts src/pages/EvaluationCenter.mappers.ts src/components/evaluation/AgentTraceViewer.tsx src/components/evaluation/AgentTraceViewer.test.tsx src/pages/EvaluationCenter.mappers.test.ts
git commit -m "feat(evaluation-ui): expose agentic v9 evidence and budget trace"
```

---

### Task 14: Add Representative Shadow Benchmark and Release Gates

**Files:**
- Create: `tests/test_agentic_v9_representative_cases.py`
- Create: `scripts/compare_agentic_v8_v9.py`
- Create: `docs/agentic-v9-operations.md`
- Modify: `docs/design-docs/agentic-semantic-router-v8.md`
- Modify: `agent.md`

**Interfaces:**
- Produces fail-closed v8/v9 comparison JSON with tokens, phases, calls, latency, RAGAS, slots, unsupported claims, and provenance failures.
- Does not automatically change the default.

- [ ] **Step 1: Add Q9/Q15/Q16 invariant tests**

Q9: `bounded_compare`, Graph off, at most four calls, zero subtask answers, one final generation. Q15: `exact_structured`, extract `0.877`, `0.837`, calculate `0.040`, Graph/Multi-Query/general CRAG off. Q16: three initial retrieval groups, Graph off, repair only `theorem_m_range`.

- [ ] **Step 2: Implement fail-closed comparison CLI**

```text
--database <evaluation.db>
--v8-campaign <id>
--v9-campaign <id>
--output <comparison.json>

Required gates:
accounting_complete == true
agentic_naive_token_ratio <= 4.0
correctness_delta_vs_v8 >= 0
relevancy >= 0.70
p95_latency_ms < 25000
full_subtask_answer_count == 0
final_answer_generation_count <= 1
provenance_missing_rate == 0
```

Exit 1 when required data is missing or a gate fails; never replace missing values with zero.

- [ ] **Step 3: Run focused backend verification**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_*.py tests/test_rag_retrieval_generation_split.py tests/test_agentic_evaluation_service.py tests/test_agentic_chat_service.py tests/test_rag_modes_agentic.py tests/test_campaign_engine.py -q
.\.venv\Scripts\python.exe -m ruff check core data_base agents evaluation tests
```

Expected: PASS/exit 0.

- [ ] **Step 4: Run full backend suite**

Run: `.\.venv\Scripts\python.exe -m pytest`

Expected: exit 0. If an existing legacy experiment/RAGAS dependency blocks collection, report exact pass/skip/fail counts and blocking module; do not claim full-suite success.

- [ ] **Step 5: Run fresh smoke campaign**

Use Q9, Q15, Q16, modes `naive,agentic`, one repeat, complete RAGAS, identical model/corpus/prompt/index/evaluator snapshots. Export comparison JSON. Do not compare historical campaigns with different snapshots as an apples-to-apples gate.

- [ ] **Step 6: Document operation and rollback**

```text
AGENTIC_EXECUTION_VERSION=v8  # rollback/default before promotion
AGENTIC_EXECUTION_VERSION=v9  # evidence-first runtime
AGENTIC_V9_SHADOW=true        # explicit research campaigns only
```

Document profiles, stop reasons, qualified partial answers, and promotion prerequisites.

- [ ] **Step 7: Commit**

```powershell
git add tests/test_agentic_v9_representative_cases.py scripts/compare_agentic_v8_v9.py docs/agentic-v9-operations.md docs/design-docs/agentic-semantic-router-v8.md agent.md
git commit -m "test(agentic-v9): add evidence-first release gates"
```

---

## Review Checkpoints

- **A — after Task 3:** Setup authority, phase sampling, token ledger, synthesis trace, visual accounting.
- **B — after Task 5:** legacy wrapper parity and zero provenance borrowing.
- **C — after Task 8:** Q9/Q15/Q16 routing, evidence extraction, sufficiency, targeted repair without final generation.
- **D — after Task 10:** zero subtask answers and at most one final generation.
- **E — after Task 13:** Evaluation, Chat, and UI are adapters over one core; old campaigns remain readable.
- **Release — Task 14:** promote only after a fresh apples-to-apples campaign passes all fail-closed gates.

## Explicit Non-Goals

- No model or retrieval-backend replacement.
- No multi-model phase routing or Graph database migration.
- No removal of v8/v7 rollback paths before promotion observation.
- No pricing fields; evaluation remains token-only.
- No full-answer regeneration after verifier failure.
- No comparison of incompatible campaign snapshots as one benchmark.

## Plan Self-Review Result

- Tasks 1–14 cover every approved design requirement.
- Phase ranges are resolved to fixed values: retrieval judge `0.10`, final answer `0.25`.
- Critical order is trust foundation -> retrieval split -> provenance -> contracts/evidence -> sufficiency -> one generation -> shared core -> adapters -> release gates.
- Every task contains a failing test, implementation target, verification command, and isolated commit.
- No placeholders, model changes, pricing work, or unbounded loops remain.
