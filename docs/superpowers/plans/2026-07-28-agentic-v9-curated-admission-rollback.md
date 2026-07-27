# Agentic v9 Curated Admission Rollback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore non-empty authorized retrieval evidence by removing strict evidence extraction from the production admission path without reverting the source/locator safety work in commit `97e9cdb`.

**Architecture:** Perform a semantic partial revert rather than `git revert 97e9cdb`. Production v9 passes authorized, slot-compatible candidate packets through `prose_curate`; the strict extractor remains available as a library but cannot erase runtime evidence. The production contract no longer reserves an unused `evidence_extract` provider call.

**Tech Stack:** Python 3.11, pytest, FastAPI runtime adapters, Pydantic Agentic v9 schemas.

## Global Constraints

- Source authorization remains fail-closed.
- Explicit structured-locator mismatches remain rejected.
- Missing structured metadata remains eligible for semantic evidence.
- Reranking remains fail-soft.
- No per-slot LLM calls are added.
- Visual extraction behavior is unchanged.
- Do not run `git revert 97e9cdb`; later locator fixes depend on parts of that commit.

---

### Task 1: Restore candidate evidence admission

**Files:**
- Modify: `tests/test_agentic_v9_campaign_runtime.py`
- Modify: `tests/test_agentic_v9_contract_planner.py`
- Modify: `evaluation/agentic_v9_campaign_runtime.py:519-554`
- Modify: `data_base/agentic_v9/contract_planner.py:245`

**Interfaces:**
- Consumes: `tuple[EvidencePacket, ...]` produced by deterministic candidate projection after source and structured-locator filtering.
- Produces: the same packet tuple for sufficiency, packing, final synthesis, and `agent_trace["agentic_v9"]["evidence_packets"]`.

- [ ] **Step 1: Write the failing runtime regression test**

Replace the curated-only expectation with a production-path assertion that
authorized reranked candidates survive malformed evidence extraction. Add this
provider beside the existing test providers:

```python
class _MalformedEvidenceThenFinalProvider:
    def __init__(self) -> None:
        self.ainvoke = AsyncMock(side_effect=self._respond)
        self.extraction_calls = 0

    async def _respond(self, messages):
        content = messages[-1]["content"]
        if isinstance(content, str) and "Source evidence:" in content:
            self.extraction_calls += 1
            return SimpleNamespace(
                content={"invalid": "not an evidence packet response"},
                usage_metadata={"input_tokens": 12, "output_tokens": 7},
            )

        payload = json.loads(content)
        packets = payload["packed_evidence_packets"]
        return SimpleNamespace(
            content={
                "supported_findings": [
                    {
                        "slot_id": packet["slot_ids"][0],
                        "statement": packet["statement"],
                        "evidence_ids": [packet["evidence_id"]],
                    }
                    for packet in packets
                ],
                "unresolved_requirements": [],
            },
            usage_metadata={"input_tokens": 12, "output_tokens": 7},
        )
```

Then add:

```python
@pytest.mark.asyncio
async def test_runtime_preserves_authorized_candidates_when_strict_curation_is_malformed(
    monkeypatch,
) -> None:
    provider = _MalformedEvidenceThenFinalProvider()
    scope = ResolvedSourceScope(authorized_doc_ids=["doc-authorized"])
    contract = QueryContract(
        contract_version="2",
        route="single_lookup",
        intent="bind an ordinary source fact",
        required_slots=[
            RequiredSlot(slot_id="S1", description="State the authorized fact.")
        ],
        evidence_extraction_required=True,
        max_retrieval_rounds=1,
        max_repair_rounds=0,
        max_llm_calls=2,
        runtime_token_budget=50_000,
        resolved_source_scope=scope,
        slot_plan_status="complete",
    )

    async def admission(**_kwargs):
        return V9AdmissionContract(source_scope=scope, contract=contract)

    monkeypatch.setattr(
        "evaluation.agentic_v9_campaign_runtime.build_v9_admission_contract",
        admission,
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=AsyncMock(
            return_value=[
                Document(
                    page_content="Authorized ordinary evidence.",
                    metadata={
                        "doc_id": "doc-authorized",
                        "chunk_id": "chunk-authorized",
                    },
                ),
                Document(
                    page_content="Blocked evidence.",
                    metadata={"doc_id": "doc-blocked", "chunk_id": "chunk-blocked"},
                ),
            ]
        ),
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="What is the authorized fact?",
        user_id="user-a",
        authorized_doc_ids=["doc-authorized"],
        setup_snapshot=_setup(),
        trace_id="candidate-admission-fallback",
    )

    packets = result.agent_trace["agentic_v9"]["evidence_packets"]
    assert packets
    assert {packet["source"]["doc_id"] for packet in packets} == {"doc-authorized"}
    assert result.documents
    assert provider.extraction_calls == 1
```

- [ ] **Step 2: Write the failing planner contract test**

Add:

```python
@pytest.mark.asyncio
async def test_production_contract_does_not_require_strict_evidence_extraction() -> None:
    case = _questions()["Q5"]
    doc_ids = [f"doc-{index}" for index, _ in enumerate(case["source_docs"], 1)]
    contract = await QuestionContractPlanner().plan(
        question=case["question"],
        authorized_source_names=case["source_docs"],
        authorized_source_doc_ids=doc_ids,
        authorized_source_name_to_doc_ids={
            name: [doc_id]
            for name, doc_id in zip(case["source_docs"], doc_ids, strict=True)
        },
        setup_policy={"max_llm_calls": 5, "max_output_tokens": 8192},
    )
    assert contract.evidence_extraction_required is False
```

- [ ] **Step 3: Run the two focused tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_agentic_v9_campaign_runtime.py::test_runtime_preserves_authorized_candidates_when_strict_curation_is_malformed `
  tests/test_agentic_v9_contract_planner.py::test_production_contract_does_not_require_strict_evidence_extraction `
  -q
```

Expected: the planner test fails because production currently sets
`evidence_extraction_required=True`; the runtime expectation exposes any
remaining extraction call or candidate erasure.

- [ ] **Step 4: Implement the minimal production rollback**

In `AgenticV9CampaignRuntime.execute`, keep optional extraction for manually
injected/research contracts, but make it fail-soft:

```python
async def prose_curate(
    _: str, contract: QueryContract, packets: tuple[EvidencePacket, ...]
) -> tuple[EvidencePacket, ...]:
    visual_packets = [
        packet for packet in packets if packet.source.asset_id is not None
    ]
    text_candidates = [
        packet for packet in packets if packet.source.asset_id is None
    ]
    curated_text = text_candidates
    if contract.evidence_extraction_required:
        controller = state["budget_controller"]
        assert isinstance(controller, RunBudgetController)
        extractor = EvidenceExtractor(
            BudgetedLlmInvoker(
                controller=controller,
                provider_factory=self._provider_factory,
                observer=llm_call_observer,
                provider_name=provider_name,
                model_name=model_name,
                capture_policy=setup_snapshot.get("prompt_capture_policy"),
            )
        )
        extracted = await extractor.extract(
            contract,
            text_candidates,
            repairs_complete=True,
            question=question,
        )
        if extracted:
            curated_text = extracted
    effective = [*curated_text, *visual_packets]
    state["evidence_packets"] = effective
    return tuple(effective)
```

Do not delete `EvidenceExtractor`; it remains available for explicitly enabled
research contracts. The key rollback is that an empty/malformed extraction no
longer replaces valid candidates with an empty list.

In the production contract planner, set:

```python
evidence_extraction_required=False,
```

This prevents preflight and the budget controller from reserving an unused
provider call.

- [ ] **Step 5: Update existing expectations**

Update only tests whose assertions deliberately required the production
evidence-extraction call:

```python
assert provider.ainvoke.await_count == 1
assert len(observer.calls) == 1
assert observer.calls[0].phase == "final_answer"
```

Retain the standalone `EvidenceExtractor` unit tests. They continue to verify
the optional research component independently.

- [ ] **Step 6: Run focused and subsystem tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_agentic_v9_campaign_runtime.py `
  tests/test_agentic_v9_contract_planner.py `
  tests/test_agentic_v9_evidence_extractor.py `
  tests/test_agentic_v9_budget_feasibility.py `
  -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Run static verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff check `
  evaluation/agentic_v9_campaign_runtime.py `
  data_base/agentic_v9/contract_planner.py `
  tests/test_agentic_v9_campaign_runtime.py `
  tests/test_agentic_v9_contract_planner.py
.\.venv\Scripts\python.exe -m compileall -q `
  evaluation/agentic_v9_campaign_runtime.py `
  data_base/agentic_v9/contract_planner.py
git diff --check
```

Expected: every command exits with status 0.

- [ ] **Step 8: Commit the partial rollback**

```powershell
git add `
  evaluation/agentic_v9_campaign_runtime.py `
  data_base/agentic_v9/contract_planner.py `
  tests/test_agentic_v9_campaign_runtime.py `
  tests/test_agentic_v9_contract_planner.py
git commit -m "fix(agentic-v9): make evidence binding fail-soft"
```

- [ ] **Step 9: Validate with the five-question smoke set**

Run Q5, Q7, Q11, Q14, and Q16 with Agentic v9 once each.

Accept only if:

- all five runs contain at least one effective evidence packet;
- no unauthorized source appears;
- ordinary prose does not become `insufficient` solely because locator metadata
  is unavailable;
- Q14/Q16 may remain `qualified_partial` when genuinely missing evidence;
- no `evidence_extract` provider call is charged in the production token
  breakdown.

Do not run the full 16-question campaign until this smoke gate passes.
