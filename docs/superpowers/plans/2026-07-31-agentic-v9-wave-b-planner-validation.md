# Agentic v9 Wave B Planner Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent invalid comparison subjects and make Agentic v9 graph-planner calls budgeted, observable, and fully attributable without changing retrieval, reranking, context packing, or synthesis.

**Architecture:** Keep the existing recall-oriented comparison marker as a prefilter, then require the single comparison-planner response to label and ground independent subjects before applying the comparison overlay. Separately, thread the existing `BudgetedLlmInvoker` through the default graph locator into `GenericGraphRouter`, preserving legacy callers and all graph route semantics.

**Tech Stack:** Python 3.13, asyncio, Pydantic v2, LangChain provider adapters, pytest, Ruff

## Global Constraints

- Do not change hybrid retrieval candidate counts, queries, or execution order.
- Do not change reranker model, limits, scores, `top_k`, or fail-soft behavior.
- Do not change per-subject document limits or corrective retrieval behavior.
- Do not change final context packing, diversity policy, synthesis prompt, response parsing, or generation count.
- Do not change source authorization or graph retrieval algorithms.
- The comparison planner remains answer-free and uses at most one provider call.
- Subject validation must not make an additional provider call.
- Comparison and graph-planner failures remain fail-soft and must not fail the run.
- Every Agentic v9 provider call must pass through `RunBudgetController` and the LLM-call observer.
- Add type hints to every new or modified function signature.
- Keep each production concern in its own commit.

---

## File Structure

### Task 1: Comparison eligibility and subject validation

- Modify `data_base/agentic_v9/schemas.py`
  - add the safe `invalid_subjects` fallback classification.
- Modify `data_base/agentic_v9/comparison_planner.py`
  - parse semantic subject roles and explicit question spans;
  - validate grounded independent subjects;
  - fail soft when fewer than two subjects remain.
- Modify `prompts/agentic_v9_comparison_planner.json`
  - distinguish independent entities from claims, capabilities, conditions, metrics, and dimensions.
- Modify `tests/test_agentic_v9_comparison_planner.py`
  - unit coverage for valid and invalid subject roles and grounding.
- Modify `tests/test_agentic_v9_campaign_runtime.py`
  - prove invalid subjects preserve the original runtime contract and valid comparisons still specialize.

### Task 2: Budgeted graph planner and accounting reconciliation

- Modify `data_base/rag_graph_locator.py`
  - forward an optional injected `LlmInvoker` to the graph bundle locator.
- Modify `data_base/RAG_QA_service.py`
  - thread the optional invoker through the graph bundle and route-decision boundaries.
- Modify `evaluation/agentic_v9_campaign_runtime.py`
  - inject a `BudgetedLlmInvoker` only into the default Agentic v9 graph locator.
- Modify `tests/test_rag_graph_locator.py`
  - prove optional invoker forwarding and legacy-call compatibility.
- Modify `tests/test_graphrag_integration.py`
  - prove the graph bundle resolver gives the injected invoker to `GenericGraphRouter`.
- Modify `tests/test_agentic_v9_campaign_runtime.py`
  - prove `graph_route` is observed and runtime tokens reconcile.

---

### Task 1: Validate Comparison Eligibility and Independent Subjects

**Files:**
- Modify: `data_base/agentic_v9/schemas.py:48-54`
- Modify: `data_base/agentic_v9/comparison_planner.py:40-170`
- Modify: `prompts/agentic_v9_comparison_planner.json`
- Test: `tests/test_agentic_v9_comparison_planner.py`
- Test: `tests/test_agentic_v9_campaign_runtime.py`

**Interfaces:**
- Consumes: `ComparisonSubject`, `ComparisonPlan`, and `ComparisonPlannerOutcome`.
- Produces: `ComparisonPlannerFallbackReason` including `"invalid_subjects"`.
- Produces: one internal `_PlannerSubjectPayload` carrying `subject_role` and `question_span`.
- Preserves: `ComparisonPlanner.plan(question: str, authorized_source_names: Sequence[str], timeout_seconds: float) -> ComparisonPlannerOutcome`.
- Preserves: `apply_comparison_overlay(contract, plan) -> QueryContract`.

- [ ] **Step 1: Add failing unit tests for semantic role and question grounding**

Add the following role-aware helper shape to `tests/test_agentic_v9_comparison_planner.py`:

```python
def _planner_subject(
    subject_id: str,
    display_name: str,
    *,
    subject_role: str = "entity",
    question_span: str | None = None,
    aliases: list[str] | None = None,
    retrieval_query: str | None = None,
) -> dict[str, object]:
    return {
        "subject_id": subject_id,
        "display_name": display_name,
        "aliases": aliases or [],
        "retrieval_query": retrieval_query or display_name,
        "subject_role": subject_role,
        "question_span": question_span or display_name,
    }
```

Update `_payload` so every valid subject includes:

```python
"subject_role": "entity",
"question_span": "nnMamba",
```

Add focused tests:

```python
@pytest.mark.asyncio
async def test_one_entity_claim_arbitration_is_not_subject_comparison() -> None:
    question = (
        "關於 MedSAM-2 的自動化程度，單一提示詞分割與"
        "初始 bounding box 品質這兩種說法是否互斥？"
    )
    response = json.dumps(
        {
            "is_comparison": True,
            "subjects": [
                _planner_subject("medsam-2", "MedSAM-2"),
                _planner_subject(
                    "one-prompt",
                    "單一提示詞分割",
                    subject_role="capability",
                ),
                _planner_subject(
                    "prompt-quality",
                    "初始 bounding box 品質",
                    subject_role="condition",
                ),
            ],
            "dimensions": ["自動化程度"],
        },
        ensure_ascii=False,
    )

    outcome = await ComparisonPlanner(llm_invoker=_Invoker(response)).plan(
        question=question,
        authorized_source_names=[],
        timeout_seconds=1,
    )

    assert outcome.status == "fallback"
    assert outcome.fallback_reason == "invalid_subjects"
    assert outcome.plan is None


@pytest.mark.asyncio
async def test_unanchored_entity_subject_is_rejected() -> None:
    response = _payload(
        subjects=[
            _planner_subject("nnmamba", "nnMamba"),
            _planner_subject("invented", "InventedModel"),
        ]
    )

    outcome = await ComparisonPlanner(llm_invoker=_Invoker(response)).plan(
        question="比較 nnMamba 與 EfficientMedNeXt-L",
        authorized_source_names=[],
        timeout_seconds=1,
    )

    assert outcome.fallback_reason == "invalid_subjects"


@pytest.mark.asyncio
async def test_valid_three_entity_lineage_comparison_remains_planned() -> None:
    question = "判斷 SAM、SegmentAnyBone、SegVol 是否屬於同一技術脈絡。"
    response = _payload(
        subjects=[
            _planner_subject("sam", "SAM"),
            _planner_subject("segmentanybone", "SegmentAnyBone"),
            _planner_subject("segvol", "SegVol"),
        ],
        dimensions=["技術脈絡"],
    )

    outcome = await ComparisonPlanner(llm_invoker=_Invoker(response)).plan(
        question=question,
        authorized_source_names=[],
        timeout_seconds=1,
    )

    assert outcome.status == "planned"
    assert [subject.subject_id for subject in outcome.plan.subjects] == [
        "sam",
        "segmentanybone",
        "segvol",
    ]
```

- [ ] **Step 2: Run the comparison-planner tests and confirm the new tests fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_comparison_planner.py -q
```

Expected: the new payload fields are rejected or ignored by the current schema, and the Q3-shaped response incorrectly remains planned.

- [ ] **Step 3: Add the typed semantic-role payload and fallback reason**

In `data_base/agentic_v9/schemas.py`, extend the fallback literal:

```python
ComparisonPlannerFallbackReason = Literal[
    "timeout",
    "provider_error",
    "invalid_response",
    "schema_violation",
    "not_comparison",
    "invalid_subjects",
]
```

In `data_base/agentic_v9/comparison_planner.py`, import `Literal` and `unicodedata`, then add:

```python
class _PlannerSubjectPayload(ComparisonSubject):
    """Provider subject plus answer-free semantic grounding metadata."""

    subject_role: Literal[
        "entity",
        "claim",
        "capability",
        "condition",
        "metric",
        "dimension",
    ]
    question_span: str = Field(min_length=1, max_length=160)
```

Change `_PlannerPayload.subjects` to:

```python
subjects: list[_PlannerSubjectPayload] = Field(default_factory=list, max_length=4)
```

Add deterministic helpers:

```python
def _normalized_identity(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _contains_explicit_span(question: str, span: str) -> bool:
    normalized_question = unicodedata.normalize("NFKC", question).casefold()
    normalized_span = unicodedata.normalize("NFKC", span).strip().casefold()
    if not normalized_span:
        return False
    for match in re.finditer(re.escape(normalized_span), normalized_question):
        before = normalized_question[match.start() - 1] if match.start() else ""
        after = (
            normalized_question[match.end()]
            if match.end() < len(normalized_question)
            else ""
        )
        left_boundary_required = (
            normalized_span[0].isascii() and normalized_span[0].isalnum()
        )
        right_boundary_required = (
            normalized_span[-1].isascii() and normalized_span[-1].isalnum()
        )
        left_ok = not left_boundary_required or not (
            before.isascii() and before.isalnum()
        )
        right_ok = not right_boundary_required or not (
            after.isascii() and after.isalnum()
        )
        if left_ok and right_ok:
            return True
    return False


def _validated_subjects(
    question: str,
    candidates: Sequence[_PlannerSubjectPayload],
) -> list[ComparisonSubject]:
    accepted: list[ComparisonSubject] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.subject_role != "entity":
            continue
        if not _contains_explicit_span(question, candidate.question_span):
            continue
        span_identity = _normalized_identity(candidate.question_span)
        names = [candidate.display_name, *candidate.aliases]
        if span_identity not in {_normalized_identity(name) for name in names}:
            continue
        identity = _normalized_identity(candidate.display_name)
        if not identity or identity in seen:
            continue
        seen.add(identity)
        accepted.append(
            ComparisonSubject.model_validate(
                candidate.model_dump(
                    exclude={"subject_role", "question_span"}
                )
            )
        )
    return accepted
```

After `_PlannerPayload` validation and `is_comparison` handling, build the plan with:

```python
subjects = _validated_subjects(question, payload.subjects)
if len(subjects) < 2:
    return _fallback("invalid_subjects", started_at)
try:
    plan = ComparisonPlan(
        subjects=subjects,
        dimensions=payload.dimensions,
        qualification=payload.qualification,
    )
    _reject_invented_numbers(question, plan)
except (ValidationError, ValueError):
    return _fallback("schema_violation", started_at)
```

- [ ] **Step 4: Tighten the answer-free planner prompt**

Replace `prompts/agentic_v9_comparison_planner.json` with a single JSON object whose system instruction requires the new fields:

```json
{
  "system": "Identify explicit independent comparison subjects and requested dimensions for retrieval. Do not answer, choose a winner, infer values, or name source files/document IDs. Return exactly one JSON object. Set is_comparison=true only when the question compares, relates, or jointly judges at least two independent entities such as models, methods, datasets, or documents. Two claims, capabilities, conditions, prompt types, or metrics about one entity are not independent subjects. For each candidate emit subject_role as one of entity, claim, capability, condition, metric, or dimension, and copy question_span exactly from the question. For an entity comparison use {\"is_comparison\":true,\"subjects\":[{\"subject_id\":\"normalized_id\",\"display_name\":\"entity name\",\"aliases\":[],\"retrieval_query\":\"entity-specific query\",\"subject_role\":\"entity\",\"question_span\":\"exact question span\"}],\"dimensions\":[\"dimension\"],\"qualification\":\"optional scope\"}. Include 2 to 4 candidates. Every retrieval_query must name its subject. Preserve numeric or locator terms only when present in the question. Otherwise return {\"is_comparison\":false,\"subjects\":[],\"dimensions\":[]}.",
  "user_template": "Question: {question}\nRuntime-authorized source names, when available: {authorized_source_names}\nClassify independent subjects and dimensions only; do not answer."
}
```

- [ ] **Step 5: Run comparison-planner tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_comparison_planner.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Add runtime fail-soft and frozen-behavior regression tests**

In `tests/test_agentic_v9_campaign_runtime.py`, add this Q3-shaped runtime test:

```python
@pytest.mark.asyncio
async def test_invalid_comparison_subjects_preserve_base_contract_and_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider()
    provider.ainvoke.side_effect = [
        SimpleNamespace(
            content=json.dumps(
                {
                    "is_comparison": True,
                    "subjects": [
                        {
                            "subject_id": "medsam-2",
                            "display_name": "MedSAM-2",
                            "aliases": [],
                            "retrieval_query": "MedSAM-2 automation",
                            "subject_role": "entity",
                            "question_span": "MedSAM-2",
                        },
                        {
                            "subject_id": "one-prompt",
                            "display_name": "單一提示詞分割",
                            "aliases": [],
                            "retrieval_query": "單一提示詞分割",
                            "subject_role": "capability",
                            "question_span": "單一提示詞分割",
                        },
                        {
                            "subject_id": "prompt-quality",
                            "display_name": "初始 bounding box 品質",
                            "aliases": [],
                            "retrieval_query": "初始 bounding box 品質",
                            "subject_role": "condition",
                            "question_span": "初始 bounding box 品質",
                        },
                    ],
                    "dimensions": ["自動化程度"],
                    "qualification": None,
                },
                ensure_ascii=False,
            ),
            usage_metadata={"input_tokens": 20, "output_tokens": 10},
        ),
        SimpleNamespace(
            content="The evidence supports a qualified answer.",
            usage_metadata={"input_tokens": 12, "output_tokens": 7},
        ),
    ]
    retrieve_documents = AsyncMock(
        return_value=[
            Document(
                page_content="MedSAM-2 evidence.",
                metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
            )
        ]
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=retrieve_documents,
        provider_factory=lambda _purpose: provider,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question=(
            "關於 MedSAM-2 的自動化程度，單一提示詞分割與"
            "初始 bounding box 品質這兩種說法是否互斥？"
        ),
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot=_setup(),
        trace_id="invalid-comparison-subjects",
    )

    v9 = result.agent_trace["agentic_v9"]
    assert v9["comparison_planner"]["status"] == "fallback"
    assert v9["comparison_planner"]["fallback_reason"] == "invalid_subjects"
    assert "comparison_plan" not in v9["query_contract"]
    assert retrieve_documents.await_count == 1
    assert result.documents
    assert provider.ainvoke.await_count == 2
```

Extend `test_v9_comparison_planner_overlays_subject_tasks_and_caps_each_at_two`
so both subjects include `subject_role` and `question_span`, while keeping these
existing frozen assertions:

```python
assert retrieve_documents.await_count == 2
assert [row["selected_count"] for row in v9["retrieval_diagnostics"]] == [2, 2]
assert v9["comparison_planner"]["status"] == "planned"
```

- [ ] **Step 7: Run focused runtime tests and lint**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_comparison_planner.py tests/test_agentic_v9_campaign_runtime.py -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9/schemas.py data_base/agentic_v9/comparison_planner.py tests/test_agentic_v9_comparison_planner.py tests/test_agentic_v9_campaign_runtime.py
```

Expected: all tests and Ruff checks pass.

- [ ] **Step 8: Commit Task 1**

```powershell
git add data_base/agentic_v9/schemas.py data_base/agentic_v9/comparison_planner.py prompts/agentic_v9_comparison_planner.json tests/test_agentic_v9_comparison_planner.py tests/test_agentic_v9_campaign_runtime.py
git commit -m "fix(agentic-v9): validate comparison subjects"
```

---

### Task 2: Route Graph Planning Through the Budgeted Provider Boundary

**Files:**
- Modify: `data_base/rag_graph_locator.py:20-105`
- Modify: `data_base/RAG_QA_service.py:485-510`
- Modify: `data_base/RAG_QA_service.py:666-742`
- Modify: `evaluation/agentic_v9_campaign_runtime.py:103-153`
- Modify: `evaluation/agentic_v9_campaign_runtime.py:308-390`
- Modify: `evaluation/agentic_v9_campaign_runtime.py:1088-1115`
- Test: `tests/test_rag_graph_locator.py`
- Test: `tests/test_graphrag_integration.py`
- Test: `tests/test_agentic_v9_campaign_runtime.py`

**Interfaces:**
- Consumes: `LlmInvoker` protocol and `BudgetedLlmInvoker`.
- Produces: `locate_graph_sources` with keyword `llm_invoker: LlmInvoker | None = None`.
- Produces: `get_graph_evidence_bundle` with keyword `llm_invoker: LlmInvoker | None = None`.
- Produces: `_resolve_graph_route_decision` with keyword `llm_invoker: LlmInvoker | None = None`.
- Preserves: the public legacy graph APIs when no invoker is supplied.
- Preserves: the existing five-positional-argument injected `GraphLocator` test seam.

- [ ] **Step 1: Add failing graph-locator forwarding tests**

In `tests/test_rag_graph_locator.py`, add:

```python
@pytest.mark.asyncio
async def test_locator_forwards_optional_llm_invoker_to_bundle_locator() -> None:
    invoker = object()
    bundle_locator = AsyncMock(
        return_value=GraphEvidenceBundle(query="q", route="none")
    )

    await locate_graph_sources(
        question="q",
        user_id="user-1",
        vector_documents=[],
        requested_doc_ids=None,
        graph_execution_hints=None,
        required_modalities=[],
        evidence_mode="locator_to_chunk",
        bundle_locator=bundle_locator,
        llm_invoker=invoker,
    )

    bundle_locator.assert_awaited_once_with(
        question="q",
        user_id="user-1",
        search_mode="generic",
        graph_execution_hints=None,
        chunk_lookup=bundle_locator.call_args.kwargs["chunk_lookup"],
        llm_invoker=invoker,
    )
```

Retain the existing `test_locator_returns_only_resolved_source_documents_not_raw_graph_content`
assertion showing that calls without an invoker do not receive an extra keyword.

- [ ] **Step 2: Add a failing graph-router injection test**

In `tests/test_graphrag_integration.py`, add a focused test around
`data_base.RAG_QA_service._resolve_graph_route_decision`:

```python
@pytest.mark.asyncio
async def test_graph_route_resolver_injects_v9_invoker(monkeypatch) -> None:
    invoker = object()
    router = AsyncMock()
    router.route.return_value = GraphRouteDecision(
        query_kind="relation",
        path="local-first",
    )
    router_factory = Mock(return_value=router)
    monkeypatch.setattr(
        "data_base.RAG_QA_service.GenericGraphRouter",
        router_factory,
    )
    status = SimpleNamespace(
        community_level_counts={},
        community_count=0,
    )

    decision, _, _ = await _resolve_graph_route_decision(
        "ambiguous relation question",
        "generic",
        status,
        None,
        llm_invoker=invoker,
    )

    router_factory.assert_called_once_with(llm_invoker=invoker)
    assert decision.path == "local-first"
```

Import the private resolver directly for this production-boundary regression;
do not test prompt content here.

In `tests/test_agentic_v9_provider_boundary.py`, add:

```python
class _FailingInvoker:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def invoke(
        self,
        *,
        phase: str,
        purpose: str,
        messages: list[dict[str, object]],
    ) -> object:
        del purpose, messages
        self.calls.append(phase)
        raise TimeoutError("graph route unavailable")


@pytest.mark.asyncio
async def test_graph_fast_path_does_not_invoke_provider() -> None:
    invoker = _RecordingInvoker(SimpleNamespace(content="unused"))

    decision = await GenericGraphRouter(llm_invoker=invoker).route(
        "Summarize the overall themes.",
        has_communities=True,
    )

    assert decision.router_reason == "summary_keywords"
    assert invoker.calls == []


@pytest.mark.asyncio
async def test_graph_provider_failure_returns_safe_route_fallback() -> None:
    invoker = _FailingInvoker()

    decision = await GenericGraphRouter(llm_invoker=invoker).route(
        "Explain the implications of this material in depth",
        has_communities=True,
    )

    assert invoker.calls == ["graph_route"]
    assert decision.router_reason == "llm_router_fallback"
    assert decision.path == "blended"
```

- [ ] **Step 3: Run the new graph tests and confirm they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rag_graph_locator.py tests/test_graphrag_integration.py -q
```

Expected: failures report unexpected `llm_invoker` arguments or missing resolver parameters.

- [ ] **Step 4: Thread the optional invoker through graph source location**

In `data_base/rag_graph_locator.py`, import `LlmInvoker` and extend the signature:

```python
async def locate_graph_sources(
    *,
    question: str,
    user_id: str,
    vector_documents: list[Document],
    requested_doc_ids: Optional[list[str]],
    graph_execution_hints: Optional[dict[str, Any]],
    required_modalities: list[str],
    evidence_mode: str,
    bundle_locator: GraphBundleLocator,
    search_mode: str = "generic",
    chunk_lookup: Optional[ChunkLookup] = None,
    claim_scope_approver: Optional[ClaimScopeApprover] = None,
    graph_chunk_ratio: float = 0.35,
    llm_invoker: LlmInvoker | None = None,
) -> GraphSourceLocatorResult:
```

Build kwargs without changing legacy calls:

```python
bundle_kwargs: dict[str, Any] = {
    "question": question,
    "user_id": user_id,
    "search_mode": search_mode,
    "graph_execution_hints": graph_execution_hints,
    "chunk_lookup": lookup,
}
if llm_invoker is not None:
    bundle_kwargs["llm_invoker"] = llm_invoker
bundle = await bundle_locator(**bundle_kwargs)
```

- [ ] **Step 5: Inject the invoker into `GenericGraphRouter`**

In `data_base/RAG_QA_service.py`, import `LlmInvoker` and extend:

```python
async def _resolve_graph_route_decision(
    question: str,
    search_mode: str,
    status: Any,
    graph_execution_hints: Optional[Dict[str, Any]],
    llm_invoker: LlmInvoker | None = None,
) -> Tuple[GraphRouteDecision, bool, bool]:
```

Change only the generic router construction:

```python
decision = await GenericGraphRouter(llm_invoker=llm_invoker).route(
    question,
    has_communities=has_communities,
    hints=hints,
)
```

Extend `_get_graph_evidence_bundle` and
`get_graph_evidence_bundle` with the same optional keyword and forward it
to `_resolve_graph_route_decision`. Leave `_get_graph_context` and
legacy callers unchanged so they continue to pass `None`.

- [ ] **Step 6: Preserve the injected graph-locator compatibility seam**

In `evaluation/agentic_v9_campaign_runtime.py`, add:

```python
self._uses_default_graph_locator = graph_locator is None
self._graph_locator = graph_locator or _locate_graph_documents
```

Do not change the existing `GraphLocator` callable alias. In the retrieval
stage, branch only for the default locator:

```python
if self._uses_default_graph_locator:
    controller = state["budget_controller"]
    assert isinstance(controller, RunBudgetController)
    located = await _locate_graph_documents(
        task.query,
        user_id,
        docs,
        list(task.source_scope.authorized_doc_ids),
        state["contract"],
        llm_invoker=BudgetedLlmInvoker(
            controller=controller,
            provider_factory=self._provider_factory,
            observer=llm_call_observer,
            provider_name=str(setup_snapshot.get("provider") or "unknown"),
            model_name=str(setup_snapshot.get("model_name") or "unknown"),
        ),
    )
else:
    located = await self._graph_locator(
        task.query,
        user_id,
        docs,
        list(task.source_scope.authorized_doc_ids),
        state["contract"],
    )
```

Extend only the private default adapter:

```python
async def _locate_graph_documents(
    question: str,
    user_id: str,
    vector_documents: list[Document],
    authorized_doc_ids: list[str],
    contract: QueryContract,
    *,
    llm_invoker: LlmInvoker | None = None,
) -> GraphSourceLocatorResult:
```

Forward `llm_invoker` into `locate_graph_sources`.

- [ ] **Step 7: Run graph boundary tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rag_graph_locator.py tests/test_graphrag_integration.py tests/test_agentic_v9_provider_boundary.py -q
```

Expected: all tests pass, including the pre-existing direct
`GenericGraphRouter` injected-invoker provider-boundary test.

- [ ] **Step 8: Add the Q14-shaped accounting regression**

In `tests/test_agentic_v9_campaign_runtime.py`, import
`GraphSourceLocatorResult` and `LlmInvoker`, then add a local observer:

```python
class _RecordingObserver:
    def __init__(self) -> None:
        self.calls: list[object] = []
        self.partial_reasons: list[str] = []

    async def on_terminal_attempt(self, observation: object) -> bool:
        self.calls.append(observation)
        return True

    def mark_partial(self, reason: str) -> None:
        self.partial_reasons.append(reason)
```

Add the complete graph-relational runtime test:

```python
@pytest.mark.asyncio
async def test_v9_graph_route_usage_is_budgeted_observed_and_reconciled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider()
    provider.ainvoke.side_effect = [
        SimpleNamespace(
            content='{"query_kind":"relation","path":"local-first"}',
            usage_metadata={
                "input_tokens": 5,
                "output_tokens": 2,
                "total_tokens": 7,
            },
        ),
        SimpleNamespace(
            content="Graph-aware evidence answer.",
            usage_metadata={
                "input_tokens": 12,
                "output_tokens": 7,
                "total_tokens": 19,
            },
        ),
    ]
    observer = _RecordingObserver()
    retrieve_documents = AsyncMock(
        return_value=[
            Document(
                page_content="Source-backed relationship evidence.",
                metadata={"doc_id": "doc-1", "chunk_id": "chunk-1"},
            )
        ]
    )

    async def observed_graph_locator(
        question: str,
        user_id: str,
        vector_documents: list[Document],
        authorized_doc_ids: list[str],
        contract: QueryContract,
        *,
        llm_invoker: LlmInvoker | None = None,
    ) -> GraphSourceLocatorResult:
        assert question
        assert user_id == "user-a"
        assert authorized_doc_ids == ["doc-1"]
        assert contract.route == "graph_relational"
        assert llm_invoker is not None
        await llm_invoker.invoke(
            phase="graph_route",
            purpose="graph_extraction",
            messages=[{"role": "user", "content": question}],
        )
        return GraphSourceLocatorResult(
            documents=vector_documents,
            resolved_source_documents=vector_documents,
            resolved_source_doc_ids=["doc-1"],
            resolved_source_chunk_ids=["chunk-1"],
            candidate_item_ids=[],
            resolved_item_ids=[],
            scope_approved_item_ids=[],
            scored_item_ids=[],
            packed_item_ids=[],
            route="local-first",
            path="source_expand",
            fallback=None,
            graph_latency_ms=1,
            bundle=None,
            chunk_lookup=SimpleNamespace(),
            resolved_chunks=[],
            scoped_chunks=[],
            graph_documents=[],
        )

    monkeypatch.setattr(
        runtime_module,
        "_locate_graph_documents",
        observed_graph_locator,
    )
    runtime = AgenticV9CampaignRuntime(
        retrieve_documents=retrieve_documents,
        provider_factory=lambda _purpose: provider,
        llm_call_observer=observer,
        document_reference_resolver=_identity_reference_resolver,
    )

    result = await runtime.execute(
        question="Trace the relationship path from ModelA to ModelB.",
        user_id="user-a",
        authorized_doc_ids=["doc-1"],
        setup_snapshot=_setup(),
        trace_id="observed-graph-route",
    )

    assert [call.phase for call in observer.calls] == [
        "graph_route",
        "final_answer",
    ]
    assert sum(call.usage["total_tokens"] for call in observer.calls) == 26
    assert result.usage["total_tokens"] == 26
    assert observer.partial_reasons == []
    assert result.agent_trace["agentic_v9"]["retrieval_diagnostics"]
    assert result.agent_trace["execution_profile"] == (
        runtime_module.agentic_v9_execution_profile(open_user_corpus=False)
    )
    assert result.agent_trace["context_policy_version"] == (
        runtime_module.AGENTIC_V9_CONTEXT_POLICY_VERSION
    )
```

- [ ] **Step 9: Run the complete focused Wave B suite and Ruff**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_comparison_planner.py tests/test_agentic_v9_campaign_runtime.py tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_phase_policy.py tests/test_rag_graph_locator.py tests/test_graphrag_integration.py tests/test_evaluation_analytics_context.py -q
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9/schemas.py data_base/agentic_v9/comparison_planner.py data_base/rag_graph_locator.py data_base/RAG_QA_service.py evaluation/agentic_v9_campaign_runtime.py tests/test_agentic_v9_comparison_planner.py tests/test_agentic_v9_campaign_runtime.py tests/test_rag_graph_locator.py tests/test_graphrag_integration.py
```

Expected: all tests and Ruff checks pass.

- [ ] **Step 10: Confirm frozen retrieval and synthesis configuration**

Run:

```powershell
git diff HEAD~2 -- evaluation/agentic_v9_campaign_runtime.py data_base/rag_filtering.py data_base/reranker.py data_base/agentic_v9/context_packer.py prompts
```

Confirm:

- no changes to retrieval counts or reranker parameters;
- no changes to context packer;
- the only prompt change is `agentic_v9_comparison_planner.json`;
- no synthesis prompt or final-answer logic changed.

- [ ] **Step 11: Commit Task 2**

```powershell
git add data_base/rag_graph_locator.py data_base/RAG_QA_service.py evaluation/agentic_v9_campaign_runtime.py tests/test_rag_graph_locator.py tests/test_graphrag_integration.py tests/test_agentic_v9_campaign_runtime.py
git commit -m "fix(agentic-v9): budget graph planner calls"
```

- [ ] **Step 12: Record smoke verification criteria**

After deployment, run the same 16-question Agentic v9 smoke configuration used
for the 2026-07-31 export and verify:

```text
Q3 comparison_planner.status = fallback
Q3 comparison_planner.fallback_reason = invalid_subjects or not_comparison
Q4 comparison_planner.status = planned
Q14 comparison_planner.status = planned
Q14 graph ambiguity call phase = graph_route
Q14 runtime total tokens = sum(persisted measured LLM call total tokens)
campaign token accounting = complete
campaign phase attribution = complete
```

Do not run a formal benchmark until these smoke conditions pass.
