# Evaluation Multi-Query and Graph Locator Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Evaluation Center Advanced, Graph family, and Agentic execution HyDE-free, use Multi-Query where expansion is intended, and upgrade the main Graph/Agentic graph paths to source-backed locator-to-chunk evidence.

**Architecture:** Add a dependency-neutral `evaluation/retrieval_profiles.py` policy module consumed by standard campaign modes and Agentic route construction. Preserve `rag_answer_question()` backward compatibility with an explicit CRAG rewrite strategy, keep legacy raw GraphRAG only under `graph_raw_current`, and version every changed evaluation baseline.

**Tech Stack:** Python 3.13, FastAPI backend modules, LangChain documents/messages, Pydantic models, asyncio, pytest, unittest.mock, Ruff.

## Global Constraints

- Scope is Evaluation Center backend execution only; ordinary chat and user-facing Deep Research defaults must not change.
- Native/Naive remains plain FAISS with no query expansion.
- Every Evaluation Advanced, Graph family, and Agentic path must have `enable_hyde=False`.
- Main Graph and Agentic graph routes must use locator-to-chunk with provenance gating and must not inject raw graph text.
- `graph_raw_current` remains the explicit raw-legacy control.
- Evaluation Agentic CRAG uses Multi-Query; the generic `rag_answer_question()` CRAG default remains HyDE for backward compatibility.
- Multi-Query failure falls back to the original question; graph failure retains vector/hybrid documents.
- Agentic context policy remains `v4_semantic_router_gate`; other modes remain `v3_answer_aware_pack`.
- No frontend route, payload, response schema, or type change is permitted.
- Follow TDD: observe each new test fail before adding its production implementation.

## File Structure

- Create `evaluation/retrieval_profiles.py`: immutable-by-construction Evaluation query, graph, and execution-profile policies.
- Create `tests/test_evaluation_retrieval_profiles.py`: unit contract for policy factories and version strings.
- Modify `data_base/RAG_QA_service.py`: CRAG rewrite strategy, Multi-Query corrective retrieval, and graph fallback observability.
- Modify `evaluation/rag_modes.py`: apply no-HyDE policy, wire main Graph locator-to-chunk, and persist standard-mode profiles.
- Modify `evaluation/agentic_evaluation_service.py`: v8 profile and route-specific Multi-Query/locator policy.
- Modify `tests/test_rag_retrieval_logic.py`: CRAG compatibility and Multi-Query corrective retrieval.
- Modify `tests/test_rag_modes_agentic.py`: standard mode policy, graph hints, and persisted profile contracts.
- Modify `tests/test_agentic_evaluation_service.py`: Agentic route matrix and CRAG rewrite contract.
- Modify `tests/test_graph_context_packing.py`: source-expand fallback behavior and reason telemetry.
- Modify `docs/BACKEND.md`: current retrieval and Graph evaluation behavior.
- Modify `docs/generated/api-surface.md`: current profile/version snapshot.
- Create `docs/design-docs/agentic-semantic-router-v8.md`: v8 successor decision record.
- Modify `docs/design-docs/index.md`: link the v8 design record.
- Create `docs/exec-plans/completed/2026-07-evaluation-multiquery-graph-locator.md`: completed milestone evidence.
- Modify `docs/exec-plans/completed/index.md`: link the completed milestone.

---

### Task 1: Shared Evaluation Retrieval Policy

**Files:**
- Create: `evaluation/retrieval_profiles.py`
- Create: `tests/test_evaluation_retrieval_profiles.py`

**Interfaces:**
- Consumes: `data_base.indexing_service.DEFAULT_PRODUCTION_INDEXING_PROFILE`.
- Produces: `multi_query_settings() -> dict[str, bool]`, `no_query_expansion_settings() -> dict[str, bool]`, `locator_to_chunk_graph_hints(...) -> dict[str, Any]`, `apply_no_hyde_policy(...) -> dict[str, dict[str, Any]]`, `evaluation_execution_profile(mode: str) -> str | None`, and profile constants.

- [ ] **Step 1: Write failing policy tests**

Create `tests/test_evaluation_retrieval_profiles.py`:

```python
from evaluation.retrieval_profiles import (
    ADVANCED_EVAL_PROFILE,
    AGENTIC_EVAL_PROFILE,
    GRAPH_EVAL_PROFILE,
    apply_no_hyde_policy,
    evaluation_execution_profile,
    locator_to_chunk_graph_hints,
    multi_query_settings,
    no_query_expansion_settings,
)


def test_query_policy_factories_return_fresh_explicit_settings() -> None:
    first = multi_query_settings()
    second = multi_query_settings()
    assert first == {"enable_hyde": False, "enable_multi_query": True}
    assert no_query_expansion_settings() == {
        "enable_hyde": False,
        "enable_multi_query": False,
    }
    assert first is not second


def test_locator_to_chunk_hints_are_source_backed_and_not_auto_gated() -> None:
    hints = locator_to_chunk_graph_hints(
        stage_hint="exploration",
        task_type="graph_analysis",
    )
    assert hints["graph_evidence_mode"] == "locator_to_chunk"
    assert hints["stage_hint"] == "exploration"
    assert hints["task_type_hint"] == "graph_analysis"
    assert hints["prefer_global"] is True
    assert hints["prefer_local"] is False
    assert hints["graph_feature_flags"] == {
        "graph_raw_current_enabled": False,
        "graph_evidence_locator_enabled": True,
        "graph_provenance_gate_enabled": True,
        "graph_to_chunk_enabled": True,
        "graph_auto_gate_enabled": False,
    }


def test_no_hyde_policy_does_not_mutate_input() -> None:
    source = {
        "naive": {"enable_hyde": False, "enable_multi_query": False},
        "advanced": {"enable_hyde": True, "enable_multi_query": True},
        "graph_raw_current": {"enable_hyde": True, "enable_multi_query": True},
        "future_unrelated": {"enable_hyde": True, "enable_multi_query": False},
    }
    normalized = apply_no_hyde_policy(source)
    assert source["advanced"]["enable_hyde"] is True
    assert normalized["naive"]["enable_hyde"] is False
    assert normalized["advanced"]["enable_hyde"] is False
    assert normalized["graph_raw_current"]["enable_hyde"] is False
    assert normalized["future_unrelated"]["enable_hyde"] is True


def test_execution_profiles_version_changed_modes() -> None:
    assert evaluation_execution_profile("naive") is None
    assert evaluation_execution_profile("advanced") == ADVANCED_EVAL_PROFILE
    assert evaluation_execution_profile("graph") == GRAPH_EVAL_PROFILE
    assert evaluation_execution_profile("agentic") == AGENTIC_EVAL_PROFILE
    assert evaluation_execution_profile("graph_raw_current").startswith(
        "graph_raw_current_eval_v2_multiquery_"
    )
```

- [ ] **Step 2: Run tests and confirm the module is missing**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_retrieval_profiles.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'evaluation.retrieval_profiles'`.

- [ ] **Step 3: Implement the policy module**

Create `evaluation/retrieval_profiles.py`:

```python
"""Versioned retrieval policies shared by Evaluation Center execution paths."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from data_base.indexing_service import DEFAULT_PRODUCTION_INDEXING_PROFILE

EVALUATION_RETRIEVAL_POLICY_VERSION = "v2_multiquery_locator"

ADVANCED_EVAL_PROFILE = (
    f"advanced_eval_v2_multiquery_{DEFAULT_PRODUCTION_INDEXING_PROFILE}"
)
GRAPH_EVAL_PROFILE = (
    f"graph_eval_v2_multiquery_locator_{DEFAULT_PRODUCTION_INDEXING_PROFILE}"
)
AGENTIC_EVAL_PROFILE = (
    f"agentic_eval_v8_multiquery_locator_{DEFAULT_PRODUCTION_INDEXING_PROFILE}"
)

GRAPH_ABLATION_MODES = frozenset(
    {
        "graph_raw_current",
        "graph_provenance_gated",
        "graph_locator_to_chunk",
        "graph_locator_claim_gate",
        "always_no_graph",
        "always_graph_locator",
        "router_auto_graph",
        "oracle_graph_router",
        "graph_local_first",
        "graph_global_first",
        "graph_blended",
        "graph_path_pruned",
        "graph_planning_only",
    }
)


def multi_query_settings() -> dict[str, bool]:
    return {"enable_hyde": False, "enable_multi_query": True}


def no_query_expansion_settings() -> dict[str, bool]:
    return {"enable_hyde": False, "enable_multi_query": False}


def locator_to_chunk_graph_hints(
    *,
    stage_hint: str | None = None,
    task_type: str | None = None,
) -> dict[str, Any]:
    hints: dict[str, Any] = {
        "graph_evidence_mode": "locator_to_chunk",
        "graph_feature_flags": {
            "graph_raw_current_enabled": False,
            "graph_evidence_locator_enabled": True,
            "graph_provenance_gate_enabled": True,
            "graph_to_chunk_enabled": True,
            "graph_auto_gate_enabled": False,
        },
    }
    if stage_hint is not None:
        hints["stage_hint"] = stage_hint
        hints["prefer_global"] = stage_hint == "exploration"
        hints["prefer_local"] = stage_hint == "verification" and task_type != "graph_analysis"
    if task_type is not None:
        hints["task_type_hint"] = task_type
        if stage_hint is None:
            hints["prefer_global"] = task_type == "graph_analysis"
            hints["prefer_local"] = False
    return hints


def apply_no_hyde_policy(
    modes: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    normalized = deepcopy({name: dict(config) for name, config in modes.items()})
    changed_modes = {"advanced", "graph", "agentic", *GRAPH_ABLATION_MODES}
    for name, config in normalized.items():
        if name in changed_modes:
            config["enable_hyde"] = False
    return normalized


def evaluation_execution_profile(mode: str) -> str | None:
    if mode == "advanced":
        return ADVANCED_EVAL_PROFILE
    if mode == "graph":
        return GRAPH_EVAL_PROFILE
    if mode == "agentic":
        return AGENTIC_EVAL_PROFILE
    if mode in GRAPH_ABLATION_MODES:
        return f"{mode}_eval_v2_multiquery_{DEFAULT_PRODUCTION_INDEXING_PROFILE}"
    return None
```

- [ ] **Step 4: Run the policy tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_retrieval_profiles.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Run Ruff on the new files**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff check evaluation/retrieval_profiles.py tests/test_evaluation_retrieval_profiles.py
.\.venv\Scripts\python.exe -m ruff format --check evaluation/retrieval_profiles.py tests/test_evaluation_retrieval_profiles.py
```

Expected: both commands exit 0. If the format check fails, run Ruff format on those two files and repeat both checks.

- [ ] **Step 6: Commit the shared policy**

```powershell
git add evaluation/retrieval_profiles.py tests/test_evaluation_retrieval_profiles.py
git commit -m "feat(evaluation): centralize retrieval policies"
```

---

### Task 2: CRAG Multi-Query Corrective Retrieval

**Files:**
- Modify: `data_base/RAG_QA_service.py:1696-1959`
- Modify: `tests/test_rag_retrieval_logic.py`

**Interfaces:**
- Consumes: existing `transform_query_multi()`, `transform_query_with_hyde()`, `invoke_retriever_queries_async()`, and `reciprocal_rank_fusion()`.
- Produces: `CragRewriteMode`, `_build_crag_queries(question, rewrite_mode) -> list[str]`, and the backward-compatible `crag_rewrite_mode` argument on `rag_answer_question()`.

- [ ] **Step 1: Add failing CRAG contract tests**

Append to `tests/test_rag_retrieval_logic.py`:

```python
def test_rag_answer_question_preserves_hyde_crag_default() -> None:
    signature = inspect.signature(rag_answer_question)
    assert signature.parameters["crag_rewrite_mode"].default == "hyde"


@pytest.mark.asyncio
async def test_crag_multi_query_correction_uses_rrf_without_hyde() -> None:
    initial = Document(page_content="weak evidence", metadata={"doc_id": "doc-1"})
    corrected_a = Document(page_content="model A evidence", metadata={"doc_id": "doc-1"})
    corrected_b = Document(page_content="model B evidence", metadata={"doc_id": "doc-2"})
    retriever = Mock()
    llm = Mock()
    llm.ainvoke = AsyncMock(return_value=SimpleNamespace(content="corrected answer"))
    retrieve = AsyncMock(
        side_effect=[
            [[initial]],
            [[corrected_a], [corrected_b, corrected_a]],
        ]
    )

    with (
        patch("data_base.RAG_QA_service.get_llm", return_value=llm),
        patch("data_base.RAG_QA_service.get_llm_usage_metrics", return_value={}),
        patch(
            "data_base.RAG_QA_service.get_user_retriever",
            new=AsyncMock(return_value=retriever),
        ),
        patch(
            "data_base.RAG_QA_service.invoke_retriever_queries_async",
            new=retrieve,
        ),
        patch(
            "data_base.RAG_QA_service.transform_query_multi",
            new=AsyncMock(return_value=["question", "comparison variant"]),
        ) as multi_query,
        patch(
            "data_base.RAG_QA_service.transform_query_with_hyde",
            new=AsyncMock(return_value="hypothetical answer"),
        ) as hyde,
        patch(
            "data_base.RAG_QA_service.DocumentReranker.is_initialized",
            return_value=False,
        ),
        patch(
            "agents.evaluator.RAGEvaluator.grade_documents",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "data_base.RAG_QA_service.fetch_document_filenames",
            new=AsyncMock(return_value={"doc-1": "a.pdf", "doc-2": "b.pdf"}),
        ),
    ):
        result = await rag_answer_question(
            question="question",
            user_id="user-1",
            enable_crag=True,
            crag_rewrite_mode="multi_query",
            return_docs=True,
        )

    assert result.answer == "corrected answer"
    multi_query.assert_awaited_once_with("question", enabled=True)
    hyde.assert_not_awaited()
    assert retrieve.await_args_list[1].args[1] == ["question", "comparison variant"]
    assert {document.page_content for document in result.documents} == {
        "model A evidence",
        "model B evidence",
    }


@pytest.mark.asyncio
async def test_crag_none_correction_reuses_original_question() -> None:
    from data_base.RAG_QA_service import _build_crag_queries

    with (
        patch(
            "data_base.RAG_QA_service.transform_query_multi",
            new=AsyncMock(),
        ) as multi_query,
        patch(
            "data_base.RAG_QA_service.transform_query_with_hyde",
            new=AsyncMock(),
        ) as hyde,
    ):
        queries = await _build_crag_queries("original", "none")

    assert queries == ["original"]
    multi_query.assert_not_awaited()
    hyde.assert_not_awaited()
```

- [ ] **Step 2: Run the new CRAG tests and confirm failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rag_retrieval_logic.py -q
```

Expected: failures identify the missing `crag_rewrite_mode` parameter and `_build_crag_queries` helper.

- [ ] **Step 3: Add the CRAG rewrite type and query builder**

In `data_base/RAG_QA_service.py`, near `ProgressCallback`, add:

```python
CragRewriteMode = Literal["hyde", "multi_query", "none"]


async def _build_crag_queries(
    question: str,
    rewrite_mode: CragRewriteMode,
) -> List[str]:
    if rewrite_mode == "multi_query":
        return await transform_query_multi(question, enabled=True)
    if rewrite_mode == "hyde":
        rewritten = await transform_query_with_hyde(question, enabled=True)
        return [rewritten or question]
    return [question]
```

Add the backward-compatible argument after the existing `progress_callback` argument so no existing positional parameter changes meaning:

```python
    progress_callback: Optional[ProgressCallback] = None,
    crag_rewrite_mode: CragRewriteMode = "hyde",
```

Document it in the function docstring:

```python
        crag_rewrite_mode: Corrective query policy used only when CRAG rejects
                           the initial retrieval. Defaults to legacy HyDE.
```

- [ ] **Step 4: Replace the fixed HyDE corrective branch**

Replace the corrective query construction and single-batch selection with:

```python
                corrected_queries = await _build_crag_queries(
                    question,
                    crag_rewrite_mode,
                )
                corrected_batches = await invoke_retriever_queries_async(
                    retriever,
                    corrected_queries,
                )
                if len(corrected_batches) == 1:
                    corrected_docs = corrected_batches[0]
                else:
                    corrected_docs = reciprocal_rank_fusion(corrected_batches)
```

Do not alter the existing document scoping, reranking, insufficient-retrieval response, progress events, or fallback-on-exception behavior below this block.

- [ ] **Step 5: Run CRAG and query-transformer regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rag_retrieval_logic.py tests/test_query_transformer.py tests/test_deep_research_persistence.py -q
```

Expected: all selected tests pass. The Deep Research test protects the existing generic CRAG caller while the default remains HyDE.

- [ ] **Step 6: Lint and commit CRAG behavior**

```powershell
.\.venv\Scripts\python.exe -m ruff check data_base/RAG_QA_service.py tests/test_rag_retrieval_logic.py
.\.venv\Scripts\python.exe -m ruff format --check data_base/RAG_QA_service.py tests/test_rag_retrieval_logic.py
git add data_base/RAG_QA_service.py tests/test_rag_retrieval_logic.py
git commit -m "feat(rag): support multi-query CRAG correction"
```

Expected: checks exit 0 and the commit contains only the CRAG interface/behavior and its tests.

---

### Task 3: Standard Evaluation Mode Wiring and Profiles

**Files:**
- Modify: `evaluation/rag_modes.py:20-275`
- Modify: `tests/test_rag_modes_agentic.py`
- Test: `tests/test_evaluation_retrieval_profiles.py`

**Interfaces:**
- Consumes: `apply_no_hyde_policy()`, `locator_to_chunk_graph_hints()`, and `evaluation_execution_profile()` from Task 1.
- Produces: normalized `RAG_MODES`, an explicit locator-to-chunk main Graph config, and persisted execution profiles for changed standard/ablation modes.

- [ ] **Step 1: Add failing standard-mode tests**

Add these imports to the existing import block in `tests/test_rag_modes_agentic.py`, then append the test functions:

```python
from evaluation.retrieval_profiles import (
    ADVANCED_EVAL_PROFILE,
    GRAPH_ABLATION_MODES,
    GRAPH_EVAL_PROFILE,
    evaluation_execution_profile,
)
from evaluation.rag_modes import RAG_MODES


def test_all_changed_evaluation_modes_disable_hyde() -> None:
    changed_modes = {"advanced", "graph", "agentic", *GRAPH_ABLATION_MODES}
    assert changed_modes.issubset(RAG_MODES)
    assert all(RAG_MODES[mode]["enable_hyde"] is False for mode in changed_modes)


def test_advanced_and_main_graph_use_multi_query() -> None:
    assert RAG_MODES["advanced"]["enable_multi_query"] is True
    assert RAG_MODES["graph"]["enable_multi_query"] is True


def test_main_graph_uses_locator_to_chunk_policy() -> None:
    hints = RAG_MODES["graph"]["graph_execution_hints"]
    assert hints["graph_evidence_mode"] == "locator_to_chunk"
    assert hints["graph_feature_flags"] == {
        "graph_raw_current_enabled": False,
        "graph_evidence_locator_enabled": True,
        "graph_provenance_gate_enabled": True,
        "graph_to_chunk_enabled": True,
        "graph_auto_gate_enabled": False,
    }
    assert RAG_MODES["graph_raw_current"]["graph_execution_hints"][
        "graph_evidence_mode"
    ] == "raw_current"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_profile"),
    [
        ("advanced", ADVANCED_EVAL_PROFILE),
        ("graph", GRAPH_EVAL_PROFILE),
        (
            "graph_locator_to_chunk",
            evaluation_execution_profile("graph_locator_to_chunk"),
        ),
    ],
)
async def test_changed_standard_modes_persist_execution_profile(
    mode: str,
    expected_profile: str | None,
) -> None:
    assert expected_profile is not None
    test_case = EvaluationCase(
        id="Q-profile",
        question="Compare models",
        ground_truth="comparison",
        source_docs=[],
        requires_multi_doc_reasoning=False,
    )
    mock_result = RAGResult(
        answer="answer",
        source_doc_ids=["doc-1"],
        documents=[Document(page_content="context")],
    )
    with patch(
        "evaluation.rag_modes.run_with_retry",
        new=AsyncMock(return_value=mock_result),
    ):
        result = await run_campaign_case(
            test_case=test_case,
            user_id="user-1",
            mode=mode,
            model_config={"model_name": "gemini-2.5-flash"},
        )

    assert result.execution_profile == expected_profile
```

- [ ] **Step 2: Run mode tests and confirm old policy fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rag_modes_agentic.py -q
```

Expected: failures show HyDE remains enabled, main Graph lacks locator hints, and standard results have no execution profile.

- [ ] **Step 3: Normalize `RAG_MODES` with the shared policy**

Add imports in `evaluation/rag_modes.py`:

```python
from evaluation.retrieval_profiles import (
    apply_no_hyde_policy,
    evaluation_execution_profile,
    locator_to_chunk_graph_hints,
)
```

Immediately after the existing `RAG_MODES` dictionary closes, normalize a copy and override only the main Graph policy:

```python
RAG_MODES = apply_no_hyde_policy(RAG_MODES)
RAG_MODES["graph"]["graph_execution_hints"] = locator_to_chunk_graph_hints()
```

Do not change `graph_raw_current` hints or any other Graph ablation flags.

- [ ] **Step 4: Persist the changed standard-mode profile**

Before constructing `BenchmarkExecutionResult`, compute:

```python
    execution_profile = (result.agent_trace or {}).get(
        "execution_profile"
    ) or evaluation_execution_profile(mode)
```

Then replace the current inline trace lookup with:

```python
        execution_profile=execution_profile,
```

Keep both context policy version constants unchanged.

- [ ] **Step 5: Run standard mode, graph policy, and schema tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rag_modes_agentic.py tests/test_graph_ablation_conditions.py tests/test_graph_anchor_contract.py tests/test_evaluation_retrieval_profiles.py -q
```

Expected: all selected tests pass, including legacy raw and structured ablation contracts.

- [ ] **Step 6: Lint and commit standard mode wiring**

```powershell
.\.venv\Scripts\python.exe -m ruff check evaluation/rag_modes.py tests/test_rag_modes_agentic.py
.\.venv\Scripts\python.exe -m ruff format --check evaluation/rag_modes.py tests/test_rag_modes_agentic.py
git add evaluation/rag_modes.py tests/test_rag_modes_agentic.py
git commit -m "feat(evaluation): use multi-query graph locator baseline"
```

---

### Task 4: Agentic Multi-Query and Locator Route Matrix

**Files:**
- Modify: `evaluation/agentic_evaluation_service.py:59-65,742-821`
- Modify: `tests/test_agentic_evaluation_service.py`
- Modify: `tests/test_rag_modes_agentic.py`

**Interfaces:**
- Consumes: Task 1 profile constants/factories and Task 2 `crag_rewrite_mode` argument.
- Produces: Agentic v8 profile and a route matrix with no HyDE, Multi-Query on comparison/visual/generic-graph routes, and locator-to-chunk on Graph routes.

- [ ] **Step 1: Add failing Agentic route-matrix tests**

Add `MicroRoute` and `RouteProfile` to the existing imports from
`evaluation.agentic_evaluation_service`, then replace the narrow CRAG route test in
`tests/test_agentic_evaluation_service.py` with:

```python
@pytest.mark.parametrize(
    ("route_profile", "micro_route", "expected_multi_query", "expected_graph"),
    [
        ("hybrid_exact", "direct_point_access", False, False),
        ("hybrid_compare", "broad_context_rag", True, False),
        ("graph_global", "broad_context_rag", False, True),
        ("visual_verify", "visual_evidence_path", True, False),
        ("generic_graph", "broad_context_rag", True, True),
    ],
)
def test_agentic_route_kwargs_use_multi_query_and_locator_policy(
    route_profile: RouteProfile,
    micro_route: MicroRoute,
    expected_multi_query: bool,
    expected_graph: bool,
) -> None:
    service = AgenticEvaluationService()
    kwargs = service._route_kwargs(
        route_profile=route_profile,
        micro_route=micro_route,
        enable_reranking=True,
        enable_visual_verification=False,
        task_type="graph_analysis" if expected_graph else "rag",
        stage_hint="exploration",
    )

    assert kwargs["enable_hyde"] is False
    assert kwargs["enable_multi_query"] is expected_multi_query
    assert kwargs["enable_crag"] is True
    assert kwargs["crag_rewrite_mode"] == "multi_query"
    assert kwargs["plain_mode"] is False
    assert kwargs["mode_hints"]["retrieval_policy"]["target_k"] in {4, 8}
    if expected_graph:
        hints = kwargs["graph_execution_hints"]
        assert hints["graph_evidence_mode"] == "locator_to_chunk"
        assert hints["graph_feature_flags"]["graph_to_chunk_enabled"] is True
        assert hints["graph_feature_flags"]["graph_raw_current_enabled"] is False
    else:
        assert kwargs["enable_graph_rag"] is False
```

Add a v8 assertion near the existing completed-trace assertion:

```python
    assert AGENTIC_EVAL_PROFILE.startswith(
        "agentic_eval_v8_multiquery_locator_"
    )
```

- [ ] **Step 2: Run Agentic tests and confirm failures**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_evaluation_service.py tests/test_rag_modes_agentic.py -q
```

Expected: route cases fail on HyDE, missing CRAG rewrite mode, old graph hints, and the v7 profile.

- [ ] **Step 3: Import the shared Agentic policy**

Replace the locally constructed Agentic profile with imports:

```python
from evaluation.retrieval_profiles import (
    AGENTIC_EVAL_PROFILE,
    locator_to_chunk_graph_hints,
    multi_query_settings,
    no_query_expansion_settings,
)
```

Delete the old local assignment:

```python
AGENTIC_EVAL_PROFILE = f"agentic_eval_v7_semantic_router_{DEFAULT_PRODUCTION_INDEXING_PROFILE}"
```

Remove the now-unused `DEFAULT_PRODUCTION_INDEXING_PROFILE` import from this module.

- [ ] **Step 4: Rebuild `_route_kwargs()` from explicit policies**

Initialize the shared kwargs with no expansion and Multi-Query CRAG:

```python
        kwargs: dict[str, Any] = {
            **no_query_expansion_settings(),
            "enable_reranking": enable_reranking,
            "enable_crag": True,
            "crag_rewrite_mode": "multi_query",
            "plain_mode": False,
            "return_docs": True,
            "enable_visual_verification": enable_visual_verification,
            "mode_hints": {
                "question_intent": self._active_question_intent,
                "strategy_tier": self._active_strategy_tier,
                "task_type": task_type,
                "stage_hint": stage_hint,
                "route_profile": route_profile,
                "micro_route": micro_route,
                "retrieval_policy": retrieval_policy,
            },
        }
```

Use these exact route updates:

```python
        if route_profile == "hybrid_exact":
            kwargs.update(
                {
                    **no_query_expansion_settings(),
                    "enable_reranking": False,
                    "enable_graph_rag": False,
                }
            )
        elif route_profile == "hybrid_compare":
            kwargs.update(
                {
                    **multi_query_settings(),
                    "enable_graph_rag": False,
                }
            )
        elif route_profile == "graph_global":
            kwargs.update(
                {
                    **no_query_expansion_settings(),
                    "enable_graph_rag": True,
                    "graph_search_mode": "generic",
                    "graph_execution_hints": locator_to_chunk_graph_hints(
                        stage_hint=stage_hint,
                        task_type=task_type,
                    ),
                }
            )
        elif route_profile == "visual_verify":
            kwargs.update(
                {
                    **multi_query_settings(),
                    "enable_graph_rag": False,
                    "enable_visual_verification": True,
                }
            )
        else:
            kwargs.update(
                {
                    **multi_query_settings(),
                    "enable_graph_rag": True,
                    "graph_search_mode": "generic",
                    "graph_execution_hints": locator_to_chunk_graph_hints(
                        stage_hint=stage_hint,
                        task_type=task_type,
                    ),
                }
            )
```

Do not change the shared `ResearchExecutionCore._graph_execution_hints()` method; Deep Research still owns its existing policy.

- [ ] **Step 5: Run Agentic behavior and trace regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_evaluation_service.py tests/test_rag_modes_agentic.py tests/test_research_execution_core_generic.py tests/test_deep_research.py -q
```

Expected: Agentic tests use v8/Multi-Query/locator policy and shared Deep Research tests remain unchanged.

- [ ] **Step 6: Lint and commit Agentic routing**

```powershell
.\.venv\Scripts\python.exe -m ruff check evaluation/agentic_evaluation_service.py tests/test_agentic_evaluation_service.py tests/test_rag_modes_agentic.py
.\.venv\Scripts\python.exe -m ruff format --check evaluation/agentic_evaluation_service.py tests/test_agentic_evaluation_service.py tests/test_rag_modes_agentic.py
git add evaluation/agentic_evaluation_service.py tests/test_agentic_evaluation_service.py tests/test_rag_modes_agentic.py
git commit -m "feat(evaluation): route agentic retrieval through multi-query"
```

---

### Task 5: Graph Source-Expand Fallback Observability

**Files:**
- Modify: `data_base/RAG_QA_service.py:1266-1289,2054-2140`
- Modify: `tests/test_graph_context_packing.py:225-327`

**Interfaces:**
- Consumes: existing `GraphContextDetails`, `GraphRouteDecision`, `GraphEvidenceLifecycle`, and `_record_graph_observability()`.
- Produces: `_graph_fallback_context_details(...) -> GraphContextDetails` and stable safe reason codes `no_packed_graph_chunks` / `source_expand_failed`.

- [ ] **Step 1: Add failing fallback-reason tests**

Extend the existing lookup-failure test to patch `_record_graph_observability` and assert:

```python
        patch(
            "data_base.RAG_QA_service._record_graph_observability",
            new=AsyncMock(),
        ) as record,
```

After the existing vector fallback assertions, add:

```python
    record.assert_awaited_once()
    details = record.await_args.kwargs["graph_context_details"]
    assert "fallback=no_packed_graph_chunks" in details.route_decision.router_reason
```

Add a separate bundle-failure test:

```python
@pytest.mark.asyncio
async def test_graph_source_expand_exception_records_safe_fallback_reason() -> None:
    retriever = Mock()
    vector_document = Document(
        page_content="Vector source",
        metadata={"doc_id": "doc-1", "chunk_id": "vector"},
    )
    retriever.invoke.return_value = [vector_document]
    llm = Mock()
    llm.ainvoke = AsyncMock(return_value=SimpleNamespace(content="answer"))
    record = AsyncMock()

    with (
        patch("data_base.RAG_QA_service.get_llm", return_value=llm),
        patch("data_base.RAG_QA_service.get_llm_usage_metrics", return_value={}),
        patch(
            "data_base.RAG_QA_service.get_user_retriever",
            new=AsyncMock(return_value=retriever),
        ),
        patch(
            "data_base.RAG_QA_service.fetch_document_filenames",
            new=AsyncMock(return_value={"doc-1": "doc.pdf"}),
        ),
        patch(
            "data_base.RAG_QA_service._get_graph_evidence_bundle",
            new=AsyncMock(side_effect=OSError("index unavailable")),
        ),
        patch(
            "data_base.RAG_QA_service._get_graph_context",
            new=AsyncMock(),
        ) as legacy,
        patch(
            "data_base.RAG_QA_service._record_graph_observability",
            new=record,
        ),
    ):
        result = await rag_answer_question(
            question="q",
            user_id="user-1",
            return_docs=True,
            enable_graph_rag=True,
            graph_execution_hints={
                "graph_evidence_mode": "locator_to_chunk",
                "graph_feature_flags": {
                    "graph_evidence_locator_enabled": True,
                    "graph_provenance_gate_enabled": True,
                    "graph_to_chunk_enabled": True,
                },
            },
        )

    assert result.documents == [vector_document]
    legacy.assert_not_awaited()
    record.assert_awaited_once()
    details = record.await_args.kwargs["graph_context_details"]
    assert details.route_decision.path == "skip"
    assert "fallback=source_expand_failed" in details.route_decision.router_reason
    assert "index unavailable" not in details.route_decision.router_reason
```

- [ ] **Step 2: Run graph context tests and confirm missing reasons**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_graph_context_packing.py -q
```

Expected: the new assertions fail because empty packing has no explicit fallback code and source-expand exceptions do not write observability.

- [ ] **Step 3: Add safe fallback detail construction**

After `_graph_context_details_for_bundle()`, add:

```python
def _graph_fallback_context_details(
    *,
    reason: str,
    graph_latency_ms: int,
    lifecycle: GraphEvidenceLifecycle,
) -> GraphContextDetails:
    return GraphContextDetails(
        route_decision=GraphRouteDecision(
            query_kind="relation",
            path="skip",
            router_reason=(
                f"strategy=source_expand; fallback={reason}; "
                f"{lifecycle.to_router_reason()}"
            ),
        ),
        matched_entity_ids=[],
        community_ids=[],
        candidate_evidence_count=len(lifecycle.candidate_item_ids),
        graph_latency_ms=graph_latency_ms,
    )
```

In `_graph_context_details_for_bundle()`, after the existing reason parts are created, add:

```python
    if not lifecycle.packed_item_ids:
        reason_parts.append("fallback=no_packed_graph_chunks")
```

- [ ] **Step 4: Record source-expand exceptions without leaking exception text**

Before the source-expand `try`, initialize:

```python
            bundle: GraphEvidenceBundle | None = None
```

Replace the source-expand exception handler body with:

```python
            except (KeyError, OSError, RuntimeError, ValueError) as exc:
                logger.warning(
                    "Graph-to-chunk expansion failed; retaining vector retrieval: %s",
                    exc,
                )
                candidate_item_ids = (
                    _unique_graph_item_ids(
                        [item.item_id for item in bundle.evidence_items]
                    )
                    if bundle is not None
                    else []
                )
                fallback_lifecycle = GraphEvidenceLifecycle(
                    candidate_item_ids=candidate_item_ids,
                    resolved_item_ids=[],
                    scope_approved_item_ids=[],
                    scored_item_ids=[],
                    packed_item_ids=[],
                    used_as_locator=True,
                    graph_to_chunk_attempted=True,
                )
                graph_context_details = _graph_fallback_context_details(
                    reason="source_expand_failed",
                    graph_latency_ms=max(
                        int((time.perf_counter() - graph_started_at) * 1000),
                        0,
                    ),
                    lifecycle=fallback_lifecycle,
                )
                await _record_graph_observability(
                    question=question,
                    graph_search_mode=graph_search_mode,
                    graph_execution_hints=graph_execution_hints,
                    mode_hints=mode_hints,
                    graph_context_details=graph_context_details,
                    graph_evidence_units=graph_evidence_units,
                    lifecycle=fallback_lifecycle,
                )
```

The logged exception remains operator-only; the persisted/router reason uses the stable safe code.

- [ ] **Step 5: Run GraphRAG focused regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_graph_context_packing.py tests/test_graph_auto_gate.py tests/test_graph_evidence_bundle_wrapper.py tests/test_evaluation_graph_events.py -q
```

Expected: all selected GraphRAG tests pass, raw legacy remains isolated, and locator failures retain vector documents.

- [ ] **Step 6: Lint and commit fallback observability**

```powershell
.\.venv\Scripts\python.exe -m ruff check data_base/RAG_QA_service.py tests/test_graph_context_packing.py
.\.venv\Scripts\python.exe -m ruff format --check data_base/RAG_QA_service.py tests/test_graph_context_packing.py
git add data_base/RAG_QA_service.py tests/test_graph_context_packing.py
git commit -m "feat(graphrag): record locator fallback reasons"
```

---

### Task 6: Current-State Documentation and Full Verification

**Files:**
- Modify: `docs/BACKEND.md`
- Modify: `docs/generated/api-surface.md`
- Create: `docs/design-docs/agentic-semantic-router-v8.md`
- Modify: `docs/design-docs/index.md`
- Create: `docs/exec-plans/completed/2026-07-evaluation-multiquery-graph-locator.md`
- Modify: `docs/exec-plans/completed/index.md`

**Interfaces:**
- Consumes: final constants and verified commands from Tasks 1–5.
- Produces: current system-of-record documentation and exact verification evidence.

- [ ] **Step 1: Update the backend runtime guide**

In `docs/BACKEND.md`, replace the stale Agentic profile and add the exact current behavior:

```markdown
- Evaluation retrieval expansion is mode-scoped:
  - `naive` keeps the plain no-expansion baseline
  - `advanced` uses Multi-Query plus hybrid retrieval/reranking
  - main `graph` uses Multi-Query plus provenance-gated locator-to-chunk evidence
  - every Graph ablation disables HyDE while retaining its named graph intervention
  - `graph_raw_current` is the only intentional raw-legacy graph control
- Evaluation `agentic` uses `agentic_eval_v8_multiquery_locator_recursive_baseline`:
  - no Agentic route invokes HyDE
  - compare/visual/generic-graph routes use Multi-Query
  - CRAG corrective retrieval uses Multi-Query with original-question fallback
  - selected graph routes resolve graph evidence to source chunks
```

Keep the existing semantic classifier, complexity, tier-shift, and fact-state bullets; update only their profile heading and retrieval-policy details.

- [ ] **Step 2: Update the generated API inventory**

In `docs/generated/api-surface.md`, replace the old v7 profile paragraph with:

```markdown
- Evaluation retrieval profiles are versioned per changed mode: `advanced_eval_v2_multiquery_recursive_baseline`, `graph_eval_v2_multiquery_locator_recursive_baseline`, `<ablation_mode>_eval_v2_multiquery_recursive_baseline`, and `agentic_eval_v8_multiquery_locator_recursive_baseline`.
- Evaluation Advanced, Graph family, and Agentic disable HyDE. Main Graph and Agentic graph routes use provenance-gated locator-to-chunk evidence; `graph_raw_current` remains the explicit raw-legacy control.
- Agentic CRAG corrective retrieval uses Multi-Query and falls back to the original question when query generation fails.
```

- [ ] **Step 3: Add the v8 design successor and index entry**

Create `docs/design-docs/agentic-semantic-router-v8.md`:

```markdown
# Agentic Semantic Router v8 Retrieval Policy

## Decision

Evaluation Agentic v8 retains the v7 semantic classifier, complexity tiers, micro-routing, dynamic tier shift, reverse-pruning, and structured fact-state. It changes the retrieval baseline to `agentic_eval_v8_multiquery_locator_<index_profile>`.

## Query policy

- HyDE is disabled on every Agentic route.
- `hybrid_exact` and `graph_global` do not expand the query.
- `hybrid_compare`, `visual_verify`, and `generic_graph` use Multi-Query.
- CRAG corrective retrieval uses Multi-Query; failure returns to the original question.

## Graph policy

The micro-router still decides whether GraphRAG is required. Once selected, graph evidence is a locator: provenance-eligible evidence resolves to source chunks, those chunks are scored and merged with vector retrieval, and raw graph text is not answer evidence.

## Compatibility

Ordinary chat and user-facing Deep Research retain their previous defaults. Historical v7 and unprofiled rows remain readable and are not compared as if they used the v8 retrieval policy.
```

Add to `docs/design-docs/index.md`:

```markdown
- `docs/design-docs/agentic-semantic-router-v8.md`: HyDE-free Multi-Query routing, CRAG correction, and locator-to-chunk Graph evidence
```

- [ ] **Step 4: Run the complete focused verification set**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_evaluation_retrieval_profiles.py tests/test_rag_retrieval_logic.py tests/test_query_transformer.py tests/test_rag_modes_agentic.py tests/test_agentic_evaluation_service.py tests/test_graph_context_packing.py tests/test_graph_auto_gate.py tests/test_graph_evidence_bundle_wrapper.py tests/test_evaluation_graph_events.py tests/test_research_execution_core_generic.py tests/test_deep_research.py -q
```

Expected: exit 0 with no failed tests. Preserve the literal terminal summary line for the completed milestone document.

- [ ] **Step 5: Run production backend lint and format checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff check evaluation/retrieval_profiles.py evaluation/rag_modes.py evaluation/agentic_evaluation_service.py data_base/RAG_QA_service.py tests/test_evaluation_retrieval_profiles.py tests/test_rag_retrieval_logic.py tests/test_rag_modes_agentic.py tests/test_agentic_evaluation_service.py tests/test_graph_context_packing.py
.\.venv\Scripts\python.exe -m ruff format --check evaluation/retrieval_profiles.py evaluation/rag_modes.py evaluation/agentic_evaluation_service.py data_base/RAG_QA_service.py tests/test_evaluation_retrieval_profiles.py tests/test_rag_retrieval_logic.py tests/test_rag_modes_agentic.py tests/test_agentic_evaluation_service.py tests/test_graph_context_packing.py
```

Expected: both commands exit 0.

- [ ] **Step 6: Run the full backend regression suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected: exit 0. If a known environment-only collection or provider dependency failure occurs, save the exact failing command/output, confirm the focused suite remains green, and report the full-suite blocker without suppressing or weakening tests.

- [ ] **Step 7: Write the completed milestone with actual evidence**

After Steps 4–6, create `docs/exec-plans/completed/2026-07-evaluation-multiquery-graph-locator.md`. Copy the literal terminal summary line from each verification command into the Verification section; if the full suite has an environment-only blocker, include the literal failing command, failure class, and first actionable error line instead.

```markdown
# Evaluation Multi-Query and Graph Locator Upgrade

- Status: Completed
- Design: `docs/superpowers/specs/2026-07-17-evaluation-multiquery-graph-locator-design.md`
- Implementation plan: `docs/superpowers/plans/2026-07-17-evaluation-multiquery-graph-locator.md`

## Delivered behavior

- Evaluation Advanced, Graph family, and Agentic execute without HyDE.
- Multi-Query drives intended expansion and Agentic CRAG correction.
- Main Graph and Agentic graph routes resolve graph evidence to source chunks.
- `graph_raw_current` remains the legacy control.
- Changed result rows carry versioned execution profiles.

## Verification

- Focused pytest: include the literal pytest summary line from Step 4.
- Ruff check/format: include the literal success lines and exit codes from Step 5.
- Full backend pytest: include the literal pytest summary line from Step 6. For an environment-only failure, include the literal command, failure class, and first actionable error line.
```

Do not stage the milestone file unless every verification bullet contains observed output and none contains instructional prose.

Add to `docs/exec-plans/completed/index.md`:

```markdown
- `docs/exec-plans/completed/2026-07-evaluation-multiquery-graph-locator.md`
```

- [ ] **Step 8: Check docs and working-tree scope**

Run:

```powershell
git diff --check
git status --short
```

Expected: `git diff --check` exits 0. Status contains only the expected documentation files after Tasks 1–5 have already been committed.

- [ ] **Step 9: Commit documentation and verification evidence**

```powershell
git add docs/BACKEND.md docs/generated/api-surface.md docs/design-docs/agentic-semantic-router-v8.md docs/design-docs/index.md docs/exec-plans/completed/2026-07-evaluation-multiquery-graph-locator.md docs/exec-plans/completed/index.md
git commit -m "docs: record evaluation retrieval v8 baseline"
```

- [ ] **Step 10: Final clean-state verification**

Run:

```powershell
git status --short
git log -6 --oneline
```

Expected: clean status and one focused commit per task plus the earlier approved design commit.
