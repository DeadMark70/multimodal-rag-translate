# RAG Pipeline Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the legacy RAG answer flow into a low-complexity functional pipeline while preserving the `data_base.RAG_QA_service` public contract and all existing behavior.

**Architecture:** `data_base/RAG_QA_service.py` becomes an explicit-signature facade that delegates through the `data_base.rag_pipeline` module. `rag_pipeline.py` composes existing retrieval, filtering, CRAG, Graph, and generation owners through four stage functions and small private outcome dataclasses; public result contracts move to `rag_pipeline_schemas.py` and remain re-exported by the facade.

**Tech Stack:** Python 3.13, pytest, Ruff, FastAPI threadpool helpers, LangChain `Document`.

## Global Constraints

- Preserve the exact `rag_answer_question()` parameter order, names, annotations, keyword defaults, return annotation, error messages, return shapes, progress events, prompts, feature flags, and fallbacks.
- Keep `initialize_llm_service()` in `data_base/RAG_QA_service.py`.
- Preserve facade re-exports for `RAGResult`, `ProgressCallback`, `get_graph_evidence_bundle`, `GraphContextDetails`, `GraphNeedDecision`, `GraphExecutionStrategy`, and `GraphEvidenceLifecycle`.
- Do not add a runner class, state machine, dependency-injection container, shared pipeline framework, or third-party dependency.
- Do not change retrieval counts, ranking behavior, CRAG policy, Graph routing, prompt text, or generated answer behavior.
- Do not merge runtime parent-chunk expansion with `data_base/context_enricher.py`.
- `data_base/rag_pipeline.py` must not import `data_base/RAG_QA_service.py`; Graph runtime must not import either module.
- `run_rag_pipeline()` and the four main stage functions must each have Ruff C901 complexity 10 or lower.
- The repository complexity ratchet must not increase in total score or finding count.
- Do not start the `CampaignEngine` refactor in this plan.

---

## File Structure

- Create `data_base/rag_pipeline.py`: legacy RAG orchestration and its stage-local helpers.
- Modify `data_base/rag_pipeline_schemas.py`: own and export `RAGResult` and `ProgressCallback` alongside the existing boundary schemas.
- Modify `data_base/RAG_QA_service.py`: retain startup behavior and compatibility re-exports; delegate answer execution.
- Modify `quality/ruff-complexity-baseline.json`: remove the facade orchestration finding and move the unchanged parent-expansion finding to its new owner.
- Modify focused RAG/Graph tests: retarget private imports and patches to the module that performs the lookup.
- Modify `docs/design-docs/retrieval-and-indexing.md`: document the facade/pipeline boundary.

---

### Task 1: Move Public Result Contracts to the Schema Owner

**Files:**
- Modify: `data_base/rag_pipeline_schemas.py:1-44`
- Modify: `data_base/RAG_QA_service.py:9-119`
- Test: `tests/test_rag_retrieval_generation_split.py:1-90`

**Interfaces:**
- Consumes: existing `RAGResult` field order and defaults from `data_base.RAG_QA_service`.
- Produces: `data_base.rag_pipeline_schemas.RAGResult` and `ProgressCallback`; facade re-exports with object identity preserved.

- [x] **Step 1: Write the failing schema-ownership test**

Add this test to `tests/test_rag_retrieval_generation_split.py`:

```python
def test_pipeline_schemas_own_public_result_contract_and_facade_reexports_it() -> None:
    schemas = import_module("data_base.rag_pipeline_schemas")
    facade = import_module("data_base.RAG_QA_service")

    assert facade.RAGResult is schemas.RAGResult
    assert facade.ProgressCallback is schemas.ProgressCallback
    assert schemas.RAGResult.__module__ == "data_base.rag_pipeline_schemas"
```

- [x] **Step 2: Run the ownership test and confirm the red state**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_rag_retrieval_generation_split.py::test_pipeline_schemas_own_public_result_contract_and_facade_reexports_it -q
```

Expected: FAIL because `rag_pipeline_schemas` does not yet export `RAGResult` or `ProgressCallback`.

- [x] **Step 3: Move the contracts without changing their shape**

Add the required imports and exact definitions to `rag_pipeline_schemas.py`:

```python
from typing import Any, Awaitable, Callable, Dict, List, NamedTuple, Optional


class RAGResult(NamedTuple):
    """Result from RAG question answering with optional documents."""

    answer: str
    source_doc_ids: List[str]
    documents: List[Document]
    usage: Dict[str, int] = {}
    thought_process: Optional[str] = None
    tool_calls: List[dict] = []
    agent_trace: Optional[dict] = None
    visual_verification_meta: Optional[Dict[str, Any]] = None


ProgressCallback = Callable[[str, Optional[Dict[str, Any]]], Awaitable[None]]
```

Set the schema module export list to:

```python
__all__ = [
    "GeneratedRagAnswer",
    "ProgressCallback",
    "RAGResult",
    "RagRetrievalResult",
]
```

Remove the local definitions from `RAG_QA_service.py` and explicitly re-export them:

```python
from data_base.rag_pipeline_schemas import (
    ProgressCallback as ProgressCallback,
    RAGResult as RAGResult,
)
```

- [x] **Step 4: Run contract and consumer tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_rag_retrieval_generation_split.py tests\test_agentic_chat_service.py tests\test_rag_modes_agentic.py tests\test_rag_ask_stream.py -q
```

Expected: PASS with the existing `RAGResult` construction and `isinstance` behavior unchanged.

- [x] **Step 5: Run focused Ruff checks**

Run:

```powershell
.venv\Scripts\python.exe -m ruff check data_base\rag_pipeline_schemas.py data_base\RAG_QA_service.py tests\test_rag_retrieval_generation_split.py --select E9,F63,F7,F82,F401,F841
```

Expected: PASS.

- [x] **Step 6: Commit the schema ownership change**

```powershell
git add -- data_base/rag_pipeline_schemas.py data_base/RAG_QA_service.py tests/test_rag_retrieval_generation_split.py
git commit -m "refactor(rag): centralize pipeline result contracts"
```

---

### Task 2: Add the Retrieval Stage and Explicit Outcome Contract

**Files:**
- Create: `data_base/rag_pipeline.py`
- Modify: `tests/test_rag_retrieval_generation_split.py`
- Test: `tests/test_rag_retrieval_logic.py`
- Test: `tests/test_reranker_logic.py`

**Interfaces:**
- Consumes: `retrieve_hybrid_documents()`, `filter_and_rerank_retrieval()`, `get_user_retriever_async()`, `RAGResult`, and `ProgressCallback`.
- Produces: `_run_retrieval_stage -> RetrievalStageOutcome` and `_terminal_result -> LegacyRagResponse`.

- [x] **Step 1: Write failing retrieval-stage contract tests**

Add imports and these tests to `tests/test_rag_retrieval_generation_split.py`:

```python
@pytest.mark.asyncio
async def test_retrieval_stage_returns_terminal_result_without_a_retriever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = import_module("data_base.rag_pipeline")
    monkeypatch.setattr(
        pipeline,
        "get_user_retriever_async",
        AsyncMock(return_value=None),
    )

    outcome = await pipeline._run_retrieval_stage(
        question="question",
        user_id="user-1",
        doc_ids=None,
        enable_reranking=False,
        enable_hyde=False,
        enable_multi_query=False,
        plain_mode=True,
        mode_hints=None,
        progress_callback=None,
        return_docs=True,
    )

    assert outcome.documents == []
    assert outcome.terminal_result == pipeline.RAGResult(
        "抱歉，您還沒有建立任何知識庫文件，請先上傳 PDF。",
        [],
        [],
    )


def test_terminal_result_preserves_legacy_tuple_and_rag_result_shapes() -> None:
    pipeline = import_module("data_base.rag_pipeline")

    assert pipeline._terminal_result("message", ["doc-1"], False) == (
        "message",
        ["doc-1"],
    )
    assert pipeline._terminal_result("message", ["doc-1"], True) == pipeline.RAGResult(
        "message",
        ["doc-1"],
        [],
    )
```

- [x] **Step 2: Run the new tests and confirm the red state**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_rag_retrieval_generation_split.py -k "retrieval_stage or terminal_result" -q
```

Expected: FAIL because `data_base.rag_pipeline` does not exist.

- [x] **Step 3: Create the pipeline contracts and retrieval stage**

Create `data_base/rag_pipeline.py` with these public/internal contracts:

```python
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from data_base.rag_pipeline_schemas import ProgressCallback, RAGResult

LegacyRagResponse = tuple[str, list[str]] | RAGResult


@dataclass(slots=True)
class RetrievalStageOutcome:
    documents: list[Document]
    retriever: Any | None = None
    reranker_available: bool = False
    target_k: int = 0
    terminal_result: LegacyRagResponse | None = None


def _terminal_result(
    message: str,
    source_doc_ids: list[str],
    return_docs: bool,
) -> LegacyRagResponse:
    if return_docs:
        return RAGResult(message, source_doc_ids, [])
    return (message, source_doc_ids)
```

Implement this exact retrieval-stage signature:

```python
async def _run_retrieval_stage(
    *,
    question: str,
    user_id: str,
    doc_ids: list[str] | None,
    enable_reranking: bool,
    enable_hyde: bool,
    enable_multi_query: bool,
    plain_mode: bool,
    mode_hints: dict[str, Any] | None,
    progress_callback: ProgressCallback | None,
    return_docs: bool,
) -> RetrievalStageOutcome:
```

Move the behavior currently in `RAG_QA_service.py:508-596` into this function. Keep these operations in the same order:

1. Resolve `_resolve_retrieval_policy(mode_hints)` and `retrieval_k` with the current caps and defaults.
2. Await `get_user_retriever_async(user_id, retrieval_k, plain_mode=plain_mode)`.
3. Return the exact no-knowledge-base terminal response when it returns `None`.
4. Call `retrieve_hybrid_documents()` with the current transformer and query-executor arguments.
5. Preserve the current retrieval exception types and message.
6. Preserve the empty-retrieval message.
7. Run `filter_and_rerank_retrieval()` through `run_in_threadpool` with the current `target_k` and candidate cap.
8. Preserve requested-document empty handling, retrieval logging, reranking progress, and inactive-reranker logging.
9. Return selected documents, retriever, reranker availability, and target size with `terminal_result=None`.

Keep `_resolve_retrieval_policy()` and `_emit_progress()` private in `rag_pipeline.py`; copy their current behavior mechanically for now. Do not remove the facade flow until Task 4.

- [x] **Step 4: Run the retrieval-stage tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_rag_retrieval_generation_split.py tests\test_rag_retrieval_logic.py tests\test_reranker_logic.py -q
```

Expected: PASS.

- [x] **Step 5: Enforce the retrieval-stage complexity target**

Run:

```powershell
.venv\Scripts\python.exe -m ruff check data_base\rag_pipeline.py --select C901
```

Expected: no C901 report for `_run_retrieval_stage`. If the function exceeds 10, extract only logging/progress formatting into `_record_retrieval_selection()` without moving filtering or ranking algorithms into the pipeline.

- [x] **Step 6: Commit the independently tested retrieval stage**

```powershell
git add -- data_base/rag_pipeline.py tests/test_rag_retrieval_generation_split.py
git commit -m "refactor(rag): add pipeline retrieval stage"
```

---

### Task 3: Add CRAG and Graph Stages

**Files:**
- Modify: `data_base/rag_pipeline.py`
- Modify: `tests/test_rag_retrieval_generation_split.py`
- Test: `tests/test_graph_auto_gate.py`
- Test: `tests/test_graph_context_packing.py`
- Test: `tests/test_rag_graph_evidence_docs.py`
- Test: `tests/test_rag_retrieval_logic.py`

**Interfaces:**
- Consumes: `RetrievalStageOutcome`, existing `rag_crag`, `rag_filtering`, `rag_graph_runtime`, and `rag_graph_locator` functions.
- Produces: `_run_crag_stage -> CragStageOutcome` and `_run_graph_stage -> GraphStageOutcome`.

- [x] **Step 1: Write failing CRAG and Graph stage tests**

Add these outcome-contract tests to `tests/test_rag_retrieval_generation_split.py`:

```python
@pytest.mark.asyncio
async def test_disabled_crag_stage_passes_documents_through() -> None:
    pipeline = import_module("data_base.rag_pipeline")
    document = Document(page_content="evidence", metadata={"doc_id": "doc-1"})

    outcome = await pipeline._run_crag_stage(
        question="question",
        documents=[document],
        retriever=SimpleNamespace(),
        enable_crag=False,
        crag_rewrite_mode="hyde",
        doc_ids=None,
        enable_reranking=False,
        reranker_available=False,
        target_k=3,
        progress_callback=None,
        return_docs=True,
    )

    assert outcome.documents == [document]
    assert outcome.terminal_result is None


@pytest.mark.asyncio
async def test_disabled_graph_stage_returns_neutral_context() -> None:
    pipeline = import_module("data_base.rag_pipeline")
    document = Document(page_content="evidence", metadata={"doc_id": "doc-1"})

    outcome = await pipeline._run_graph_stage(
        question="question",
        user_id="user-1",
        documents=[document],
        doc_ids=None,
        enable_graph_rag=False,
        graph_search_mode="generic",
        graph_execution_hints=None,
        mode_hints=None,
        return_docs=True,
        progress_callback=None,
    )

    assert outcome.documents == [document]
    assert outcome.graph_context == ""
    assert outcome.graph_evidence_documents == []
```

- [x] **Step 2: Run the stage tests and confirm the red state**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_rag_retrieval_generation_split.py -k "crag_stage or graph_stage" -q
```

Expected: FAIL because the outcome types and stage functions are absent.

- [x] **Step 3: Add explicit CRAG and Graph outcomes**

Add these dataclasses to `rag_pipeline.py`:

```python
@dataclass(slots=True)
class CragStageOutcome:
    documents: list[Document]
    terminal_result: LegacyRagResponse | None = None


@dataclass(slots=True)
class GraphStageOutcome:
    documents: list[Document]
    graph_context: str = ""
    graph_evidence_documents: list[Document] = field(default_factory=list)
```

Import `field` from `dataclasses`.

- [x] **Step 4: Implement the CRAG stage mechanically**

Add this signature:

```python
async def _run_crag_stage(
    *,
    question: str,
    documents: list[Document],
    retriever: Any,
    enable_crag: bool,
    crag_rewrite_mode: CragRewriteMode,
    doc_ids: list[str] | None,
    enable_reranking: bool,
    reranker_available: bool,
    target_k: int,
    progress_callback: ProgressCallback | None,
    return_docs: bool,
) -> CragStageOutcome:
```

Move `RAG_QA_service.py:598-636` into this function. Preserve the disabled pass-through, all `run_corrective_retrieval()` dependency arguments, the insufficient message, rewrite progress, and the broad exception fallback to the original `documents`.

- [x] **Step 5: Implement Graph strategy dispatch without a large conditional**

Add this main signature:

```python
async def _run_graph_stage(
    *,
    question: str,
    user_id: str,
    documents: list[Document],
    doc_ids: list[str] | None,
    enable_graph_rag: bool,
    graph_search_mode: str,
    graph_execution_hints: dict[str, Any] | None,
    mode_hints: dict[str, Any] | None,
    return_docs: bool,
    progress_callback: ProgressCallback | None,
) -> GraphStageOutcome:
```

Move `RAG_QA_service.py:638-813` while splitting strategy bodies into these private helpers:

```python
async def _run_graph_skip_strategy(
    *,
    question: str,
    documents: list[Document],
    graph_search_mode: str,
    graph_execution_hints: dict[str, Any] | None,
    mode_hints: dict[str, Any] | None,
    progress_callback: ProgressCallback | None,
    strategy: rag_graph_runtime.GraphExecutionStrategy,
) -> GraphStageOutcome:
    """Record an explicit Graph skip and return unchanged documents."""


async def _run_graph_source_expand_strategy(
    *,
    question: str,
    user_id: str,
    documents: list[Document],
    doc_ids: list[str] | None,
    graph_search_mode: str,
    graph_execution_hints: dict[str, Any] | None,
    mode_hints: dict[str, Any] | None,
    progress_callback: ProgressCallback | None,
    strategy: rag_graph_runtime.GraphExecutionStrategy,
) -> GraphStageOutcome:
    """Locate source chunks from Graph evidence and record observability."""


async def _run_graph_raw_legacy_strategy(
    *,
    question: str,
    user_id: str,
    documents: list[Document],
    graph_search_mode: str,
    graph_execution_hints: dict[str, Any] | None,
    mode_hints: dict[str, Any] | None,
    progress_callback: ProgressCallback | None,
    return_docs: bool,
) -> GraphStageOutcome:
    """Load legacy raw Graph context and optional evaluation evidence."""
```

The main Graph stage must only resolve flags and strategy, then dispatch `skip`, `source_expand`, or `raw_legacy`. Each helper copies its corresponding current branch without changing progress payloads, lifecycle construction, evidence projection, or observability calls. Use module attribute lookups such as `rag_graph_runtime._get_graph_context` so private Graph tests can patch the implementation owner.

- [x] **Step 6: Run stage and existing Graph/CRAG behavior tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_rag_retrieval_generation_split.py tests\test_graph_auto_gate.py tests\test_graph_context_packing.py tests\test_rag_graph_evidence_docs.py tests\test_rag_retrieval_logic.py -q
```

Expected: PASS. Production still uses the facade body, while the new stages have direct contract coverage.

- [x] **Step 7: Enforce stage complexity**

Run:

```powershell
.venv\Scripts\python.exe -m ruff check data_base\rag_pipeline.py --select C901
```

Expected: no C901 report for `_run_crag_stage`, `_run_graph_stage`, or the three strategy helpers.

- [x] **Step 8: Commit the optional stages**

```powershell
git add -- data_base/rag_pipeline.py tests/test_rag_retrieval_generation_split.py
git commit -m "refactor(rag): add CRAG and Graph pipeline stages"
```

---

### Task 4: Add Generation, Switch the Facade, and Retarget Tests

**Files:**
- Modify: `data_base/rag_pipeline.py`
- Modify: `data_base/RAG_QA_service.py:1-849`
- Modify: `quality/ruff-complexity-baseline.json`
- Modify: `tests/test_evaluation_phase_attribution.py`
- Modify: `tests/test_graph_auto_gate.py`
- Modify: `tests/test_graph_context_packing.py`
- Modify: `tests/test_rag_graph_evidence_docs.py`
- Modify: `tests/test_rag_retrieval_generation_split.py`
- Modify: `tests/test_rag_retrieval_logic.py`
- Modify: `tests/test_reranker_logic.py`
- Modify: `tests/test_visual_tool_trigger.py`

**Interfaces:**
- Consumes: all three stage outcomes and `generate_legacy_answer_from_evidence()`.
- Produces: `run_rag_pipeline()` with the exact legacy signature; facade `rag_answer_question()` with the same signature and delegation only.

- [x] **Step 1: Write failing facade delegation and signature tests**

Add these tests to `tests/test_rag_retrieval_generation_split.py`:

```python
def test_facade_and_pipeline_answer_signatures_do_not_drift() -> None:
    facade = import_module("data_base.RAG_QA_service")
    pipeline = import_module("data_base.rag_pipeline")

    assert inspect.signature(facade.rag_answer_question) == inspect.signature(
        pipeline.run_rag_pipeline
    )


@pytest.mark.asyncio
async def test_facade_delegates_through_pipeline_module_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facade = import_module("data_base.RAG_QA_service")
    expected = facade.RAGResult("answer", ["doc-1"], [])
    delegate = AsyncMock(return_value=expected)
    monkeypatch.setattr(facade.rag_pipeline, "run_rag_pipeline", delegate)

    result = await facade.rag_answer_question(
        "question",
        "user-1",
        return_docs=True,
    )

    assert result is expected
    delegate.assert_awaited_once()
    assert delegate.await_args.kwargs["question"] == "question"
    assert delegate.await_args.kwargs["user_id"] == "user-1"
    assert delegate.await_args.kwargs["return_docs"] is True
```

Import `inspect` at the top of the test module.

- [x] **Step 2: Run the delegation tests and confirm the red state**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_rag_retrieval_generation_split.py -k "signatures_do_not_drift or delegates_through_pipeline" -q
```

Expected: FAIL because `run_rag_pipeline()` is absent and the facade still owns the body.

- [x] **Step 3: Add generation-stage helpers and preserve legacy projection**

Move `_format_history_for_prompt()`, `_resolve_intent_hint()`, `_intent_constraints_for_prompt()`, `_expand_short_chunks()`, and its constants from the facade into `rag_pipeline.py` without behavior changes.

Add this signature:

```python
async def _run_generation_stage(
    *,
    question: str,
    user_id: str,
    llm: Any,
    graph_outcome: GraphStageOutcome,
    history: Optional[List["ChatMessage"]],
    mode_hints: Optional[Dict[str, Any]],
    plain_mode: bool,
    enable_visual_verification: bool,
    progress_callback: Optional[ProgressCallback],
    return_docs: bool,
) -> LegacyRagResponse:
```

Move `RAG_QA_service.py:815-849` into this function. Preserve advanced-mode parent expansion, history formatting, intent constraints, source ID order, thought-process-dependent Graph evidence projection, usage, tool calls, and visual metadata. Call generation through the owner module and pass the owner lookup for image encoding:

```python
generated = await rag_generation.generate_legacy_answer_from_evidence(
    question=question,
    user_id=user_id,
    documents=graph_outcome.documents,
    llm=llm,
    graph_context=graph_outcome.graph_context,
    history_section=(f"\n{_format_history_for_prompt(history)}\n" if history else ""),
    intent_constraints=_intent_constraints_for_prompt(question, mode_hints),
    plain_mode=plain_mode,
    enable_visual_verification=enable_visual_verification,
    progress_callback=progress_callback,
    image_encoder=rag_generation._encode_image,
)
```

- [x] **Step 4: Add the low-complexity orchestration function**

Import `Any`, `Dict`, `List`, `Optional`, `Tuple`, `TYPE_CHECKING`, and `Union` from `typing`, and use the same type-checking-only `ChatMessage` import as the facade. Define `run_rag_pipeline()` with the exact current `rag_answer_question()` signature and this orchestration body:

```python
async def run_rag_pipeline(
    question: str,
    user_id: str,
    doc_ids: Optional[List[str]] = None,
    history: Optional[List["ChatMessage"]] = None,
    enable_reranking: bool = False,
    enable_hyde: bool = False,
    enable_multi_query: bool = False,
    enable_crag: bool = False,
    return_docs: bool = False,
    enable_graph_rag: bool = False,
    graph_search_mode: str = "generic",
    graph_execution_hints: Optional[Dict[str, Any]] = None,
    mode_hints: Optional[Dict[str, Any]] = None,
    enable_visual_verification: bool = False,
    plain_mode: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
    crag_rewrite_mode: CragRewriteMode = "hyde",
) -> Union[Tuple[str, List[str]], RAGResult]:
    llm = _resolve_pipeline_llm()
    if llm is None:
        return _terminal_result(
            "抱歉，AI 模型尚未初始化 (API Key 可能有誤)。",
            [],
            return_docs,
        )

    retrieval = await _run_retrieval_stage(
        question=question,
        user_id=user_id,
        doc_ids=doc_ids,
        enable_reranking=enable_reranking,
        enable_hyde=enable_hyde,
        enable_multi_query=enable_multi_query,
        plain_mode=plain_mode,
        mode_hints=mode_hints,
        progress_callback=progress_callback,
        return_docs=return_docs,
    )
    if retrieval.terminal_result is not None:
        return retrieval.terminal_result

    crag = await _run_crag_stage(
        question=question,
        documents=retrieval.documents,
        retriever=retrieval.retriever,
        enable_crag=enable_crag,
        crag_rewrite_mode=crag_rewrite_mode,
        doc_ids=doc_ids,
        enable_reranking=enable_reranking,
        reranker_available=retrieval.reranker_available,
        target_k=retrieval.target_k,
        progress_callback=progress_callback,
        return_docs=return_docs,
    )
    if crag.terminal_result is not None:
        return crag.terminal_result

    graph = await _run_graph_stage(
        question=question,
        user_id=user_id,
        documents=crag.documents,
        doc_ids=doc_ids,
        enable_graph_rag=enable_graph_rag,
        graph_search_mode=graph_search_mode,
        graph_execution_hints=graph_execution_hints,
        mode_hints=mode_hints,
        return_docs=return_docs,
        progress_callback=progress_callback,
    )
    return await _run_generation_stage(
        question=question,
        user_id=user_id,
        llm=llm,
        graph_outcome=graph,
        history=history,
        mode_hints=mode_hints,
        plain_mode=plain_mode,
        enable_visual_verification=enable_visual_verification,
        progress_callback=progress_callback,
        return_docs=return_docs,
    )
```

Implement `_resolve_pipeline_llm() -> Any | None` with the existing `get_llm("rag_qa")` call, caught `RuntimeError`, `KeyError`, and `ValueError`, and the current error log. Returning `None` is private stage state; `run_rag_pipeline()` constructs the unchanged public failure projection shown above.

- [x] **Step 5: Replace the facade body with explicit delegation**

In `RAG_QA_service.py`, retain module documentation, startup imports/state, `initialize_llm_service()`, and required compatibility exports. Import the module:

```python
from data_base import rag_pipeline
```

Keep the exact current `rag_answer_question()` signature and delegate every argument explicitly:

```python
async def rag_answer_question(
    question: str,
    user_id: str,
    doc_ids: Optional[List[str]] = None,
    history: Optional[List["ChatMessage"]] = None,
    enable_reranking: bool = False,
    enable_hyde: bool = False,
    enable_multi_query: bool = False,
    enable_crag: bool = False,
    return_docs: bool = False,
    enable_graph_rag: bool = False,
    graph_search_mode: str = "generic",
    graph_execution_hints: Optional[Dict[str, Any]] = None,
    mode_hints: Optional[Dict[str, Any]] = None,
    enable_visual_verification: bool = False,
    plain_mode: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
    crag_rewrite_mode: CragRewriteMode = "hyde",
) -> Union[Tuple[str, List[str]], RAGResult]:
    return await rag_pipeline.run_rag_pipeline(
        question=question,
        user_id=user_id,
        doc_ids=doc_ids,
        history=history,
        enable_reranking=enable_reranking,
        enable_hyde=enable_hyde,
        enable_multi_query=enable_multi_query,
        enable_crag=enable_crag,
        return_docs=return_docs,
        enable_graph_rag=enable_graph_rag,
        graph_search_mode=graph_search_mode,
        graph_execution_hints=graph_execution_hints,
        mode_hints=mode_hints,
        enable_visual_verification=enable_visual_verification,
        plain_mode=plain_mode,
        progress_callback=progress_callback,
        crag_rewrite_mode=crag_rewrite_mode,
    )
```

Remove facade-owned orchestration helpers after their tests have been retargeted. Small compatibility aliases may remain only when a non-test runtime consumer still imports them.

- [x] **Step 6: Retarget private imports and patches to lookup owners**

Apply these ownership rules across the listed tests:

```text
data_base.RAG_QA_service.get_llm
    -> data_base.rag_pipeline.get_llm

data_base.RAG_QA_service.get_user_retriever
    -> data_base.rag_pipeline.get_user_retriever_async

data_base.RAG_QA_service._expand_short_chunks
    -> data_base.rag_pipeline._expand_short_chunks

data_base.RAG_QA_service.ParentDocumentStore
    -> data_base.rag_pipeline.ParentDocumentStore

data_base.RAG_QA_service._get_graph_context
data_base.RAG_QA_service._get_graph_evidence_bundle
data_base.RAG_QA_service._record_graph_observability
    -> corresponding data_base.rag_graph_runtime names

data_base.RAG_QA_service.locate_graph_sources
    -> data_base.rag_graph_locator.locate_graph_sources

data_base.RAG_QA_service.get_llm_usage_metrics
data_base.RAG_QA_service.fetch_document_filenames
data_base.RAG_QA_service._encode_image
    -> corresponding data_base.rag_generation names

data_base.RAG_QA_service._build_crag_queries
    -> data_base.rag_crag.build_crag_queries

data_base.RAG_QA_service._rerank_documents_for_generation
data_base.RAG_QA_service._limit_rerank_candidates
    -> corresponding data_base.rag_filtering names

data_base.RAG_QA_service._parse_visual_tool_request
    -> data_base.rag_generation.parse_legacy_visual_tool_request
```

When a pipeline function uses a module attribute lookup, patch that owner module. When it imports a provider symbol directly, patch `data_base.rag_pipeline.<symbol>`. Update `_build_crag_queries` tests to pass the real owner's required `hyde_transformer` and `multi_query_transformer` keyword arguments instead of recreating a facade wrapper.

Update `test_legacy_wrapper_delegates_generation_without_exposing_visual_synthesis` so it asserts `generate_legacy_answer_from_evidence(` is owned by `rag_pipeline.py`, while `RAG_QA_service.py` contains `rag_pipeline.run_rag_pipeline(` and still contains neither visual-verification loop implementation. Retarget `test_legacy_wrapper_preserves_empty_retrieval_projection` to patch the pipeline LLM and retriever lookups while continuing to call the facade.

- [x] **Step 7: Update the complexity baseline**

In `quality/ruff-complexity-baseline.json`:

- Remove `data_base/RAG_QA_service.py::rag_answer_question` with score 30.
- Move `data_base/RAG_QA_service.py::_expand_short_chunks` with score 12 to `data_base/rag_pipeline.py::_expand_short_chunks` with the same score.
- Do not add entries for `run_rag_pipeline()` or any main stage.

- [x] **Step 8: Run focused integration tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_rag_retrieval_generation_split.py tests\test_rag_retrieval_logic.py tests\test_reranker_logic.py tests\test_evaluation_phase_attribution.py tests\test_graph_auto_gate.py tests\test_graph_context_packing.py tests\test_rag_graph_evidence_docs.py tests\test_visual_tool_trigger.py -q
```

Expected: PASS with facade imports, signatures, return values, progress payloads, Graph evidence, CRAG fallback, and visual verification unchanged.

- [x] **Step 9: Run complexity and Ruff checks**

Run:

```powershell
.venv\Scripts\python.exe -m ruff check data_base\RAG_QA_service.py data_base\rag_pipeline.py --select C901
.venv\Scripts\python.exe scripts\check_complexity_ratchet.py --check
.venv\Scripts\python.exe -m ruff check data_base\RAG_QA_service.py data_base\rag_pipeline.py tests\test_rag_retrieval_generation_split.py --select E9,F63,F7,F82,F401,F841
```

Expected: no C901 findings for the facade, `run_rag_pipeline()`, or the four main stages; complexity ratchet passes with no new finding; import/unused checks pass.

- [x] **Step 10: Commit the facade switch**

```powershell
git add -- data_base/RAG_QA_service.py data_base/rag_pipeline.py quality/ruff-complexity-baseline.json tests/test_evaluation_phase_attribution.py tests/test_graph_auto_gate.py tests/test_graph_context_packing.py tests/test_rag_graph_evidence_docs.py tests/test_rag_retrieval_generation_split.py tests/test_rag_retrieval_logic.py tests/test_reranker_logic.py tests/test_visual_tool_trigger.py
git commit -m "refactor(rag): delegate facade to staged pipeline"
```

---

### Task 5: Document the Boundary and Run the Complete Verification Gate

**Files:**
- Modify: `docs/design-docs/retrieval-and-indexing.md:7-18`
- Verify: all files changed by Tasks 1–4

**Interfaces:**
- Consumes: the completed facade and pipeline boundary.
- Produces: maintained architecture documentation and a fully verified isolated refactor.

- [x] **Step 1: Update the architecture documentation**

Replace the single retrieval/generation path entry in `docs/design-docs/retrieval-and-indexing.md` with:

```markdown
- Legacy RAG compatibility facade: `data_base/RAG_QA_service.py`
- Legacy RAG orchestration: `data_base/rag_pipeline.py`
- Retrieval, filtering, CRAG, Graph, and generation stages: their focused `data_base/rag_*.py` owners
```

Add this design rule:

```markdown
- `RAG_QA_service.py` preserves public imports and delegates execution; new orchestration logic belongs in `rag_pipeline.py` or the focused stage owner.
```

- [x] **Step 2: Run all focused RAG and Graph tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_rag_retrieval_generation_split.py tests\test_rag_retrieval_pipeline.py tests\test_rag_retrieval_logic.py tests\test_rag_filtering.py tests\test_rag_crag.py tests\test_rag_qa_prompts.py tests\test_rag_pipeline_prompts.py tests\test_reranker_logic.py tests\test_visual_tool_trigger.py tests\test_graph_auto_gate.py tests\test_graph_context_packing.py tests\test_rag_graph_evidence_docs.py tests\test_graphrag_integration.py -q
```

Expected: PASS.

- [x] **Step 3: Run the complete backend test gate with warning budget**

Run:

```powershell
.venv\Scripts\python.exe scripts\run_pytest_with_warning_budget.py --max-warnings 56 -- -q
```

Expected: all tests pass, warning count is 56 or lower, and no external API access is required.

- [x] **Step 4: Run generated artifact and documentation checks**

Run:

```powershell
.venv\Scripts\python.exe scripts\sync_openapi_artifacts.py --check
.venv\Scripts\python.exe scripts\check_markdown_links.py
```

Expected: OpenAPI artifacts are current and Markdown links are valid.

- [x] **Step 5: Run repository-wide quality gates**

Run:

```powershell
.venv\Scripts\python.exe scripts\check_complexity_ratchet.py --check
.venv\Scripts\python.exe -m ruff check . --select E9,F63,F7,F82,F401,F841
git diff --check
```

Expected: complexity ratchet passes with the 30-point facade finding removed, Ruff passes, and Git reports no whitespace errors.

- [x] **Step 6: Review scope and commit the documentation**

Run:

```powershell
git status --short
git diff --stat
git diff -- docs/design-docs/retrieval-and-indexing.md
```

Confirm the work contains only the RAG pipeline extraction and its tests/docs, with no `CampaignEngine` changes. Then commit:

```powershell
git add -- docs/design-docs/retrieval-and-indexing.md
git commit -m "docs: document RAG pipeline boundary"
```

- [x] **Step 7: Perform the post-commit check**

Run:

```powershell
git status --porcelain=v1
git log -5 --oneline --decorate
```

Expected: clean worktree and the plan's isolated commits at `HEAD`; do not push unless the user explicitly requests it.
