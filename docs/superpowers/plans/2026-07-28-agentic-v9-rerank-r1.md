# Agentic v9 Rerank R1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change Agentic v9 from Hybrid 8 → LLM 8 to Hybrid 8 → Jina rerank 8 → LLM 4 per retrieval task, with fail-soft Hybrid top-4 fallback.

**Architecture:** Keep candidate retrieval and selection inside the existing `filter_and_rerank_retrieval` boundary. Make the boundary honor `target_k` when reranking is requested but unavailable, then enable strict reranking only in the Agentic v9 production adapter and catch failures there. Attach bounded reranking diagnostics to selected document metadata so each task's trace proves whether scoring executed or fell back.

**Tech Stack:** Python 3.13, asyncio, LangChain `Document`, local Jina Reranker v3, Pydantic retrieval schemas, pytest, Ruff.

## Global Constraints

- Change only Agentic v9 runtime behavior; Naive remains `plain_mode=True` and `enable_reranking=False`.
- Keep Wave S open-user-corpus authorization unchanged.
- Use `k=8`, `max_candidates=8`, and `target_k=4` for every Agentic v9 retrieval task.
- The four-chunk limit is per retrieval task, not per run.
- Reranker failure must return original Hybrid top 4, never all 8 and never fail the campaign unit.
- Do not fabricate rerank scores.
- Do not change routing, slots, Graph, Visual, corrective retrieval, synthesis, provider configuration, or RAGAS.
- Do not add an execution flag or parallel experiment profile.

---

### Task 1: Make the existing reranking boundary enforce fail-soft target size

**Files:**
- Modify: `data_base/rag_filtering.py:152-184`
- Modify: `tests/test_rag_filtering.py:90-155`

**Interfaces:**
- Consumes: `filter_and_rerank_retrieval(..., enable_reranking=True, target_k=int)`
- Produces: unavailable-reranker selection that preserves original order and returns at most `target_k` documents with `None` scores

- [ ] **Step 1: Replace the unavailable-reranker expectations with failing top-k tests**

Update the existing unavailable tests to require bounded fail-soft selection:

```python
def test_unavailable_reranker_preserves_original_top_k_with_none_scores() -> None:
    first = Document(page_content="First", metadata={"doc_id": "one"})
    second = Document(page_content="Second", metadata={"doc_id": "two"})
    retrieval = RagRetrievalResult(documents=[first, second])

    result = filter_and_rerank_retrieval(
        "question",
        retrieval,
        enable_reranking=True,
        reranker_available=False,
        target_k=1,
    )

    assert result.documents == [first]
    assert result.metadata["reranking"]["post_rerank_ranks"][0]["score"] is None
    assert result.metadata["reranking"]["rejected_candidates"][0]["reason"] == (
        "selection_limit"
    )


def test_unavailable_reranker_caps_candidates_to_requested_target() -> None:
    documents = [
        Document(page_content=f"Document {index}", metadata={"doc_id": str(index)})
        for index in range(8)
    ]

    result = filter_and_rerank_retrieval(
        "question",
        RagRetrievalResult(documents=documents),
        enable_reranking=True,
        reranker_available=False,
        target_k=4,
        max_candidates=8,
    )

    assert result.documents == documents[:4]
    assert all(
        row["score"] is None
        for row in result.metadata["reranking"]["post_rerank_ranks"]
    )
```

- [ ] **Step 2: Run the two tests and verify the current behavior fails**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -p no:cacheprovider `
  tests\test_rag_filtering.py::test_unavailable_reranker_preserves_original_top_k_with_none_scores `
  tests\test_rag_filtering.py::test_unavailable_reranker_caps_candidates_to_requested_target -q
```

Expected: both tests fail because the current unavailable branch returns every filtered document.

- [ ] **Step 3: Implement the minimal bounded fallback**

Change only the `enable_reranking` unavailable branch in `_select_documents`:

```python
if enable_reranking:
    selected = documents[:target_k]
    return (
        selected,
        _post_rerank_rows(selected, documents, {}),
        _selection_rejections(documents, selected),
    )
```

Do not change the successful scoring branch or the `enable_reranking=False`
multi-document behavior.

- [ ] **Step 4: Run filtering tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests\test_rag_filtering.py -q
.venv\Scripts\python.exe -m ruff check --no-cache `
  data_base\rag_filtering.py tests\test_rag_filtering.py
```

Expected: all tests and Ruff pass.

- [ ] **Step 5: Commit the boundary fix**

```powershell
git add -- data_base/rag_filtering.py tests/test_rag_filtering.py
git commit -m "fix(rag): bound unavailable reranker fallback"
```

---

### Task 2: Enable Agentic v9 rerank 8 → 4 with task-level diagnostics

**Files:**
- Modify: `evaluation/agentic_v9_campaign_runtime.py:521-565`
- Modify: `tests/test_agentic_v9_campaign_runtime.py`
- Modify: `tests/test_rag_modes_agentic.py`

**Interfaces:**
- Consumes: Task 1 bounded `filter_and_rerank_retrieval`
- Produces: `_retrieve_documents(user_id: str, question: str, authorized_doc_ids: list[str]) -> list[Document]` returning at most four annotated documents
- Metadata contract on each selected document:

```python
{
    "agentic_v9_reranking": {
        "status": "executed" | "fallback",
        "fallback_reason": str | None,
        "candidate_count": int,
        "selected_count": int,
        "pre_rerank_rank": int,
        "post_rerank_rank": int,
        "rerank_score": float | None,
    }
}
```

- [ ] **Step 1: Add failing successful-rerank and fallback tests**

Patch the production dependencies used by `_retrieve_documents` and verify all
three terminal paths:

```python
@pytest.mark.asyncio
async def test_v9_retrieval_reranks_eight_to_four(monkeypatch) -> None:
    documents = [
        Document(page_content=f"chunk-{index}", metadata={"doc_id": "doc-1"})
        for index in range(8)
    ]
    monkeypatch.setattr(
        runtime_module,
        "get_user_retriever_async",
        AsyncMock(return_value=object()),
    )
    monkeypatch.setattr(
        runtime_module,
        "retrieve_hybrid_documents",
        AsyncMock(return_value=PipelineRetrievalResult(documents=documents)),
    )
    monkeypatch.setattr(DocumentReranker, "is_initialized", lambda: True)
    monkeypatch.setattr(
        DocumentReranker,
        "get_instance",
        lambda: SimpleNamespace(
            rerank_with_scores_strict=lambda _query, docs, _top_k: [
                (docs[index], float(8 - index)) for index in reversed(range(8))
            ]
        ),
    )

    selected = await runtime_module._retrieve_documents(
        "user-a", "question", ["doc-1"]
    )

    assert len(selected) == 4
    assert all(
        doc.metadata["agentic_v9_reranking"]["status"] == "executed"
        for doc in selected
    )
    assert all(
        doc.metadata["agentic_v9_reranking"]["rerank_score"] is not None
        for doc in selected
    )
```

Add corresponding tests in the same file for:

```python
DocumentReranker.is_initialized() is False
```

and:

```python
rerank_with_scores_strict raises RuntimeError("reranker failed")
```

Both must assert:

```python
selected == documents[:4]
status == "fallback"
rerank_score is None
```

Use stable fallback reasons `reranker_unavailable` and `reranker_error`;
do not place exception text in document metadata.

- [ ] **Step 2: Add a failing multiple-task integration assertion**

Extend the existing v9 runtime integration test with two compiled retrieval
tasks and make the injected retrieval adapter return four annotated documents
per call:

```python
assert retrieve_documents.await_count == 2
assert all(
    item["selected_count"] <= 4
    for item in result.agent_trace["agentic_v9"]["retrieval_diagnostics"]
)
```

Assert the two diagnostics retain distinct `task_id` values. Do not add a
production-only test seam.

- [ ] **Step 3: Add a Naive invariance test**

Add or strengthen this preset assertion in `tests/test_rag_modes_agentic.py`:

```python
def test_naive_remains_plain_without_reranking() -> None:
    assert RAG_MODES["naive"]["plain_mode"] is True
    assert RAG_MODES["naive"]["enable_reranking"] is False
    assert RAG_MODES["naive"]["enable_hyde"] is False
    assert RAG_MODES["naive"]["enable_multi_query"] is False
```

- [ ] **Step 4: Run the new tests and verify they fail**

Run the exact new test node IDs plus the Naive invariant:

```powershell
.venv\Scripts\python.exe -m pytest -p no:cacheprovider `
  tests\test_agentic_v9_campaign_runtime.py `
  tests\test_rag_modes_agentic.py::test_naive_remains_plain_without_reranking -q
```

Expected: the new production retrieval tests fail because v9 currently uses
`enable_reranking=False` and returns up to eight documents.

- [ ] **Step 5: Add a private metadata annotation helper**

In `evaluation/agentic_v9_campaign_runtime.py`, add:

```python
def _annotate_rerank_selection(
    selection: PipelineRetrievalResult,
    *,
    status: str,
    fallback_reason: str | None,
) -> list[Document]:
    reranking = dict(selection.metadata.get("reranking") or {})
    rows = list(reranking.get("post_rerank_ranks") or [])
    candidate_count = int(reranking.get("candidate_count") or 0)
    selected_count = len(selection.documents)
    annotated: list[Document] = []
    for post_rank, document in enumerate(selection.documents, start=1):
        row = rows[post_rank - 1] if post_rank <= len(rows) else {}
        annotated.append(
            Document(
                page_content=document.page_content,
                metadata={
                    **dict(document.metadata),
                    "agentic_v9_reranking": {
                        "status": status,
                        "fallback_reason": fallback_reason,
                        "candidate_count": candidate_count,
                        "selected_count": selected_count,
                        "pre_rerank_rank": int(
                            row.get("pre_rerank_rank") or post_rank
                        ),
                        "post_rerank_rank": post_rank,
                        "rerank_score": (
                            row.get("score") if status == "executed" else None
                        ),
                    },
                },
            )
        )
    return annotated
```

Alias `data_base.rag_pipeline_schemas.RagRetrievalResult` as
`PipelineRetrievalResult` to distinguish it from the Agentic v9 trace schema.

- [ ] **Step 6: Enable strict reranking in a worker thread and implement fail-soft**

Update `_retrieve_documents`:

```python
raw = await retrieve_hybrid_documents(
    question,
    retriever,
    enable_hyde=False,
    enable_multi_query=False,
)
try:
    selection = await asyncio.to_thread(
        filter_and_rerank_retrieval,
        question,
        raw,
        doc_ids=authorized_doc_ids,
        enable_reranking=True,
        target_k=4,
        max_candidates=8,
        strict_reranking=True,
    )
    scored = any(
        row.get("score") is not None
        for row in selection.metadata["reranking"]["post_rerank_ranks"]
    )
    status = "executed" if scored else "fallback"
    fallback_reason = None if scored else "reranker_empty_result"
except Exception:  # noqa: BLE001 -- fail-soft stage boundary
    selection = filter_and_rerank_retrieval(
        question,
        raw,
        doc_ids=authorized_doc_ids,
        enable_reranking=True,
        reranker_available=False,
        target_k=4,
        max_candidates=8,
    )
    status = "fallback"
    fallback_reason = "reranker_error"
```

Before entering the `try`, check `DocumentReranker.is_initialized()`. If false,
skip the strict call and run the bounded fallback with
`fallback_reason="reranker_unavailable"`.

Use `asyncio.to_thread` only for actual model scoring. The unavailable fallback
is deterministic and may remain synchronous.

- [ ] **Step 7: Project diagnostics into the task trace**

Add `"retrieval_diagnostics": []` to runtime `state`. In the `retrieve` stage,
after `_retrieve_documents` returns, read the annotations and append one
task-owned projection:

```python
rows = [
    dict(document.metadata["agentic_v9_reranking"])
    for document in docs
    if isinstance(document.metadata.get("agentic_v9_reranking"), dict)
]
state["retrieval_diagnostics"].append(
    {
        "task_id": task.task_id,
        "status": rows[0]["status"] if rows else "not_instrumented",
        "fallback_reason": rows[0]["fallback_reason"] if rows else None,
        "candidate_count": rows[0]["candidate_count"] if rows else len(docs),
        "selected_count": len(docs),
        "selected": rows,
    }
)
```

Add this list to `trace["agentic_v9"]["retrieval_diagnostics"]`. This explicit
trace field is required because `TaskRetrievalResult` is internal to the core
and is not otherwise materialized in the final `RAGResult`.

- [ ] **Step 8: Run focused and regression tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -p no:cacheprovider `
  tests\test_rag_filtering.py `
  tests\test_agentic_v9_campaign_runtime.py `
  tests\test_agentic_v9_provider_boundary.py `
  tests\test_agentic_v9_full_rollback.py `
  tests\test_rag_modes_agentic.py -q

.venv\Scripts\python.exe -m ruff check --no-cache `
  data_base\rag_filtering.py `
  evaluation\agentic_v9_campaign_runtime.py `
  tests\test_rag_filtering.py `
  tests\test_agentic_v9_campaign_runtime.py `
  tests\test_rag_modes_agentic.py

git diff --check
```

Expected: all focused tests, Ruff, and whitespace checks pass.

- [ ] **Step 9: Commit R1 runtime behavior**

```powershell
git add -- `
  evaluation/agentic_v9_campaign_runtime.py `
  tests/test_agentic_v9_campaign_runtime.py `
  tests/test_rag_modes_agentic.py
git commit -m "feat(evaluation): rerank agentic v9 evidence to four"
```

---

### Task 3: Final verification and deployment handoff

**Files:**
- Verify only; no production file changes expected

**Interfaces:**
- Consumes: Task 1 and Task 2 commits
- Produces: a local branch ready for the user to push and smoke-test

- [ ] **Step 1: Inspect committed scope**

Run:

```powershell
git status --short
git log -3 --oneline
git show --stat --oneline HEAD
git diff HEAD~2..HEAD -- `
  data_base/rag_filtering.py `
  evaluation/agentic_v9_campaign_runtime.py `
  evaluation/rag_modes.py
```

Expected: no Naive preset change and no unrelated tracked change.

- [ ] **Step 2: Repeat completion verification**

Run the Task 2 Step 8 test and Ruff commands again after commits. Do not rely on
pre-commit output.

- [ ] **Step 3: Report smoke criteria**

Tell the user to verify one small deployed Agentic v9 campaign first:

```text
reranker_status = executed or fallback
candidate_count <= 8 per retrieval task
selected_count <= 4 per retrieval task
fallback score = N/A/null, never 0
Failed Runs = 0
Tokens = complete
Phase attribution = complete
```

Do not run the full 16-question evaluation locally because it requires the
user's deployed corpus and provider credentials.
