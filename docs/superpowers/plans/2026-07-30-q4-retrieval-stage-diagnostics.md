# Q4 Retrieval Stage Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist enough v9 retrieval-stage metadata to tell whether an expected source was absent from retrieval candidates or excluded by reranking, without altering runtime selection.

**Architecture:** The generic retrieval boundary enriches its existing `reranking` metadata with a document-ID-only description of the optional soft candidate-diversity tail. The v9 runtime copies this existing metadata into selected-document annotations and includes it in its retrieval diagnostic projection, which is already persisted for evaluation analysis.

**Tech Stack:** Python 3.11, LangChain `Document`, pytest, existing evaluation export/trace contracts.

## Global Constraints

- Do not read or use `expected_sources` at runtime.
- Do not add LLM calls, embeddings, reranker calls, document text, or a hard diversity gate.
- Do not modify Native RAG behavior or existing final context selection.
- Store only stable document IDs, counts, booleans, and the existing diversity policy identifier.
- Stage and commit only files named in this plan; preserve all existing untracked files.

---

### Task 1: Persist soft-diversity diagnostics through the v9 retrieval trace

**Files:**
- Modify: `data_base/rag_filtering.py:72-115`
- Modify: `evaluation/agentic_v9_campaign_runtime.py:709-783`
- Modify: `tests/test_rag_filtering.py:161-192`
- Modify: `tests/test_agentic_v9_campaign_runtime.py:393-467`

**Interfaces:**
- Consumes: `filter_and_rerank_retrieval(..., diversify_rerank_candidates=True)` and its existing `metadata["reranking"]` mapping.
- Produces: `metadata["reranking"]["candidate_diversification"]` with `policy`, `applied`, `retrieved_doc_ids`, `candidate_doc_ids`, `represented_doc_ids_before_tail`, and `admitted_doc_ids`.
- Produces: `_retrieval_diagnostic_projection(task_id, documents)["candidate_diversification"]` as the same sanitized mapping.

- [ ] **Step 1: Write the failing retrieval-boundary test**

```python
assert result.metadata["reranking"]["candidate_diversification"] == {
    "policy": "tail_source_diversity_r1",
    "applied": True,
    "retrieved_doc_ids": ["doc-primary", "doc-alternate"],
    "candidate_doc_ids": ["doc-primary", "doc-alternate"],
    "represented_doc_ids_before_tail": ["doc-primary"],
    "admitted_doc_ids": ["doc-alternate"],
}
```

Use the existing multi-document candidate test fixture with an alternate
document that enters only through the reserved diversity tail.

- [ ] **Step 2: Run the retrieval test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_rag_filtering.py::test_candidate_diversification_reserves_tail_candidates_for_other_documents -q`

Expected: FAIL because the metadata currently contains only `policy` and `applied`.

- [ ] **Step 3: Write the failing runtime propagation test**

```python
assert diagnostics["candidate_diversification"] == {
    "policy": "tail_source_diversity_r1",
    "applied": True,
    "retrieved_doc_ids": ["doc-primary", "doc-alternate"],
    "candidate_doc_ids": ["doc-primary", "doc-alternate"],
    "represented_doc_ids_before_tail": ["doc-primary"],
    "admitted_doc_ids": ["doc-alternate"],
}
```

Construct a selected `Document` with `agentic_v9_reranking` metadata so the
test checks the public diagnostic projection, not a mock.

- [ ] **Step 4: Run the runtime test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_agentic_v9_campaign_runtime.py::test_retrieval_diagnostic_projection_retains_candidate_diversification -q`

Expected: FAIL because `_retrieval_diagnostic_projection` currently omits the field.

- [ ] **Step 5: Implement the minimal metadata propagation**

In `rag_filtering.py`, derive ordered unique document IDs from the fixed
prefix and the tail candidates admitted by `_limit_rerank_candidates`; omit
missing IDs. Keep the current selection list byte-for-byte unchanged.

In `agentic_v9_campaign_runtime.py`, copy the sanitized mapping from the
generic reranking metadata into `agentic_v9_reranking`, then return it from
`_retrieval_diagnostic_projection`. Do not put this metadata in final prompt
contexts or use it to alter scores.

- [ ] **Step 6: Run focused tests**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_rag_filtering.py tests/test_agentic_v9_campaign_runtime.py -q`

Expected: PASS, including the new retrieval and runtime-propagation tests.

- [ ] **Step 7: Run scoped lint**

Run: `./.venv/Scripts/python.exe -m ruff check data_base/rag_filtering.py evaluation/agentic_v9_campaign_runtime.py tests/test_rag_filtering.py tests/test_agentic_v9_campaign_runtime.py`

Expected: PASS.

- [ ] **Step 8: Commit the implementation**

```powershell
git add -- data_base/rag_filtering.py evaluation/agentic_v9_campaign_runtime.py tests/test_rag_filtering.py tests/test_agentic_v9_campaign_runtime.py
git commit -m "feat(agentic-v9): trace candidate diversity decisions"
```
