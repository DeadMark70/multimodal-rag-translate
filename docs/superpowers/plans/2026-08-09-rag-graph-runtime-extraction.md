# RAG Graph Runtime Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the GraphRAG policy, context, evidence, and observability cluster from `data_base/RAG_QA_service.py` without changing the public RAG API or answer behavior.

**Architecture:** `data_base/rag_graph_runtime.py` becomes the owner of Graph-specific types and helpers. `data_base/RAG_QA_service.py` continues orchestrating `rag_answer_question` and re-exports the existing public Graph contract during this batch.

**Tech Stack:** Python 3.13, pytest, Ruff.

## Global Constraints

- Do not change `rag_answer_question` behavior, signature, prompt text, output schemas, feature flags, or error messages.
- Do not introduce dependency containers or new third-party packages.
- Graph runtime must not import `data_base.RAG_QA_service`.
- Keep this batch independent from the later `rag_pipeline.py` extraction.

---

### Task 1: Establish the Graph runtime ownership contract

**Files:**
- Modify: `tests/test_rag_retrieval_generation_split.py`

**Interfaces:**
- Produces: a regression test requiring `data_base.rag_graph_runtime` to own Graph types and `get_graph_evidence_bundle`, with facade identity preserved.

- [x] Add the ownership/re-export test.
- [x] Run the focused test and verify it fails because `data_base.rag_graph_runtime` does not exist.

### Task 2: Extract and verify the Graph runtime

**Files:**
- Create: `data_base/rag_graph_runtime.py`
- Modify: `data_base/RAG_QA_service.py`
- Modify: Graph-focused tests under `tests/`

**Interfaces:**
- Consumes: existing `graph_rag` services, feature flags, schemas, and `data_base.rag_graph_locator.locate_graph_sources`.
- Produces: `GraphContextDetails`, `GraphNeedDecision`, `GraphExecutionStrategy`, `GraphEvidenceLifecycle`, and `get_graph_evidence_bundle` from `data_base.rag_graph_runtime`.

- [x] Move the Graph cluster mechanically without changing its logic.
- [x] Re-export the existing public Graph contract from `RAG_QA_service.py`.
- [x] Retarget private Graph helper imports and patches to the owning module.
- [x] Run the focused Graph tests, Ruff, complexity ratchet, and the complete backend test gate.
- [x] Review the final diff and commit the isolated refactor.
