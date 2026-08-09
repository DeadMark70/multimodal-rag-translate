# RAG Pipeline Extraction Design

## Status

Approved for implementation planning on 2026-08-09.

## Context

`data_base/RAG_QA_service.py` is the compatibility entry point for the legacy RAG answer path. The earlier Graph runtime extraction moved Graph-specific policy, evidence, and observability into `data_base/rag_graph_runtime.py`, but `rag_answer_question()` still owns retrieval, filtering, CRAG, Graph routing, context expansion, generation, and legacy result projection.

The facade is currently about 849 lines. `rag_answer_question()` spans about 413 lines and has Ruff C901 complexity 30. Moving that body unchanged would reduce the facade size without improving the flow's maintainability.

Existing stage modules already own most domain logic:

- `data_base/rag_retrieval.py`: query expansion and hybrid retrieval.
- `data_base/rag_filtering.py`: document filtering and reranking.
- `data_base/rag_crag.py`: corrective retrieval.
- `data_base/rag_graph_runtime.py`: Graph policy, evidence, and observability.
- `data_base/rag_generation.py`: prompt construction, LLM generation, and visual verification.

The missing boundary is a small orchestration module that composes those stages without reimplementing them.

## Goals

- Make `data_base/RAG_QA_service.py` a compatibility facade of roughly 120–200 lines.
- Move answer orchestration into `data_base/rag_pipeline.py`.
- Reduce `run_rag_pipeline()` and each of its four main stage functions to Ruff C901 complexity 10 or lower.
- Preserve the exact public signature, defaults, return shapes, error messages, progress events, prompts, feature flags, and fallbacks of `rag_answer_question()`.
- Keep tests coupled to the module that performs each dependency lookup instead of maintaining facade-only test seams.
- Keep the change mechanical and independently reviewable before any `CampaignEngine` refactor.

## Non-goals

- Change retrieval counts, ranking behavior, CRAG policy, Graph routing, prompts, or generated answer quality.
- Replace the functional flow with a runner class, state machine, dependency-injection container, or shared pipeline framework.
- Merge runtime parent-chunk expansion with `data_base/context_enricher.py`. The former expands retrieved chunks from `ParentDocumentStore`; the latter adds LLM-generated prefixes during indexing and has a different lifecycle.
- Rewrite `_expand_short_chunks` behavior or remove its existing complexity-baseline entry.
- Start the `CampaignEngine` decomposition.
- Add third-party dependencies.

## Considered Approaches

### 1. Mechanical move

Move `rag_answer_question()` unchanged into `rag_pipeline.py` and delegate from the facade.

This is the smallest patch, but it merely moves C901 30 to another file. It does not make stages independently readable or testable, so it is rejected.

### 2. Functional stage pipeline

Create one orchestration function and four private stage functions. Use small private outcome value objects for explicit stage results and terminal legacy responses.

This preserves the existing procedural design while exposing clear boundaries. It reduces complexity without introducing a framework and is the selected approach.

### 3. Runner class with injected dependencies

Create a `RagPipelineRunner` and inject retrievers, rerankers, Graph services, generators, and callbacks.

This would provide strong isolation but would add construction and dependency plumbing throughout the application. The current code does not need multiple pipeline implementations, so this approach is rejected as unnecessary complexity.

## Module Ownership

### `data_base/RAG_QA_service.py`

The facade retains:

- `rag_answer_question()` with its exact current signature and defaults.
- `initialize_llm_service()` and its startup behavior.
- Re-exports for `RAGResult`, `ProgressCallback`, `get_graph_evidence_bundle`, `GraphContextDetails`, `GraphNeedDecision`, `GraphExecutionStrategy`, and `GraphEvidenceLifecycle`.
- Only small compatibility aliases that are deliberately retained; it does not retain orchestration implementations solely for tests.

The facade imports the pipeline module and performs module-lookup delegation:

```python
from data_base import rag_pipeline


async def rag_answer_question(...):
    return await rag_pipeline.run_rag_pipeline(...)
```

Using a module lookup ensures a test can patch `data_base.rag_pipeline.run_rag_pipeline`. The facade signature is written explicitly and protected by a signature-drift test.

### `data_base/rag_pipeline.py`

The module owns:

- `run_rag_pipeline()`.
- Retrieval policy, history formatting, intent constraints, and runtime parent-chunk expansion used only by this legacy orchestration path.
- Four private stage functions:
  - `_run_retrieval_stage()`
  - `_run_crag_stage()`
  - `_run_graph_stage()`
  - `_run_generation_stage()`
- Small private outcome types used to transport stage results.
- Construction of legacy terminal responses without changing their messages or shapes.

The pipeline calls existing stage modules; it does not copy their domain implementations.

### `data_base/rag_pipeline_schemas.py`

This module becomes the owner of `RAGResult` and `ProgressCallback` in addition to the existing `RagRetrievalResult` and `GeneratedRagAnswer` schemas. The definitions move without field, order, default, or identity changes. `RAG_QA_service.py` re-exports them so existing consumers keep importing from the facade.

### Existing stage modules

The existing retrieval, filtering, CRAG, Graph, and generation modules retain their current responsibilities. Tests for private helpers import or patch those owners directly. Pipeline integration tests patch names looked up by `rag_pipeline.py`.

`rag_graph_runtime.py` must not import `rag_pipeline.py` or `RAG_QA_service.py`. `rag_pipeline.py` must not import `RAG_QA_service.py`.

## Stage Contracts

The implementation uses three private outcome dataclasses rather than positional tuples or shared mutable state.

### `RetrievalStageOutcome`

- `documents`: selected documents, if retrieval can continue.
- `retriever`: retriever needed by optional CRAG.
- `reranker_available`: runtime reranker state.
- `target_k`: resolved generation-context target.
- `terminal_result`: existing legacy tuple or `RAGResult` when the pipeline must stop.

This stage owns retriever creation, hybrid retrieval, document filtering, reranking, retrieval progress events, and retrieval-related terminal responses.

### `CragStageOutcome`

- `documents`: original or corrected documents.
- `terminal_result`: existing insufficient-retrieval response when CRAG intentionally stops the pipeline.

When CRAG raises unexpectedly, this stage preserves current behavior: log a warning and continue with the original documents.

### `GraphStageOutcome`

- `documents`: documents after optional Graph source expansion.
- `graph_context`: legacy raw Graph prompt context when applicable.
- `graph_evidence_documents`: Graph evidence projected for evaluation returns.

This stage preserves `skip`, `source_expand`, and `raw_legacy` behavior. It delegates Graph policy, evidence, and observability to `rag_graph_runtime.py` and source location to `rag_graph_locator.py`.

## Data Flow

```text
RAG_QA_service.rag_answer_question
    -> rag_pipeline.run_rag_pipeline
        -> resolve LLM
        -> _run_retrieval_stage
        -> _run_crag_stage
        -> _run_graph_stage
        -> _run_generation_stage
        -> legacy tuple or RAGResult
```

`run_rag_pipeline()` only coordinates stage results:

1. Resolve the LLM using the existing provider and preserve the current initialization failure response.
2. Run retrieval. Return immediately when `terminal_result` is present.
3. Run CRAG. Return immediately when `terminal_result` is present.
4. Run Graph enhancement and carry its documents and context forward.
5. Generate the answer and project it into the existing public return type.

The orchestration function contains no Graph strategy implementation, prompt implementation, reranking algorithm, or CRAG rewrite policy.

## Error and Fallback Compatibility

The extraction preserves existing exception boundaries:

- LLM acquisition catches `RuntimeError`, `KeyError`, and `ValueError` and returns the current model-not-initialized response.
- Retrieval catches `RuntimeError` and `ValueError` and returns the current retrieval-error response.
- Empty knowledge bases, empty retrieval, and empty requested-document results keep their current messages and source-ID behavior.
- CRAG catches its current broad exception boundary, logs a warning, and continues with the original retrieval.
- Graph does not gain a new broad catch. Existing Graph runtime and locator behavior remains authoritative.
- Generation continues to own provider failures and visual-verification fallback behavior.

A small internal helper may centralize construction of terminal legacy projections, but each call site supplies its explicit message and source IDs. It must not normalize distinct failure cases into one response.

## Compatibility Rules

- Existing application imports from `data_base.RAG_QA_service` remain valid.
- `RAGResult` remains the same named-tuple contract, including field order and defaults.
- `rag_answer_question()` keeps the same parameter order, names, annotations, keyword defaults, and return annotation.
- `initialize_llm_service()` remains in the facade because it is a startup compatibility entry rather than pipeline execution.
- Existing Graph public re-export identity remains unchanged.
- Private facade helpers are not compatibility guarantees. Tests using them move to the owning module unless a small alias is intentionally retained for a known runtime consumer.
- No caller is migrated to import `run_rag_pipeline()` directly in this batch.

## Complexity Budget

- `RAG_QA_service.rag_answer_question`: delegation only; no C901 finding.
- `rag_pipeline.run_rag_pipeline`: C901 10 or lower.
- `_run_retrieval_stage`: C901 10 or lower.
- `_run_crag_stage`: C901 10 or lower.
- `_run_graph_stage`: C901 10 or lower.
- `_run_generation_stage`: C901 10 or lower.
- `_expand_short_chunks`: behavior remains unchanged; its complexity-baseline entry moves to its new owning path if still required.
- The repository complexity ratchet must not increase in total score or finding count.

## Testing Strategy

### Boundary tests

- Require `data_base.rag_pipeline` to exist and own `run_rag_pipeline()`.
- Verify the facade calls the pipeline through module lookup.
- Compare facade and pipeline signatures to detect drift.
- Verify `RAGResult` and `ProgressCallback` facade re-export identity.
- Preserve the existing Graph runtime re-export identity checks.

### Stage and behavior tests

- Retarget existing retrieval, reranking, CRAG, Graph, context packing, visual verification, and generation patches to the module that performs the lookup.
- Preserve tests for all existing terminal messages and return shapes.
- Preserve Graph strategy coverage for `skip`, `source_expand`, and `raw_legacy`.
- Preserve CRAG insufficient and exception-fallback coverage.
- Preserve advanced-mode parent expansion and visual-verification behavior.
- Add only missing stage-contract tests; do not duplicate behavior already covered by existing integration tests.

### Verification gates

- Focused RAG, CRAG, Graph, reranking, context-packing, and visual-verification tests.
- Ruff import and unused-variable selectors.
- Ruff C901 and the repository complexity ratchet.
- Complete backend pytest suite with the warning budget.
- OpenAPI, generated documentation, and Markdown drift checks.
- `git diff --check`.

## Delivery Boundary

The implementation is one isolated refactor commit after its tests and verification gates pass. It does not include behavior changes or the subsequent `CampaignEngine` extraction.
