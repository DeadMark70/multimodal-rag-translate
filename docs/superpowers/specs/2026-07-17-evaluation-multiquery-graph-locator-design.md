# Evaluation Multi-Query and Graph Locator Upgrade Design

- Date: 2026-07-17
- Status: Approved design
- Scope: Evaluation Center backend execution only

## Context

Evaluation Center currently exposes `naive`, `advanced`, `graph`, and `agentic` modes. The active implementation has two baseline problems:

1. `advanced`, `graph`, and several Agentic route profiles set both HyDE and Multi-Query to true. `rag_answer_question()` uses an `if/elif` branch, so HyDE always wins and Multi-Query does not run on those paths.
2. The main `graph` mode and Agentic `generic_graph` route do not supply structured graph feature flags. Defaults resolve them to `graph_evidence_mode="raw_current"` and `GraphExecutionStrategy("raw_legacy")`, which inserts graph-generated text directly into the answer prompt instead of resolving graph evidence back to source chunks.

Agentic also has a hidden HyDE path: CRAG corrective retrieval invokes `transform_query_with_hyde()` whenever the initial evidence is graded irrelevant.

## Goals

- Disable HyDE across Evaluation Center `advanced`, every Graph family mode, and Agentic execution.
- Make Multi-Query the query-expansion policy where those routes currently request expansion.
- Replace Agentic CRAG's HyDE correction with Multi-Query correction.
- Upgrade the main `graph` mode and Agentic graph route to provenance-aware locator-to-chunk retrieval.
- Keep `graph_raw_current` as the explicit legacy graph control.
- Version changed benchmark profiles so historical and new campaign results remain distinguishable.
- Preserve safe fallback to original-query/vector retrieval when query generation or graph resolution fails.

## Non-goals

- Do not change Native/Naive RAG behavior.
- Do not change user-facing Deep Research or ordinary chat defaults.
- Do not make `router_auto_graph` the main Graph benchmark.
- Do not promote claim-gated graph retrieval to the default.
- Do not change RAGAS context packing or its context policy versions.
- Do not change frontend request or response schemas.

## Considered Approaches

### 1. Shared evaluation retrieval policy module — selected

Create a small policy module used by both standard campaign modes and Agentic route construction. It owns query-expansion settings, locator-to-chunk graph hints, and execution-profile naming. This prevents duplicated dictionaries from drifting.

### 2. Edit mode dictionaries in place

This has the smallest initial diff but duplicates the same policy in `rag_modes.py` and `agentic_evaluation_service.py`. A later change could update only one execution surface and silently invalidate comparisons.

### 3. Change global RAG defaults and precedence

Making Multi-Query globally outrank HyDE and enabling structured GraphRAG by default would also change chat and Deep Research. That exceeds the Evaluation Center scope and risks unversioned user-facing behavior changes.

## Architecture

### Evaluation retrieval policy

Add `evaluation/retrieval_profiles.py` with fresh-dictionary factories rather than shared mutable dictionaries:

- Multi-Query expansion: `enable_hyde=False`, `enable_multi_query=True`.
- No expansion: `enable_hyde=False`, `enable_multi_query=False`.
- Locator-to-chunk graph hints:
  - `graph_evidence_mode="locator_to_chunk"`
  - `graph_raw_current_enabled=False`
  - `graph_evidence_locator_enabled=True`
  - `graph_provenance_gate_enabled=True`
  - `graph_to_chunk_enabled=True`
  - `graph_auto_gate_enabled=False`
- Optional Agentic stage/task hints are merged into a new dictionary for each call.
- Execution profile names are generated from mode, retrieval policy version, and `DEFAULT_PRODUCTION_INDEXING_PROFILE`.

The module must not import `rag_modes.py` or `agentic_evaluation_service.py`, avoiding the existing import direction where `rag_modes.py` imports the Agentic service.

### Standard campaign modes

- `naive`: unchanged; no expansion, plain FAISS baseline.
- `advanced`: Multi-Query + hybrid retrieval + reranking; no HyDE and no GraphRAG.
- `graph`: Multi-Query + hybrid retrieval + reranking + explicit locator-to-chunk GraphRAG.
- Every graph ablation mode: HyDE disabled and its existing Multi-Query setting retained.
- `graph_raw_current`: continues to set raw-current graph flags and remains the only intentional raw-legacy comparison mode.
- Other graph ablations retain their evidence, usage-policy, or query-strategy intervention; this change only removes HyDE from their query baseline.

### Agentic route policy

Agentic keeps semantic classification, complexity budgeting, planning, drill-down, fact-state pruning, and synthesis unchanged. Route behavior becomes:

| Route profile | Query expansion | Graph policy |
| --- | --- | --- |
| `hybrid_exact` | none | disabled |
| `hybrid_compare` | Multi-Query | disabled |
| `graph_global` | none | locator-to-chunk when Graph is enabled |
| `visual_verify` | Multi-Query | disabled |
| `generic_graph` | Multi-Query | locator-to-chunk |

The Agentic micro-router remains responsible for deciding whether a subtask needs GraphRAG. Once it chooses a graph route, graph evidence must resolve to source chunks rather than raw graph text.

### CRAG corrective retrieval

Extend `rag_answer_question()` with a backward-compatible parameter:

```python
crag_rewrite_mode: Literal["hyde", "multi_query", "none"] = "hyde"
```

Behavior:

- `hyde`: preserve current behavior for existing non-evaluation callers.
- `multi_query`: generate the original question plus query variants, retrieve each variant, and fuse multiple result lists with RRF.
- `none`: retry with the original question only.

Evaluation Agentic passes `crag_rewrite_mode="multi_query"`. Multi-Query generation failure already returns `[question]`, providing deterministic original-query fallback. Standard Advanced and Graph modes keep CRAG disabled.

## Graph Data Flow

The upgraded main Graph and Agentic graph path is:

1. Hybrid vector/BM25 retrieval obtains initial candidate chunks.
2. Generic Graph Router selects local, global, or blended graph search according to the question and Agentic stage hints.
3. Graph evidence items act as locators rather than final answer text.
4. `ChunkAnchorResolver` resolves eligible graph items to source document chunks.
5. Provenance and document-scope filters reject unresolved or out-of-scope evidence.
6. Located chunks are scored and merged with vector chunks using the existing graph chunk ratio.
7. The answer prompt receives source-backed chunks.

`graph_raw_current` intentionally bypasses steps 3–6 and remains available for legacy comparison.

## Versioning

- Bump Agentic to `agentic_eval_v8_multiquery_locator_<index_profile>`.
- Persist `advanced_eval_v2_multiquery_<index_profile>` for Advanced rows.
- Persist `graph_eval_v2_multiquery_locator_<index_profile>` for the main Graph rows.
- Persist `<ablation_mode>_eval_v2_multiquery_<index_profile>` for Graph ablation rows.
- Naive execution behavior and historical rows are not rewritten.
- Historical rows with no execution profile remain readable as legacy results.
- Keep Agentic `context_policy_version="v4_semantic_router_gate"` and other modes at `v3_answer_aware_pack`; evaluator packing is unchanged.

## Fallback and Failure Semantics

- Multi-Query generation failure returns the original question and continues retrieval.
- A failed individual retrieval follows the existing retrieval error policy.
- Empty or failed graph locator/anchor resolution retains vector/hybrid documents and does not fail the answer.
- Missing Graph index retains vector/hybrid documents.
- Graph observability must identify the requested locator-to-chunk policy and record a skip/fallback reason when no graph-backed chunk reaches final context.
- `graph_raw_current` failures retain their existing best-effort behavior.

## Observability and Persistence

- Campaign results persist the new execution profile.
- Agentic result and agent trace use the same v8 profile.
- Agentic trace metadata continues to record route profile and micro-route.
- Graph events snapshot locator-to-chunk evidence mode and feature flags.
- Graph evidence lifecycle records candidates, resolved anchors, scope-approved items, scored chunks, and packed chunks.
- No raw prompts, credentials, or new sensitive content are added to observability.

## Test Strategy

Implementation follows test-first development.

### Mode policy tests

- Assert Advanced, every Graph family mode, and Agentic have no `enable_hyde=True` path.
- Assert Advanced and main Graph use Multi-Query.
- Assert Naive remains unchanged.
- Assert main Graph has locator, provenance, and graph-to-chunk flags enabled while raw-current and auto-gate are disabled.
- Assert `graph_raw_current` still resolves to raw legacy.

### Agentic route tests

- Assert exact route uses no expansion.
- Assert compare and visual routes use Multi-Query.
- Assert generic graph uses Multi-Query and locator-to-chunk hints.
- Assert graph-global has no query expansion and uses source-backed graph policy when invoked.
- Assert no Agentic route enables HyDE.

### CRAG tests

- Assert Evaluation Agentic passes `crag_rewrite_mode="multi_query"`.
- Exercise a low-relevance CRAG result and assert Multi-Query is called, HyDE is not called, and multiple retrieval batches use RRF.
- Assert Multi-Query failure falls back to the original question.
- Preserve coverage for the default HyDE mode so non-evaluation callers remain backward compatible.

### Campaign/version tests

- Assert Advanced, Graph, Graph ablation, and Agentic results persist the expected execution profiles.
- Assert Agentic result and trace profiles match.
- Assert context policy versions remain unchanged.

### Graph integration tests

- Assert main Graph and Agentic graph calls enter `source_expand`, not `raw_legacy`.
- Assert resolved source chunks merge with vector chunks.
- Assert missing graph, empty anchors, and graph exceptions preserve vector results and emit a fallback reason.

### Verification commands

Run focused tests first, then the production backend regression suite. Report exact pass/fail/skip counts. If the full suite hits a documented environment-only collection failure, preserve the focused evidence and report the blocker without weakening assertions.

## Documentation Updates

- Update `docs/BACKEND.md` with the new query and graph policies.
- Update `docs/generated/api-surface.md` with the v8 Agentic and v2 standard profiles.
- Add `docs/design-docs/agentic-semantic-router-v8.md` as the retrieval-policy successor and link it from `docs/design-docs/index.md`; keep the v7 document unchanged as historical design context.
- Update the relevant execution-plan index when implementation is completed.
- No frontend inventory change is required because routes, payloads, and UI controls do not change.

## Acceptance Criteria

- No Evaluation Advanced, Graph family, or Agentic execution path invokes HyDE, including CRAG correction.
- Advanced and main Graph actually invoke Multi-Query and RRF when multiple queries are generated.
- Main Graph and Agentic graph routes use locator-to-chunk source evidence.
- Only the explicit `graph_raw_current` control intentionally uses raw legacy graph evidence.
- Safe original-query/vector fallbacks remain functional.
- Changed modes persist versioned execution profiles.
- Focused tests and production backend regression checks pass, or any environment-only full-suite blocker is reported with exact evidence.
