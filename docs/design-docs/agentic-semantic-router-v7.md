# Agentic Benchmark Semantic Router v7

## Goal
Upgrade only the evaluation `agentic` benchmark path from static keyword routing to semantic adaptive routing, while preserving backward-compatible APIs and fallback safety.

## Scope
- In scope: `evaluation/agentic_evaluation_service.py`, `agents/planner.py`, additive `mode_hints.retrieval_policy` support in `data_base/RAG_QA_service.py`, trace schema extensions.
- Out of scope: user-facing Deep Research/chat routing behavior changes.

## Core Decisions
- Baseline profile is versioned to `agentic_eval_v7_semantic_router_<index_profile>`.
- Main question classification uses a planner-purpose LLM with strict timeout (`0.5s`) and parse/validation fallback to heuristic classification.
- Rollout is controlled by `AGENTIC_SEMANTIC_ROUTER_MODE`:
  - `off`: keep heuristic routing behavior.
  - `shadow`: compute semantic decisions and emit trace telemetry, but keep execution behavior unchanged.
  - `active`: apply semantic classifier and micro-routing decisions.
- Complexity mapping is fixed:
  - `1 -> tier1, subtask=1, iterations=0`
  - `2 -> tier1, subtask=1, iterations=1`
  - `3 -> tier2, subtask=2, iterations=1`
  - `4 -> tier3, subtask=3, iterations=1`
  - `5 -> tier3, subtask=4, iterations=2`

## Routing Design
- Added micro-route layer before existing route profiles:
  - `direct_point_access` -> `hybrid_exact`
  - `broad_context_rag` -> `hybrid_compare` or graph-capable profile when relation-heavy
  - `visual_evidence_path` -> `visual_verify`
- Each micro-route emits additive retrieval policy hints via internal `mode_hints.retrieval_policy`:
  - direct: `retrieval_k=6`, `target_k=4`
  - broad: `target_k=8`
  - visual: `target_k=8`

## Iteration Control
- Retrieval quality gate now includes semantic completeness signals:
  - coverage gap status
  - claim support ratio
  - context relevance ratio
  - aggregate `semantic_gate_score`
- After first exploration round, strategy can shift:
  - `downshift` when evidence is strong and gaps are closed.
  - `upshift` when gaps remain with weak semantic support.
- Follow-up generation includes reverse-pruning:
  - prune tasks with high overlap against `fact_state` and no gap-targeting value.
  - optional fast LLM duplicate check for high-overlap candidates.

## Trace Additions
Additive trace fields:
- `classifier_decision`
- `complexity_score`
- `tier_shift`
- `pruned_followups`
- `semantic_gate_score`
- per-step metadata includes `micro_route`.

## Backward Compatibility
- No external HTTP contract changes.
- `mode_hints.retrieval_policy` is additive and optional.
- On classifier timeout/parse/model errors, routing falls back without interrupting benchmark execution.
