# Evaluation Setup-authoritative LLM runtime design

## Context

Evaluation Setup currently normalizes the selected model's thinking controls for
the outer evaluation request, but GraphRAG adds its own runtime override inside
that request. The nested override can select a Graph-specific thinking family
from a different model than the Setup-selected model. With thinking disabled in
Setup, GraphRAG can still add `thinking_level` or `thinking_budget`; this caused
the latest campaign's Gemini 2.5 Graph calls to fail with:

`400 INVALID_ARGUMENT: Thinking level is not supported for this model.`

The fix must make the Setup snapshot the only authority for model and thinking
configuration across naive, advanced, graph, and agentic execution.

## Design

### 1. Preserve Setup intent in the runtime context

`normalize_model_config_for_runtime` will emit an explicit thinking-enabled
state in addition to model parameters. A disabled Setup must remain observable
inside nested GraphRAG calls; absence of a budget or level must not be
interpreted as permission to apply a Graph default.

The effective model name and thinking-enabled state are request-scoped and must
not use the process-global Graph default when an Evaluation Setup model exists.

### 2. Make GraphRAG derive a compatible parameter set

`graph_rag_llm_runtime_override` will read the current request-scoped Setup
context and derive the final provider parameters from that model:

- thinking disabled: clear both `thinking_budget` and `thinking_level`, and do
  not send either parameter to the provider;
- thinking enabled + Gemini 2.5 family: retain only the Setup budget;
- thinking enabled + Gemini 3 family: retain only the Setup level;
- unsupported model family: retain no thinking control unless the provider
  capability explicitly allows one.

Nested overrides must clear conflicting keys rather than merge a stale level
with a budget. The actual model passed to the LLM constructor must equal the
Setup-selected model.

Graph-specific extraction/community defaults remain available only when a
non-evaluation GraphRAG operation has no Setup runtime context. They must never
override an active evaluation request.

### 3. Regression coverage

Add unit tests covering the final constructor arguments for:

- Gemini 2.5 with thinking disabled;
- Gemini 2.5 with an enabled budget;
- Gemini 3 with thinking disabled;
- Gemini 3 with an enabled level;
- an outer Setup model that differs from the Graph default;
- nested overrides clearing incompatible/stale thinking keys.

Retain the existing GraphRAG default tests for non-evaluation usage. Add one
evaluation-context regression that proves a disabled Setup does not emit a
thinking parameter from `graph_rag_llm_runtime_override`.

## Error handling and compatibility

The provider must never receive both thinking families for one call. Existing
Graph fallback behavior remains unchanged for genuine provider failures, but a
configuration mismatch should be prevented before the provider call. Existing
campaign data is not rewritten; the corrected accounting appears on newly
executed campaigns.

## Verification

Run the focused factory, model-capability, and GraphRAG tests first, then the
backend evaluation/accounting regression suite. A fresh campaign using the
Setup shown in the report must produce no `INVALID_ARGUMENT` Graph usage
events, and `graph_reasoning` token accounting must be measured whenever the
provider calls succeed.
