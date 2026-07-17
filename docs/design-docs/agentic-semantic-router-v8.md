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
