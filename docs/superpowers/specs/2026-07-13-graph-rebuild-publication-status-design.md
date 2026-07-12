# Graph Rebuild Publication Status Design

## Goal

Remove the ambiguous completed-state warning on the GraphRAG page while preserving the safety warning before a staged graph is published.

## Cause

`GraphRebuildStatusResponse.live_graph_unchanged` defaults to `true`. The status projection does not override it, so a completed and published rebuild is incorrectly reported as still using the old graph.

## Design

The backend status projection will explicitly set `live_graph_unchanged` from the durable manifest: it is `false` only when the manifest state is `completed`, which is set only after the staging graph has been published.

The frontend will render mutually exclusive publication notices:

- When `live_graph_unchanged` is true, retain the existing informational warning that queries use the old graph.
- When the rebuild is `completed` and `live_graph_unchanged` is false, show a success notice that queries now use the new graph.

The persisted progress card remains visible after completion as job history; no rebuild lifecycle or graph data is changed.

## Verification

- Backend test: completed manifest projects `live_graph_unchanged=false`; an in-progress manifest remains `true`.
- Frontend test: completed status renders the new-graph notice and does not render the old-graph warning; running status continues to show the warning.
