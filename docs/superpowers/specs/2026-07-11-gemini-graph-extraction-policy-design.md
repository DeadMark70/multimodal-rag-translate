# Gemini Graph Extraction Policy Design

## Goal

Replace the current implicit GraphRAG Gemini thinking configuration with an explicit, auditable extraction policy. A full rebuild uses Gemini 3.1 Flash-Lite at medium thinking for document extraction and low thinking for community summaries. Operators can selectively retry one document at high thinking after inspecting graph quality.

## Scope

This change covers the existing Google Gemini provider only. It does not add an OpenRouter adapter, automatic quality-based escalation, changes to the graph schema-v1 vocabulary, or changes to query-time GraphRAG routing.

## Policy Contract

| Operation | LLM purpose | Provider | Model | Thinking level |
| --- | --- | --- | --- | --- |
| Full graph rebuild | `graph_extraction` | Google Gemini | `gemini-3.1-flash-lite` | `medium` |
| Normal one-document retry | `graph_extraction` | Google Gemini | `gemini-3.1-flash-lite` | `medium` |
| High-precision one-document retry | `graph_extraction` | Google Gemini | `gemini-3.1-flash-lite` | `high` |
| Community regeneration after any rebuild or retry | `community_summary` | Google Gemini | `gemini-3.1-flash-lite` | `low` |

The valid extraction profiles are `standard` and `high_precision`. `standard` is the default for every entry point. `high_precision` is available only on the single-document retry API. A full rebuild must never silently select `high_precision`.

The legacy Gemini 2.5 behavior remains compatible: if a caller explicitly supplies a non-Gemini-3 model, the factory continues to use the existing purpose-specific `thinking_budget` values rather than sending `thinking_level`.

## Architecture

### LLM policy boundary

`core/llm_factory.py` owns a typed GraphRAG purpose profile instead of inferring `high` from any model name beginning with `gemini-3`. The profile resolves the model and the thinking level for `graph_extraction` and `community_summary`.

`graph_rag_llm_runtime_override(...)` gains an optional extraction profile argument. It applies `medium` for a standard extraction and `high` only when the caller explicitly requests `high_precision`. The community purpose always resolves to low. The existing task-local `ContextVar` remains the concurrency boundary, so one document retry cannot alter another background job.

### Extraction and maintenance flow

`run_graph_extraction(...)`, `extract_and_add_to_graph(...)`, and the extraction calls beneath them accept an `ExtractionProfile` value. The full rebuild task passes `standard`. The retry task passes the profile submitted to the API.

The retry operation retains its current transactional behavior: it builds against a temporary graph, removes and re-extracts only the requested document, rebuilds communities, and promotes the temporary snapshot only after success. A failed high-precision retry keeps the live graph unchanged.

Community creation keeps its current invocation path but resolves its independent low-thinking policy. It does not inherit high thinking from the document retry that triggered it.

### Extraction manifests

The existing `GraphExtractionRunManifest` becomes live persisted data. It records one completed extraction run with these fields:

```text
extraction_run_id
graph_extraction_version
extractor_provider = "google"
extractor_model
thinking_level
extraction_profile
prompt_version
schema_version
doc_id
chunk_hashes
temperature
validated
created_at
graph_snapshot_version
```

The `graph_snapshot_version` is assigned when the run is saved into the promoted graph state. A manifest only represents a completed extraction whose results are present in that graph state; a failed temporary retry does not append a live record.

`GraphStore` stores manifests in a new versioned sidecar named `graph.extraction_runs.json`. It loads legacy graphs without that file as an empty history. Snapshot save, temporary-copy maintenance, atomic promotion, and snapshot restoration must copy/hash this sidecar together with other graph sidecars.

The document-status API exposes the most recent successful extraction profile, model, thinking level, prompt version, and timestamp for each document. The graph status response additionally exposes the active default extraction and community policies, so the UI can distinguish a legacy graph from one rebuilt under this policy.

## API Contract

The retry route becomes:

```http
POST /graph/documents/{doc_id}/retry
Content-Type: application/json

{"extraction_profile":"standard"}
```

`extraction_profile` is optional and defaults to `standard`; the only accepted values are `standard` and `high_precision`. Invalid values return the API's normal 422 validation response. A started response includes the requested profile so the client can display the operation it started.

Existing callers that send no JSON body remain valid and receive standard extraction.

## UI Contract

The GraphRAG document list retains its normal retry action and adds an explicit high-precision retry action. Both actions use the same existing progress/status polling. The high-precision action states that it re-extracts only the selected document, refreshes communities, and preserves the current graph if the job fails; it requires confirmation before issuing the request.

Document rows display the latest extraction policy metadata when available. Older graphs show an explicit legacy/unrecorded state rather than fabricated values. The page-level graph status displays the configured default extraction and community policies.

## Error Handling and Compatibility

- The retry endpoint validates the profile before scheduling work.
- The runtime policy validates that a profile is only used with `graph_extraction`; community summaries reject or ignore extraction-profile input at the factory boundary rather than inheriting it.
- Missing manifest sidecars, malformed legacy manifest entries, or unavailable metadata must not block graph reads. Invalid entries are logged and skipped.
- A failure to persist a manifest is a graph-write failure: it prevents promotion of the temporary graph, preserving the current snapshot and the correspondence between graph evidence and its provenance.
- Legacy snapshots remain readable. They have no run history and report metadata as unavailable.

## Tests and Acceptance Criteria

1. The factory resolves Gemini 3.1 Flash-Lite with medium thinking for standard graph extraction and low thinking for community summaries.
2. A high-precision profile resolves high thinking only for graph extraction, and a full rebuild passes standard extraction explicitly.
3. Explicit Gemini 2.5 model overrides retain the existing `thinking_budget` behavior.
4. Normal and high single-document retries pass the selected profile through API, maintenance, service, and extractor layers without affecting other jobs.
5. Each successful extraction produces a manifest containing provider, model, thinking level, prompt version, schema version, document id, hashes, validation result, and snapshot version.
6. Manifest data survives GraphStore reload, temporary-copy retry, atomic snapshot promotion, and snapshot restoration. Legacy stores without the sidecar remain readable.
7. A failed high-precision retry leaves the live graph and its manifest history unchanged.
8. The frontend sends the selected profile, shows the high-precision confirmation, renders metadata when present, and continues to work against a legacy backend response.
9. Backend targeted pytest and ruff checks, plus frontend typecheck/lint/unit tests, pass before each task commit and at final verification.
