# Evaluation Center Observability and Export v2 Design

> Status: approved on 2026-08-12. This design is based on the current backend
> and frontend code contracts. The local SQLite data is explicitly not treated
> as evidence of what exists in the server environment.

## Goal

Make every Evaluation Center value traceable to a real backend projection,
remove or clearly label values that are not instrumented, give all selected-run
panels one canonical observability contract, and replace the current export
payload with a clean Schema v2 that can optionally contain every run's complete
sanitized observability.

The result must remain easy to reason about:

1. one authoritative projection per concept;
2. no duplicated analytics formulas in the frontend or export path;
3. no unknown value represented as zero;
4. no placeholder presented as monitored data; and
5. one focused commit per implementation task, with a real-environment stop
   after each wave.

## Evidence Basis

The repair is driven by static code and contract inspection only:

- frontend API calls, types, mappers, and mounted component trees;
- backend routes, response schemas, analytics services, repositories, and
  observability write paths;
- unit, integration, contract, redaction, and repository tests; and
- generated API/UI documentation where it can be checked against code.

Local `.db` content, local campaign counts, and local historical rows are not
used to claim whether production data is present or absent. Production data
availability is verified only at the checkpoint that follows each wave.

## Scope

This design covers the active Evaluation Center and the APIs it calls:

- Campaign Overview;
- Question Analysis;
- Run Trace;
- Retrieval Evidence;
- Agent Behavior;
- Claim Evidence;
- Router Lab;
- Ablation, human evaluation, errors, stage warnings, and export;
- selected-campaign durable job visibility; and
- JSON export Schema v2.

It also covers observability recording changes that are required to distinguish
measured, persisted, derived, heuristic, unavailable, and not-instrumented
values on those surfaces.

## Non-goals

- Rewriting analytics endpoints that the active Evaluation Center does not use.
- Adding monetary pricing or cost UI to a token-focused product surface.
- Inventing historical telemetry that was never recorded.
- Exposing raw provider responses, secrets, stack traces, or unrestricted
  persistence payloads.
- Replacing the current persistence technology or adding new observability
  tables. The design uses existing normalized rows and JSON-backed metadata.
- Adding streaming export in this repair. Full observability is opt-in and the
  initial v2 response remains a normal JSON download.
- Preserving the current export response shape. There are no known external
  export consumers, so Schema v2 replaces it directly.

## Design Principles

### Canonical projections

The backend owns metric formulas, comparability rules, availability, and
provenance. The frontend formats those projections and may calculate only
presentation-level values such as row totals when every required input is
known.

The export service calls the same backend projections as the panels. It must
not contain a second implementation of RAGAS selection, token completeness,
router utility, evidence coverage, or ablation formulas.

### Null is not zero

A nullable metric remains `null` until the backend has evidence that its value
is zero. The client renders an unavailable state instead of coercing missing or
non-finite values to `0`.

### Availability is explicit but lightweight

The system does not wrap every scalar in a generic metric object. Existing
typed nullable fields remain intact. A section or observability entity carries
small metadata where the distinction matters:

```text
availability.status = complete | partial | not_instrumented |
                      not_available | not_applicable
availability.reasons = string[]
provenance = measured | persisted | derived | heuristic
```

`provenance` describes how the value was produced. `availability` describes
whether the projection is usable. A heuristic value may be available, but it
must never be labelled measured.

### Safe typed fields instead of raw payload dependence

Interactive UI contracts expose explicit, allow-listed fields. A panel must
not recover business data from an arbitrary `payload` object, especially when
the route intentionally redacts that object.

## Canonical Run Observability

### Single active contract

The campaign-scoped endpoint remains the authoritative selected-run API:

```text
GET /api/evaluation/campaigns/{campaign_id}/runs/{run_id}/observability
```

`EvaluationRunObservabilityDetail` is extended to include the versioned
`agentic_v9` envelope. It already contains normalized graph events, graph
evidence, accounting diagnostics, run summary, and evidence coverage that the
older `/runs/{run_id}/detail` response does not model completely.

The existing `/runs/{run_id}/detail` route may remain for existing internal
callers, but no new Evaluation Center or export behavior will depend on it.
This repair does not create another run-detail endpoint.

### Service boundary

The route's current assembly logic moves into one backend service projection.
The logical flow is:

```text
normalized persistence and campaign result
    -> canonical observability assembly
    -> policy-based safe projection
       -> interactive selected-run response
       -> export v2 run observability
```

The canonical assembly owns field selection and status calculation. The safe
projection owns redaction. This keeps one data model while still allowing the
export request's explicit prompt/trace options.

### Required fields

The canonical run projection contains:

- `run_summary`;
- `accounting_diagnostics`;
- `trace_events`;
- `llm_calls`;
- `retrieval_events` and `retrieval_chunks`;
- `context_packs`;
- `tool_calls`;
- `routing_decisions`;
- `graph_events`, `graph_evidence_items`, and graph observability status;
- `claims`;
- `human_ratings`;
- `evidence_coverage` and evidence coverage status; and
- `agentic_v9` when the persisted execution version supports it.

The service must populate `human_ratings`; a declared field that is never
assembled is not considered implemented.

### Redaction boundary

The interactive response always uses the safe policy:

- trace and LLM arbitrary payloads remain redacted;
- prompt content is limited to the existing safe preview;
- claim evidence is represented by typed evidence references;
- repair diagnostics use typed `repair_action` and `post_repair_status` fields;
- raw provider error bodies and stack traces are excluded; and
- graph/retrieval locators expose only normalized document, chunk, page, and
  asset identifiers already allowed by the evaluation contract.

Export may include explicitly requested raw trace or full prompt material only
through the existing authorized options and the export redaction policy. The
interactive endpoint never accepts those options.

## Panel Corrections

### Run Trace and v9 panels

The frontend `RunDetailResponse` type and API client are aligned to the
canonical observability schema. The v9 trace, evidence explorer, atomic slot
alignment, claim repair, context pack, and execution route all read the same
`agentic_v9` object.

Selected-run requests are protected against stale responses: switching
campaign or run cannot allow a slower previous request to replace the current
detail.

### Claim Evidence

Claim rows receive safe typed evidence references and repair fields. The
frontend no longer reads claim evidence or repair state from a payload that the
backend clears.

An empty claim list is not automatically evidence of zero unsupported claims.
The claim projection carries availability/provenance so `no claims extracted`
can be distinguished from `claim extraction ran and found zero claims`.

Per-slot graph data is shown only when a real per-slot contract exists. Until
then, the table omits that synthetic column and shows one capability notice for
the section.

### Retrieval Evidence

Normalized retrieval rows expose explicit measurement/provenance state.
Result-context reconstruction is labelled `derived`; text or expected-document
matching is labelled `heuristic`. Dense score, BM25 score, page, and modality
remain null when their write path did not record them.

`used_in_context` and `used_in_answer` are never promoted to measured values
unless the recorder supplies measured attribution. The frontend no longer uses
redacted payload depth as the authority for whether a top-level field is real.

Graph evidence mappers consume the backend's normalized fields, including
`source_doc_ids`, `source_chunk_ids`, `pages`, and `asset_ids`, instead of
looking for unrelated singular aliases.

### Router Lab

Campaign router analysis and selected-run execution route are distinct:

- campaign router analysis remains explicitly `retrospective` unless a future
  benchmark contract proves otherwise; and
- selected-run execution route comes from that run's v9 observability.

Entering Router Lab directly loads the run list and the current selected run's
observability. It does not depend on the user visiting Run Trace first, and a
campaign switch clears the previous execution route.

Saved tokens, quality loss/gain, regret, oracle label, and confusion matrix are
rendered only when the backend response contains authoritative values. The
frontend does not manufacture a formula, oracle source, or empty matrix.

### Question Analysis

The hard-coded Router Selected Mode and empty ablation flags are removed from
the current table and mapper. They may return only through a future typed
backend contract backed by persisted data.

### V9 evidence capability gaps

The UI removes per-row values that are currently hard-coded rather than
instrumented, including cited state, per-slot token count, per-source token
count, and per-slot graph state. One section-level capability notice explains
what is not instrumented. This is clearer and smaller than repeating `N/A` in
every row.

### Counts and operational status

Missing or non-finite Ablation and human-evaluation counts render as
unavailable, not `0`. Run summaries and active Evaluation Center APIs follow
the same rule.

The existing durable `EvaluationJobPanel` is mounted in the selected campaign's
operation area. It reuses existing job APIs and polling behavior; no new tab or
API is introduced. The progress event's `latest_result_id` must either contain
the actual newest persisted result identifier or be absent from the contract;
this design selects the former so polling consumers can refresh deterministically.

Overview no longer performs an errors request whose result it does not render.
Errors and stage warnings remain in the diagnostics surface that consumes them.

## Export Schema v2

### Request

The existing export endpoint remains:

```text
POST /api/evaluation/campaigns/{campaign_id}/export
```

The redaction/content options remain available. The request adds:

```text
include_run_observability: boolean = false
```

The frontend checkbox label is **Include all run observability** and includes a
**Larger file** hint. It is off by default.

### Response

Schema v2 directly replaces the previous response:

```json
{
  "schema_version": "2.0",
  "export_metadata": {
    "exported_at": "2026-08-12T00:00:00Z",
    "options": {
      "include_run_observability": false
    },
    "redaction": {},
    "availability_warnings": []
  },
  "campaign": {},
  "sections": {
    "overview": { "availability": {}, "data": {} },
    "question_analysis": { "availability": {}, "data": {} },
    "agent_behavior": { "availability": {}, "data": {} },
    "router_analysis": { "availability": {}, "data": {} },
    "ablation": { "availability": {}, "data": {} },
    "human_evaluation": { "availability": {}, "data": {} },
    "diagnostics": { "availability": {}, "data": {} }
  },
  "runs": [
    {
      "result": {},
      "ragas_metrics": {},
      "observability": {
        "included": false,
        "availability": {},
        "data": null
      }
    }
  ]
}
```

Each `sections.*.data` value is the same canonical response projection used by
the corresponding panel. Conditional sections, such as release metrics for a
campaign without a compatible benchmark, use `not_applicable` rather than an
empty object that looks complete.

`diagnostics.data` contains campaign errors and stage warnings. Human
evaluation contains the human-versus-automatic comparison and evaluation
queue. Ablation contains condition comparison and graph-family summaries.

### Optional complete run observability

When `include_run_observability` is false, every run still includes its result,
finite official RAGAS metrics, token/accounting status, and latency summary.
`observability.included` is false and `data` is null.

When true, every campaign run contains the complete sanitized canonical run
projection. The export loads normalized data with campaign-level bulk
repository methods, groups rows by run identifier, and assembles runs without
calling a per-run endpoint or issuing a per-run database query.

The response is all-or-error. It does not silently truncate runs or event
arrays. Availability warnings describe legitimately unavailable instrumentation,
not transport truncation.

### Filename

The filename identifies both scope and content policy:

```text
{campaign_id}-summary-redacted-v2.json
{campaign_id}-observability-redacted-v2.json
{campaign_id}-summary-custom-v2.json
{campaign_id}-observability-custom-v2.json
```

`observability` is selected when all run observability is included. `custom` is
selected when full prompts or raw trace payloads are explicitly enabled. The
response's redaction metadata remains the authoritative audit record.

## Performance and Failure Handling

- Panel reads remain lazy by tab.
- Router Lab explicitly loads only the run list and selected run required for
  its execution-route card.
- Summary export uses bounded campaign projections and does not load detailed
  run observability.
- Full export uses one bulk load per normalized entity family, not one query per
  run.
- A failed export leaves the previous preview untouched and displays the
  established Evaluation Center error state.
- The frontend always revokes its temporary object URL after starting a
  download.
- The backend fails the request if a canonical required projection cannot be
  assembled; optional or uninstrumented sections use explicit availability.

## Delivery Waves

### Wave 1: Canonical run observability

1. Backend canonical observability service, v9 envelope, human ratings, and
   response contract.
2. Frontend selected-run API/types, stale-request protection, and v9 consumers.
3. Safe claim fields, graph evidence mapping, and retrieval provenance.

Checkpoint: deploy/push these commits and validate a real v9 campaign across
Run Trace, Retrieval Evidence, Claim Evidence, and the execution-route view.

### Wave 2: Panel truthfulness and operations

1. Separate retrospective router analysis from actual selected-run execution.
2. Remove unsupported Question, Router, V9 Evidence, and Claim placeholders.
3. Repair null/zero and availability semantics on active panels and APIs.
4. Mount durable job visibility and populate progress `latest_result_id`.

Checkpoint: validate all eight panels, direct Router Lab entry, campaign
switching, missing instrumentation, and durable job polling in the real system.

### Wave 3: Export Schema v2

1. Backend request/response types and canonical campaign sections.
2. Optional bulk all-run observability, redaction, and parity tests.
3. Frontend option, filename, strict types, download, and preview.

Checkpoint: download summary and full-observability artifacts from a real
campaign, inspect redaction, compare section values with the panels, and confirm
all campaign runs are present in the full artifact.

### Wave 4: Integration and cleanup

1. Cross-panel/export contract parity tests.
2. API, frontend, generated UI, and operational verification documentation.
3. Remove only duplicate helpers and mappers made obsolete by this repair.

Checkpoint: run the full regression set and repeat the production smoke
checklist before release acceptance.

## Commit and Checkpoint Protocol

Every implementation task follows this sequence:

1. add a failing test;
2. implement the smallest complete change;
3. run the task's focused frontend and/or backend tests;
4. update affected API/UI documentation in the same change set; and
5. create exactly one focused commit.

A task does not share a commit with another task. Existing unrelated working
tree changes are never staged.

At the end of a wave, the implementation agent provides:

- commit hashes and subjects;
- focused and wave-level test results;
- contract or persistence changes;
- the exact real-system verification checklist; and
- known availability limitations that production data may reveal.

The agent then stops. It must not begin the next wave until the user explicitly
accepts the checkpoint. A real-environment failure creates a corrective task
and a separate commit inside the same wave.

## Test Strategy

### Backend

- Contract tests prove the canonical observability response includes v9,
  human ratings, graph evidence, evidence status, and accounting diagnostics.
- Analytics tests distinguish measured, derived, heuristic, unavailable, and
  zero values.
- Repository-spy tests prove full export uses campaign bulk methods and no
  per-run query loop.
- Export parity tests compare every section with its canonical analytics
  service using the same fixture.
- Redaction tests cover every combination that permits full prompts or raw
  trace payloads and prove secrets/provider bodies remain excluded.
- Authorization tests prove campaign and run ownership on interactive and
  export paths.

### Frontend

- API contract tests cover the canonical observability route and new export
  request flag.
- Mapper tests prove normalized graph fields and explicit retrieval provenance.
- Integration tests enter every run-dependent tab directly and switch campaign
  while requests are in flight.
- Component tests prove unsupported metrics are absent, capability notices are
  singular, and unknown counts are not zero.
- Export tests cover default-off behavior, all four filename classes, preview
  stability on failure, and object URL cleanup.
- Durable job tests prove the panel mounts only for a selected campaign and
  refreshes when a terminal result identifier arrives.

### Real environment

Automated tests establish contract correctness, not production instrumentation
coverage. Each wave therefore ends with an explicit real-server verification
using at least:

- one agentic v9 run with graph or visual evidence;
- one run without optional instrumentation;
- one failed or partial-stage run when available;
- direct navigation to Router Lab; and
- one campaign large enough to exercise multi-run export.

## Acceptance Criteria

1. Every active Evaluation Center panel value is backed by a typed backend
   field or clearly identified as unavailable/derived/heuristic.
2. No hard-coded monitoring placeholder is presented as observed data.
3. Selected-run v9 data loads from the canonical campaign observability route
   on first entry and after campaign/run switches.
4. Claim, graph, and retrieval panels do not depend on redacted arbitrary
   payloads.
5. Unknown counts and token values remain unknown rather than becoming zero.
6. Router retrospective analysis is never presented as actual execution
   performance.
7. Export Schema v2 contains the canonical campaign sections used by the
   panels.
8. The default export omits detailed run observability and records that choice.
9. The opt-in export contains complete sanitized observability for every
   campaign run without N+1 loading or silent truncation.
10. Export filename and metadata correctly describe scope and redaction.
11. Durable evaluation job state is visible for the selected campaign.
12. Every task has one focused commit, and implementation stops after each wave
    for real-system verification.
