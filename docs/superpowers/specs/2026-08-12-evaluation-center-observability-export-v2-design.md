# Evaluation Center Observability and Export v2 Design

> Status: Wave 1 completed and validated; Wave 2-4 revision approved on
> 2026-08-13. This design is based on the current backend and frontend code
> contracts. Local SQLite content is explicitly not treated as evidence of what
> exists in the server environment.

> Revision note: the Wave 2-4 design was re-audited against the post-Wave 1
> code. The revision removes planned interfaces that do not exist, narrows
> repairs to APIs used by the active Evaluation Center, and separates the
> interactive-safe observability projection from the richer export projection.

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

## Post-Wave 1 Baseline

Wave 1 is complete. The campaign-scoped observability route is the canonical
selected-run contract, v9 data is nullable for non-v9 runs, normalized claim
and evidence projections are typed, text-bearing interactive fields are
bounded and credential-redacted, and frontend stale-request protection is in
place. Wave 2-4 must build on those contracts rather than reimplement them.

## Wave 2 Panel Corrections

### Active API contract hygiene

Only contracts used by the mounted Evaluation Center are repaired in this
track. `EvaluationRunListItem.total_tokens` becomes nullable so an unknown run
total cannot be serialized as zero. The unused run-diff and repeat-stability
APIs are not expanded or refactored as part of this feature.

`CampaignProgressEvent.latest_result_id` is removed from backend and frontend
contracts. The server never populated it, the mounted client does not consume
it, and durable-job refresh does not require a result identifier.

Every public schema change is synchronized through the backend OpenAPI
artifacts and the frontend contract pin. A stale generated contract is a
blocking failure, not documentation debt to defer.

### Router Lab

Campaign router analysis and selected-run execution route remain separate:

- the campaign endpoint returns only recorded retrospective decisions and is
  typed accordingly; actual execution rows are not mixed into this response;
- the selected-run route is `agentic_v9.contract.route`; the optional
  `agentic_v9.contract.route_decision` describes provenance but is not required
  to display a valid route; and
- direct entry loads router analysis, the campaign run list, and the selected
  run observability without requiring a previous visit to Run Trace.

The frontend removes fields the backend does not produce: tier, complexity,
saved tokens, quality loss/gain, latency comparison, token comparison, regret,
utility formula, oracle label, and confusion matrix. Retrospective failures do
not erase a valid selected-run route, and selected-run failures do not erase
retrospective analysis. A campaign switch clears the old run route immediately
while preserving the active tab.

### Question and capability placeholders

Question Analysis removes the hard-coded Router Selected Mode and empty
ablation flags from types, mappers, tables, and heatmaps.

The UI removes repeated values that are not instrumented by the current
contract:

- v9 evidence packet `Cited`;
- context-pack per-slot and per-source token counts;
- claim alignment per-slot graph state; and
- Agent Behavior atomic completeness while the backend can only return the
  experimental/uninstrumented placeholder.

Each affected section may show one scoped capability notice. The design does
not introduce a generic capability framework or pretend that capability flags
already exist in the API.

### Numeric semantics

Missing and non-finite Ablation, human-evaluation, and condition sample counts
render `N/A`; a measured zero remains `0`. A zero-millisecond normalized or
legacy trace duration renders `0 ms`, not `n/a`.

Overview stops requesting errors that it does not render. Errors and stage
warnings remain owned by the diagnostics surface.

### Durable jobs

The existing `EvaluationJobPanel` is mounted for the selected campaign and
shows an explicit empty state. Existing polling and job APIs are reused.

Campaign inventory loading becomes a stable operation so a terminal-job
refresh does not reset the active tab. Terminal notification is one-shot per
terminal job. It refreshes campaign inventory and invalidates the selected
campaign's currently loaded panel data; it does not depend on
`latest_result_id`.

## Export Schema v2

### Three projection layers

The post-Wave 1 interactive response is intentionally safe and has already
discarded arbitrary payloads. Export therefore cannot be implemented by
serializing `EvaluationRunObservabilityDetail` and then trying to restore raw
content. The backend uses three explicit layers:

```text
CampaignObservabilitySnapshot (internal, persistence-derived)
    -> interactive projector -> EvaluationRunObservabilityDetail
    -> export policy projector -> ExportRunObservabilityDataV2
```

The internal snapshot is not an HTTP model. It contains the typed normalized
rows and exact result/source-attempt relationships required by both projectors.
The two public projectors share selection and analytics helpers but enforce
different allow lists.

### Request and content policy

The endpoint remains:

```text
POST /api/evaluation/campaigns/{campaign_id}/export
```

The five existing content flags remain, and the request adds:

```text
include_run_observability: boolean = false
```

The frontend checkbox is labelled **Include all run observability**, includes a
**Larger file** hint, and is off by default. It always sends the boolean
explicitly.

All export modes are sanitized. Enabling raw trace or full prompts authorizes
only the corresponding stored, allow-listed content. Provider response bodies,
credentials, authorization headers, stack traces, and unrestricted error or
payload objects are never exported. Disabling answers or excerpts also removes
equivalent content nested inside trace and v9 projections. Fields suppressed by
policy remain present as `null` where a fixed v2 shape requires them.

The policy boundaries are exact:

- `include_raw_trace_payloads` controls only the sanitized allow-listed
  `trace_events[].payload`; it never opens retrieval, tool, claim, routing, or
  provider payload objects;
- `include_answers=false` clears the result answer, answer preview, claim
  statements, final-claim statements, and evidence-coverage fact text while
  preserving identifiers and statuses;
- `include_retrieved_excerpts=false` clears retrieval excerpts and v9 evidence
  packet statements while preserving document, chunk, page, asset, and slot
  locators; and
- a full prompt is present only when it was captured at execution and
  `include_full_prompts=true`.

### Typed response

Schema v2 directly replaces the old response. All top-level keys and section
names are required and typed:

```json
{
  "schema_version": "2.0",
  "export_metadata": {
    "exported_at": "2026-08-13T00:00:00Z",
    "options": { "include_run_observability": false },
    "redaction": {},
    "availability_warnings": []
  },
  "campaign": {},
  "sections": {
    "overview": { "availability": {}, "data": {
      "research_summary": {},
      "release_metrics": { "availability": {}, "data": null }
    }},
    "question_analysis": { "availability": {}, "data": {} },
    "agent_behavior": { "availability": {}, "data": {} },
    "router_analysis": { "availability": {}, "data": {} },
    "ablation": { "availability": {}, "data": {} },
    "human_evaluation": { "availability": {}, "data": {
      "comparison": {}, "queue": {}
    }},
    "diagnostics": { "availability": {}, "data": {
      "errors": {}, "stage_warnings": {}
    }}
  },
  "runs": [{
    "result": {},
    "ragas_metrics": {},
    "accounting": {},
    "latency": {},
    "observability": {
      "included": false,
      "availability": {},
      "data": null
    }
  }]
}
```

`campaign` is a small allow-listed identity/configuration snapshot rather than
the complete runtime campaign object. It contains the campaign ID, name,
lifecycle status, benchmark identity, selected modes, repeat count, and
timestamps only. `result` is an
export-specific fixed model rather than the runtime `CampaignResult`, because
redaction can make answer and reference fields null. It includes the run and
question identifiers, question text, execution identity/version, status,
answer/reference content subject to policy, source locators, token/latency
scalars, and timestamp; it excludes arbitrary token usage, snapshots, and
derived-metric dictionaries. `sections` uses the same
authoritative services as the active panels: research summary and optional
release metrics, research question comparison, agent behavior, retrospective
router analysis, ablation, human comparison/queue, errors, and stage warnings.
Legacy overview and comparison helpers are not used.

When release metrics are not applicable, the nested release section has
`availability.status="not_applicable"` and retains the typed report returned by
the canonical release service when available; `data=null` is reserved for a
truly absent optional projection.

Official RAGAS selection and accounting completeness are shared typed helpers,
not duplicated export SQL. Official scores must match the result's current
source attempt and evaluator identity. Each run includes accounting and latency
even when detailed observability is disabled.

### Optional all-run observability

When `include_run_observability` is false, every run has
`observability.included=false`, an explicit `not_applicable` reason, and
`data=null`.

When true, every campaign result receives one `ExportRunObservabilityDataV2`
built from a campaign-level snapshot. Every normalized entity family and
accounting family is loaded a bounded number of times, grouped in memory by run
and exact `source_attempt_id`, and never loaded through a per-run query loop.
Multiple attempt materializations must not overwrite one another in a
run-keyed dictionary.

Required section or run-container failure makes the export fail as one request.
Optional missing instrumentation produces explicit availability. Runs and
event arrays are never silently omitted or truncated.

### Frontend contract and filename

The frontend uses strict named v2 interfaces and an export-specific run/result
type. It validates the received v2 shape at the API boundary before creating a
download. It does not extend the response with `Record<string, unknown>` or
reuse `CampaignResult` for redacted export data.

The filename identifies scope and content policy:

```text
{campaign_id}-summary-redacted-v2.json
{campaign_id}-observability-redacted-v2.json
{campaign_id}-summary-custom-v2.json
{campaign_id}-observability-custom-v2.json
```

`custom` means full prompts or raw trace payloads were explicitly requested.
Server `export_metadata` remains authoritative for the preview and audit
record. A summary preview never fabricates a zero LLM-call count when detailed
calls were not requested.

## Performance and Failure Handling

- Panel reads remain lazy by tab.
- Router direct entry loads only router analysis, the run list, and one selected
  run detail.
- Summary export does not load detailed run observability.
- Full export uses one campaign snapshot and bounded loaders; query count does
  not grow linearly with the number of runs.
- Independent campaign sections may be assembled concurrently, but a required
  section failure returns no partial v2 artifact.
- A failed frontend export keeps the previous preview, creates no download, and
  restores controls.
- The frontend revokes every temporary object URL.

## Contract Synchronization

Backend schema tasks update and verify all three generated artifacts:

- `openapi.json`;
- `contracts/openapi-contract.json`; and
- `docs/generated/api-surface.md`.

The frontend pins the resulting backend HEAD contract and runs
`contract:check`. Generated UI docs are synchronized only after component and
API tests pass. Contract drift blocks the wave checkpoint.

## Delivery Waves

### Wave 1: Canonical run observability — complete

Canonical backend projection, safe typed evidence, frontend consumption,
stale-request protection, and two consolidated-review correction rounds were
implemented and validated before this revision.

### Wave 2: Panel truthfulness and operations

1. Repair generated API baseline, remove the unused progress field, and
   preserve nullable run-list tokens.
2. Type and filter retrospective router analysis.
3. Make direct Router entry and Question Analysis truthful.
4. Remove uninstrumented capability placeholders.
5. Correct missing-count and zero-duration rendering.
6. Mount durable jobs with stable, one-shot refresh behavior.

Checkpoint: validate all eight panels, direct Router entry, campaign switching,
missing instrumentation, zero versus unknown, and durable polling in the real
system. Stop before Export v2.

### Wave 3: Export Schema v2

1. Define typed public v2 schemas, export policy, and synchronized OpenAPI.
2. Build the internal campaign snapshot and shared official RAGAS/accounting
   projections with no-N+1 proof.
3. Compose every authoritative section and replace the old export route with
   all-or-error behavior.
4. Add the strict frontend consumer, default-off option, deterministic
   filename, and server-authoritative preview.

Checkpoint: download default and full artifacts, inspect policy metadata,
compare panel values, and prove every campaign run is present. Stop before the
release parity wave.

### Wave 4: Parity and release gate

1. Use authenticated HTTP tests over one durable fixture to prove complete
   serialized panel/export parity and bounded query behavior.
2. Lock frontend runtime-contract rejection and focused cross-campaign/UI
   behavior.
3. Synchronize backend/frontend/generated documentation and run release gates.

Checkpoint: run the complete regression, contract, lint, build, docs, and real
environment checklist before release acceptance.

## Commit and Checkpoint Protocol

Every task adds a failing test, implements the smallest complete change, runs
focused checks, updates directly affected docs, and creates exactly one focused
commit. Tasks never share a commit. Unrelated working-tree changes are never
staged.

Execution uses a fresh implementation subagent per task. Per the approved
workflow, there is one consolidated code review at the end of each wave rather
than a review after every task. Any Critical or Important finding is corrected
inside the same wave with a separate commit and one scoped re-review.

At each checkpoint the agent reports commit hashes, exact test totals, contract
changes, query-count evidence where relevant, and a safe real-system checklist,
then stops. The next wave starts only after explicit user acceptance.

## Test Strategy

### Backend

- Schema tests cover nullable run tokens and the removed progress field.
- Router tests prove the campaign response is typed and retrospective-only.
- Export policy tests cover the complete boolean-option matrix and permanent
  secret/provider/error exclusions.
- Repository spies fail on every per-run/per-attempt loader and prove bounded
  campaign loads, exact attempt selection, and no truncation.
- Authenticated HTTP parity tests compare complete serialized named sections,
  not a selected subset of convenient fields.
- Authorization tests cover campaign/run ownership on interactive and export
  paths.

### Frontend

- Router integration tests cover direct entry, valid contract route without
  route-decision provenance, failure isolation, campaign switching, and stale
  requests.
- Component tests prove unsupported columns are absent, capability notices are
  scoped, missing counts are `N/A`, and zero durations are `0 ms`.
- Durable-job tests cover selected-campaign mount, empty state, one-shot
  terminal refresh, and tab preservation.
- Export tests cover runtime shape rejection, default-off requests, all four
  filenames, server-authoritative previews, failure stability, and URL cleanup.
- Contract and generated-doc checks run against committed backend artifacts.

### Real environment

Automated tests establish contract correctness, not production instrumentation
coverage. Each checkpoint uses at least one v9 run with evidence, one run
without optional instrumentation, one legacy/v8 run, one failed/partial run if
available, direct Router navigation, and a campaign large enough for all-run
export.

## Acceptance Criteria

1. Every active panel value is backed by a typed backend field or clearly shown
   as unavailable, derived, or heuristic.
2. No hard-coded monitoring placeholder is presented as observed data.
3. Router retrospective decisions never masquerade as actual route performance;
   actual route comes from `agentic_v9.contract.route`.
4. Direct Router entry and campaign switching cannot show stale route data.
5. Unknown active run totals and UI counts remain unknown; measured zero remains
   zero.
6. Durable job state is visible and terminal refresh does not reset the active
   tab or depend on a nonexistent result ID.
7. Export v2 has required named typed sections and export-specific redacted run
   models.
8. Default export explicitly omits detailed run observability.
9. Opt-in export contains sanitized observability for every result, uses the
   exact source attempt, and has no N+1 or silent truncation.
10. Provider bodies, credentials, unrestricted errors, and stack traces never
    appear in any export option combination.
11. Export metadata, preview, and filename accurately describe the artifact.
12. Backend OpenAPI, frontend contract pin, generated docs, tests, lint, and
    build are synchronized at the final gate.
13. Every task has one focused commit and execution stops after each wave for
    real-system validation.
