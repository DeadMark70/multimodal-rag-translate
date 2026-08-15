# Agentic v9 smoke verification

This tool is a release-verification aid, not a benchmark launcher. It never calls
a provider or an evaluation API by default.

## Status meanings

- `pass`: every checked release requirement is present and consistent in the
  supplied export.
- `fail`: the export contradicts a safety requirement, for example it emits a
  supported final claim for an unresolved slot.
- `partial`: an artifact exists but lacks required observability or exact token
  reconciliation evidence.
- `not_executed`: no external campaign export was supplied. This is the expected
  status for this implementation delivery; it is never equivalent to `pass`.

## 1. Inspect the dry-run plan

Run this from the backend repository root:

```powershell
.\.venv\Scripts\python.exe scripts\run_agentic_v9_smoke.py
```

It prints five fixed question IDs (`Q5`, `Q7`, `Q11`, `Q14`, `Q16`), one
Agentic v9 repeat, and a `not_executed` offline-verification result. It sends no
HTTP request. Add `--include-naive` only to plan a paired Naive arm.

## 2. Execute externally only when authorized

Execution is intentionally opt-in and requires all four controls: a base URL, a
named Evaluation Setup preset, an authentication header supplied by the operator,
and this exact confirmation string.

```powershell
.\.venv\Scripts\python.exe scripts\run_agentic_v9_smoke.py `
  --execute `
  --base-url "https://evaluation.example" `
  --preset "approved-release-preset" `
  --auth-header "Authorization: Bearer <operator-token>" `
  --confirm-execute I_UNDERSTAND_EXECUTE
```

The command first resolves the five test cases and the named preset, then creates
the campaign. Do not use this step without external approval. This repository's
automated tests use an injected fake transport and do not execute it.

## 3. Export the campaign

After the authorized campaign completes, use Evaluation Center's redacted export
control. Preserve the downloaded JSON unchanged; do not add model secrets or raw
credentials to it.

## 4. Verify offline and write a manifest

```powershell
.\.venv\Scripts\python.exe scripts\run_agentic_v9_smoke.py `
  --artifact .\downloads\campaign-redacted.json `
  --manifest .\verification\agentic-v9-release.json `
  --backend-commit <backend-commit> `
  --frontend-commit <frontend-commit> `
  --setup-snapshot .\verification\evaluation-setup.json `
  --dataset-identity golden-v2
```

The offline verifier uses a tolerant JSON reader: it accepts current export shape
and harmless field nesting differences, but it fails closed on missing evidence.
For each Agentic v9 run it checks contract v2 and actual route rationale, atomic
slots and final resolutions, targeted repair traces for missing slots,
phase-linked provider attempts, exact-or-explicitly-partial token reconciliation,
capture availability against the recorded setup, and unsupported final claims.

### Active Atomic Contract V2 and Wave 2 Qualification Invariants

- **Wave 2 Quote-Qualified Profile**: New open-corpus and explicit-scope runs end
  in `finalpack_r1_active_atomic_contract_v2_quote_qualified_v1`. Historical
  profiles (`retrieval_safe` and prior) are read under their respective contracts.
- **Evidence Qualification Before Sufficiency**: Every sufficiency-supported
  slot resolution must have at least one packet verified by
  `is_qualified_evidence()`. Unvalidated or candidate-only packets cannot satisfy
  slots.
- **Qualification Metrics Bounds**: The profile requires non-negative integers
  for `candidate_packet_count`, `qualified_packet_count`,
  `qualification_round_count`, and `qualification_provider_call_count`. Raw
  candidate count may exceed qualified count (`candidate_packet_count >= qualified_packet_count`).
  `qualification_provider_call_count` must equal the count of persisted LLM calls
  with `phase="evidence_extract"`. Provider failure or error cannot increase
  qualified count or promote unvalidated candidates.
- **Qualification Outcome Integrity**: Current Wave 2 runs use
  `not_attempted`, `deterministic`, `provider_qualified`, `no_match`,
  `provider_failed`, or `invalid_response`. Provider failures require
  `provider_attempt_failed`; invalid responses require
  `invalid_provider_response`; budget rejection is
  `not_attempted/budget_not_admitted`. Successful and valid no-match outcomes
  carry no failure code. Historical `not_enabled` remains readable but is not
  valid for the current quote-qualified profile.
- **Positive Control & Insufficient Regressions**: Positive controls Q5/Q23
  require qualified evidence packets; exports where all runs collapse to
  `insufficient` fail the smoke verification.
- **Deterministic Route & Atomic Overlay**: The admission contract retains deterministic router ownership while the atomic overlay decomposes the prompt into sequential evidence slots (`S1..Sn`, $1 \le n \le 8$), synthesis obligations, response constraints, and comparison plans.
- **Evidence Slots vs Obligations/Constraints**: Slots represent verifiable source-backed units of evidence (`required_slots`). Obligations (`synthesis_obligations`) and constraints (`response_constraints`) govern answer composition and reasoning structure rather than raw retrieval targets.
- **Provider Call Accounting**: Active atomic execution budgets at most 1 `contract_planning` provider call (`purpose="atomic_contract_planning"`) and 0 active `comparison_plan` provider calls (`atomic_planner_call_count <= 1`, `comparison_planner_call_count == 0`).
- **Comparison Subject Grounding**: When a comparison plan is present, all comparison subject `evidence_slot_ids` must reference valid slot IDs declared in `required_slots`.
- **Degraded V2 Behavior**: When the atomic planner encounters a provider failure, schema violation, or unparseable output, runtime gracefully degrades to a deterministic fallback contract (`contract_version="2"`, fallback `route_reason`, and a single catch-all slot `S1`), ensuring fail-closed safety without aborting execution.
- **Planner Diagnostics**: Active v2 profile runs must export typed
  `planner_diagnostics`. A `degraded` outcome must declare
  `retrieval_query_strategy="safe_fallback_original_question"` and exactly one
  compiled retrieval task, with matching `slot_plan_status="degraded"` and
  `slot_plan_source="safe_fallback"` contract provenance; its only `S1` query
  is the normalized, 512-character-bounded original question. The same profile
  requires integer planner-call metrics
  (`atomic_planner_call_count` in `0..1` and
  `comparison_planner_call_count=0`). `deterministic` and `planned` outcomes
  use `atomic_slots` with no failure stage/code; `deterministic` records no
  provider response and zero atomic calls, while `planned` records a provider
  response and one atomic call. The smoke gate does not require `planned`: a
  correctly diagnosed degraded fallback remains valid until the real-server
  canary establishes provider behavior.
- **Instrumentation Bounds**: Runtime strictly records
  `slot_binding_method="task_target_inherited"` and one explicit qualification
  outcome. Falsified, contradictory, obsolete, or uninstrumented values are
  rejected during current-profile verification.

The manifest records supplied backend/frontend commit IDs, the setup snapshot and
hash, dataset identity, SHA-256 input artifact hashes, requirement statuses, and
residual failures. Use
[`verification/agentic-v9-release-template.json`](verification/agentic-v9-release-template.json)
as a schema-shaped placeholder when no artifact exists.

For filesystem safety, `--manifest` refuses both an existing file and a symlink.
Choose a new output path for every verification run; this CLI intentionally has no
overwrite flag.

## 5. Run the contract-planner staging checkpoint

Do not run this checkpoint from a developer machine. After the Wave 1 commit is
pushed to the real server environment, export the exact Evaluation Setup model
configuration as a JSON object that validates against `evaluation.schemas.ModelConfig`.
An authorized operator runs the two planner commands and the evidence
qualification construction check first. Only the final command performs the
single evidence provider request:

```powershell
.\.venv\Scripts\python.exe scripts\agentic_v9_contract_planner_canary.py --schema current --model-config-json <real-server-model-config.json>
.\.venv\Scripts\python.exe scripts\agentic_v9_contract_planner_canary.py --schema minimal --model-config-json <real-server-model-config.json>
.\.venv\Scripts\python.exe scripts\agentic_v9_evidence_qualification_canary.py --model-config-json <real-server-model-config.json>
.\.venv\Scripts\python.exe scripts\agentic_v9_evidence_qualification_canary.py --model-config-json <real-server-model-config.json> --invoke
```

Each canary validates the file before provider construction and applies the same
model normalization, runtime override, phase policy, and shared schema binding
boundary as its campaign stage. The evidence construction command binds the
provider but never invokes it; `--invoke` uses the campaign's budgeted extractor
and parser. Its task-local `max_retries=0` override disables provider retries,
so one invocation is one wire-level attempt; normal campaign calls retain the
provider default. Each command writes one sanitized JSON
document containing only success, safe failure information, relevant package
versions, model identifier, and response availability/count metadata. It never
includes the model-config body, prompt, response body, key, or raw exception.
Exit code `0` means the bound response passed local validation; a nonzero exit
identifies a sanitized failure.

The campaign and `current` planner canary share a compact provider-generation
projection of the canonical planner schema so Gemini does not receive redundant
Pydantic validation metadata. The canonical `_PlannerDecision` model and the
planner's semantic checks remain the authoritative post-response acceptance
boundary; compaction does not relax extra-field, enum, length, count, numeric,
route, source-scope, or dependency validation.

Record the two complete JSON documents here without adding provider output:

```text
current: <sanitized JSON result>
minimal: <sanitized JSON result>
```

Select exactly one correction branch from the observed pair:

| Current schema | Minimal schema | Required correction |
| --- | --- | --- |
| fails | succeeds | Replace only unsupported schema constructs with a Google-supported reduced response schema; retain strict local Pydantic validation of the full domain result. |
| fails | fails | Correct provider/model/config/deployment wiring; do not weaken the schema. |
| succeeds | succeeds | Treat as deployment/version drift; pin the server package/config combination and add a parity test. |
| succeeds | fails | Stop: the canary is invalid or its minimal fixture is malformed; repair the canary before production changes. |

The local SDK accepting the production schema is not evidence for any branch.
Do not change production provider binding or schema configuration until both
real-server results are recorded and reconciled with official provider support.

