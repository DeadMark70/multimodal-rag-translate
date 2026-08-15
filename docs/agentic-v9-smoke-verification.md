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

### Active Atomic Contract V2 Invariants

- **Deterministic Route & Atomic Overlay**: The admission contract retains deterministic router ownership while the atomic overlay decomposes the prompt into sequential evidence slots (`S1..Sn`, $1 \le n \le 8$), synthesis obligations, response constraints, and comparison plans.
- **Evidence Slots vs Obligations/Constraints**: Slots represent verifiable source-backed units of evidence (`required_slots`). Obligations (`synthesis_obligations`) and constraints (`response_constraints`) govern answer composition and reasoning structure rather than raw retrieval targets.
- **Provider Call Accounting**: Active atomic execution budgets at most 1 `contract_planning` provider call (`purpose="atomic_contract_planning"`) and 0 active `comparison_plan` provider calls (`atomic_planner_call_count <= 1`, `comparison_planner_call_count == 0`).
- **Comparison Subject Grounding**: When a comparison plan is present, all comparison subject `evidence_slot_ids` must reference valid slot IDs declared in `required_slots`.
- **Degraded V2 Behavior**: When the atomic planner encounters a provider failure, schema violation, or unparseable output, runtime gracefully degrades to a deterministic fallback contract (`contract_version="2"`, fallback `route_reason`, and a single catch-all slot `S1`), ensuring fail-closed safety without aborting execution.
- **Instrumentation Bounds**: Runtime strictly records `slot_binding_method="task_target_inherited"` and `semantic_qualification="not_enabled"`. Falsified or uninstrumented values are rejected during verification.

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
An authorized operator then runs exactly these two commands, once each:

```powershell
.\.venv\Scripts\python.exe scripts\agentic_v9_contract_planner_canary.py --schema current --model-config-json <real-server-model-config.json>
.\.venv\Scripts\python.exe scripts\agentic_v9_contract_planner_canary.py --schema minimal --model-config-json <real-server-model-config.json>
```

The canary validates the file before provider construction, applies the same model
normalization, runtime override, contract-planning phase policy, and shared schema
binding boundary as the campaign. Its task-local `max_retries=0` override disables
provider retries, so its single wrapper invocation is also exactly one wire-level
attempt; normal campaign calls retain the provider default. Each invocation writes
one sanitized JSON document containing only success, schema, failure
stage/code, relevant package versions, model identifier, and whether a response was
received. It never includes the model-config body, prompt, response body, key, or
raw exception or import traceback. Exit code `0` means the bound response passed
local validation; a nonzero exit identifies the failed stage. Missing, unreadable,
malformed, non-UTF-8, or schema-invalid model configuration fails before the
provider-dependent stack is imported.

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

