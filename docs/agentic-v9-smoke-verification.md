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

The manifest records supplied backend/frontend commit IDs, the setup snapshot and
hash, dataset identity, SHA-256 input artifact hashes, requirement statuses, and
residual failures. Use
[`verification/agentic-v9-release-template.json`](verification/agentic-v9-release-template.json)
as a schema-shaped placeholder when no artifact exists.
