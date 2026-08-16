# Agentic v9 Grounded Verification Corrective — Task 2 Report

## Scope

- Base commit: `613da11ffa410807f31acf5e80f55e5e1bd98772`
- Scope limited to post-contract feasibility and runtime claim-verifier reservation.
- No Q5/Q23 dependency, model pinning, frontend/export work, or Task 3 anchor changes.

## RED

Ran the brief's focused RED command before production changes:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_budget_feasibility.py tests/test_agentic_v9_campaign_runtime.py -k "claim_verifier and (feasibility or budget or provider_calls or at_most_once)" -q
```

RED was observed as expected: the new feasibility keyword was not accepted and
the runtime ledger assertions had no `claim_verifier` entry (8 failures).

## Implementation

1. Extended `validate_post_contract_feasibility()` with the exact keyword-only
   input `claim_verifier_provider_calls: int = 0`.
2. Applied strict integer/boolean validation for values outside `{0, 1}` and
   returned `CONFIGURATION_INCOMPATIBLE` with reason
   `invalid_claim_verifier_provider_calls`.
3. Added one pending `claim_verifier` provider call to the required/pending
   ledger when the value is `1`, so call and token reservations include it.
4. Passed `claim_verifier_provider_calls=1` in all three active runtime
   feasibility attempts: planner-admitted, planner-fallback, and deterministic.
5. Added feasibility, invalid-input, two-call rejection, runtime wiring, and
   two-pending-claims-at-most-once coverage. Runtime still performs zero
   verifier calls when Task 1 produces no pending claims.

## GREEN and verification

Focused Task 2 filter after implementation:

```text
9 passed, 65 deselected, 23 warnings
```

Affected suites from the brief:

```text
101 passed, 23 warnings
```

Ruff on all four modified source/test files:

```text
All checks passed!
```

`git diff --check` passed. It emitted only Git's existing LF/CRLF conversion
warnings for the four modified files.

The end-to-end fixture confirms two pending claims produce exactly one
`purpose="claim_verifier"` observer attempt and one runtime claim-verifier
metric. Its test contract uses `max_llm_calls=5` to accommodate the controller's
planner/evidence/final/verifier sequence; production controller behavior was not
changed.

## Self-audit

- Modified only the four brief-listed implementation/test files; this report is
  the required Task 2 artifact.
- Preserved fail-closed behavior and the existing one-batch verifier contract.
- No Task 3 anchors or unrelated UI/export paths were changed.
- Remaining warnings are dependency-level Pydantic deprecations from `storage3`.

## Corrective round 1 — route-budget regression

### RED

Added a real `RoutePlanner` → `validate_post_contract_feasibility()` regression
for visual `exact_structured` and visual `graph_relational` contracts, including
the active runtime reservations (`evidence_qualification_provider_calls=1` and
`claim_verifier_provider_calls=1`). Before the production change:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_route_planner.py::test_visual_route_budgets_admit_grounded_completion -q --disable-warnings
```

RED was precise: both cases failed with
`required_provider_calls_exceed_call_budget`:

- visual `exact_structured`: required 4 provider calls, route cap 3;
- visual `graph_relational`: required 5 provider calls, route cap 4.

### Minimal GREEN change

Only the two affected production route caps were raised; retrieval rounds,
repair rounds, token budgets, and all other route caps were unchanged.

| Route | Visual path required calls | Nonvisual required calls | Correct route cap |
| --- | ---: | ---: | ---: |
| `exact_structured` | 4 | 3 | 4 |
| `graph_relational` | 5 | 4 | 5 |

The production matrix now uses `max_llm_calls=4` for `exact_structured` and
`max_llm_calls=5` for `graph_relational`; the caps cover their visual and
nonvisual phase sets without a global increase.

### Corrective verification

```text
tests/test_agentic_v9_route_planner.py: 10 passed
route planner + feasibility + campaign runtime: 84 passed
```

Ruff and `git diff --check` remain clean. The existing 23 dependency-level
Pydantic deprecation warnings are unchanged.

## Corrective round 2 — remaining visual-capable routes

### RED

Extended the same real `RoutePlanner` → post-contract feasibility regression to
visual `multi_document_exact` and visual `multi_hop` contracts:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_route_planner.py::test_visual_route_budgets_admit_grounded_completion -q --disable-warnings
```

RED was precise: the two new cases failed with
`required_provider_calls_exceed_call_budget`:

- visual `multi_document_exact`: required 4 provider calls, route cap 3;
- visual `multi_hop`: required 4 provider calls, route cap 3.

### Minimal GREEN change and complete route matrix

Only `multi_document_exact` and `multi_hop` route caps were raised from 3 to 4.
All retrieval rounds, repair rounds, token budgets, and other route caps remain
unchanged.

| Route | Nonvisual required calls | Visual required calls | Route cap |
| --- | ---: | ---: | ---: |
| `single_lookup` | 3 | n/a | 3 |
| `bounded_compare` | 3 | n/a | 3 |
| `exact_structured` | 3 | 4 | 4 |
| `multi_document_exact` | 3 | 4 | 4 |
| `multi_hop` | 3 | 4 | 4 |
| `graph_relational` | 4 | 5 | 5 |

### Corrective verification

```text
visual route regression: 4 passed
route planner + feasibility + budget controller + budgeted llm + campaign runtime: 113 passed
```

Ruff and `git diff --check` remain clean. The existing 23 dependency-level
Pydantic deprecation warnings are unchanged.
