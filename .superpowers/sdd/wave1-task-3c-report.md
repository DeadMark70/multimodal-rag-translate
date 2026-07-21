# Wave 1 — Task 3C report

Commit: `refactor(agentic-v9): centralize provider invocation`

## Scope and implementation

- Added `BudgetedLlmInvoker`, the concrete implementation of the existing
  `LlmInvoker.invoke(phase, purpose, messages)` protocol. It delegates every
  request to `invoke_budgeted_llm`, preserving Task 3's reserve-before-provider
  accounting gate.
- Added v9-only model-path adapters for multi-query/HyDE rewrite, CRAG
  relevance judgment, visual and evidence extraction, conflict arbitration,
  claim verification, and final rendering. Each receives an injected invoker;
  none receives a provider or calls `ainvoke`.
- Added an optional injected-invoker route to `GenericGraphRouter`. Its legacy
  no-argument construction retains the existing v8 provider path; the v9 route
  sends the graph fallback through `phase="graph_route"`.

## Red / green proof

- Red: `tests/test_agentic_v9_provider_boundary.py` initially failed collection
  with `ModuleNotFoundError: data_base.agentic_v9.model_paths`.
- Green: the implemented adapter module, concrete invoker, graph injection, and
  AST guard passed focused verification.

## Verification

```powershell
..\..\.venv\Scripts\python.exe -m pytest tests\test_agentic_v9_provider_boundary.py tests\test_agentic_v9_budget_controller.py tests\test_agentic_v9_budgeted_llm.py tests\test_agentic_v9_phase_policy.py tests\test_agentic_v9_budget_feasibility.py tests\test_graphrag_integration.py tests\test_query_transformer.py -q
```

Result: `46 passed` (23 existing third-party `storage3` Pydantic deprecation
warnings).

```powershell
..\..\.venv\Scripts\python.exe -m ruff check data_base\agentic_v9\budgeted_llm.py data_base\agentic_v9\model_paths.py graph_rag\generic_mode.py tests\test_agentic_v9_provider_boundary.py
```

Result: `All checks passed!`

## Files

- `data_base/agentic_v9/budgeted_llm.py`
- `data_base/agentic_v9/model_paths.py`
- `graph_rag/generic_mode.py`
- `tests/test_agentic_v9_provider_boundary.py`
- `.superpowers/sdd/wave1-task-3c-report.md`

## Residual risk

- There is no v9 orchestration runtime in this task's baseline; later retrieval
  and execution wiring must construct these v9 adapters with the run-scoped
  `BudgetedLlmInvoker`. Legacy v8 wrappers intentionally remain unmodified.
- The AST guard permits the sole direct provider call in `budgeted_llm.py`, the
  prescribed budget gateway; all other v9 modules are forbidden from bypassing
  it.
