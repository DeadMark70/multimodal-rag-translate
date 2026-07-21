# Wave 1 — Task 3 report

Commit: `feat(agentic-v9): enforce atomic provider budgets`

## Modified files

- `data_base/agentic_v9/budget_controller.py`
- `data_base/agentic_v9/budgeted_llm.py`
- `core/llm_usage_context.py`
- `core/llm_usage_callback.py`
- `core/llm_factory.py`
- `tests/test_agentic_v9_budget_controller.py`
- `tests/test_agentic_v9_budgeted_llm.py`
- `tests/test_llm_usage_callback.py`
- `tests/test_llm_factory_override.py`

## Red / green evidence

- Red: the first budget-controller test failed collection with
  `ModuleNotFoundError: data_base.agentic_v9.budget_controller`.
- Red: the invoker test failed collection with
  `ModuleNotFoundError: data_base.agentic_v9.budgeted_llm`.
- Red: the final-provider fallback test raised the expected unavailable-provider
  `RuntimeError` before the deterministic qualified partial was implemented.
- Red: callback missing usage reconciled to zero before the callback preserved
  missing usage for the controller's conservative estimate.
- Green: all focused budget, callback, factory, phase, feasibility, and usage
  normalization tests passed after the implementation and formatting pass.

## Verification

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m pytest tests\test_agentic_v9_budget_controller.py tests\test_agentic_v9_budgeted_llm.py tests\test_llm_usage_callback.py tests\test_llm_factory_override.py tests\test_agentic_v9_phase_policy.py tests\test_agentic_v9_budget_feasibility.py tests\test_evaluation_token_normalizers.py tests\test_evaluation_token_cost.py -q
```

Result: `81 passed` (third-party `storage3` Pydantic deprecation warnings only).

```powershell
D:\flutterserver\pdftopng\.venv\Scripts\python.exe -m ruff check data_base\agentic_v9\budget_controller.py data_base\agentic_v9\budgeted_llm.py core\llm_usage_context.py core\llm_usage_callback.py core\llm_factory.py tests\test_agentic_v9_budget_controller.py tests\test_agentic_v9_budgeted_llm.py tests\test_llm_usage_callback.py tests\test_llm_factory_override.py
```

Result: `All checks passed!`

## Residual risks

- Task 3C must inject `invoke_budgeted_llm` into every v9 provider path; this
  task deliberately does not modify adapters or direct v8 call paths.
- The final-input reservation is supplied by the later context-packing layer;
  this task enforces the envelope it receives.
