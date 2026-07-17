# Evaluation Multi-Query and Graph Locator Upgrade

- Status: Completed
- Design: `docs/superpowers/specs/2026-07-17-evaluation-multiquery-graph-locator-design.md`
- Implementation plan: `docs/superpowers/plans/2026-07-17-evaluation-multiquery-graph-locator.md`

## Delivered behavior

- Evaluation Advanced, Graph family, and Agentic execute without HyDE.
- Multi-Query drives intended expansion and Agentic CRAG correction.
- Main Graph and Agentic graph routes resolve graph evidence to source chunks.
- `graph_raw_current` remains the legacy control.
- Changed result rows carry versioned execution profiles.

## Verification

- Focused pytest: `122 passed, 23 warnings in 7.65s` (exit 0).
- Ruff check: `All checks passed!` (exit 0).
- Ruff format: `9 files already formatted` (exit 0).
- Full backend pytest command: `.\\.venv\\Scripts\\python.exe -m pytest tests -q`.
- Full backend pytest failure class: `KeyboardInterrupt` after 82.09 seconds; 378 tests had passed at interruption. Cause undetermined.
- Full backend pytest first actionable error line: `C:\\Users\\user\\AppData\\Local\\Programs\\Python\\Python313\\Lib\\asyncio\\runners.py:124: KeyboardInterrupt`.
- Full backend pytest observed terminal summary: `378 passed, 44 warnings in 82.09s (0:01:22)`.
- Diagnostic full backend pytest command: `.\\.venv\\Scripts\\python.exe -X faulthandler -m pytest tests -vv -o faulthandler_timeout=60`.
- Diagnostic full backend pytest failure class: `KeyboardInterrupt` after 52.16 seconds; 377 tests had passed at interruption. Cause undetermined.
- Diagnostic full backend pytest last running test: `tests/test_graph_maintenance_lock.py::test_non_owner_cannot_release_maintenance_lock`.
- Diagnostic full backend pytest first actionable error line: `C:\\Users\\user\\AppData\\Local\\Programs\\Python\\Python313\\Lib\\pathlib\\_local.py:722: KeyboardInterrupt`.
- Diagnostic full backend pytest observed terminal summary: `377 passed, 44 warnings in 52.16s`.
