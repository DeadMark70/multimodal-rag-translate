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
- Full backend pytest: `805 passed, 57 warnings in 73.44s (0:01:13)` (exit 0).
- The earlier interruption was traced to Windows use of `os.kill(pid, 0)` in Graph maintenance liveness checks and resolved by commit `7da6156`.
