# Evaluation Multi-Query and Graph Locator Upgrade

- Status: Completed
- Design: `docs/superpowers/specs/2026-07-17-evaluation-multiquery-graph-locator-design.md`
- Implementation plan: `docs/superpowers/plans/2026-07-17-evaluation-multiquery-graph-locator.md`

## Delivered behavior

- Evaluation Advanced, Graph family, and Agentic execute without HyDE.
- Multi-Query drives intended expansion and Agentic CRAG correction.
- Main Graph and Agentic graph routes resolve graph evidence to source chunks.
- `graph_raw_current` remains the legacy control.
- Public Agentic chat retains its pre-v8 retrieval policy; the v8 policy is scoped to Evaluation Center.
- Successful and failed result rows carry versioned execution profiles, including captured trace profiles when available.
- Graph maintenance uses a non-destructive Windows process-liveness probe.

## Verification

- Final focused pytest: `45 passed, 25 warnings in 10.38s` (exit 0).
- Ruff check: `All checks passed!` (exit 0).
- Ruff format: `10 files already formatted` (exit 0).
- Full backend pytest command: `.\\.venv\\Scripts\\python.exe -m pytest`.
- Full backend pytest: `817 passed, 58 warnings in 89.54s (0:01:29)` (exit 0).
- The earlier interruption was traced to Windows use of `os.kill(pid, 0)` in Graph maintenance liveness checks and resolved by commit `7da6156`.
- Final whole-change review found and resolved public-chat policy leakage and missing failure-path profiles in commit `b85f719`; direct campaign-engine coverage was added in commit `403374a` and re-review approved it with no remaining findings.
