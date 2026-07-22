# Wave 5 Task 13B report

- Added token-only `POST /api/evaluation/campaigns/preflight`, which validates owned Golden v2 questions against their expected routes through both v9 feasibility stages and returns per-question `configuration_incompatible` issues.
- Added `GET /api/evaluation/runs/{run_id}/detail`. It preserves all legacy detail fields and exposes the typed, optional `RunDetailResponse.agentic_v9` envelope; historical/non-v9 runs return `null`.
- Regenerated the scoped OpenAPI contract additions. No money fields were introduced.
- Verified: `pytest tests/test_evaluation_analytics_api.py tests/test_agentic_v9_budget_feasibility.py -q -p no:cacheprovider` (9 passed); Ruff clean; `git diff --check` clean.
