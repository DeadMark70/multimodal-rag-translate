# Task 8: strict research analytics and authenticated API

## Delivered

- Added the version-2 typed research-summary contract in `evaluation/accounting_schemas.py`.
- Added `ResearchAnalyticsService`, using campaign/result/RAGAS/accounting repositories directly rather than legacy analytics or `RagasEvaluator`.
- Added authenticated `GET /api/evaluation/campaigns/{campaign_id}/research-summary`.
- Kept missing token, quality, and price totals nullable; RAGAS batch usage is reported only as evaluation overhead.
- Added regression coverage for nearest-rank percentile behaviour and HTTP null serialization.

## Verification

`python -m pytest tests/test_evaluation_research_analytics.py tests/test_evaluation_research_api.py tests/test_evaluation_analytics_api.py tests/test_evaluation_api.py -q`

Result: `17 passed` (third-party Pydantic deprecation warnings only).

`python -m ruff check evaluation/accounting_schemas.py evaluation/research_analytics.py evaluation/router.py tests/test_evaluation_research_analytics.py tests/test_evaluation_research_api.py`

Result: all checks passed.

## Remaining risk

The focused fixtures cover contract/percentile/API behaviour. Broader literal durable-DB fixtures for retry, mixed evaluator signatures, and failed RAGAS work-item state should be expanded before frontend rollout.
