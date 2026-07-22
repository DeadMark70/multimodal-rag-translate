# Wave 4 Task 12B report

## Delivered

- Added `execution_policy.py`, an isolated v9 runtime boundary with typed
  retrieval/LLM/visual semaphores, phase timeouts, TaskGroup sibling
  cancellation, explicit retry numbering, and a shared campaign/SSE
  cancellation signal.
- Set the initial `ExecutionPolicy` defaults to retrieval `3`, LLM `2`, visual
  `1`; route/judge `2s`; extraction `8s`; and final `15s`.
- SSE disconnects mark the shared signal, and cancellation stops already
  running adapter work rather than only preventing subsequent work.
- A cancelled provider call now reconciles its admitted reservation
  conservatively before re-raising `CancelledError`, preserving the existing
  v9 budget ledger contract.
- v8 execution modules and `agents/*` were not changed.

## TDD evidence

The new policy test initially failed during collection because
`data_base.agentic_v9.execution_policy` did not exist.  The completed tests
cover default limits, semaphore/timeout enforcement, numbered retries,
TaskGroup cancellation, campaign cancellation, SSE disconnect cancellation,
and reservation reconciliation after provider cancellation.

## Verification

```text
python -m ruff check --no-cache [changed v9 files]
All checks passed!

python -m ruff format --no-cache --check [changed v9 files]
5 files already formatted

python -m pytest tests/test_agentic_v9_execution_policy.py -q -p no:cacheprovider
7 passed, 1 existing pytest configuration warning

python -m pytest (all test_agentic_v9_*.py) -q -p no:cacheprovider
163 passed, 1 existing pytest configuration warning

python -m pytest tests/test_research_execution_core_generic.py \
  tests/test_agentic_evaluation_service.py -q -p no:cacheprovider
28 passed, 24 existing third-party/Pytest warnings

git diff --check
(no diff errors)
```
