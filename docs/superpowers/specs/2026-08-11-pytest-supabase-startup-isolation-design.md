# Pytest Supabase Startup Isolation Design

## Problem

Production startup correctly fails closed when `init_supabase()` cannot create a client. The CI test job intentionally supplies no Supabase credentials, so every test that enters the real FastAPI lifespan now fails before reaching its assertions. Local runs can hide the issue when `config.env` contains valid credentials.

The failing tests are not obsolete. They share one missing test boundary around startup initialization.

## Design

Add one autouse fixture to `tests/conftest.py`. During pytest execution it replaces `supabase_client.init_supabase()` with a truthy local sentinel. This lets the application lifespan complete without credentials or network access.

The fixture changes test infrastructure only:

- Production `core/app_factory.py` remains fail closed.
- CI does not receive fake Supabase credentials.
- Individual TestClient modules do not need repeated patches.
- The existing negative readiness test continues to patch `init_supabase()` to return `None` inside that test, overriding the autouse fixture and preserving fail-closed coverage.
- No test is deleted or skipped.

## Verification

1. Reproduce the current readiness failure with `TEST_MODE=true`, fake providers enabled, network blocked, and blank Supabase variables.
2. Add the autouse fixture and rerun the readiness success and fail-closed tests; both must pass.
3. Run the TestClient-heavy affected suites from the CI failure summary.
4. Run the complete warning-budget pytest command under the same credential-free CI environment.
5. Run Ruff on the changed test infrastructure.

## Non-goals

- Changing production readiness semantics.
- Adding a fake Supabase service implementation.
- Supplying placeholder secrets in CI.
- Removing or weakening existing API tests.
