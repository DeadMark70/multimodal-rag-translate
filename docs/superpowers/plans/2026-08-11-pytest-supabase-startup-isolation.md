# Pytest Supabase Startup Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the backend pytest suite independent of local Supabase credentials while preserving production fail-closed startup behavior.

**Architecture:** Add one autouse fixture at the shared pytest boundary. The fixture replaces only `supabase_client.init_supabase` with a truthy local sentinel; the existing negative readiness test may still override it with `None` to exercise the real fail-closed branch.

**Tech Stack:** Python 3.13, pytest, pytest `monkeypatch`, FastAPI `TestClient`, Ruff

## Global Constraints

- Do not modify `core/app_factory.py` or production readiness semantics.
- Do not add Supabase credentials or placeholder secrets to CI.
- Do not delete, skip, or weaken existing tests.
- Keep the change limited to `tests/conftest.py` plus this plan.
- Verify with blank `SUPABASE_URL` and `SUPABASE_KEY` under the CI test flags.

---

### Task 1: Isolate Supabase startup in pytest

**Files:**
- Modify: `tests/conftest.py`
- Verify: `tests/test_health_api.py`
- Reference: `.github/workflows/no-external-api-test.yml`

**Interfaces:**
- Consumes: pytest's built-in `monkeypatch` fixture and `supabase_client.init_supabase(force: bool = False)`.
- Produces: autouse fixture `stub_supabase_startup(monkeypatch) -> object` that supplies a truthy startup-only sentinel during every test.

- [ ] **Step 1: Reproduce the credential-free startup failure**

Run in PowerShell from the backend repository:

```powershell
$env:TEST_MODE='true'
$env:USE_FAKE_PROVIDERS='true'
$env:CI_BLOCK_EXTERNAL_NETWORK='true'
$env:SUPABASE_URL=''
$env:SUPABASE_KEY=''
.\.venv\Scripts\python.exe -m pytest `
  tests/test_health_api.py::test_ready_is_200_while_lifespan_is_active `
  tests/test_health_api.py::test_lifespan_fails_when_supabase_client_is_unavailable `
  -q
```

Expected RED: `test_ready_is_200_while_lifespan_is_active` fails with `RuntimeError: Critical dependency initialization failed`; the explicit fail-closed test passes.

- [ ] **Step 2: Add the minimal autouse fixture**

Add this fixture in `tests/conftest.py` under `Mock Fixtures`:

```python
@pytest.fixture(autouse=True)
def stub_supabase_startup(monkeypatch: pytest.MonkeyPatch) -> object:
    """Keep app startup credential-free while preserving explicit failure tests."""
    test_client = object()
    monkeypatch.setattr(
        "supabase_client.init_supabase",
        lambda force=False: test_client,
    )
    return test_client
```

Do not patch `core.app_factory._initialize_external_clients`: the negative readiness test must remain able to replace `supabase_client.init_supabase` with `None` inside its own patch context.

- [ ] **Step 3: Verify readiness success and fail-closed coverage together**

Run the Step 1 command again.

Expected GREEN: `2 passed`. The success test uses the autouse sentinel; the negative test's inner `patch(..., return_value=None)` overrides it and still observes the production `RuntimeError`.

- [ ] **Step 4: Verify representative TestClient suites under CI conditions**

With the same environment variables still set, run:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_agentic_chat_service.py `
  tests/test_api_contracts_v3.py `
  tests/test_conversations_api.py `
  tests/test_rag_ask_stream.py `
  -q
```

Expected: all selected tests pass with no `Critical dependency initialization failed` startup errors.

- [ ] **Step 5: Run the exact CI-equivalent full suite and Ruff**

```powershell
.\.venv\Scripts\python.exe scripts/run_pytest_with_warning_budget.py --max-warnings 56 -- -q
.\.venv\Scripts\python.exe -m ruff check tests/conftest.py
```

Expected: full pytest exits `0`, warning count stays at or below `56`, and Ruff exits `0`.

- [ ] **Step 6: Verify scope and commit**

```powershell
git diff --check
git status --short
git add tests/conftest.py
git commit -m "test(startup): isolate Supabase initialization"
```

Expected: only `tests/conftest.py` is part of the implementation commit; no production or workflow file changes.
