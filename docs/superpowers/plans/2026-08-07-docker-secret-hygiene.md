# Docker Secret Hygiene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep backend credentials out of Docker image layers and keep generated SQLite databases out of Git while preserving runtime `env_file` injection.

**Architecture:** Enforce the security boundary at the two repository-native context filters: `.dockerignore` for Docker build inputs and `.gitignore` for Git candidates. Add a focused pytest module that checks the explicit Docker exclusions and delegates Git matching to `git check-ignore`, while retaining `config.env.example` as the tracked developer template.

**Tech Stack:** Docker ignore files, Git ignore files, Python 3.11+, pytest, Git CLI, Docker Compose YAML

## Global Constraints

- Do not read, print, rewrite, move, commit, or rotate credential values.
- Keep `D:\flutterserver\docker-compose.yml` unchanged and preserve `env_file: ./pdftopng/config.env`.
- Keep `config.env.example` tracked.
- Do not delete existing generated databases.
- Limit production changes to `.dockerignore` and `.gitignore`.

---

## File Structure

- Create `tests/test_secret_hygiene.py`: focused regression checks for Docker exclusions, Git ignore behavior, and the tracked environment template.
- Modify `.dockerignore`: exclude production environment-file naming patterns from the backend build context.
- Modify `.gitignore`: exclude `.pytest-tmp/` and the `data/evaluation.remote.db*` SQLite family.

### Task 1: Exclude Credentials from the Docker Build Context

**Files:**
- Create: `tests/test_secret_hygiene.py`
- Modify: `.dockerignore`

**Interfaces:**
- Consumes: repository root resolved as `Path(__file__).resolve().parents[1]`
- Produces: `test_dockerignore_excludes_production_environment_files()` proving the explicit build-context boundary

- [ ] **Step 1: Write the failing Docker exclusion test**

Create `tests/test_secret_hygiene.py` with:

```python
"""Regression tests for repository secret and generated-data hygiene."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _active_rules(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_dockerignore_excludes_production_environment_files() -> None:
    rules = _active_rules(PROJECT_ROOT / ".dockerignore")
    required = {"config.env", ".env", ".env.*", "*.env"}

    assert required <= set(rules), f"Missing Docker secret exclusions: {required - set(rules)}"
    assert "!config.env" not in rules
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
python -m pytest tests/test_secret_hygiene.py::test_dockerignore_excludes_production_environment_files -q
```

Expected: FAIL with `Missing Docker secret exclusions` listing the four absent patterns.

- [ ] **Step 3: Add the minimal Docker exclusions**

Append this dedicated block to `.dockerignore` after the existing cache exclusions and before generated-data exclusions:

```gitignore
# Runtime secrets are injected by Docker Compose env_file, never copied into images.
config.env
.env
.env.*
*.env
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```powershell
python -m pytest tests/test_secret_hygiene.py::test_dockerignore_excludes_production_environment_files -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit the Docker boundary**

```powershell
git add -- .dockerignore tests/test_secret_hygiene.py
git commit -m "fix(security): exclude secrets from backend image"
```

### Task 2: Ignore Generated Evaluation Databases

**Files:**
- Modify: `tests/test_secret_hygiene.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: Git CLI available on `PATH`
- Produces: `test_generated_databases_are_ignored_and_env_template_is_tracked()` covering actual Git matcher behavior and the legitimate developer template

- [ ] **Step 1: Add the failing Git hygiene test**

Add the import and helper below to `tests/test_secret_hygiene.py`:

```python
import subprocess


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
```

Then add:

```python
def test_generated_databases_are_ignored_and_env_template_is_tracked() -> None:
    generated_paths = (
        ".pytest-tmp/example.db",
        "data/evaluation.remote.db",
        "data/evaluation.remote.db-shm",
        "data/evaluation.remote.db-wal",
    )

    for generated_path in generated_paths:
        result = _git("check-ignore", "-q", "--", generated_path)
        assert result.returncode == 0, f"Git does not ignore {generated_path}"

    tracked_template = _git("ls-files", "--error-unmatch", "config.env.example")
    assert tracked_template.returncode == 0, "config.env.example must remain tracked"
```

- [ ] **Step 2: Run the Git hygiene test and verify RED**

Run:

```powershell
python -m pytest tests/test_secret_hygiene.py::test_generated_databases_are_ignored_and_env_template_is_tracked -q
```

Expected: FAIL with `Git does not ignore .pytest-tmp/example.db`.

- [ ] **Step 3: Add the minimal Git exclusions**

Under the existing `# Test artifacts` section in `.gitignore`, add:

```gitignore
.pytest-tmp/
```

Under the existing evaluation database exclusions, add:

```gitignore
data/evaluation.remote.db*
```

- [ ] **Step 4: Run the complete secret-hygiene module and verify GREEN**

Run:

```powershell
python -m pytest tests/test_secret_hygiene.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit the Git hygiene boundary**

```powershell
git add -- .gitignore tests/test_secret_hygiene.py
git commit -m "fix(security): ignore generated evaluation databases"
```

### Task 3: Verify Security Closure and Preserved Runtime Behavior

**Files:**
- Verify: `.dockerignore`
- Verify: `.gitignore`
- Verify: `tests/test_secret_hygiene.py`
- Verify unchanged: `D:\flutterserver\docker-compose.yml`

**Interfaces:**
- Consumes: completed Task 1 and Task 2 commits
- Produces: evidence that secrets are excluded from images, generated databases are ignored, and existing environment-based runtime behavior still passes

- [ ] **Step 1: Inspect the candidate diff for scope**

Run:

```powershell
git diff 881bef9..HEAD -- .dockerignore .gitignore tests/test_secret_hygiene.py
git status --short
```

Expected: only the approved ignore rules and regression tests are changed by the implementation commits; pre-existing unrelated untracked documentation remains untouched.

- [ ] **Step 2: Re-run the original security-boundary checks**

Run:

```powershell
git check-ignore -v -- config.env .pytest-tmp/example.db data/evaluation.remote.db data/evaluation.remote.db-shm data/evaluation.remote.db-wal
```

Expected: every path is matched by `.gitignore`; no credential values are printed.

Run:

```powershell
$rules = Get-Content -Encoding utf8 .dockerignore
@('config.env', '.env', '.env.*', '*.env') | ForEach-Object { if ($_ -notin $rules) { throw "Missing Docker exclusion: $_" } }
```

Expected: exit code `0`.

- [ ] **Step 3: Confirm runtime secret injection remains unchanged**

Run:

```powershell
Select-String -Path '..\docker-compose.yml' -Pattern 'env_file:|\.\/pdftopng\/config\.env'
```

Expected: matches for `env_file:` and `./pdftopng/config.env`.

- [ ] **Step 4: Run focused and compatibility tests**

Run:

```powershell
python -m pytest tests/test_secret_hygiene.py tests/test_dependency_env.py tests/test_supabase_client_lazy_init.py -q
```

Expected: all selected tests pass with no failures.

- [ ] **Step 5: Run the backend suite**

Run:

```powershell
python -m pytest -q
```

Expected: full suite passes. If the suite is blocked by unavailable GPU, model, or external-service dependencies, preserve the focused passing evidence and report the exact blocked tests and errors without weakening the security result.

- [ ] **Step 6: Perform change-aware bypass review**

Re-read `.dockerignore` from top to bottom and confirm no later negation re-includes `config.env`. Re-read `.gitignore` and confirm the new database patterns are not negated later. Confirm `config.env.example` remains tracked and `D:\flutterserver\docker-compose.yml` remains unchanged.

- [ ] **Step 7: Report outcome**

Report `fixed` only if the focused regression tests, ignore checks, Compose preservation check, and existing environment/Supabase tests all pass. State Docker CLI availability separately; static build-context closure is sufficient for this patch, while inspecting previously built or published images remains an operational follow-up.
