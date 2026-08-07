# Docker Secret and Generated-Database Hygiene Design

## Context

The FastAPI backend loads production credentials from `config.env`. Git already ignores that file, and the root Docker Compose configuration injects it at container startup with `env_file`. However, the backend Dockerfile copies the entire build context into `/app`, while `.dockerignore` does not exclude environment files. A built backend image therefore contains `/app/config.env` in addition to receiving the same values at runtime.

The backend worktree also contains generated SQLite files under `.pytest-tmp/` and `data/evaluation.remote.db`. Existing Git ignore rules cover `.pytest_cache/` and selected evaluation database names, but not these paths, leaving them eligible for accidental commits.

## Security Invariants

1. Production environment files must not enter the Docker build context or any image layer.
2. The running backend must continue receiving its credentials through Docker Compose `env_file` and normal process environment variables.
3. The tracked `config.env.example` template must remain available to developers.
4. Generated test databases and the remote evaluation database family must be ignored by Git.
5. Tests must detect removal or weakening of the relevant ignore controls.

## Selected Approach

Apply the smallest repository-native configuration change:

- Add explicit environment-file exclusions to `.dockerignore`: `config.env`, `.env`, `.env.*`, and `*.env`.
- Add `.pytest-tmp/` and `data/evaluation.remote.db*` to `.gitignore`. The wildcard covers SQLite sidecars such as `-shm` and `-wal`.
- Keep the outer `docker-compose.yml` unchanged. Its existing `env_file: ./pdftopng/config.env` remains the runtime secret-injection boundary.
- Do not move, rewrite, print, commit, or rotate any credential as part of this patch.

Moving secrets outside the repository and adopting Docker, Kubernetes, or cloud secret stores remain valid future hardening options, but they require deployment-specific decisions outside this fix.

## Test Design

Add `tests/test_secret_hygiene.py` with focused checks:

1. Read `.dockerignore` and require explicit coverage for `config.env`, `.env`, `.env.*`, and `*.env`; reject a later explicit negation for `config.env`.
2. Run `git check-ignore` separately for `.pytest-tmp/example.db`, `data/evaluation.remote.db`, and representative `-shm` and `-wal` sidecars. This tests Git's real ignore behavior instead of reimplementing its matcher.
3. Use `git ls-files --error-unmatch config.env.example` to prove the developer template remains tracked.

The new tests must be observed failing before the ignore files are changed and passing afterward. Existing environment-loading and Supabase lazy-initialization tests will verify that runtime credential lookup behavior remains intact.

## Validation

Verification will proceed in this order:

1. Inspect the final diff and confirm it contains only the approved ignore rules, regression test, design, and implementation-plan artifacts.
2. Run the focused secret-hygiene test.
3. Recheck `git check-ignore` for `config.env`, generated test databases, and remote evaluation database sidecars.
4. Confirm the outer Compose file still references `./pdftopng/config.env` through `env_file`.
5. Run existing environment-dependency and Supabase lazy-initialization tests.
6. Run the relevant backend test suite if the focused checks pass and the local environment supports it.

## Scope and Non-Goals

This change does not alter application APIs, provider integrations, credential values, Docker Compose topology, frontend configuration, database contents, or deployment-server paths. It does not delete existing generated databases. Credential rotation is required only as a separate operational action when an image containing the current secrets was shared or pushed.
