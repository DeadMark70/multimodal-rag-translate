# SECURITY

## Security Baseline

Backend APIs are the source of truth for authentication, authorization boundaries, path safety, and error handling.

## Implemented Controls

1. User-facing routes rely on `Depends(get_current_user_id)`.
2. Request validation and standardized error envelopes run through FastAPI + `core/errors.py`.
3. `doc_id`-based file/resource paths are UUID-typed on current document lifecycle endpoints.
4. Upload-root policies and PDF validation are centralized in `core/uploads.py`.
5. CORS origins are explicit and overrideable through `CORS_ORIGINS`.
6. Runtime code does not support env-flag auth bypass; auth mocking is test-only via dependency overrides.
7. `/health/live` reports process liveness, while `/health/ready` returns `503` until startup completes and during shutdown.
8. Frontend Nginx applies per-client request and connection limits to public API route classes, preserves unbuffered SSE proxying, and returns a stable JSON `429` with `Retry-After`.
9. Backend request audit records contain bounded request IDs and safe route/status/timing/client/user-hash metadata; they exclude query strings, credentials, request/response bodies, prompts, answers, filenames, and source content.
10. The root deployment Compose file binds backend diagnostics to `127.0.0.1:8000` while frontend Nginx reaches `backend:8000` over the Docker network.
11. Both deployment containers use bounded Docker JSON log rotation (`10m` across `5` files).

## Current Limits

1. The system is not a full RBAC or tenant-isolation platform beyond the current auth boundaries.
2. HTTP does not encrypt JWTs, prompts, uploaded documents, citations, or answers. HTTPS termination or a trusted VPN remains the next P0; the controls above do not provide transport confidentiality.
3. Nginx limits are per client address in the current single-gateway deployment; this release does not add distributed rate limiting or a new proxy tier.
4. Safe audit records remain in container logs; there is no centralized audit database, anomaly detector, or log shipping system.
5. SSE recovery is bounded and in-process. Persistent replay, `Last-Event-ID`, and cross-process resume are not implemented.

## Deployment Hardening Priorities

1. Keep auth dependencies explicit on every protected endpoint.
2. Keep OpenAPI security requirements aligned with protected endpoints (including utility/discovery routes).
3. Maintain UUID/path-safety coverage for document and graph maintenance flows.
4. Terminate HTTPS or require a trusted VPN before relying on the deployment for transport confidentiality.
5. Keep backend host binding loopback-only and expose APIs publicly only through frontend Nginx.
6. Review edge limits and safe audit metadata against real traffic without logging user or source content.
7. Keep dependency and env validation checks aligned with real imports and startup requirements.
