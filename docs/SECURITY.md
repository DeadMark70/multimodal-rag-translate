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

## Current Limits

1. The system is not a full RBAC or tenant-isolation platform beyond the current auth boundaries.
2. Security telemetry and anomaly detection are still limited.
3. Public deployment still needs broader abuse protection and audit visibility.

## Deployment Hardening Priorities

1. Keep auth dependencies explicit on every protected endpoint.
2. Keep OpenAPI security requirements aligned with protected endpoints (including utility/discovery routes).
3. Maintain UUID/path-safety coverage for document and graph maintenance flows.
4. Add stronger rate limiting, audit logging, and abuse protection for public exposure.
5. Keep dependency and env validation checks aligned with real imports and startup requirements.
