# SECURITY

## Current Security Baseline

Backend enforces JWT-based user identity checks on user-facing routers.

## Implemented Controls

1. Auth dependency (`get_current_user_id`) is used across user routers.
2. Secrets are loaded from environment; no hardcoded keys expected.
3. Upload flows validate file types and use safer path handling patterns.
4. Public health endpoint remains intentionally minimal.

## Known Gaps

1. Some historical endpoints have used string `doc_id` and need strict UUID coverage.
2. Internal controls are not a full RBAC/tenant isolation system.
3. Security telemetry and anomaly detection are limited.

## Required Upgrade Path For Public Deployment

1. Enforce UUID/path-safety for all file/resource IDs.
2. Add stronger authorization boundaries and audit logging.
3. Add rate limiting and abuse protection.
4. Expand security tests and CI security checks.

