# Runtime And Router Boundaries

## Purpose

Describe how the FastAPI app is assembled and where cross-cutting concerns belong.

## Runtime Assembly

- Entry: `main.py`
- App factory: `core/app_factory.py`
- Shared core seams:
  - auth
  - errors
  - providers
  - uploads
  - Supabase repository helpers

## Router Rules

- Routers register under fixed prefixes and should not import other routers.
- Shared behavior belongs in service/helper modules.
- User-facing routes should keep auth dependency explicit.

## Middleware And Errors

- Request-id middleware attaches `X-Request-Id`.
- Global handlers normalize HTTP, validation, app, and unexpected errors into one envelope.
