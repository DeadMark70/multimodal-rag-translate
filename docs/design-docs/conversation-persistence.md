# Conversation Persistence

## Purpose

Describe the backend persistence model for conversations and chat messages.

## Ownership

- Router: `conversations/router.py`
- Schemas: `conversations/schemas.py`
- Service/repository seams: `conversations/service.py`, `conversations/repository.py`

## Rules

- Conversations and messages are stored separately from UI-local state.
- Metadata is the compatibility seam for mode snapshots and Deep Research restore data.
- Backend contracts should stay explicit so frontend restore logic can avoid guessing.
