# Conversations (conversations) Technical Documentation

## 1. Technical Implementation Details

### Core Logic
The `conversations` module provides a RESTful API for managing user chat history. It abstracts the storage of conversations and individual messages, allowing the frontend to maintain persistent chat sessions.

1.  **Conversation Management (`router.py`)**:
    -   **CRUD Operations**: Create, Read, Update, Delete for `conversations` table.
    -   **Ownership**: All queries are scoped by `user_id` to ensure data privacy.
    -   **Ordering**: Conversations are returned sorted by `updated_at` (descending) to show the most recent chats first.

2.  **Message Management**:
    -   **Storage**: Messages are stored in a separate `messages` table linked by foreign key to `conversations`.
    -   **Retrieval**: Fetching a conversation detail automatically joins and retrieves all associated messages in chronological order.
    -   **Cascade Delete**: Deleting a conversation relies on database-level cascading to remove associated messages (implied by the delete logic comment).

3.  **Data Models**:
    -   **Conversation**: Holds metadata (`title`, `type`, `metadata`) and timestamps.
    -   **Message**: Holds `role` (user/assistant), `content`, and `metadata`.

### Algorithms
-   **Direct Database Access**: Uses `supabase_client` to perform SQL-like operations on the Postgres backend via PostgREST. No complex in-memory algorithms are used; logic relies on efficient database querying.

## 2. Codebase Map

| File Path | Responsibility |
| :--- | :--- |
| `conversations/router.py` | API endpoints for listing, creating, retrieving, updating, and deleting conversations and messages. |
| `conversations/schemas.py` | Pydantic models for request validation and response serialization. |

## 3. Usage Guide

**⚠️ IMPORTANT: All commands must be executed within the project's virtual environment (`.venv`).**

### API Usage
**List Conversations:**
`GET /api/conversations`

**Create Conversation:**
`POST /api/conversations`
-   **Body**: `{"title": "New Chat", "type": "chat"}`

**Get Details (with messages):**
`GET /api/conversations/{conversation_id}`

**Add Message:**
`POST /api/conversations/{conversation_id}/messages`
-   **Body**: `{"role": "user", "content": "Hello"}`

### Standalone Testing
Integration testing requires a running Supabase instance.
To test Pydantic models or mock the router:

```bash
# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Example: Run specific tests if available
# pytest tests/test_conversations_api.py
```

## 4. Dependencies

### Internal Modules
-   `core`: Authentication (`get_current_user_id`).
-   `supabase_client`: Database connection.

### External Libraries
-   `fastapi`: API framework.
-   `pydantic`: Schema definition.
-   `postgrest`: Error handling for database calls.
