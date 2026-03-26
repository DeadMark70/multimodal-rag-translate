# Conversation API

## User Outcomes

- List conversations for the current user.
- Create, read, update, and delete conversation records.
- Append messages to an existing conversation.

## Acceptance Notes

- Conversation metadata is the persistence seam for frontend restore behavior.
- Message persistence and conversation metadata updates must stay explicit and typed.
