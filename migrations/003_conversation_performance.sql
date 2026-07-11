-- Keep local/reference conversation schema aligned with the Supabase index.
CREATE INDEX IF NOT EXISTS idx_conversations_user_updated_id
ON conversations (user_id, updated_at DESC, id DESC);
