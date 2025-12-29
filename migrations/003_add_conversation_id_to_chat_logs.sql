-- Migration: 003_add_conversation_id_to_chat_logs.sql
-- Description: Add conversation_id and role columns to chat_logs
-- Date: 2025-12-28

-- Add conversation_id column (nullable for backward compatibility)
ALTER TABLE chat_logs 
ADD COLUMN IF NOT EXISTS conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE;

-- Add role column to distinguish user/assistant messages
ALTER TABLE chat_logs 
ADD COLUMN IF NOT EXISTS role TEXT DEFAULT 'user' CHECK (role IN ('user', 'assistant'));

-- Create index for conversation_id lookups
CREATE INDEX IF NOT EXISTS idx_chat_logs_conversation_id ON chat_logs(conversation_id);

-- Optional: Update existing records to have role based on content
-- (Run this manually if needed)
-- UPDATE chat_logs SET role = 'user' WHERE question IS NOT NULL;
-- UPDATE chat_logs SET role = 'assistant' WHERE answer IS NOT NULL AND question IS NULL;
