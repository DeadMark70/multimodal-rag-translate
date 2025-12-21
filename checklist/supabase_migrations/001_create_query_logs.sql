-- ============================================
-- Query Logs Table for 3R Dashboard Statistics
-- ============================================
-- Purpose: Store RAG query history for analytics
-- Retention: Permanent (per user decision)

-- Create query_logs table
CREATE TABLE IF NOT EXISTS query_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    question TEXT NOT NULL,
    answer TEXT,
    doc_ids TEXT[],
    faithfulness TEXT CHECK (faithfulness IN ('grounded', 'hallucinated', 'uncertain')),
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    has_history BOOLEAN DEFAULT false,
    response_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add comment
COMMENT ON TABLE query_logs IS 'Stores RAG query history for dashboard analytics';

-- Index: User's queries ordered by time (for user history)
CREATE INDEX IF NOT EXISTS idx_query_logs_user_time 
ON query_logs(user_id, created_at DESC);

-- Index: Dashboard stats (recent queries)
CREATE INDEX IF NOT EXISTS idx_query_logs_created_at 
ON query_logs(created_at DESC);

-- Index: Faithfulness analysis
CREATE INDEX IF NOT EXISTS idx_query_logs_faithfulness 
ON query_logs(faithfulness);

-- ============================================
-- Row Level Security (RLS)
-- ============================================
-- Enable RLS
ALTER TABLE query_logs ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see their own logs
CREATE POLICY "Users can view own query logs"
ON query_logs
FOR SELECT
USING (auth.uid() = user_id);

-- Policy: Users can insert their own logs
CREATE POLICY "Users can insert own query logs"
ON query_logs
FOR INSERT
WITH CHECK (auth.uid() = user_id);

-- ============================================
-- Usage Examples
-- ============================================
-- 
-- Insert new log:
-- INSERT INTO query_logs (user_id, question, answer, faithfulness, confidence)
-- VALUES ('user-uuid', 'question text', 'answer text', 'grounded', 0.85);
--
-- Get user's recent queries:
-- SELECT * FROM query_logs 
-- WHERE user_id = 'user-uuid' 
-- ORDER BY created_at DESC 
-- LIMIT 50;
--
-- Get accuracy stats:
-- SELECT 
--   faithfulness,
--   COUNT(*) as count,
--   AVG(confidence) as avg_confidence
-- FROM query_logs
-- WHERE created_at > NOW() - INTERVAL '7 days'
-- GROUP BY faithfulness;
