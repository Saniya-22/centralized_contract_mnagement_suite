-- User feedback table for thumbs up/down (query_id and user_id as UUIDs).
-- Also created automatically at app startup via VectorQueries.init_user_feedback_table().
-- Usage: psql -h HOST -p PORT -U USER -d DB -f scripts/setup_user_feedback.sql

CREATE TABLE IF NOT EXISTS user_feedback (
    id                  SERIAL PRIMARY KEY,
    user_id             UUID NOT NULL,
    query_id            UUID NOT NULL,
    feedback_response   TEXT NOT NULL CHECK (feedback_response IN ('good', 'bad')),
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_user_feedback_user_id ON user_feedback (user_id);
CREATE INDEX IF NOT EXISTS idx_user_feedback_query_id ON user_feedback (query_id);
CREATE INDEX IF NOT EXISTS idx_user_feedback_created_at ON user_feedback (created_at);
