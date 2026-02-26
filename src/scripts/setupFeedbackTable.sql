-- Setup script for Chat Feedback table
-- Usage: psql -h localhost -p 5432 -U kumarravi -d daedalus -f setupFeedbackTable.sql

CREATE TABLE IF NOT EXISTS chat_feedback (
    id SERIAL PRIMARY KEY,
    rating VARCHAR(10) NOT NULL,          -- 'up' or 'down'
    message TEXT,                          -- user's typed feedback
    screenshot TEXT,                       -- base64 data URI
    query TEXT,                            -- the user's original question
    response TEXT,                         -- the AI response text
    agent_name VARCHAR(100),
    session_id VARCHAR(255),
    user_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_feedback_rating ON chat_feedback (rating);
CREATE INDEX IF NOT EXISTS idx_chat_feedback_created_at ON chat_feedback (created_at);
