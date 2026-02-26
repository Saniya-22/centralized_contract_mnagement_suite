-- Setup script for Chat History table
-- Usage: psql -h localhost -p 5432 -U kumarravi -d daedalus -f setupChatHistory.sql

CREATE TABLE IF NOT EXISTS chat_history (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255),
    role VARCHAR(20) NOT NULL,            -- 'user' or 'assistant'
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_history_session ON chat_history (session_id, created_at);
