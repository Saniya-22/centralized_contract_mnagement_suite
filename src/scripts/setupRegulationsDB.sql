-- Setup script for Government Regulations RAG Database
-- Requires PostgreSQL with pgvector extension
--
-- Usage: psql -U your_user -d your_db -f setupRegulationsDB.sql
-- psql -h localhost -p 5432 -U kumarravi -d daedalus -f setupRegulationsDB.sql

-- Enable pgvector extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- Dense embeddings table (for semantic search)
-- Uses OpenAI text-embedding-ada-002 (1536 dimensions)
CREATE TABLE IF NOT EXISTS embeddings_dense (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    text TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(1536),
    search_vector tsvector,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS idx_embeddings_dense_vector
ON embeddings_dense
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create index for namespace filtering
CREATE INDEX IF NOT EXISTS idx_embeddings_dense_namespace
ON embeddings_dense (namespace);

-- Create GIN index for JSONB metadata queries
CREATE INDEX IF NOT EXISTS idx_embeddings_dense_metadata
ON embeddings_dense
USING GIN (metadata);

-- Create GIN index for full-text search
CREATE INDEX IF NOT EXISTS idx_embeddings_dense_search_vector
ON embeddings_dense
USING GIN (search_vector);

-- Trigger to auto-populate search_vector on insert/update
CREATE OR REPLACE FUNCTION embeddings_dense_search_vector_trigger()
RETURNS trigger AS $$
BEGIN
    NEW.search_vector := to_tsvector('english', COALESCE(NEW.text, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_embeddings_dense_search_vector ON embeddings_dense;
CREATE TRIGGER trg_embeddings_dense_search_vector
    BEFORE INSERT OR UPDATE OF text ON embeddings_dense
    FOR EACH ROW
    EXECUTE FUNCTION embeddings_dense_search_vector_trigger();

-- Sparse embeddings table (for BM25 search)
CREATE TABLE IF NOT EXISTS embeddings_sparse (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    text TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding JSONB NOT NULL,  -- Format: {"indices": [...], "values": [...]}
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for namespace filtering
CREATE INDEX IF NOT EXISTS idx_embeddings_sparse_namespace
ON embeddings_sparse (namespace);

-- Create GIN index for JSONB metadata queries
CREATE INDEX IF NOT EXISTS idx_embeddings_sparse_metadata
ON embeddings_sparse
USING GIN (metadata);

-- Create GIN index for sparse embedding indices (for efficient BM25 lookup)
CREATE INDEX IF NOT EXISTS idx_embeddings_sparse_embedding
ON embeddings_sparse
USING GIN (embedding);

-- Helpful views for monitoring

-- View to see document counts by namespace and source
CREATE OR REPLACE VIEW v_regulation_stats AS
SELECT
    namespace,
    metadata->>'source' as source,
    metadata->>'part' as part,
    COUNT(*) as chunk_count
FROM embeddings_dense
WHERE namespace = 'public-regulations'
GROUP BY namespace, metadata->>'source', metadata->>'part'
ORDER BY source, part;

-- View to see recent ingestion activity
CREATE OR REPLACE VIEW v_recent_ingestions AS
SELECT
    namespace,
    metadata->>'source' as source,
    metadata->>'filename' as filename,
    COUNT(*) as chunks,
    MAX(created_at) as last_updated
FROM embeddings_dense
GROUP BY namespace, metadata->>'source', metadata->>'filename'
ORDER BY last_updated DESC
LIMIT 20;

-- Grant permissions (adjust role name as needed)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON embeddings_dense TO your_app_role;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON embeddings_sparse TO your_app_role;
-- GRANT SELECT ON v_regulation_stats TO your_app_role;
-- GRANT SELECT ON v_recent_ingestions TO your_app_role;

-- Sample queries for verification:

-- Check document counts after ingestion
-- SELECT * FROM v_regulation_stats;

-- Check recent ingestion activity
-- SELECT * FROM v_recent_ingestions;

-- Test semantic search (replace with actual embedding)
-- SELECT id, namespace, metadata->>'source' as source,
--        1 - (embedding <=> '[0.1, 0.2, ...]'::vector) as similarity
-- FROM embeddings_dense
-- WHERE namespace = 'public-regulations'
-- ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
-- LIMIT 5;

COMMENT ON TABLE embeddings_dense IS 'Dense vector embeddings for semantic search using OpenAI text-embedding-ada-002';
COMMENT ON TABLE embeddings_sparse IS 'Sparse BM25 embeddings for keyword-based search';
