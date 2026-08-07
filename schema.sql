-- Reference schema for the weather RAG app on Lakebase (Postgres 17 + pgvector).
-- Run this manually in the Databricks SQL editor before the first sync/embed.

-- pgvector must be enabled once per database.
CREATE EXTENSION IF NOT EXISTS vector;

-- Normalized weather documents. One row per NWS alert or forecast period.
CREATE TABLE IF NOT EXISTS weather_documents (
    id             TEXT PRIMARY KEY,
    location       TEXT NOT NULL,
    source_type    TEXT NOT NULL CHECK (source_type IN ('alert', 'forecast')),
    headline       TEXT,
    narrative_text TEXT NOT NULL,
    issued_at      TIMESTAMPTZ,
    payload        JSONB,
    synced_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Chunk-level embeddings. all-MiniLM-L6-v2 emits 384-dimensional vectors.
CREATE TABLE IF NOT EXISTS weather_embeddings (
    id          BIGSERIAL PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES weather_documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chunk_text  TEXT NOT NULL,
    embedding   vector(384) NOT NULL,
    model_name  TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

-- HNSW index for approximate nearest-neighbor search under cosine distance.
CREATE INDEX IF NOT EXISTS weather_embeddings_embedding_hnsw
    ON weather_embeddings
    USING hnsw (embedding vector_cosine_ops);
