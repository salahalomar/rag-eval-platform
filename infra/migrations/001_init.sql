-- 001_init.sql — extensions only.
--
-- The core schema (papers, chunks, embeddings, query_logs) lands in 002 during Phase 1.
-- It is deliberately not folded into this file: 001 is applied the moment Phase 0 is
-- reviewed, and migrations here are forward-only, so this file is never edited again.
--
-- vector   — pgvector, for the embeddings table and its HNSW index (Phase 2).
-- pg_trgm  — trigram matching, for fuzzy title/author lookup alongside the tsvector
--            lexical arm (Phase 3). Cheap to enable now, awkward to add later once
--            the database has grown.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
