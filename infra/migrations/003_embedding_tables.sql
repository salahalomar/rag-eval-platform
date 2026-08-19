-- 003_embedding_tables.sql — one embeddings table per vector dimension, plus HNSW.
--
-- Why a table per dimension rather than one table with several vector columns:
-- pgvector fixes dimensionality on the column, and an HNSW index requires it, so
-- supporting a 768-d model for the Phase 7 ablation needs either extra columns or extra
-- tables. Extra columns mean a NULL on every row, a CASE in every query, partial indexes
-- and a table twice as wide for no benefit. Separate tables keep exactly one code path,
-- parameterised by a table name that a model→dimension registry resolves.
--
-- embeddings_768 is created empty and unused today. It costs a few kilobytes and means
-- the bge-base ablation arm needs no schema change at the point it is finally run.
--
-- Adding a 1024-d model later is this file again with one number changed, plus one line
-- in rag.index.embed.MODEL_DIMENSIONS.

ALTER TABLE embeddings RENAME TO embeddings_384;

-- The embedding cache key.
--
-- The plan specifies caching by `content_sha256`, and that is wrong whenever contextual
-- headers are enabled: the string handed to the model is `embed_input`, which prefixes
-- the paper title and section path onto the content. Two chunks can therefore share a
-- `content_sha256` while embedding to genuinely different vectors, and a cache keyed on
-- content would hand back a vector computed under a different header — a silent,
-- invisible corruption of exactly the ablation arm that header is there to test.
--
-- Computed by the application, exactly as content_sha256 already is. A Postgres
-- generated column was the first choice and is not possible: sha256() takes bytea, and
-- the only text-to-bytea conversion, convert_to(), is STABLE rather than IMMUTABLE
-- because its result depends on the server encoding -- which generated columns forbid.
-- md5() is immutable and would work, but computing one digest in Python and the other in
-- SQL, with different algorithms, is a worse thing to explain than this comment.
--
-- Nullable because migrations are forward-only and must be safe on a database that
-- already holds chunks. Rows written before this migration carry NULL and need
-- re-ingesting; every row written after it is populated.
ALTER TABLE chunks ADD COLUMN embed_input_sha256 TEXT;

CREATE INDEX chunks_embed_input_sha_idx ON chunks (embed_input_sha256);

CREATE TABLE embeddings_768 (
    chunk_id        BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    model           TEXT NOT NULL,
    dim             INT NOT NULL,
    vec             VECTOR(768) NOT NULL,
    PRIMARY KEY (chunk_id, model)
);

-- m and ef_construction are the values from the plan. They are build-time parameters:
-- changing them requires rebuilding the index, unlike ef_search which is a per-session
-- knob and is swept in `rag bench-index`.
--
-- Vectors are stored L2-normalised, so cosine distance and inner product rank
-- identically; cosine ops are used anyway because the stored form is then a detail
-- rather than a correctness requirement of every query.
CREATE INDEX embeddings_384_hnsw_idx ON embeddings_384
    USING hnsw (vec vector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE INDEX embeddings_768_hnsw_idx ON embeddings_768
    USING hnsw (vec vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Lets the embedder find chunks that still need a vector for a given model without
-- scanning the whole table.
CREATE INDEX embeddings_384_model_idx ON embeddings_384 (model);
CREATE INDEX embeddings_768_model_idx ON embeddings_768 (model);
