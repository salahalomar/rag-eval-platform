-- 002_core_schema.sql — papers, chunks, embeddings, query_logs.
--
-- Numbered 002 rather than folded into 001 because migrations are forward-only and 001
-- was already applied at the end of Phase 0. Editing it would be a schema divergence
-- between any database that had already run it and any that had not.

CREATE TABLE papers (
    id              TEXT PRIMARY KEY,          -- arxiv id, e.g. '2401.02385v2'
    title           TEXT NOT NULL,
    authors         TEXT[] NOT NULL,
    abstract        TEXT NOT NULL,
    categories      TEXT[] NOT NULL,
    published_at    DATE NOT NULL,
    pdf_sha256      TEXT NOT NULL,             -- idempotency: skip if unchanged
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
    id                  BIGSERIAL PRIMARY KEY,
    paper_id            TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    ordinal             INT NOT NULL,          -- position within paper, for this chunking
    section_path        TEXT NOT NULL,         -- '3 Method > 3.2 Training'
    content             TEXT NOT NULL,         -- raw chunk text, this is what the user sees
    embed_input         TEXT NOT NULL,         -- content, optionally with contextual header
    token_count         INT NOT NULL,          -- counted with the embedding model's tokenizer
    page_start          INT NOT NULL,          -- for citation highlighting
    page_end            INT NOT NULL,
    char_start          INT NOT NULL,          -- offsets into the paper's reconstructed text
    char_end            INT NOT NULL,
    content_sha256      TEXT NOT NULL,         -- embedding cache key
    chunk_config        JSONB NOT NULL,        -- the chunking params that produced this
    -- Identity of the chunking run, derived from the canonical form of chunk_config.
    --
    -- Not in the original schema sketch, and the reason it is here: the Phase 7 chunk
    -- size sweep requires several chunkings of the same corpus to coexist, which the
    -- unique constraint below already permits. Without a keyed, indexed identity,
    -- "retrieve only the 512-token chunking" becomes a filter over a JSONB blob --
    -- unindexable in practice and easy to get silently wrong. Cheap now, expensive to
    -- retrofit once embeddings reference these rows.
    chunk_config_sha256 TEXT NOT NULL,
    tsv                 TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (paper_id, ordinal, content_sha256, chunk_config_sha256)
);

CREATE INDEX chunks_tsv_idx    ON chunks USING GIN (tsv);
CREATE INDEX chunks_paper_idx  ON chunks (paper_id);
CREATE INDEX chunks_config_idx ON chunks (chunk_config_sha256);

-- One row per (chunk, embedding model) so that swapping the embedding model does not
-- require re-chunking. With a column on chunks instead, an embedding-model ablation
-- would vary two things at once and stop being an experiment.
CREATE TABLE embeddings (
    chunk_id        BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    model           TEXT NOT NULL,
    dim             INT NOT NULL,
    vec             VECTOR(384) NOT NULL,
    PRIMARY KEY (chunk_id, model)
);

-- HNSW index is deliberately deferred to Phase 2, where it is benchmarked with an
-- ef_search sweep against exact-scan ground truth. Building it before there is anything
-- to measure would be an unmeasured claim.

-- Every query, for observability and for latency percentiles.
CREATE TABLE query_logs (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    question        TEXT NOT NULL,
    config          JSONB NOT NULL,            -- full RetrievalConfig
    retrieved_ids   BIGINT[] NOT NULL,
    stage_timings   JSONB NOT NULL,            -- {"dense_ms":41,"lexical_ms":12,...}
    input_tokens    INT,
    output_tokens   INT,
    cost_usd        NUMERIC(10,6),
    refused         BOOLEAN NOT NULL DEFAULT false
);

CREATE INDEX query_logs_created_idx ON query_logs (created_at DESC);
