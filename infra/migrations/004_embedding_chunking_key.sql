-- 004_embedding_chunking_key.sql — denormalise the chunking identity onto embeddings.
--
-- Measured cause. Dense search joins embeddings to chunks and papers, and filters on
-- chunks.chunk_config_sha256. With the filter on a joined table, Postgres cannot answer
-- the query from an ordered index scan: it materialises all 6,386 rows, hash-joins, then
-- top-N heapsorts. EXPLAIN confirmed `Seq Scan on embeddings_384` — the HNSW index built
-- in 003 was never used by a single query.
--
--   joins in the same query   Seq Scan       56.5 ms
--   two-stage CTE             HNSW Index      1.6 ms
--
-- The restructure alone recovers the index, but leaves the chunking filter applied
-- *after* the top-k, which is only harmless while one chunking exists. Phase 7's
-- chunk-size sweep keeps several resident at once, and post-filtering would then discard
-- part of every result set and silently return fewer than final_top_k rows.
--
-- Carrying the key on the embedding row lets the filter sit inside the index scan, where
-- pgvector's iterative scan can keep walking the graph until the limit is genuinely met.
--
-- Denormalised rather than joined, which is a real cost: it is redundant with
-- chunks.chunk_config_sha256 and could drift. It cannot drift in practice because chunks
-- are insert-only and embeddings are written from the chunk row in the same transaction.

ALTER TABLE embeddings_384 ADD COLUMN chunk_config_sha256 TEXT;
ALTER TABLE embeddings_768 ADD COLUMN chunk_config_sha256 TEXT;

UPDATE embeddings_384 e
   SET chunk_config_sha256 = c.chunk_config_sha256
  FROM chunks c
 WHERE c.id = e.chunk_id;

UPDATE embeddings_768 e
   SET chunk_config_sha256 = c.chunk_config_sha256
  FROM chunks c
 WHERE c.id = e.chunk_id;

-- Composite, leading with the columns every search filters on, so the planner can use it
-- to pre-select before the vector comparison.
CREATE INDEX embeddings_384_model_chunking_idx
    ON embeddings_384 (model, chunk_config_sha256);
CREATE INDEX embeddings_768_model_chunking_idx
    ON embeddings_768 (model, chunk_config_sha256);

DROP INDEX IF EXISTS embeddings_384_model_idx;
DROP INDEX IF EXISTS embeddings_768_model_idx;
