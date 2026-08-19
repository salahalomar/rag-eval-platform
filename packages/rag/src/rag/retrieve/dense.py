"""Dense retrieval: cosine top-k over pgvector.

Two details here are load-bearing.

**The query is embedded with bge's instruction prefix, passages without it.** That
asymmetry is baked into `Encoder`, not reimplemented here, so the search path and the
indexing path cannot drift apart.

**Ties break on `chunk_id`.** Two chunks at identical distance would otherwise come back
in whatever order the scan happened to produce, and ENGINEERING.md requires two runs of
`make eval` on one commit to produce identical metrics. Unordered ties are the classic
way that quietly stops being true.
"""

import logging

import psycopg

from rag.config import RetrievalConfig
from rag.index.embed import Encoder, encoder_for, register_vector_types, table_for
from rag.retrieve.types import Candidate
from rag.telemetry import StageTimer

logger = logging.getLogger(__name__)


def apply_ef_search(conn: psycopg.Connection, ef_search: int) -> None:
    """Set HNSW's per-query search breadth for this session.

    Uses `set_config` rather than `SET`, because `SET` is parsed before parameters are
    bound and rejects a placeholder outright. `set_config` is an ordinary function call,
    so the value stays a bound parameter instead of being interpolated into SQL.

    Session-scoped (`is_local=false`): connections here run in autocommit, so a
    transaction-local setting would be discarded before the query that needs it ran.

    Iterative scan is enabled alongside it. Without it, a filtered index scan returns
    whatever survives the filter from one pass of ef_search candidates, which can be far
    fewer than the requested limit once several chunkings share the table. Relaxed order
    is sufficient because the outer query re-sorts by distance anyway.
    """
    conn.execute("SELECT set_config('hnsw.ef_search', %s, false)", (str(ef_search),))
    conn.execute("SELECT set_config('hnsw.iterative_scan', 'relaxed_order', false)")


def search(
    conn: psycopg.Connection,
    query: str,
    config: RetrievalConfig,
    *,
    encoder: Encoder | None = None,
    timer: StageTimer | None = None,
    exact: bool = False,
    top_k: int | None = None,
) -> list[Candidate]:
    """Return the `dense_top_k` nearest chunks to `query` under `config`.

    `exact` disables index scans so the query falls back to a full scan, producing the
    true nearest neighbours. That is not a production path -- it exists so the
    approximate index can be measured against ground truth rather than assumed correct.
    """
    timer = timer or StageTimer()
    register_vector_types(conn)
    encoder = encoder or encoder_for(config.embedding_model)
    limit = top_k if top_k is not None else config.dense_top_k

    with timer.stage("embed_query_ms"):
        vector = encoder.encode_query(query)

    table = table_for(config.embedding_model)
    # Two stages, and the split is load-bearing rather than stylistic.
    #
    # The obvious single-statement form -- join embeddings to chunks and papers, filter,
    # then ORDER BY distance LIMIT k -- cannot be answered from the HNSW index. Postgres
    # materialises every row, hash-joins, and top-N heapsorts. Measured on this corpus:
    #
    #   joins in one statement    Seq Scan on embeddings_384    56.5 ms
    #   inner CTE, then joins     HNSW Index Scan                1.6 ms
    #
    # So the inner query touches nothing but the embedding table, which is the shape
    # pgvector can serve from the index. Metadata joins happen afterwards, against at
    # most `limit` rows, by primary key.
    #
    # Both filters live inside the CTE deliberately. Filtering after the top-k would
    # discard part of the result set whenever more than one chunking is resident -- the
    # exact situation Phase 7's chunk-size sweep creates -- silently returning fewer than
    # `limit` rows. Migration 004 carries chunk_config_sha256 onto the embedding row so
    # the filter can sit here at all.
    #
    # The ::vector casts are required, not cosmetic: a Python list of floats is adapted
    # as double precision[], and while Postgres assignment-casts that into a vector
    # column on INSERT, operator resolution for <=> performs no such implicit cast.
    sql = f"""
        WITH nearest AS (
            SELECT chunk_id, vec <=> %(query_vec)s::vector AS distance
            FROM {table}
            WHERE model = %(model)s AND chunk_config_sha256 = %(chunking)s
            ORDER BY vec <=> %(query_vec)s::vector
            LIMIT %(limit)s
        )
        SELECT c.id, c.paper_id, p.title, c.section_path, c.content,
               c.page_start, c.page_end, c.char_start, c.char_end,
               1 - n.distance AS score
        FROM nearest n
        JOIN chunks c ON c.id = n.chunk_id
        JOIN papers p ON p.id = c.paper_id
        ORDER BY n.distance, c.id
    """

    params = {
        "query_vec": vector,
        "model": config.embedding_model,
        "chunking": config.chunking_sha256(),
        "limit": limit,
    }

    with timer.stage("dense_ms"):
        if exact:
            # Forces a sequential scan, which is exhaustive and therefore exact.
            #
            # The transaction block is not decoration: SET LOCAL is scoped to a
            # transaction, and these connections run in autocommit, so outside a block
            # the setting would be discarded before the query ran -- the "ground truth"
            # would silently come back from the very index it is meant to check.
            with conn.transaction():
                conn.execute("SET LOCAL enable_indexscan = off")
                conn.execute("SET LOCAL enable_bitmapscan = off")
                rows = conn.execute(sql, params).fetchall()
        else:
            apply_ef_search(conn, config.hnsw_ef_search)
            rows = conn.execute(sql, params).fetchall()

    candidates = [
        Candidate(
            chunk_id=int(row[0]),
            paper_id=str(row[1]),
            paper_title=str(row[2]),
            section_path=str(row[3]),
            content=str(row[4]),
            page_start=int(row[5]),
            page_end=int(row[6]),
            char_start=int(row[7]),
            char_end=int(row[8]),
            score=float(row[9]),
            rank=position,
        )
        for position, row in enumerate(rows, start=1)
    ]
    logger.debug("dense search returned %d candidates for %r", len(candidates), query[:60])
    return candidates
