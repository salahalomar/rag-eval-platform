"""Lexical retrieval over Postgres full-text search.

**This is not BM25, and it is not called BM25 anywhere.** It is `ts_rank_cd`, a
coverage-density ranking: it scores by how many query lexemes a document contains and
how tightly clustered they are, with no document-frequency term and no length saturation
curve. That makes it a genuinely different function from Okapi BM25, not an
implementation of it. It works well and it keeps everything in one datastore, which is
why it is the v1 lexical arm — but calling it BM25 would be a claim the code does not
support. Phase 6 adds a real BM25 arm and the ablation table reports both.

Query semantics are OR, not AND, and that is a deliberate choice worth defending.
`plainto_tsquery` conjoins every term, so a natural-language question would demand that
all seven of its content words appear inside one 512-token chunk. On this corpus that
returns almost nothing. Lexical search is here as the recall-oriented complement to a
dense arm that already handles semantic similarity, so the query is lexed with the same
dictionary the index used and its lexemes are OR-ed, leaving `ts_rank_cd` to rank by how
many of them each chunk actually covers.
"""

import logging

import psycopg

from rag.config import RetrievalConfig
from rag.retrieve.types import Candidate
from rag.telemetry import StageTimer

logger = logging.getLogger(__name__)

# Named here so the README and the code cannot drift. If this ever changes, the honest
# naming in the docs has to change with it.
RANKING_FUNCTION = "ts_rank_cd"

# The text search configuration. Must match the one in the `tsv` generated column, or the
# query is lexed by different rules than the index was built with and matching degrades
# silently -- stemming 'training' to 'train' on one side only, for instance.
TEXT_SEARCH_CONFIG = "english"

# ts_rank_cd normalisation flag 32 maps the raw rank r to r/(r+1), bounding it to [0,1).
# Strictly monotonic, so ordering is untouched; it exists so the score is on a comparable
# scale to the dense arm's cosine similarity when `score_floor` starts gating in Phase 4.
# An unbounded rank would make any fixed floor meaningless.
RANK_NORMALISATION = 32


def search(
    conn: psycopg.Connection,
    query: str,
    config: RetrievalConfig,
    *,
    timer: StageTimer | None = None,
    top_k: int | None = None,
) -> list[Candidate]:
    """Return the `lexical_top_k` best-matching chunks for `query`.

    A query of only stopwords, punctuation or whitespace yields no lexemes and therefore
    no results, rather than an error. That path is reached in practice -- "what about
    it?" lexes to nothing -- and an exception there would take down a request that dense
    search could still have answered.
    """
    timer = timer or StageTimer()
    limit = top_k if top_k is not None else config.lexical_top_k

    # The `terms` CTE lexes the query with the same dictionary as the index and quotes
    # each lexeme, producing `'learn' | 'rate' | 'warmup'`. Quoting matters: lexemes can
    # contain apostrophes and colons, which are tsquery operators.
    #
    # `q` is empty when the query produced no lexemes, because string_agg over zero rows
    # returns NULL. Joining through it means to_tsquery is never called with an empty
    # string -- which raises a syntax error rather than matching nothing. A WHERE guard
    # would not be enough, since Postgres does not promise to short-circuit AND.
    sql = f"""
        WITH terms AS (
            SELECT string_agg(quote_literal(lexeme), ' | ') AS expression
            FROM unnest(tsvector_to_array(to_tsvector(%(ts_config)s, %(query)s))) AS lexeme
        ),
        q AS (
            SELECT to_tsquery(%(ts_config)s, expression) AS tsquery
            FROM terms
            WHERE expression IS NOT NULL
        )
        SELECT c.id, c.paper_id, p.title, c.section_path, c.content,
               c.page_start, c.page_end, c.char_start, c.char_end,
               ts_rank_cd(c.tsv, q.tsquery, {RANK_NORMALISATION}) AS score
        FROM chunks c
        JOIN q ON TRUE
        JOIN papers p ON p.id = c.paper_id
        WHERE c.chunk_config_sha256 = %(chunking)s
          AND c.tsv @@ q.tsquery
        ORDER BY score DESC, c.id
        LIMIT %(limit)s
    """

    params = {
        "ts_config": TEXT_SEARCH_CONFIG,
        "query": query,
        "chunking": config.chunking_sha256(),
        "limit": limit,
    }

    with timer.stage("lexical_ms"):
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
    logger.debug("lexical search returned %d candidates for %r", len(candidates), query[:60])
    return candidates


def lexemes(conn: psycopg.Connection, query: str) -> list[str]:
    """The lexemes `query` reduces to under the index's dictionary.

    Exposed for inspection rather than for retrieval: when the lexical arm returns
    nothing surprising, the first question is always what the query actually became after
    stemming and stopword removal.
    """
    row = conn.execute(
        "SELECT tsvector_to_array(to_tsvector(%s, %s))", (TEXT_SEARCH_CONFIG, query)
    ).fetchone()
    return [] if row is None or row[0] is None else [str(x) for x in row[0]]
