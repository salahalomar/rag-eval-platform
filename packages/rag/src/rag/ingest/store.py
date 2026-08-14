"""Persistence for ingested papers and chunks.

Idempotency is enforced on two independent axes, because there are two different reasons
to re-run an ingest. A paper whose PDF is unchanged should not be re-parsed; but a paper
whose *chunking configuration* has changed must be re-chunked even though its PDF is
byte-identical. Keying only on `pdf_sha256` would silently skip the second case and make
the Phase 7 chunk-size sweep quietly compare a config against itself.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import psycopg

from rag.config import RetrievalConfig
from rag.ingest.arxiv import PaperMetadata
from rag.ingest.chunk import Chunk

logger = logging.getLogger(__name__)

# Below this, a "chunk" is usually a stray header, a page number or an equation fragment
# rather than anything answerable. Reported by `rag stats` as a corpus health signal.
SUSPICIOUSLY_SHORT_TOKENS = 50


@dataclass(frozen=True, slots=True)
class CorpusStats:
    """What is actually in the database, for `rag stats`."""

    papers: int
    chunks: int
    chunkings: int
    mean_chunks_per_paper: float
    token_percentiles: dict[str, float]
    fallback_papers: int
    fallback_rate: float
    short_chunks: int
    distinct_sections: int
    pages_covered: int


def paper_pdf_sha256(conn: psycopg.Connection, paper_id: str) -> str | None:
    """The digest recorded for an already-ingested paper, or None if it is new."""
    row = conn.execute("SELECT pdf_sha256 FROM papers WHERE id = %s", (paper_id,)).fetchone()
    return None if row is None else str(row[0])


def chunks_exist(conn: psycopg.Connection, paper_id: str, chunk_config_sha256: str) -> bool:
    """Whether this paper has already been chunked under this exact configuration."""
    row = conn.execute(
        "SELECT 1 FROM chunks WHERE paper_id = %s AND chunk_config_sha256 = %s LIMIT 1",
        (paper_id, chunk_config_sha256),
    ).fetchone()
    return row is not None


def upsert_paper(conn: psycopg.Connection, metadata: PaperMetadata, pdf_sha256: str) -> None:
    """Insert or refresh a paper row.

    Updates on conflict rather than doing nothing so that re-ingesting a paper whose
    arXiv metadata was corrected picks up the correction.
    """
    conn.execute(
        """
        INSERT INTO papers (id, title, authors, abstract, categories, published_at, pdf_sha256)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            title        = EXCLUDED.title,
            authors      = EXCLUDED.authors,
            abstract     = EXCLUDED.abstract,
            categories   = EXCLUDED.categories,
            published_at = EXCLUDED.published_at,
            pdf_sha256   = EXCLUDED.pdf_sha256
        """,
        (
            metadata.id,
            metadata.title,
            list(metadata.authors),
            metadata.abstract,
            list(metadata.categories),
            metadata.published_at,
            pdf_sha256,
        ),
    )


def insert_chunks(
    conn: psycopg.Connection,
    paper_id: str,
    chunks: Sequence[Chunk],
    config: RetrievalConfig,
) -> int:
    """Persist chunks, skipping any already present. Returns rows actually inserted.

    The returned count is what makes the idempotency claim checkable: a second ingest of
    an unchanged corpus must return zero, and `rag ingest` prints it.
    """
    if not chunks:
        return 0

    chunk_config = json.dumps(config.chunking_params(), sort_keys=True)
    chunk_config_sha256 = config.chunking_sha256()

    with conn.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO chunks (
                paper_id, ordinal, section_path, content, embed_input, token_count,
                page_start, page_end, char_start, char_end, content_sha256,
                chunk_config, chunk_config_sha256
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (paper_id, ordinal, content_sha256, chunk_config_sha256) DO NOTHING
            """,
            [
                (
                    paper_id,
                    chunk.ordinal,
                    chunk.section_path,
                    chunk.content,
                    chunk.embed_input,
                    chunk.token_count,
                    chunk.page_start,
                    chunk.page_end,
                    chunk.char_start,
                    chunk.char_end,
                    chunk.content_sha256,
                    chunk_config,
                    chunk_config_sha256,
                )
                for chunk in chunks
            ],
        )
        return max(0, cursor.rowcount)


def corpus_stats(conn: psycopg.Connection) -> CorpusStats:
    """Summarise the corpus.

    The section-detection fallback rate is derived rather than stored: a paper whose
    chunks all sit in the literal section 'Body' is one where heading detection gave up.
    Deriving it keeps the schema smaller and cannot drift out of sync with reality.
    """
    papers = _scalar_int(conn, "SELECT count(*) FROM papers")
    chunks = _scalar_int(conn, "SELECT count(*) FROM chunks")
    chunkings = _scalar_int(conn, "SELECT count(DISTINCT chunk_config_sha256) FROM chunks")
    distinct_sections = _scalar_int(conn, "SELECT count(DISTINCT section_path) FROM chunks")
    short_chunks = _scalar_int(
        conn, "SELECT count(*) FROM chunks WHERE token_count < %s", (SUSPICIOUSLY_SHORT_TOKENS,)
    )
    pages_covered = _scalar_int(
        conn, "SELECT coalesce(sum(page_end - page_start + 1), 0) FROM chunks"
    )
    fallback_papers = _scalar_int(
        conn,
        """
        SELECT count(*) FROM (
            SELECT paper_id FROM chunks
            GROUP BY paper_id
            HAVING bool_and(section_path = 'Body')
        ) AS fell_back
        """,
    )

    percentiles: dict[str, float] = {}
    if chunks:
        row = conn.execute(
            """
            SELECT
                min(token_count),
                percentile_cont(0.05) WITHIN GROUP (ORDER BY token_count),
                percentile_cont(0.25) WITHIN GROUP (ORDER BY token_count),
                percentile_cont(0.50) WITHIN GROUP (ORDER BY token_count),
                percentile_cont(0.75) WITHIN GROUP (ORDER BY token_count),
                percentile_cont(0.95) WITHIN GROUP (ORDER BY token_count),
                max(token_count)
            FROM chunks
            """
        ).fetchone()
        if row is not None:
            labels = ("min", "p05", "p25", "p50", "p75", "p95", "max")
            percentiles = {label: float(value) for label, value in zip(labels, row, strict=True)}

    papers_with_chunks = _scalar_int(conn, "SELECT count(DISTINCT paper_id) FROM chunks")
    return CorpusStats(
        papers=papers,
        chunks=chunks,
        chunkings=chunkings,
        mean_chunks_per_paper=chunks / papers_with_chunks if papers_with_chunks else 0.0,
        token_percentiles=percentiles,
        fallback_papers=fallback_papers,
        fallback_rate=fallback_papers / papers_with_chunks if papers_with_chunks else 0.0,
        short_chunks=short_chunks,
        distinct_sections=distinct_sections,
        pages_covered=pages_covered,
    )


def sample_chunks(
    conn: psycopg.Connection, limit: int, *, seed: float = 0.42
) -> list[dict[str, Any]]:
    """A reproducible random sample of chunks, for eyeballing ingest quality.

    Seeded so that `rag stats --sample` shows the same rows on the same corpus; an
    unseeded sample makes it impossible to tell whether a change fixed something or the
    dice simply landed differently.
    """
    conn.execute("SELECT setseed(%s)", (seed,))
    rows = conn.execute(
        """
        SELECT c.id, c.paper_id, p.title, c.section_path, c.token_count,
               c.page_start, c.page_end, c.content
        FROM chunks c JOIN papers p ON p.id = c.paper_id
        ORDER BY random()
        LIMIT %s
        """,
        (limit,),
    ).fetchall()
    keys = (
        "id",
        "paper_id",
        "title",
        "section_path",
        "token_count",
        "page_start",
        "page_end",
        "content",
    )
    return [dict(zip(keys, row, strict=True)) for row in rows]


def _scalar_int(conn: psycopg.Connection, sql: str, params: tuple[object, ...] = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return 0 if row is None or row[0] is None else int(row[0])
