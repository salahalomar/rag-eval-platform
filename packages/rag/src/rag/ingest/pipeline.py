"""Orchestration: fetch -> parse -> section -> chunk -> persist.

Commits per paper rather than per run. A 150-paper ingest takes ten minutes at arXiv's
requested rate limit, and a single transaction around all of it means one malformed PDF
at paper 140 discards the other 139. Per-paper commits make the whole operation
resumable, which is the only reason the acceptance criterion "a second run inserts zero
rows" is cheap to satisfy after a partial failure.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import psycopg

from rag.config import RetrievalConfig
from rag.ingest import arxiv, store
from rag.ingest.chunk import chunk_document
from rag.ingest.parse import parse_pdf
from rag.ingest.sections import detect_sections
from rag.ingest.tokenization import TokenCounter, token_counter_for

logger = logging.getLogger(__name__)

DEFAULT_CATEGORIES = ("cs.LG", "cs.CL")
DEFAULT_PDF_CACHE = Path("data/pdfs")


@dataclass(slots=True)
class IngestReport:
    """Counts at each stage, so a run can be judged without reading the logs."""

    papers_seen: int = 0
    papers_skipped_unchanged: int = 0
    papers_ingested: int = 0
    papers_failed: int = 0
    chunks_inserted: int = 0
    section_fallbacks: int = 0
    two_column_papers: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)

    def as_lines(self) -> list[str]:
        """Human-readable summary, printed at the end of `rag ingest`."""
        lines = [
            f"  papers seen            {self.papers_seen}",
            f"  ingested               {self.papers_ingested}",
            f"  skipped (unchanged)    {self.papers_skipped_unchanged}",
            f"  failed                 {self.papers_failed}",
            f"  chunks inserted        {self.chunks_inserted}",
            f"  section fallbacks      {self.section_fallbacks}",
            f"  two-column papers      {self.two_column_papers}",
        ]
        lines.extend(f"    ! {paper_id}: {reason}" for paper_id, reason in self.failures[:10])
        return lines


def ingest_paper(
    conn: psycopg.Connection,
    metadata: arxiv.PaperMetadata,
    pdf_path: Path,
    config: RetrievalConfig,
    counter: TokenCounter,
    report: IngestReport,
) -> None:
    """Parse, chunk and persist one paper. Commits on success, rolls back on failure."""
    pdf_sha256 = arxiv.sha256_of(pdf_path)
    chunk_config_sha256 = config.chunking_sha256()

    unchanged = store.paper_pdf_sha256(conn, metadata.id) == pdf_sha256
    if unchanged and store.chunks_exist(conn, metadata.id, chunk_config_sha256):
        report.papers_skipped_unchanged += 1
        logger.debug("skipping %s: pdf and chunking config both unchanged", metadata.id)
        return

    with conn.transaction():
        document = parse_pdf(pdf_path)
        section_map = detect_sections(document)
        chunks = chunk_document(
            document,
            section_map,
            paper_title=metadata.title,
            config=config,
            counter=counter,
        )
        store.upsert_paper(conn, metadata, pdf_sha256)
        inserted = store.insert_chunks(conn, metadata.id, chunks, config)

    report.papers_ingested += 1
    report.chunks_inserted += inserted
    report.section_fallbacks += int(section_map.used_fallback)
    report.two_column_papers += int(bool(document.two_column_pages))
    logger.info(
        "ingested %s: %d chunks (%d new), %d sections%s",
        metadata.id,
        len(chunks),
        inserted,
        len(section_map.sections),
        ", section detection fell back" if section_map.used_fallback else "",
    )


def ingest(
    conn: psycopg.Connection,
    *,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
    limit: int = 150,
    config: RetrievalConfig | None = None,
    cache_dir: Path = DEFAULT_PDF_CACHE,
) -> IngestReport:
    """Ingest the most recent `limit` papers from `categories`.

    A failure on one paper is logged and counted, never fatal: a corpus build that dies
    on a single malformed PDF after eight minutes of rate-limited downloading is a bad
    trade against simply reporting which paper failed.
    """
    config = config or RetrievalConfig()
    counter = token_counter_for(config.embedding_model)
    report = IngestReport()

    for metadata, pdf_path in arxiv.fetch_corpus(categories, limit, cache_dir):
        report.papers_seen += 1
        try:
            ingest_paper(conn, metadata, pdf_path, config, counter, report)
        except (psycopg.Error, ValueError, RuntimeError) as exc:
            report.papers_failed += 1
            report.failures.append((metadata.id, f"{type(exc).__name__}: {exc}"))
            logger.exception("failed to ingest %s", metadata.id)

    logger.info("ingest complete: %s", report)
    return report
