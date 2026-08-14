"""Command line entry point: `rag ingest`, `rag stats`.

Kept in the library rather than in `apps/` because ingestion and inspection have to be
runnable without the web layer -- from a notebook, from CI, or from the eval harness.
"""

import argparse
import logging
import sys
from pathlib import Path

from rag.config import RetrievalConfig
from rag.db import connect
from rag.ingest import store
from rag.ingest.pipeline import DEFAULT_CATEGORIES, DEFAULT_PDF_CACHE, ingest
from rag.settings import get_settings


def _configure_logging(verbose: bool) -> None:
    configured = getattr(logging, get_settings().log_level, logging.INFO)
    logging.basicConfig(
        level=logging.DEBUG if verbose else configured,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _cmd_ingest(args: argparse.Namespace) -> int:
    categories = tuple(args.category) if args.category else DEFAULT_CATEGORIES
    config = RetrievalConfig(
        chunk_tokens=args.chunk_tokens,
        chunk_overlap_pct=args.chunk_overlap_pct,
    )

    print(f"ingesting up to {args.limit} papers from {', '.join(categories)}")
    print(f"chunking identity: {config.chunking_sha256()[:12]}  {config.chunking_params()}")

    with connect() as conn:
        report = ingest(
            conn,
            categories=categories,
            limit=args.limit,
            config=config,
            cache_dir=args.cache_dir,
        )

    print("\ningest report:")
    for line in report.as_lines():
        print(line)
    return 1 if report.papers_ingested == 0 and report.papers_skipped_unchanged == 0 else 0


def _cmd_stats(args: argparse.Namespace) -> int:
    with connect() as conn:
        stats = store.corpus_stats(conn)
        samples = store.sample_chunks(conn, args.sample) if args.sample else []

    print("corpus")
    print(f"  papers                 {stats.papers}")
    print(f"  chunks                 {stats.chunks}")
    print(f"  distinct chunkings     {stats.chunkings}")
    print(f"  mean chunks / paper    {stats.mean_chunks_per_paper:.1f}")
    print(f"  distinct section paths {stats.distinct_sections}")
    print(f"  page-spans covered     {stats.pages_covered}")
    print()
    print("chunk token counts (of embed_input, the string the model encodes)")
    if stats.token_percentiles:
        for label, value in stats.token_percentiles.items():
            print(f"  {label:<22} {value:.0f}")
    else:
        print("  (no chunks yet)")
    print()
    print("health signals")
    print(
        f"  section fallbacks      {stats.fallback_papers} papers "
        f"({stats.fallback_rate:.1%}) — heading detection gave up, all text in 'Body'"
    )
    print(
        f"  short chunks           {stats.short_chunks} "
        f"(< {store.SUSPICIOUSLY_SHORT_TOKENS} tokens; usually stray headers or equations)"
    )

    for sample in samples:
        print()
        print(
            f"--- chunk {sample['id']}  [{sample['paper_id']}]  "
            f"pages {sample['page_start']}-{sample['page_end']}  "
            f"{sample['token_count']} tokens"
        )
        print(f"    paper:   {sample['title'][:88]}")
        print(f"    section: {sample['section_path']}")
        body = " ".join(str(sample["content"]).split())
        print(f"    text:    {body[:400]}{'...' if len(body) > 400 else ''}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to a subcommand."""
    parser = argparse.ArgumentParser(prog="rag", description="rag-eval-platform tooling")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="fetch, parse and chunk arXiv papers")
    ingest_parser.add_argument(
        "--category", action="append", help="arXiv category; repeatable (default: cs.LG cs.CL)"
    )
    ingest_parser.add_argument("--limit", type=int, default=150, help="papers to ingest")
    ingest_parser.add_argument("--chunk-tokens", type=int, default=RetrievalConfig().chunk_tokens)
    ingest_parser.add_argument(
        "--chunk-overlap-pct", type=float, default=RetrievalConfig().chunk_overlap_pct
    )
    ingest_parser.add_argument("--cache-dir", type=Path, default=DEFAULT_PDF_CACHE)
    ingest_parser.set_defaults(func=_cmd_ingest)

    stats_parser = subparsers.add_parser("stats", help="summarise what is in the corpus")
    stats_parser.add_argument(
        "--sample", type=int, default=0, help="also print N random chunks (seeded, reproducible)"
    )
    stats_parser.set_defaults(func=_cmd_stats)

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
