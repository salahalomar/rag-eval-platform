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
from rag.index.benchmark import DEFAULT_EF_SEARCH_SWEEP, benchmark_index
from rag.index.embed import DEFAULT_BATCH_SIZE, embed_corpus, table_for
from rag.ingest import store
from rag.ingest.arxiv import read_manifest
from rag.ingest.pipeline import DEFAULT_CATEGORIES, DEFAULT_PDF_CACHE, ingest
from rag.retrieve import lexical, retrieve
from rag.retrieve.types import RetrievalResult
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

    ids = read_manifest(args.ids_file) if args.ids_file else None
    if ids is not None:
        print(f"ingesting {len(ids)} papers pinned by {args.ids_file}")
    else:
        print(f"ingesting up to {args.limit} papers from {', '.join(categories)}")
        print("  (unpinned: 'most recent N' is a different set of papers every day —")
        print("   use --ids-file once the corpus is fixed)")
    print(f"chunking identity: {config.chunking_sha256()[:12]}  {config.chunking_params()}")

    with connect() as conn:
        report = ingest(
            conn,
            categories=categories,
            limit=args.limit,
            config=config,
            cache_dir=args.cache_dir,
            ids=ids,
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


def _cmd_manifest(args: argparse.Namespace) -> int:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, title FROM papers ORDER BY published_at DESC, id DESC"
        ).fetchall()
    if not rows:
        print("no papers ingested; nothing to pin")
        return 1

    lines = [
        "# Frozen corpus for rag-eval-platform.",
        "#",
        "# Rebuild with: rag ingest --ids-file <this file>",
        "#",
        "# Pinned because a category search returns 'the most recent N', which is a",
        "# different set of papers every day. The golden set binds questions to chunk",
        "# ids, so the corpus behind those ids has to be nameable, not merely",
        "# describable.",
        f"# {len(rows)} papers.",
        "",
    ]
    lines.extend(f"{paper_id}  # {str(title)[:90]}" for paper_id, title in rows)
    args.path.parent.mkdir(parents=True, exist_ok=True)
    args.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} paper ids to {args.path}")
    return 0


def _cmd_embed(args: argparse.Namespace) -> int:
    config = RetrievalConfig(embedding_model=args.model)
    print(f"embedding with {config.embedding_model} into {table_for(config.embedding_model)}")
    with connect() as conn:
        report = embed_corpus(
            conn,
            model=config.embedding_model,
            chunk_config_sha256=None if args.all_chunkings else config.chunking_sha256(),
            batch_size=args.batch_size,
        )
    print("\nembed report:")
    for line in report.as_lines():
        print(line)
    return 0


def _print_candidates(result: RetrievalResult, *, verbose: bool) -> None:
    for candidate in result:
        print(
            f"  {candidate.rank:>2}. {candidate.score:.5f}  "
            f"chunk {candidate.chunk_id:<7} [{candidate.paper_id}]  "
            f"p{candidate.page_start}-{candidate.page_end}  "
            f"{candidate.section_path[:44]}"
        )
        if verbose:
            body = " ".join(candidate.content.split())
            print(f"       {candidate.paper_title[:96]}")
            print(f"       {body[:240]}{'...' if len(body) > 240 else ''}")


def _cmd_search(args: argparse.Namespace) -> int:
    base = RetrievalConfig(
        embedding_model=args.model,
        hnsw_ef_search=args.ef_search,
        final_top_k=args.top_k,
        rrf_k=args.rrf_k,
    )
    modes = ("lexical_only", "dense_only", "rrf") if args.compare else (args.mode,)

    print(f'query: "{args.query}"')
    with connect() as conn:
        print(f"lexemes: {' | '.join(lexical.lexemes(conn, args.query)) or '(none)'}")
        for mode in modes:
            config = base.model_copy(update={"fusion": mode})
            result = retrieve(args.query, config, conn)
            print()
            label = f"{mode}" + (f"  (rrf_k={config.rrf_k})" if mode == "rrf" else "")
            print(f"--- {label} " + "-" * max(0, 58 - len(label)))
            print(
                f"    arms: dense={result.dense_count} lexical={result.lexical_count} "
                f"-> fused={result.fused_count}   timings: {result.timings_ms}"
            )
            if not result.candidates:
                print("    no results")
                continue
            _print_candidates(result, verbose=args.verbose or not args.compare)
    return 0


def _cmd_bench_index(args: argparse.Namespace) -> int:
    config = RetrievalConfig(embedding_model=args.model)
    print(f"benchmarking {table_for(config.embedding_model)} over {args.queries} sampled queries")
    print("(exact ground truth is a full scan per query — this takes a minute)\n")
    with connect() as conn:
        result = benchmark_index(
            conn,
            config,
            query_count=args.queries,
            ef_search_values=tuple(args.ef_search),
            rebuild=args.rebuild,
        )
    for line in result.as_lines():
        print(line)
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
    ingest_parser.add_argument(
        "--ids-file",
        type=Path,
        help="ingest exactly these arXiv ids (one per line) instead of a category search",
    )
    ingest_parser.set_defaults(func=_cmd_ingest)

    manifest_parser = subparsers.add_parser(
        "manifest", help="write the ingested corpus out as a pinned id list"
    )
    manifest_parser.add_argument("path", type=Path)
    manifest_parser.set_defaults(func=_cmd_manifest)

    stats_parser = subparsers.add_parser("stats", help="summarise what is in the corpus")
    stats_parser.add_argument(
        "--sample", type=int, default=0, help="also print N random chunks (seeded, reproducible)"
    )
    stats_parser.set_defaults(func=_cmd_stats)

    default_model = RetrievalConfig().embedding_model

    embed_parser = subparsers.add_parser("embed", help="embed chunks that lack a vector")
    embed_parser.add_argument("--model", default=default_model)
    embed_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    embed_parser.add_argument(
        "--all-chunkings",
        action="store_true",
        help="embed every stored chunking, not just the one the default config describes",
    )
    embed_parser.set_defaults(func=_cmd_embed)

    search_parser = subparsers.add_parser("search", help="retrieve over the corpus")
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int, default=10)
    search_parser.add_argument(
        "--mode",
        choices=("rrf", "dense_only", "lexical_only"),
        default=RetrievalConfig().fusion,
        help="retrieval strategy (default: rrf)",
    )
    search_parser.add_argument(
        "--compare",
        action="store_true",
        help="run all three modes over the same query, side by side",
    )
    search_parser.add_argument("--rrf-k", type=int, default=RetrievalConfig().rrf_k)
    search_parser.add_argument("--model", default=default_model)
    search_parser.add_argument("--ef-search", type=int, default=RetrievalConfig().hnsw_ef_search)
    search_parser.set_defaults(func=_cmd_search)

    bench_parser = subparsers.add_parser(
        "bench-index", help="measure HNSW recall and latency against an exact scan"
    )
    bench_parser.add_argument("--model", default=default_model)
    bench_parser.add_argument("--queries", type=int, default=50)
    bench_parser.add_argument(
        "--ef-search", type=int, nargs="+", default=list(DEFAULT_EF_SEARCH_SWEEP)
    )
    bench_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="drop and rebuild the index first, to measure build time on a populated table",
    )
    bench_parser.set_defaults(func=_cmd_bench_index)

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
