"""Measures the HNSW index instead of assuming it works.

An approximate index is a quality/latency trade, and shipping one without measuring the
quality side means publishing recall numbers that silently include an unknown amount of
approximation error. This module produces the missing half: HNSW's results are compared
against an exhaustive scan over the same corpus, which is slow but exact.

Nothing here is on the serving path. It exists to produce a table.
"""

import logging
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import psycopg

from rag.config import RetrievalConfig
from rag.index.embed import encoder_for, register_vector_types, table_for
from rag.retrieve import dense

logger = logging.getLogger(__name__)

DEFAULT_EF_SEARCH_SWEEP = (40, 100, 200)
DEFAULT_QUERY_COUNT = 50
WARMUP_QUERIES = 5
QUERY_SAMPLE_SEED = 0.17


@dataclass(frozen=True, slots=True)
class EfSearchResult:
    """One row of the ef_search sweep."""

    ef_search: int
    recall_at_10: float
    recall_at_50: float
    p50_ms: float
    p95_ms: float


@dataclass(slots=True)
class IndexBenchmark:
    """Everything measured about the index, ready to print."""

    rows: int
    index_bytes: int
    table_bytes: int
    build_seconds: float | None
    exact_p50_ms: float
    exact_p95_ms: float
    sweep: list[EfSearchResult] = field(default_factory=list)

    def as_lines(self) -> list[str]:
        """Formatted report, including the ef_search table."""
        lines = [
            f"  rows indexed           {self.rows}",
            f"  index size             {self.index_bytes / 1_048_576:.1f} MiB",
            f"  table size             {self.table_bytes / 1_048_576:.1f} MiB",
        ]
        if self.build_seconds is not None:
            lines.append(f"  index build time       {self.build_seconds:.2f} s")
        lines.append(
            f"  exact scan p50/p95     {self.exact_p50_ms:.1f} / {self.exact_p95_ms:.1f} ms"
        )
        lines.append("")
        lines.append("  ef_search  recall@10  recall@50   p50 ms   p95 ms")
        lines.append("  " + "-" * 50)
        for row in self.sweep:
            lines.append(
                f"  {row.ef_search:>9}  {row.recall_at_10:>9.3f}  {row.recall_at_50:>9.3f}"
                f"  {row.p50_ms:>7.1f}  {row.p95_ms:>7.1f}"
            )
        return lines


def sample_queries(
    conn: psycopg.Connection, count: int, *, seed: float = QUERY_SAMPLE_SEED
) -> list[str]:
    """Draw pseudo-queries from the corpus itself, reproducibly.

    The first sentence of a random chunk stands in for a real question. It is not a user
    query distribution and is not claimed to be -- for comparing an approximate index
    against an exact scan, what matters is that both see identical inputs, and that the
    inputs are the same on every run so two benchmarks can be compared at all.
    """
    conn.execute("SELECT setseed(%s)", (seed,))
    rows = conn.execute(
        """
        SELECT content FROM chunks
        WHERE token_count > 60
        ORDER BY random()
        LIMIT %s
        """,
        (count,),
    ).fetchall()
    queries = []
    for (content,) in rows:
        sentence = str(content).split(". ")[0].strip()
        queries.append(sentence[:300] if sentence else str(content)[:300])
    return queries


def index_sizes(conn: psycopg.Connection, model: str) -> tuple[int, int, int]:
    """Row count, index bytes and table bytes for `model`'s embedding table."""
    table = table_for(model)
    row = conn.execute(
        f"""
        SELECT (SELECT count(*) FROM {table} WHERE model = %s),
               pg_relation_size(%s),
               pg_relation_size(%s)
        """,
        (model, f"{table}_hnsw_idx", table),
    ).fetchone()
    if row is None:
        return (0, 0, 0)
    return (int(row[0]), int(row[1]), int(row[2]))


def rebuild_index(conn: psycopg.Connection, model: str) -> float:
    """Drop and recreate the HNSW index, returning seconds taken.

    Build time is only meaningful on a populated table. The migration creates the index
    when the table is empty, where it is instantaneous and tells nobody anything.
    """
    table = table_for(model)
    index = f"{table}_hnsw_idx"
    start = time.perf_counter()
    with conn.transaction():
        conn.execute(f"DROP INDEX IF EXISTS {index}")
        conn.execute(
            f"CREATE INDEX {index} ON {table} "
            f"USING hnsw (vec vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
        )
    elapsed = time.perf_counter() - start
    logger.info("rebuilt %s in %.2fs", index, elapsed)
    return elapsed


def _time_queries(
    conn: psycopg.Connection,
    queries: Sequence[str],
    config: RetrievalConfig,
    *,
    exact: bool,
) -> tuple[list[list[int]], float, float]:
    """Run each query, returning its chunk ids plus p50/p95 latency in milliseconds.

    Query embedding is excluded from the timing. It is a fixed cost paid identically by
    every arm, and including it would dilute exactly the difference being measured.
    """
    encoder = encoder_for(config.embedding_model)
    results: list[list[int]] = []
    latencies: list[float] = []

    for query in queries:
        vector = encoder.encode_query(query)
        precomputed = _FixedEncoder(vector, encoder.dimension)
        start = time.perf_counter()
        candidates = dense.search(conn, query, config, encoder=precomputed, exact=exact)
        latencies.append((time.perf_counter() - start) * 1000.0)
        results.append([c.chunk_id for c in candidates])

    latencies.sort()
    return results, _percentile(latencies, 0.50), _percentile(latencies, 0.95)


class _FixedEncoder:
    """Returns a precomputed query vector, so timing excludes model inference."""

    def __init__(self, vector: list[float], dimension: int) -> None:
        self._vector = vector
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector for _ in texts]

    def encode_query(self, text: str) -> list[float]:
        return self._vector


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, round(fraction * (len(sorted_values) - 1)))
    return sorted_values[index]


def _recall_at(
    approximate: Sequence[Sequence[int]], exact: Sequence[Sequence[int]], k: int
) -> float:
    """Mean overlap between the approximate and exact top-k, over all queries."""
    scores = []
    for approx_ids, exact_ids in zip(approximate, exact, strict=True):
        truth = set(exact_ids[:k])
        if not truth:
            continue
        scores.append(len(truth & set(approx_ids[:k])) / len(truth))
    return statistics.fmean(scores) if scores else 0.0


def benchmark_index(
    conn: psycopg.Connection,
    config: RetrievalConfig,
    *,
    query_count: int = DEFAULT_QUERY_COUNT,
    ef_search_values: Sequence[int] = DEFAULT_EF_SEARCH_SWEEP,
    rebuild: bool = False,
) -> IndexBenchmark:
    """Measure index size, build time, latency, and approximate-vs-exact recall."""
    register_vector_types(conn)
    build_seconds = rebuild_index(conn, config.embedding_model) if rebuild else None

    queries = sample_queries(conn, query_count)
    if not queries:
        raise RuntimeError("no chunks in the corpus to sample queries from; run `rag ingest` first")

    # Warm the page cache and the model, so the first measured query is not paying for
    # everyone else's setup.
    for query in queries[:WARMUP_QUERIES]:
        dense.search(conn, query, config)

    logger.info("computing exact ground truth over %d queries", len(queries))
    exact_ids, exact_p50, exact_p95 = _time_queries(conn, queries, config, exact=True)

    sweep = []
    for ef_search in ef_search_values:
        variant = config.model_copy(update={"hnsw_ef_search": ef_search})
        approx_ids, p50, p95 = _time_queries(conn, queries, variant, exact=False)
        sweep.append(
            EfSearchResult(
                ef_search=ef_search,
                recall_at_10=_recall_at(approx_ids, exact_ids, 10),
                recall_at_50=_recall_at(approx_ids, exact_ids, 50),
                p50_ms=p50,
                p95_ms=p95,
            )
        )
        logger.info(
            "ef_search=%d recall@10=%.3f p95=%.1fms",
            ef_search,
            sweep[-1].recall_at_10,
            p95,
        )

    rows, index_bytes, table_bytes = index_sizes(conn, config.embedding_model)
    return IndexBenchmark(
        rows=rows,
        index_bytes=index_bytes,
        table_bytes=table_bytes,
        build_seconds=build_seconds,
        exact_p50_ms=exact_p50,
        exact_p95_ms=exact_p95,
        sweep=sweep,
    )
