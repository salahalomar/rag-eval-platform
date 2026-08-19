"""Local embedding with a content-addressed cache.

Three things in here are easy to get wrong and expensive to detect afterwards.

**The bge prefix asymmetry.** bge models are trained with an instruction prefix on the
*query* side only; passages are embedded bare. Applying the prefix to both, or to
neither, produces a system that works -- it returns plausible neighbours -- while losing
several points of recall for no visible reason. It is enforced here by construction and
asserted by a test, because it cannot be spotted by reading output.

**The cache key is the embedded string, not the content.** With contextual headers on,
the model sees `embed_input`, which prefixes paper title and section path onto the
content. Keying a cache on `content_sha256` would return a vector computed under a
different header, silently corrupting the very ablation arm the header exists to test.

**Normalisation.** Vectors are stored L2-normalised so that cosine distance, inner
product and Euclidean distance all rank identically, and no query has to remember which
one the index was built for.
"""

import logging
import weakref
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import psycopg

from rag.telemetry import StageTimer

logger = logging.getLogger(__name__)

# Model → vector dimension. Also selects the storage table, so an unknown model is an
# error rather than a guess: writing 768-d vectors into a 384-d column fails loudly at
# the database, but only after an hour of encoding.
MODEL_DIMENSIONS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
}

# bge's query-side instruction. Passages must NOT receive it.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DEFAULT_BATCH_SIZE = 32
TORCH_SEED = 0


class UnknownModelError(KeyError):
    """A model with no registered dimension was requested."""


def dimension_of(model: str) -> int:
    """Vector width for `model`, or a clear error naming the registry to update."""
    try:
        return MODEL_DIMENSIONS[model]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_DIMENSIONS))
        raise UnknownModelError(
            f"no dimension registered for {model!r}. Add it to "
            f"rag.index.embed.MODEL_DIMENSIONS and create the matching embeddings table "
            f"(known: {known})"
        ) from exc


def table_for(model: str) -> str:
    """Storage table for `model`.

    Derived from the registry rather than passed in, so the table name interpolated into
    SQL below can never originate from user input.
    """
    return f"embeddings_{dimension_of(model)}"


class Encoder(Protocol):
    """Turns text into unit-length vectors."""

    @property
    def dimension(self) -> int:
        """Width of the vectors produced."""
        ...

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages, with no query prefix applied."""
        ...

    def encode_query(self, text: str) -> list[float]:
        """Embed a search query, with the model's query prefix applied."""
        ...


class SentenceTransformerEncoder:
    """bge via sentence-transformers, pinned to CPU and to eval mode."""

    def __init__(self, model_name: str, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        """Load `model_name` and put it in a deterministic, inference-only state."""
        import torch
        from sentence_transformers import SentenceTransformer

        # Seeded and switched out of training mode explicitly. sentence-transformers
        # already defaults to eval, but the ablation's credibility rests on runs being
        # reproducible, and relying on a library default for that is not worth the risk.
        torch.manual_seed(TORCH_SEED)
        torch.set_grad_enabled(False)

        self._model_name = model_name
        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name, device="cpu")
        self._model.eval()
        # sentence-transformers renamed this accessor; support both so the library does
        # not pin us to one minor version of theirs.
        accessor = (
            getattr(self._model, "get_embedding_dimension", None)
            or self._model.get_sentence_embedding_dimension
        )
        self._dimension = int(accessor() or 0)

        expected = dimension_of(model_name)
        if self._dimension != expected:
            raise ValueError(
                f"{model_name} produced {self._dimension}-d vectors, registry says {expected}"
            )

    @property
    def dimension(self) -> int:
        """Width of the vectors produced."""
        return self._dimension

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages. Deliberately no prefix -- see the module docstring."""
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in row] for row in vectors]

    def encode_query(self, text: str) -> list[float]:
        """Embed a query, with bge's query-side instruction prefix applied."""
        return self.encode_documents([BGE_QUERY_PREFIX + text])[0]

    def __repr__(self) -> str:
        """Identify the model behind the vectors."""
        return f"SentenceTransformerEncoder({self._model_name!r}, dim={self._dimension})"


@lru_cache(maxsize=2)
def encoder_for(model: str) -> SentenceTransformerEncoder:
    """Cached encoder, because loading model weights dominates a short run."""
    return SentenceTransformerEncoder(model)


@dataclass(slots=True)
class EmbedReport:
    """Where the work went, so a re-run can be shown to have done nothing."""

    chunks_total: int = 0
    already_embedded: int = 0
    cache_hits: int = 0
    encoded: int = 0
    written: int = 0

    @property
    def cache_hit_rate(self) -> float:
        """Share of needed vectors served without invoking the model."""
        considered = self.cache_hits + self.encoded
        return self.cache_hits / considered if considered else 0.0

    def as_lines(self) -> list[str]:
        """Human-readable summary for `rag embed`."""
        return [
            f"  chunks in corpus       {self.chunks_total}",
            f"  already embedded       {self.already_embedded}",
            f"  reused from cache      {self.cache_hits}",
            f"  encoded                {self.encoded}",
            f"  vectors written        {self.written}",
            f"  cache hit rate         {self.cache_hit_rate:.1%}",
        ]


# Connections already taught the pgvector types. Weak so a closed connection is not kept
# alive by this set, and keyed by identity so two connections never share an entry.
_REGISTERED_CONNECTIONS: weakref.WeakSet[psycopg.Connection] = weakref.WeakSet()


def register_vector_types(conn: psycopg.Connection) -> None:
    """Teach psycopg the pgvector types for this connection, once.

    Kept out of `connect()` on purpose: registration inspects `pg_type`, so wiring it
    into every connection would make `/health` fail on a database that has not yet run
    migration 001 -- turning "not migrated yet" into "cannot connect".

    Memoised because it costs a database round trip and search calls it on every query.
    No latency claim is attached to that: the change was measured, and the difference sat
    below this machine's run-to-run variance of roughly 5ms on a 30ms query. It removes a
    redundant round trip, which is worth doing on its own; it is not a speedup until
    something measures it as one.
    """
    if conn in _REGISTERED_CONNECTIONS:
        return

    from pgvector.psycopg import register_vector

    register_vector(conn)
    _REGISTERED_CONNECTIONS.add(conn)


def pending_chunks(
    conn: psycopg.Connection, model: str, chunk_config_sha256: str | None = None
) -> list[tuple[int, str, str, str]]:
    """Chunks with no vector for `model`.

    Returns (chunk_id, embed_input, embed_input_sha256, chunk_config_sha256). The last is
    carried onto the embedding row so dense search can filter by chunking inside the
    index scan rather than after it -- see migration 004.

    The anti-join here is what makes re-embedding an unchanged corpus a genuine no-op
    rather than a re-encode that overwrites identical rows.
    """
    table = table_for(model)
    sql = f"""
        SELECT c.id, c.embed_input, c.embed_input_sha256, c.chunk_config_sha256
        FROM chunks c
        LEFT JOIN {table} e ON e.chunk_id = c.id AND e.model = %(model)s
        WHERE e.chunk_id IS NULL
    """
    params: dict[str, object] = {"model": model}
    if chunk_config_sha256 is not None:
        sql += " AND c.chunk_config_sha256 = %(chunking)s"
        params["chunking"] = chunk_config_sha256
    sql += " ORDER BY c.id"
    return [(int(a), str(b), str(c), str(d)) for a, b, c, d in conn.execute(sql, params).fetchall()]


def cached_vectors(
    conn: psycopg.Connection, model: str, digests: Sequence[str]
) -> dict[str, list[float]]:
    """Vectors already stored for these embed-input digests, under `model`.

    Serves the case the anti-join above cannot: a chunk-size sweep produces new chunk
    ids whose embedded text is byte-identical to text already embedded under another
    chunking. Re-encoding those would be pure waste.
    """
    if not digests:
        return {}
    table = table_for(model)
    rows = conn.execute(
        f"""
        SELECT DISTINCT ON (c.embed_input_sha256) c.embed_input_sha256, e.vec
        FROM {table} e
        JOIN chunks c ON c.id = e.chunk_id
        WHERE e.model = %s AND c.embed_input_sha256 = ANY(%s)
        """,
        (model, list(set(digests))),
    ).fetchall()
    return {str(digest): _as_floats(vec) for digest, vec in rows}


def _as_floats(value: object) -> list[float]:
    """Normalise whatever pgvector hands back into a plain list of floats.

    With the vector types registered, psycopg returns a pgvector `Vector`, which is not
    iterable; without registration it returns a string. Neither is what the rest of the
    library expects, and the difference only shows up on a cache hit -- so it stayed
    latent through a full 6,386-chunk embedding run, where the cache was always empty.
    """
    to_list = getattr(value, "to_list", None)
    if callable(to_list):
        return [float(x) for x in to_list()]
    if isinstance(value, str):
        return [float(x) for x in value.strip("[]").split(",")]
    if isinstance(value, Iterable):
        return [float(x) for x in value]
    raise TypeError(f"cannot read a vector out of {type(value).__name__}")


def write_vectors(
    conn: psycopg.Connection, model: str, rows: Sequence[tuple[int, list[float], str]]
) -> int:
    """Persist (chunk_id, vector, chunk_config_sha256), ignoring any already present.

    The chunking key is written here, taken from the chunk row it came from, in the same
    transaction -- which is why the denormalised copy added in migration 004 cannot drift
    from the chunks table.
    """
    if not rows:
        return 0
    table = table_for(model)
    dim = dimension_of(model)
    with conn.cursor() as cursor:
        cursor.executemany(
            f"""
            INSERT INTO {table} (chunk_id, model, dim, vec, chunk_config_sha256)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id, model) DO NOTHING
            """,
            [(chunk_id, model, dim, vector, chunking) for chunk_id, vector, chunking in rows],
        )
        return max(0, cursor.rowcount)


def embed_corpus(
    conn: psycopg.Connection,
    *,
    model: str,
    chunk_config_sha256: str | None = None,
    encoder: Encoder | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    timer: StageTimer | None = None,
) -> EmbedReport:
    """Embed every chunk that lacks a vector for `model`.

    Commits in batches so an interrupted run keeps the vectors it already produced;
    encoding 6,000 chunks on CPU is minutes of work worth not repeating.
    """
    register_vector_types(conn)
    timer = timer or StageTimer()
    report = EmbedReport()

    report.chunks_total = _count_chunks(conn, chunk_config_sha256)
    outstanding = pending_chunks(conn, model, chunk_config_sha256)
    report.already_embedded = report.chunks_total - len(outstanding)
    if not outstanding:
        logger.info("nothing to embed: all %d chunks already have vectors", report.chunks_total)
        return report

    reusable = cached_vectors(conn, model, [digest for _, _, digest, _ in outstanding])
    encoder = encoder or encoder_for(model)

    batch: list[tuple[int, str, str, str]] = []
    seen_in_run: dict[str, list[float]] = {}

    def flush(pending: list[tuple[int, str, str, str]]) -> None:
        if not pending:
            return
        to_encode = [
            (text, digest)
            for _, text, digest, _ in pending
            if digest not in reusable and digest not in seen_in_run
        ]
        # Identical embedded strings inside one batch are encoded once, not once each.
        unique: dict[str, str] = {digest: text for text, digest in to_encode}
        if unique:
            with timer.stage("encode_ms"):
                vectors = encoder.encode_documents(list(unique.values()))
            seen_in_run.update(dict(zip(unique.keys(), vectors, strict=True)))
            report.encoded += len(unique)

        rows: list[tuple[int, list[float], str]] = []
        for chunk_id, _, digest, chunking in pending:
            vector = reusable.get(digest) or seen_in_run.get(digest)
            if vector is None:  # pragma: no cover - defensive
                continue
            if digest in reusable:
                report.cache_hits += 1
            rows.append((chunk_id, vector, chunking))

        with conn.transaction():
            report.written += write_vectors(conn, model, rows)

    for chunk in outstanding:
        batch.append(chunk)
        if len(batch) >= batch_size:
            flush(batch)
            batch = []
            logger.info("embedded %d/%d chunks", report.written, len(outstanding))
    flush(batch)

    logger.info("embedding complete: %s", report)
    return report


def _count_chunks(conn: psycopg.Connection, chunk_config_sha256: str | None) -> int:
    if chunk_config_sha256 is None:
        row = conn.execute("SELECT count(*) FROM chunks").fetchone()
    else:
        row = conn.execute(
            "SELECT count(*) FROM chunks WHERE chunk_config_sha256 = %s", (chunk_config_sha256,)
        ).fetchone()
    return 0 if row is None else int(row[0])
