"""Per-stage timing collection for the retrieval path.

Why this is in the library rather than in a benchmark script: ENGINEERING.md treats
cost and latency as features, and the ablation table carries a p95-latency column. If
timings were collected by a separate harness, the numbers published would describe that
harness rather than the code the API serves. Timing the real path is the only way the
column means what it says.

Deliberately not part of `RetrievalConfig`: timings are an output of a run, not an input
to it, and they are the one part of a result that is legitimately non-deterministic.
"""

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager


class StageTimer:
    """Accumulates wall-clock milliseconds against named retrieval stages.

    A single timer is threaded through one query so that `query_logs.stage_timings`
    ends up with one row-shaped record per request, e.g.
    `{"dense_ms": 41.2, "lexical_ms": 12.0, "rerank_ms": 180.4}`.
    """

    __slots__ = ("_elapsed_ms",)

    def __init__(self) -> None:
        """Start with no stages recorded."""
        self._elapsed_ms: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time the enclosed block and add its duration to `name`.

        Repeated entries under the same name accumulate rather than overwrite, so a
        stage that runs in a loop -- batched embedding, a per-shard search -- reports
        the total cost of the stage rather than the cost of its final iteration.

        The duration is recorded even when the block raises, because a stage that blew
        up after 900ms is exactly the stage worth seeing in the log.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            self._elapsed_ms[name] = self._elapsed_ms.get(name, 0.0) + elapsed

    def record(self, name: str, elapsed_ms: float) -> None:
        """Add a duration measured elsewhere, for stages timed by an external client.

        Used where the thing being timed reports its own duration -- an HTTP call to
        the Anthropic API, for instance -- so that vendor-reported latency and locally
        measured latency end up in the same record.
        """
        self._elapsed_ms[name] = self._elapsed_ms.get(name, 0.0) + elapsed_ms

    @property
    def timings_ms(self) -> Mapping[str, float]:
        """A read-only view of accumulated stage durations, in insertion order."""
        return self._elapsed_ms

    def total_ms(self) -> float:
        """Sum of all stages.

        Note this is a sum, not an end-to-end wall-clock measurement: nested stages are
        counted once per name, so overlapping stages would double count. Stages in this
        codebase are sequential, which keeps the two equivalent.
        """
        return sum(self._elapsed_ms.values())

    def as_dict(self, ndigits: int = 3) -> dict[str, float]:
        """Round-tripped copy suitable for the `query_logs.stage_timings` JSONB column."""
        return {name: round(ms, ndigits) for name, ms in self._elapsed_ms.items()}

    def __repr__(self) -> str:
        """Show the accumulated stages, which is what anyone debugging wants to see."""
        stages = ", ".join(f"{k}={v:.1f}ms" for k, v in self._elapsed_ms.items())
        return f"StageTimer({stages})"
