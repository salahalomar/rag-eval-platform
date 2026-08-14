"""arXiv metadata and PDF retrieval.

Deliberately polite and deliberately slow. arXiv asks for one request every three
seconds and a descriptive User-Agent; a corpus build is a one-off cost paid offline, so
there is nothing to gain by pushing harder and a working relationship with a free API to
lose. PDFs are cached on disk by identifier, which makes a re-run of a partially
completed ingest resume rather than restart.
"""

import hashlib
import logging
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

API_URL = "http://export.arxiv.org/api/query"
USER_AGENT = "rag-eval-platform/0.1 (https://github.com/salahalomar/rag-eval-platform)"

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

# arXiv's stated courtesy interval between requests.
MIN_REQUEST_INTERVAL_S = 3.0
PAGE_SIZE = 100
MAX_ATTEMPTS = 5


class ArxivError(RuntimeError):
    """A request to arXiv failed in a way that retrying did not fix."""


@dataclass(frozen=True, slots=True)
class PaperMetadata:
    """One paper as arXiv describes it, before any PDF has been fetched."""

    id: str  # versioned, e.g. '2401.02385v2'
    title: str
    authors: tuple[str, ...]
    abstract: str
    categories: tuple[str, ...]
    published_at: date
    pdf_url: str


class _RateLimiter:
    """Enforces a minimum interval between requests across the whole process."""

    __slots__ = ("_interval", "_last")

    def __init__(self, interval_s: float) -> None:
        self._interval = interval_s
        self._last = 0.0

    def wait(self) -> None:
        """Block until the configured interval has elapsed since the previous call."""
        elapsed = time.monotonic() - self._last
        if self._last and elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last = time.monotonic()


def _normalise(text: str) -> str:
    """Collapse the newlines and runs of spaces that arXiv's Atom feed embeds."""
    return " ".join(text.split())


def _entry_to_metadata(entry: ET.Element) -> PaperMetadata | None:
    raw_id = entry.findtext(f"{ATOM}id")
    title = entry.findtext(f"{ATOM}title")
    summary = entry.findtext(f"{ATOM}summary")
    published = entry.findtext(f"{ATOM}published")
    if not (raw_id and title and summary and published):
        logger.warning("skipping arXiv entry with missing required fields: %s", raw_id)
        return None

    identifier = raw_id.rsplit("/abs/", 1)[-1]
    authors = tuple(
        _normalise(name)
        for author in entry.findall(f"{ATOM}author")
        if (name := author.findtext(f"{ATOM}name"))
    )
    categories = tuple(
        term
        for category in entry.findall(f"{ATOM}category")
        if (term := category.get("term")) is not None
    )
    pdf_url = next(
        (
            href
            for link in entry.findall(f"{ATOM}link")
            if link.get("title") == "pdf" and (href := link.get("href"))
        ),
        f"https://arxiv.org/pdf/{identifier}",
    )

    return PaperMetadata(
        id=identifier,
        title=_normalise(title),
        authors=authors,
        abstract=_normalise(summary),
        categories=categories,
        published_at=datetime.fromisoformat(published.replace("Z", "+00:00")).date(),
        pdf_url=pdf_url,
    )


def _get(client: httpx.Client, limiter: _RateLimiter, url: str, **kwargs: object) -> httpx.Response:
    """GET with backoff on the transient statuses arXiv actually returns.

    Retries 429 and 5xx; a 404 means the paper is genuinely absent and retrying it four
    more times only wastes twelve seconds of the corpus build.
    """
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        limiter.wait()
        try:
            response = client.get(url, **kwargs)  # type: ignore[arg-type]
        except httpx.RequestError as exc:
            last_error = exc
        else:
            if response.status_code < 400:
                return response
            if response.status_code == 429 or response.status_code >= 500:
                last_error = httpx.HTTPStatusError(
                    f"HTTP {response.status_code}", request=response.request, response=response
                )
            else:
                raise ArxivError(f"GET {url} returned HTTP {response.status_code}")

        backoff = MIN_REQUEST_INTERVAL_S * (2 ** (attempt - 1))
        logger.warning(
            "arXiv request failed (attempt %d/%d), retrying in %.0fs: %s",
            attempt,
            MAX_ATTEMPTS,
            backoff,
            last_error,
        )
        time.sleep(backoff)

    raise ArxivError(f"GET {url} failed after {MAX_ATTEMPTS} attempts: {last_error}")


def search(
    categories: Sequence[str],
    limit: int,
    *,
    client: httpx.Client | None = None,
    limiter: _RateLimiter | None = None,
) -> list[PaperMetadata]:
    """Fetch metadata for the most recent `limit` papers in `categories`.

    Sorted by submission date descending and paginated, because arXiv truncates large
    single requests without saying so. Ordering is fixed rather than relevance-based so
    that a rebuild of the corpus on the same day yields the same papers.
    """
    query = " OR ".join(f"cat:{category}" for category in categories)
    limiter = limiter or _RateLimiter(MIN_REQUEST_INTERVAL_S)
    owns_client = client is None
    client = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60.0)

    papers: list[PaperMetadata] = []
    seen: set[str] = set()
    try:
        while len(papers) < limit:
            params = {
                "search_query": query,
                "start": len(seen),
                "max_results": min(PAGE_SIZE, limit - len(papers)),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            response = _get(client, limiter, API_URL, params=params)
            entries = ET.fromstring(response.text).findall(f"{ATOM}entry")
            if not entries:
                logger.info("arXiv returned no further entries at offset %d", len(seen))
                break

            for entry in entries:
                seen.add(entry.findtext(f"{ATOM}id") or "")
                metadata = _entry_to_metadata(entry)
                # Papers are occasionally cross-listed and returned twice across pages.
                if metadata is not None and metadata.id not in {p.id for p in papers}:
                    papers.append(metadata)
                if len(papers) == limit:
                    break
    finally:
        if owns_client:
            client.close()

    logger.info("fetched metadata for %d papers across %s", len(papers), ", ".join(categories))
    return papers


def fetch_pdf(
    metadata: PaperMetadata,
    cache_dir: Path,
    *,
    client: httpx.Client | None = None,
    limiter: _RateLimiter | None = None,
) -> Path:
    """Download the PDF, or return the cached copy if it is already on disk.

    The cache is what makes an interrupted 150-paper ingest resumable: a second run
    re-downloads nothing it already has, so the expensive part of the build is paid once.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"{metadata.id.replace('/', '_')}.pdf"
    if destination.exists() and destination.stat().st_size > 0:
        logger.debug("pdf cache hit for %s", metadata.id)
        return destination

    limiter = limiter or _RateLimiter(MIN_REQUEST_INTERVAL_S)
    owns_client = client is None
    client = client or httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=120.0, follow_redirects=True
    )
    try:
        response = _get(client, limiter, metadata.pdf_url)
        # Write via a temporary file so an interrupted download cannot leave a truncated
        # PDF that the cache check above would happily accept on the next run.
        partial = destination.with_suffix(".pdf.part")
        partial.write_bytes(response.content)
        partial.replace(destination)
    finally:
        if owns_client:
            client.close()

    logger.debug("downloaded %s (%d bytes)", metadata.id, destination.stat().st_size)
    return destination


def sha256_of(path: Path) -> str:
    """Content digest of a file, used to skip papers whose PDF has not changed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_corpus(
    categories: Sequence[str],
    limit: int,
    cache_dir: Path,
) -> Iterator[tuple[PaperMetadata, Path]]:
    """Yield each paper's metadata alongside its PDF on disk.

    A generator so that a failure on paper 140 does not discard the work done on the
    first 139 -- the pipeline persists as it goes.
    """
    limiter = _RateLimiter(MIN_REQUEST_INTERVAL_S)
    with httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=120.0, follow_redirects=True
    ) as client:
        for metadata in search(categories, limit, client=client, limiter=limiter):
            try:
                yield metadata, fetch_pdf(metadata, cache_dir, client=client, limiter=limiter)
            except ArxivError:
                logger.exception("giving up on %s", metadata.id)
