"""arXiv client tests. No network: the Atom payload is canned."""

import hashlib
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

from rag.ingest.arxiv import ATOM, _entry_to_metadata, search, sha256_of

ATOM_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.02385v2</id>
    <published>2024-01-04T18:30:00Z</published>
    <title>Attention Is All You
      Need   Again</title>
    <summary>We propose a new
      simple network architecture.</summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <link href="http://arxiv.org/abs/2401.02385v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.02385v2" rel="related"/>
    <category term="cs.LG"/>
    <category term="cs.CL"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.09999v1</id>
    <published>2024-01-03T10:00:00Z</published>
    <title>A Paper Without A PDF Link</title>
    <summary>Short abstract.</summary>
    <author><name>Grace Hopper</name></author>
    <category term="cs.LG"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.11111v1</id>
    <title>Missing Its Published Date</title>
    <summary>Short abstract.</summary>
  </entry>
</feed>
"""


def entries() -> list[ET.Element]:
    return ET.fromstring(ATOM_FEED).findall(f"{ATOM}entry")


def test_extracts_the_versioned_identifier() -> None:
    metadata = _entry_to_metadata(entries()[0])
    assert metadata is not None
    # The version suffix matters: v1 and v2 are different documents with different text.
    assert metadata.id == "2401.02385v2"


def test_collapses_the_whitespace_atom_embeds() -> None:
    metadata = _entry_to_metadata(entries()[0])
    assert metadata is not None
    assert metadata.title == "Attention Is All You Need Again"
    assert metadata.abstract == "We propose a new simple network architecture."


def test_extracts_authors_categories_and_date() -> None:
    metadata = _entry_to_metadata(entries()[0])
    assert metadata is not None
    assert metadata.authors == ("Ada Lovelace", "Alan Turing")
    assert metadata.categories == ("cs.LG", "cs.CL")
    assert metadata.published_at == date(2024, 1, 4)


def test_uses_the_declared_pdf_link_when_present() -> None:
    metadata = _entry_to_metadata(entries()[0])
    assert metadata is not None
    assert metadata.pdf_url == "http://arxiv.org/pdf/2401.02385v2"


def test_falls_back_to_a_constructed_pdf_url() -> None:
    metadata = _entry_to_metadata(entries()[1])
    assert metadata is not None
    assert metadata.pdf_url == "https://arxiv.org/pdf/2401.09999v1"


def test_skips_entries_missing_required_fields() -> None:
    # A malformed entry is dropped with a warning rather than aborting a corpus build
    # that has already spent minutes on rate-limited downloads.
    assert _entry_to_metadata(entries()[2]) is None


def test_sha256_depends_only_on_contents(tmp_path: Path) -> None:
    # This digest is the idempotency key for re-ingestion, so it has to be exactly the
    # content hash and nothing else -- not the path, not the mtime.
    same_a, same_b, different = (tmp_path / n for n in ("a.pdf", "b.pdf", "c.pdf"))
    same_a.write_bytes(b"%PDF-1.7 fake")
    same_b.write_bytes(b"%PDF-1.7 fake")
    different.write_bytes(b"%PDF-1.7 other")

    assert sha256_of(same_a) == sha256_of(same_b)
    assert sha256_of(same_a) != sha256_of(different)
    assert sha256_of(same_a) == hashlib.sha256(b"%PDF-1.7 fake").hexdigest()


def test_search_is_importable_without_network() -> None:
    # Guards the module-level import graph: `rag ingest` must not require a live network
    # merely to construct its argument parser.
    assert callable(search)
