import re
from pathlib import Path

import pytest

from rag.config import RetrievalConfig
from rag.ingest.chunk import chunk_document
from rag.ingest.parse import CONTROL_CHARACTERS, ParsedDocument, parse_pdf
from rag.ingest.sections import detect_sections
from rag.ingest.tokenization import WhitespaceTokenCounter

FIXTURE = Path(__file__).parent / "fixtures" / "sample_paper.pdf"


@pytest.fixture(scope="module")
def document() -> ParsedDocument:
    return parse_pdf(FIXTURE)


def test_reads_every_page(document: ParsedDocument) -> None:
    assert document.page_count == 4
    assert len(document.text) > 800


def test_span_offsets_address_the_text_they_describe(document: ParsedDocument) -> None:
    # Citation highlighting depends on this being exact; an off-by-one here surfaces in
    # Phase 8 as a highlight that drifts further from the claim on every page.
    for span in document.spans:
        assert document.text[span.char_start : span.char_end] == span.text


def test_spans_are_ordered_and_non_overlapping(document: ParsedDocument) -> None:
    for earlier, later in zip(document.spans, document.spans[1:], strict=False):
        assert earlier.char_end <= later.char_start


def test_columns_are_read_one_after_the_other_not_interleaved(document: ParsedDocument) -> None:
    # The fixture places three paragraphs in each column at matching heights. A naive
    # top-to-bottom block sort yields LEFTONE, RIGHTONE, LEFTTWO, RIGHTTWO... which reads
    # as alternating half-arguments. Correct ordering drains the left column first.
    order = [m.group() for m in re.finditer(r"(LEFT|RIGHT)(ONE|TWO|THREE)", document.text)]
    assert order == [
        "LEFTONE",
        "LEFTTWO",
        "LEFTTHREE",
        "RIGHTONE",
        "RIGHTTWO",
        "RIGHTTHREE",
    ]


def test_identifies_two_column_pages(document: ParsedDocument) -> None:
    assert document.two_column_pages == frozenset({1, 2})


def test_identifies_the_figure_only_page(document: ParsedDocument) -> None:
    # Page 3 is a boxed figure and a caption; page 4 is a dense references list and must
    # not be swept up with it.
    assert document.figure_only_pages == frozenset({3})


def test_joins_words_broken_across_a_line_break(document: ParsedDocument) -> None:
    assert "generation systems" in document.text
    assert "gener- ation" not in document.text
    assert "gener-ation" not in document.text


def test_body_font_size_is_the_size_most_text_is_set_in(document: ParsedDocument) -> None:
    # Headings are 11pt and the title 16pt, but the bulk of the page is 9pt body text.
    assert document.body_font_size == pytest.approx(9.0)


def test_page_range_maps_offsets_back_to_pages(document: ParsedDocument) -> None:
    first_page_span = document.spans[0]
    assert document.page_range(first_page_span.char_start, first_page_span.char_end) == (1, 1)
    assert document.page_range(0, len(document.text)) == (1, 4)


def test_sections_are_detected_without_falling_back(document: ParsedDocument) -> None:
    result = detect_sections(document)
    assert not result.used_fallback
    assert [s.path for s in result.sections] == [
        "Frontmatter",
        "Abstract",
        "1 Introduction",
        "2 Related Work",
        "3 Method",
        "3 Method > 3.2 Training",
        "4 Results",
        "References",
    ]


def test_end_to_end_produces_chunks_confined_to_their_sections(document: ParsedDocument) -> None:
    config = RetrievalConfig(chunk_tokens=40, contextual_headers=True)
    section_map = detect_sections(document)
    chunks = chunk_document(
        document,
        section_map,
        paper_title="Structure Aware Chunking",
        config=config,
        counter=WhitespaceTokenCounter(),
    )

    assert chunks
    assert "References" not in {c.section_path for c in chunks}
    assert "Figure 1" not in " ".join(c.content for c in chunks)

    by_path = {s.path: s for s in section_map.sections}
    for produced in chunks:
        owner = by_path[produced.section_path]
        assert owner.char_start <= produced.char_start
        assert produced.char_end <= owner.char_end


def test_extracted_text_carries_no_control_characters(document: ParsedDocument) -> None:
    assert not any(ord(c) < 32 and c not in "\n\t" for c in document.text)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("safe text", "safe text"),
        ("with\x00nul", "withnul"),  # aborts a Postgres insert outright
        ("bell\x07and\x1bescape", "bellandescape"),
        ("keeps\nnewlines\tand\ttabs", "keeps\nnewlines\tand\ttabs"),
        ("café — naïve", "café — naïve"),  # non-ASCII must survive untouched
    ],
)
def test_control_character_pattern_strips_only_what_it_should(raw: str, expected: str) -> None:
    # Observed on roughly a third of a three-paper sample, so this is the common case.
    assert CONTROL_CHARACTERS.sub("", raw) == expected


def test_parsing_is_deterministic() -> None:
    # Two ingests of the same corpus must produce the same chunk ids, or the embedding
    # cache and every committed eval result stop referring to the same text.
    first, second = parse_pdf(FIXTURE), parse_pdf(FIXTURE)
    assert first.text == second.text
    assert [s.char_start for s in first.spans] == [s.char_start for s in second.spans]
