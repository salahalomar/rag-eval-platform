"""PyMuPDF extraction that preserves reading order, page numbers and character offsets.

The hard part here is not extraction, it is *order*. arXiv ML papers are overwhelmingly
two-column, and PyMuPDF's default block order is roughly top-to-bottom across the whole
page -- which interleaves the two columns and produces text that reads as alternating
half-sentences. Chunks built from that are incoherent no matter how good the chunker is,
and no reranker recovers them. So blocks are grouped into columns and emitted column by
column, with full-width elements acting as band separators.

Character offsets index into the reconstructed text this module returns. They are
reproducible for a given PDF and a given PyMuPDF version, which is why the version is
pinned by the lockfile.
"""

import logging
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf

# PyMuPDF ships no type information for these two entry points, so strict mode rejects
# the calls. Narrowly ignored at each call site rather than relaxing the setting for the
# module, which would also hide genuinely untyped calls we write ourselves.

logger = logging.getLogger(__name__)

# A block wider than this fraction of the page is treated as spanning both columns
# (title, abstract, a wide table) rather than belonging to one of them.
FULL_WIDTH_RATIO = 0.65

# A page needs at least this many blocks clearly on each side before it is considered
# two-column; otherwise a single stray block would flip the reading order of a page.
MIN_BLOCKS_PER_COLUMN = 2

# Pages yielding less text than this are treated as figure-only. Deliberately generous:
# a page holding one full-width diagram and its caption is usually 100-250 characters,
# while a genuine text page on these papers clears 1,500.
FIGURE_ONLY_MAX_CHARS = 280

PYMUPDF_BLOCK_TYPE_TEXT = 0

# PyMuPDF returns whatever a PDF's font encoding yields, and a meaningful minority of
# arXiv papers produce NUL and other C0 control bytes. Postgres TEXT cannot store NUL at
# all -- the insert aborts outright -- and the remaining control characters are invisible
# noise that only pollutes the lexical index. Roughly a third of a sample of three papers
# hit this, so it is the common case, not an edge case.
#
# Stripped here, at extraction, rather than before the insert: character offsets are
# assigned in this module, and sanitising downstream would leave every offset pointing a
# few characters off from the text actually stored.
CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(slots=True)
class TextSpan:
    """A run of text sharing one font, located in the reconstructed document text."""

    text: str
    page: int  # 1-based, matching what a reader sees in a PDF viewer
    font_size: float
    bold: bool
    char_start: int
    char_end: int
    # Monotonic ids identifying the source line. Section detection needs to know which
    # spans formed one visual line -- a heading like "3.2 Training" is often three spans
    # -- and that structure is otherwise lost once text is flattened.
    block: int
    line: int


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """A PDF reduced to ordered text plus the metadata needed to cite back into it."""

    text: str
    spans: tuple[TextSpan, ...]
    page_count: int
    body_font_size: float
    figure_only_pages: frozenset[int]
    two_column_pages: frozenset[int]

    def page_range(self, char_start: int, char_end: int) -> tuple[int, int]:
        """First and last page touched by a character range.

        Used to populate `chunks.page_start` / `page_end` so a citation resolves to a
        page a reader can actually turn to.
        """
        pages = [s.page for s in self.spans if s.char_start < char_end and s.char_end > char_start]
        if not pages:
            return (1, 1)
        return (min(pages), max(pages))


def _is_bold(font_name: str, flags: int) -> bool:
    # PyMuPDF exposes bold as bit 4 of the span flags, but a good number of LaTeX fonts
    # do not set it and encode weight in the name instead.
    return bool(flags & 2**4) or "bold" in font_name.lower() or "-bd" in font_name.lower()


def _ordered_blocks(page: pymupdf.Page) -> tuple[list[dict[str, Any]], bool]:
    """Return this page's text blocks in reading order, and whether it is two-column."""
    raw = page.get_text("dict")  # type: ignore[no-untyped-call]
    blocks = [b for b in raw.get("blocks", []) if b.get("type") == PYMUPDF_BLOCK_TYPE_TEXT]
    if not blocks:
        return [], False

    page_width = page.rect.width or 1.0
    midpoint = page.rect.x0 + page_width / 2

    full_width, left, right = [], [], []
    for block in blocks:
        x0, _, x1, _ = block["bbox"]
        if (x1 - x0) / page_width > FULL_WIDTH_RATIO:
            full_width.append(block)
        elif (x0 + x1) / 2 < midpoint:
            left.append(block)
        else:
            right.append(block)

    if len(left) < MIN_BLOCKS_PER_COLUMN or len(right) < MIN_BLOCKS_PER_COLUMN:
        return sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0])), False

    # Full-width blocks split the page into horizontal bands. Within each band the left
    # column is read fully before the right, which is what a human does.
    separators = sorted(full_width, key=lambda b: b["bbox"][1])
    ordered: list[dict[str, Any]] = []
    band_top = page.rect.y0
    for separator in [*separators, None]:
        band_bottom = separator["bbox"][1] if separator is not None else page.rect.y1
        for column in (left, right):
            ordered.extend(
                sorted(
                    (b for b in column if band_top <= b["bbox"][1] < band_bottom),
                    key=lambda b: b["bbox"][1],
                )
            )
        if separator is not None:
            ordered.append(separator)
            band_top = separator["bbox"][3]

    return ordered, True


def parse_pdf(path: Path) -> ParsedDocument:
    """Extract ordered text, spans, page mapping and layout statistics from a PDF."""
    parts: list[str] = []
    spans: list[TextSpan] = []
    figure_only: set[int] = set()
    two_column: set[int] = set()
    cursor = 0
    block_id = 0
    line_id = 0

    def append(text: str, *, page: int, size: float, bold: bool, block: int, line: int) -> None:
        nonlocal cursor
        parts.append(text)
        spans.append(
            TextSpan(
                text=text,
                page=page,
                font_size=size,
                bold=bold,
                char_start=cursor,
                char_end=cursor + len(text),
                block=block,
                line=line,
            )
        )
        cursor += len(text)

    def append_separator(text: str) -> None:
        nonlocal cursor
        if not parts:
            return
        parts.append(text)
        cursor += len(text)

    def ends_with_soft_hyphen() -> bool:
        """Whether the text so far ends in a word broken across a line break."""
        if not spans:
            return False
        tail = spans[-1].text
        return len(tail) >= 2 and tail.endswith("-") and tail[-2].isalpha()

    def dehyphenate() -> None:
        r"""Drop the hyphen of a word broken across a line break, e.g. 'hyphen-\nation'.

        Left in place, 'hyphen-ation' tokenises differently from 'hyphenation' and the
        lexical arm stops matching the word at all.
        """
        nonlocal cursor
        spans[-1].text = spans[-1].text[:-1]
        spans[-1].char_end -= 1
        parts[-1] = parts[-1][:-1]
        cursor -= 1

    with pymupdf.open(path) as document:  # type: ignore[no-untyped-call]
        page_count = document.page_count
        for page_index, page in enumerate(document):
            page_number = page_index + 1
            blocks, is_two_column = _ordered_blocks(page)
            if is_two_column:
                two_column.add(page_number)

            page_chars_before = cursor
            for block in blocks:
                append_separator("\n\n")
                block_id += 1
                for line_index, line in enumerate(block.get("lines", [])):
                    if line_index:
                        # A hyphenated break joins with nothing; any other break is a
                        # wrapped line and joins with a single space.
                        if ends_with_soft_hyphen():
                            dehyphenate()
                        else:
                            append_separator(" ")
                    line_id += 1
                    for span in line.get("spans", []):
                        text = CONTROL_CHARACTERS.sub("", span.get("text", ""))
                        if not text:
                            continue
                        append(
                            text,
                            page=page_number,
                            size=float(span.get("size", 0.0)),
                            bold=_is_bold(str(span.get("font", "")), int(span.get("flags", 0))),
                            block=block_id,
                            line=line_id,
                        )

            if cursor - page_chars_before < FIGURE_ONLY_MAX_CHARS:
                figure_only.add(page_number)

    text = "".join(parts)
    logger.debug(
        "parsed %s: %d pages, %d spans, %d chars, %d two-column, %d figure-only",
        path.name,
        page_count,
        len(spans),
        len(text),
        len(two_column),
        len(figure_only),
    )

    return ParsedDocument(
        text=text,
        spans=tuple(spans),
        page_count=page_count,
        body_font_size=_body_font_size(spans),
        figure_only_pages=frozenset(figure_only),
        two_column_pages=frozenset(two_column),
    )


def _body_font_size(spans: list[TextSpan]) -> float:
    """Median font size weighted by characters, i.e. the size most of the text is set in.

    Weighted because an unweighted median over spans is skewed by the many short spans
    that headings, footnotes and math produce; body text is long runs of one size.
    """
    if not spans:
        return 0.0
    sizes: list[float] = []
    for span in spans:
        # Rounded so that near-identical sizes from different font instances collapse.
        sizes.extend([round(span.font_size, 1)] * max(1, len(span.text)))
    return float(statistics.median(sizes))
