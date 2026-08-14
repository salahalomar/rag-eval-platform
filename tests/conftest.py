from collections.abc import Sequence

from rag.ingest.parse import ParsedDocument, TextSpan


def build_document(
    paragraphs: Sequence[tuple[str, int, float, bool]],
    *,
    body_font_size: float = 10.0,
    figure_only_pages: frozenset[int] = frozenset(),
) -> ParsedDocument:
    """Assemble a ParsedDocument from (text, page, font_size, bold) paragraphs.

    Hand-built rather than parsed from a PDF so that chunker and section tests state
    their inputs explicitly. Each paragraph becomes one span on its own line, and
    paragraphs are joined with a blank line, matching what parse_pdf emits per block.
    """
    spans: list[TextSpan] = []
    parts: list[str] = []
    cursor = 0
    for index, (text, page, size, bold) in enumerate(paragraphs):
        if index:
            separator = "\n\n"
            parts.append(separator)
            cursor += len(separator)
        parts.append(text)
        spans.append(
            TextSpan(
                text=text,
                page=page,
                font_size=size,
                bold=bold,
                char_start=cursor,
                char_end=cursor + len(text),
                block=index,
                line=index,
            )
        )
        cursor += len(text)

    return ParsedDocument(
        text="".join(parts),
        spans=tuple(spans),
        page_count=max((p for _, p, _, _ in paragraphs), default=1),
        body_font_size=body_font_size,
        figure_only_pages=figure_only_pages,
        two_column_pages=frozenset(),
    )


def words(count: int, marker: str) -> str:
    """A sentence of exactly `count` whitespace-separated words, ending in a period.

    Used so WhitespaceTokenCounter yields token counts a reader can verify by counting.
    Capitalised because the sentence splitter requires a capital after the terminator --
    a lowercase fixture would silently read as one long sentence and test nothing.
    """
    body = " ".join(f"{marker}{i}" for i in range(count - 1))
    sentence = f"{body} end."
    return sentence[0].upper() + sentence[1:]
