"""Section heading detection and hierarchical section paths.

Why this earns its complexity: chunking inside section boundaries is what stops a
results table being split from the prose that explains it, and `section_path` is what
makes a citation legible ("3.2 Training" beats "page 4"). When detection fails the
chunker still works, but every chunk lands in one undifferentiated "Body" section -- so
the fallback rate is reported by `rag stats` rather than swallowed. A corpus with a high
fallback rate is the first thing to check when Recall@5 stalls.

Detection combines two weak signals into one usable one: typography (headings are set
larger or bolder than body text) and numbering (arXiv papers overwhelmingly use
`3.2 Training`). Either alone produces false positives -- bold run-in emphasis, or a
sentence that happens to start with a figure reference -- and requiring both is what
keeps precision high enough to be useful.
"""

import logging
import re
from dataclasses import dataclass

from rag.ingest.parse import ParsedDocument, TextSpan

logger = logging.getLogger(__name__)

# "3 Method", "3.2 Training", "A.1 Extra results". The capital after the number is what
# separates a heading from a cross-reference such as "3.2 shows that ...".
NUMBERED_HEADING = re.compile(r"^(?P<number>(?:\d+|[A-Z])(?:\.\d+)*)\.?\s+(?P<title>[A-Z].{0,80})$")

UNNUMBERED_HEADINGS = frozenset(
    {
        "abstract",
        "acknowledgement",
        "acknowledgements",
        "acknowledgment",
        "acknowledgments",
        "appendix",
        "background",
        "bibliography",
        "broader impact",
        "conclusion",
        "conclusions",
        "discussion",
        "ethics statement",
        "experiment",
        "experiments",
        "experimental setup",
        "future work",
        "introduction",
        "limitations",
        "method",
        "methods",
        "methodology",
        "preliminaries",
        "related work",
        "reproducibility statement",
        "results",
        "references",
    }
)

REFERENCE_TITLES = frozenset({"references", "bibliography"})

# A heading is set at least this much larger than body text, or is bold.
HEADING_SIZE_RATIO = 1.05
MAX_HEADING_CHARS = 100
MAX_HEADING_WORDS = 12

# A heading is a title phrase, and these two guards reject the things that merely look
# like one. Both were added after reading real output: a displayed equation such as
# "N X1N)⊺∈RD×D is the sample covariance matrix." satisfies the numbering pattern via its
# leading capital, and promoting it invents a section boundary in the middle of a
# derivation -- severing exactly the content this module exists to keep together.
#
# 1. Titles are mostly letters. Math lines are mostly operators and symbols.
# 2. Titles do not trail off in lowercase prose. "3.2 Training." is fine; "... matrix."
#    is a sentence.
MIN_ALPHABETIC_RATIO = 0.7

FALLBACK_SECTION_PATH = "Body"
MIN_HEADINGS_FOR_STRUCTURE = 2


@dataclass(frozen=True, slots=True)
class Section:
    """One detected section, located by character range in the document text."""

    path: str  # '3 Method > 3.2 Training'
    title: str
    number: str | None
    char_start: int
    char_end: int
    is_references: bool


@dataclass(frozen=True, slots=True)
class SectionMap:
    """Every section of one paper, plus whether detection had to give up."""

    sections: tuple[Section, ...]
    used_fallback: bool

    def body_sections(self, *, drop_references: bool) -> tuple[Section, ...]:
        """Sections worth chunking.

        A references list answers no question anyone would ask of this corpus, and it is
        dense with the title-like text that lexical search matches on -- so leaving it in
        actively costs precision rather than merely wasting space.
        """
        if not drop_references:
            return self.sections
        return tuple(s for s in self.sections if not s.is_references)


@dataclass(frozen=True, slots=True)
class _Line:
    """One visual line of the PDF, reassembled from its spans."""

    text: str
    char_start: int
    char_end: int
    font_size: float
    bold: bool


def _group_lines(spans: tuple[TextSpan, ...]) -> list[_Line]:
    """Reassemble spans into the lines they were laid out as.

    A heading is routinely split across several spans -- the number, the space and the
    title often carry different font records -- so testing spans individually finds
    nothing.
    """
    lines: list[_Line] = []
    current: list[TextSpan] = []

    def flush() -> None:
        if not current:
            return
        lines.append(
            _Line(
                text="".join(s.text for s in current).strip(),
                char_start=current[0].char_start,
                char_end=current[-1].char_end,
                font_size=max(s.font_size for s in current),
                bold=any(s.bold for s in current),
            )
        )

    for span in spans:
        if current and span.line != current[-1].line:
            flush()
            current = []
        current.append(span)
    flush()

    return [line for line in lines if line.text]


def _reads_like_a_title(text: str) -> bool:
    """Reject lines that match a heading pattern but are plainly not titles."""
    letters = sum(1 for c in text if c.isalpha() or c.isspace())
    if letters / len(text) < MIN_ALPHABETIC_RATIO:
        return False
    last_word = text.split()[-1] if text.split() else ""
    return not (text.rstrip().endswith(".") and last_word[:1].islower())


def _looks_typographically_like_heading(line: _Line, body_font_size: float) -> bool:
    if not (3 <= len(line.text) <= MAX_HEADING_CHARS):
        return False
    if len(line.text.split()) > MAX_HEADING_WORDS:
        return False
    if not _reads_like_a_title(line.text):
        return False
    return line.bold or line.font_size >= body_font_size * HEADING_SIZE_RATIO


def _parse_heading(text: str) -> tuple[str | None, str] | None:
    """Return (number, title) if `text` reads as a heading, else None."""
    match = NUMBERED_HEADING.match(text)
    if match is not None:
        return match.group("number"), text
    normalised = text.strip().lower().rstrip(".:")
    if normalised in UNNUMBERED_HEADINGS:
        return None, text.strip()
    return None


def detect_sections(document: ParsedDocument) -> SectionMap:
    """Build the section map for a parsed paper.

    Falls back to a single `Body` section when fewer than two headings are found, rather
    than emitting a structure it does not believe. One honest section beats a handful of
    spurious ones, because the chunker treats section boundaries as hard limits and a
    false boundary permanently severs a table from its caption.
    """
    headings: list[tuple[str | None, str, int]] = []  # (number, title, char_start)
    for line in _group_lines(document.spans):
        if not _looks_typographically_like_heading(line, document.body_font_size):
            continue
        parsed = _parse_heading(line.text)
        if parsed is None:
            continue
        number, title = parsed
        headings.append((number, title, line.char_start))

    if len(headings) < MIN_HEADINGS_FOR_STRUCTURE:
        logger.debug("section detection fell back to a single Body section")
        return SectionMap(
            sections=(
                Section(
                    path=FALLBACK_SECTION_PATH,
                    title=FALLBACK_SECTION_PATH,
                    number=None,
                    char_start=0,
                    char_end=len(document.text),
                    is_references=False,
                ),
            ),
            used_fallback=True,
        )

    sections: list[Section] = []
    stack: list[tuple[str | None, str]] = []
    for index, (number, title, char_start) in enumerate(headings):
        depth = number.count(".") + 1 if number else 1
        del stack[depth - 1 :]
        stack.append((number, title))
        char_end = headings[index + 1][2] if index + 1 < len(headings) else len(document.text)
        sections.append(
            Section(
                path=" > ".join(entry_title for _, entry_title in stack),
                title=title,
                number=number,
                char_start=char_start,
                char_end=char_end,
                is_references=_is_references(title),
            )
        )

    # Text before the first heading is the title block and abstract, which is some of the
    # most retrievable content in a paper -- it must not be silently discarded.
    if sections and sections[0].char_start > 0:
        sections.insert(
            0,
            Section(
                path="Frontmatter",
                title="Frontmatter",
                number=None,
                char_start=0,
                char_end=sections[0].char_start,
                is_references=False,
            ),
        )

    return SectionMap(sections=tuple(sections), used_fallback=False)


def _is_references(title: str) -> bool:
    normalised = NUMBERED_HEADING.sub(r"\g<title>", title.strip()).strip().lower().rstrip(".:")
    return normalised in REFERENCE_TITLES or title.strip().lower().rstrip(".:") in REFERENCE_TITLES
