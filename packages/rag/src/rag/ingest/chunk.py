"""Structure-aware chunking.

Fixed-size chunking is the single largest quality killer in RAG, and papers punish it
hardest: split a results table from its caption and no retriever recovers the pair. So
this chunker obeys three hard rules, in priority order:

1. Never cross a section boundary. A chunk spanning the end of Method and the start of
   Results describes neither, and embeds to a point between two meanings.
2. Never split mid-sentence. A truncated clause retrieves badly and reads worse when it
   lands in a citation panel.
3. Only then, fill up to the token budget with the configured overlap.

Overlap exists because rule 2 means chunk boundaries land wherever sentences happen to
end, and a fact stated across two sentences would otherwise be split with no chunk
containing both halves.
"""

import hashlib
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from rag.config import RetrievalConfig
from rag.ingest.parse import ParsedDocument
from rag.ingest.sections import Section, SectionMap
from rag.ingest.tokenization import TokenCounter, content_budget

logger = logging.getLogger(__name__)

# A sentence ends at .!? followed by whitespace and something that starts a new
# sentence. Deliberately conservative: over-splitting costs more than under-splitting,
# because an over-split chunk loses context the reranker cannot restore.
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])[ \t\n]+(?=[A-Z(\[\"'])")

# Tokens that end in a period without ending a sentence. Splitting after "et al." or
# "Fig." fragments exactly the citation- and figure-dense prose that matters most here.
ABBREVIATIONS = frozenset(
    {
        "al.",
        "approx.",
        "ca.",
        "cf.",
        "e.g.",
        "eq.",
        "eqs.",
        "etc.",
        "fig.",
        "figs.",
        "i.e.",
        "inc.",
        "no.",
        "pp.",
        "ref.",
        "refs.",
        "resp.",
        "sec.",
        "secs.",
        "st.",
        "tab.",
        "tabs.",
        "vs.",
        "vol.",
    }
)


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable unit of a paper."""

    ordinal: int
    section_path: str
    content: str  # what the user is shown
    embed_input: str  # what the embedding model sees
    token_count: int  # of embed_input, since that is what the budget governs
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    content_sha256: str
    # Digest of embed_input, not of content. These differ whenever contextual headers are
    # on, and the embedding cache must key on what the model actually saw.
    embed_input_sha256: str


@dataclass(frozen=True, slots=True)
class _Sentence:
    text: str
    char_start: int
    char_end: int
    tokens: int


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Split into sentences, returning each with its character offsets into `text`.

    Offsets are returned rather than recomputed by the caller because a chunk's
    `char_start`/`char_end` must survive back into the source document for citation
    highlighting, and searching for the sentence text again would find the wrong copy
    whenever a paper repeats a phrase.
    """
    sentences: list[tuple[str, int, int]] = []
    start = 0
    for match in SENTENCE_BOUNDARY.finditer(text):
        candidate_end = match.start()
        preceding = text[start:candidate_end]
        last_word = preceding.rsplit(maxsplit=1)[-1].lower() if preceding.strip() else ""
        if last_word in ABBREVIATIONS:
            continue
        stripped = preceding.strip()
        if stripped:
            offset = preceding.index(stripped[0]) if stripped else 0
            sentences.append((stripped, start + offset, start + offset + len(stripped)))
        start = match.end()

    tail = text[start:]
    stripped_tail = tail.strip()
    if stripped_tail:
        offset = tail.index(stripped_tail[0])
        sentences.append((stripped_tail, start + offset, start + offset + len(stripped_tail)))
    return sentences


def _header_for(paper_title: str, section_path: str) -> str:
    return f"{paper_title} — {section_path}\n\n"


def _hard_split(sentence: _Sentence, budget: int, counter: TokenCounter) -> list[_Sentence]:
    """Break a single over-long sentence on word boundaries.

    Reached by table dumps and equation blocks, where PyMuPDF returns hundreds of tokens
    with no terminal punctuation. Rule 2 has to yield here -- the alternative is emitting
    a chunk the embedding model truncates, which loses the tail with no error raised.

    Offsets are taken from the original string rather than from the rejoined words, so a
    piece still points at the region of the source document it came from even where the
    PDF used irregular spacing.
    """
    words = [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", sentence.text)]
    if not words:
        return []

    def build(group: list[tuple[str, int, int]]) -> _Sentence:
        text = sentence.text[group[0][1] : group[-1][2]]
        return _Sentence(
            text=text,
            char_start=sentence.char_start + group[0][1],
            char_end=sentence.char_start + group[-1][2],
            tokens=counter.count(text),
        )

    pieces: list[_Sentence] = []
    current: list[tuple[str, int, int]] = []
    for word in words:
        candidate = [*current, word]
        if current and counter.count(" ".join(w for w, _, _ in candidate)) > budget:
            pieces.append(build(current))
            current = [word]
        else:
            current = candidate
    if current:
        pieces.append(build(current))
    return pieces


def _overlap_suffix(emitted: Sequence[_Sentence], overlap_tokens: int) -> list[_Sentence]:
    """Longest trailing run of sentences fitting inside the overlap budget.

    Bounded to a proper suffix: seeding the next chunk with everything the last one
    contained would make no forward progress and loop forever.
    """
    if overlap_tokens <= 0 or len(emitted) < 2:
        return []
    suffix: list[_Sentence] = []
    total = 0
    for sentence in reversed(emitted[1:]):
        if total + sentence.tokens > overlap_tokens:
            break
        suffix.insert(0, sentence)
        total += sentence.tokens
    return suffix


def _section_sentences(
    document: ParsedDocument,
    section: Section,
    config: RetrievalConfig,
    counter: TokenCounter,
    budget: int,
) -> list[_Sentence]:
    raw = document.text[section.char_start : section.char_end]
    sentences: list[_Sentence] = []
    for text, rel_start, rel_end in split_sentences(raw):
        abs_start = section.char_start + rel_start
        abs_end = section.char_start + rel_end
        if config.drop_figure_only_pages and _is_figure_only(document, abs_start, abs_end):
            continue
        sentence = _Sentence(text, abs_start, abs_end, counter.count(text))
        if sentence.tokens > budget:
            sentences.extend(_hard_split(sentence, budget, counter))
        else:
            sentences.append(sentence)
    return sentences


def _is_figure_only(document: ParsedDocument, char_start: int, char_end: int) -> bool:
    first, last = document.page_range(char_start, char_end)
    return set(range(first, last + 1)) <= document.figure_only_pages


def chunk_document(
    document: ParsedDocument,
    section_map: SectionMap,
    *,
    paper_title: str,
    config: RetrievalConfig,
    counter: TokenCounter,
) -> list[Chunk]:
    """Split a parsed paper into retrievable chunks under `config`.

    `ordinal` is assigned across the whole paper rather than per section, so it orders
    chunks as a reader would encounter them.
    """
    chunks: list[Chunk] = []

    def emit(buffer: list[_Sentence], section_path: str, prefix: str) -> None:
        if not buffer:
            return
        # `content` is the retained text; char_start/char_end bound the region of the
        # source document it came from. The two differ where figure-only sentences were
        # dropped from inside the range, so Phase 8 highlights by locating `content` on
        # the page rather than by slicing blindly.
        content = " ".join(s.text for s in buffer)
        embed_input = f"{prefix}{content}"
        char_start, char_end = buffer[0].char_start, buffer[-1].char_end
        page_start, page_end = document.page_range(char_start, char_end)
        chunks.append(
            Chunk(
                ordinal=len(chunks),
                section_path=section_path,
                content=content,
                embed_input=embed_input,
                token_count=counter.count(embed_input),
                page_start=page_start,
                page_end=page_end,
                char_start=char_start,
                char_end=char_end,
                content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                embed_input_sha256=hashlib.sha256(embed_input.encode("utf-8")).hexdigest(),
            )
        )

    for section in section_map.body_sections(drop_references=config.drop_references):
        header = _header_for(paper_title, section.path) if config.contextual_headers else ""
        header_tokens = counter.count(header) if header else 0
        budget = content_budget(config.embedding_model, config.chunk_tokens, header_tokens)
        overlap_tokens = int(budget * config.chunk_overlap_pct)

        pending: list[_Sentence] = []
        pending_tokens = 0

        for sentence in _section_sentences(document, section, config, counter, budget):
            if pending and pending_tokens + sentence.tokens > budget:
                emit(pending, section.path, header)
                suffix = _overlap_suffix(pending, overlap_tokens)
                # An overlap that cannot fit alongside the incoming sentence is dropped
                # rather than carried: keeping it would push the next chunk past the
                # budget and back into silent truncation, which is the failure this
                # whole module exists to avoid.
                if sum(s.tokens for s in suffix) + sentence.tokens > budget:
                    suffix = []
                pending = suffix
                pending_tokens = sum(s.tokens for s in pending)
            pending.append(sentence)
            pending_tokens += sentence.tokens

        emit(pending, section.path, header)

    logger.debug("chunked '%s' into %d chunks", paper_title[:60], len(chunks))
    return chunks
