import pytest

from conftest import build_document, words
from rag.config import RetrievalConfig
from rag.ingest.chunk import Chunk, chunk_document, split_sentences
from rag.ingest.sections import Section, SectionMap
from rag.ingest.tokenization import WhitespaceTokenCounter

COUNTER = WhitespaceTokenCounter()


def section(path: str, start: int, end: int, *, references: bool = False) -> Section:
    return Section(
        path=path,
        title=path,
        number=None,
        char_start=start,
        char_end=end,
        is_references=references,
    )


def single_section(text_length: int, path: str = "1 Method") -> SectionMap:
    return SectionMap(sections=(section(path, 0, text_length),), used_fallback=False)


# --- sentence splitting -----------------------------------------------------


def test_splits_on_sentence_boundaries() -> None:
    text = "First sentence here. Second sentence here. Third one."
    assert [s for s, _, _ in split_sentences(text)] == [
        "First sentence here.",
        "Second sentence here.",
        "Third one.",
    ]


def test_offsets_point_back_into_the_source() -> None:
    text = "Alpha beta. Gamma delta. Alpha beta."
    for sentence, start, end in split_sentences(text):
        assert text[start:end] == sentence
    # The repeated first sentence must resolve to its own occurrence, not the earlier
    # identical one -- this is why offsets are returned rather than searched for.
    assert split_sentences(text)[2][1] == 25


@pytest.mark.parametrize(
    "abbreviation", ["e.g.", "i.e.", "et al.", "Fig.", "Eq.", "Sec.", "cf.", "vs."]
)
def test_does_not_split_after_abbreviations(abbreviation: str) -> None:
    text = f"We follow prior work {abbreviation} Smith and colleagues in this setup."
    assert len(split_sentences(text)) == 1


def test_handles_text_with_no_terminal_punctuation() -> None:
    assert [s for s, _, _ in split_sentences("no punctuation at all")] == ["no punctuation at all"]


def test_empty_text_yields_no_sentences() -> None:
    assert split_sentences("   ") == []


# --- chunk boundaries -------------------------------------------------------


def chunk(text: str, config: RetrievalConfig, *, sections: SectionMap | None = None) -> list[Chunk]:
    document = build_document([(text, 1, 10.0, False)])
    return chunk_document(
        document,
        sections or single_section(len(document.text)),
        paper_title="A Paper",
        config=config,
        counter=COUNTER,
    )


def test_never_exceeds_the_token_budget() -> None:
    # Budget 20; each sentence is 6 tokens, so a chunk holds at most 3.
    config = RetrievalConfig(chunk_tokens=20, chunk_overlap_pct=0.0, contextual_headers=False)
    text = " ".join(words(6, f"s{i}_") for i in range(9))
    chunks = chunk(text, config)
    assert chunks
    for produced in chunks:
        assert produced.token_count <= 20


def test_packs_up_to_the_budget_rather_than_one_sentence_per_chunk() -> None:
    config = RetrievalConfig(chunk_tokens=20, chunk_overlap_pct=0.0, contextual_headers=False)
    text = " ".join(words(6, f"s{i}_") for i in range(9))
    chunks = chunk(text, config)
    assert [c.token_count for c in chunks] == [18, 18, 18]


def test_never_splits_mid_sentence() -> None:
    config = RetrievalConfig(chunk_tokens=20, chunk_overlap_pct=0.0, contextual_headers=False)
    sentences = [words(6, f"s{i}_") for i in range(9)]
    chunks = chunk(" ".join(sentences), config)
    for produced in chunks:
        # A chunk is a run of whole sentences: it begins where one begins and ends where
        # one ends. A mid-sentence split would break one end or the other.
        assert any(produced.content.startswith(s) for s in sentences)
        assert produced.content.endswith("end.")


def test_overlap_carries_the_exact_trailing_sentences() -> None:
    # Budget 20, overlap 50% -> overlap_tokens = 10. Chunk 1 holds sentences 0,1,2.
    # Walking back from the end: sentence 2 is 6 tokens (fits in 10), sentence 1 would
    # bring the total to 12 (does not fit). So chunk 2 begins with sentence 2 exactly.
    config = RetrievalConfig(chunk_tokens=20, chunk_overlap_pct=0.5, contextual_headers=False)
    sentences = [words(6, f"s{i}_") for i in range(9)]
    chunks = chunk(" ".join(sentences), config)
    assert chunks[0].content.startswith(sentences[0])
    assert chunks[0].content.endswith(sentences[2])
    assert chunks[1].content.startswith(sentences[2])


def test_zero_overlap_produces_disjoint_chunks() -> None:
    config = RetrievalConfig(chunk_tokens=20, chunk_overlap_pct=0.0, contextual_headers=False)
    sentences = [words(6, f"s{i}_") for i in range(9)]
    chunks = chunk(" ".join(sentences), config)
    assert chunks[1].content.startswith(sentences[3])


def test_overlap_never_pushes_a_chunk_past_the_budget() -> None:
    # A large overlap plus a large incoming sentence would otherwise overflow; the
    # overlap is dropped instead, because overflow means silent truncation downstream.
    config = RetrievalConfig(chunk_tokens=20, chunk_overlap_pct=0.9, contextual_headers=False)
    text = " ".join(words(9, f"s{i}_") for i in range(8))
    for produced in chunk(text, config):
        assert produced.token_count <= 20


def test_an_over_long_sentence_is_hard_split_rather_than_emitted_whole() -> None:
    config = RetrievalConfig(chunk_tokens=10, chunk_overlap_pct=0.0, contextual_headers=False)
    chunks = chunk(words(35, "w"), config)
    assert len(chunks) > 1
    for produced in chunks:
        assert produced.token_count <= 10


# --- section boundaries -----------------------------------------------------


def test_chunks_never_cross_a_section_boundary() -> None:
    config = RetrievalConfig(chunk_tokens=100, chunk_overlap_pct=0.0, contextual_headers=False)
    first = " ".join(words(6, "a") for _ in range(3))
    second = " ".join(words(6, "b") for _ in range(3))
    document = build_document([(first, 1, 10.0, False), (second, 2, 10.0, False)])
    boundary = document.text.index(second)
    sections = SectionMap(
        sections=(
            section("1 Method", 0, boundary),
            section("2 Results", boundary, len(document.text)),
        ),
        used_fallback=False,
    )

    chunks = chunk_document(
        document, sections, paper_title="A Paper", config=config, counter=COUNTER
    )
    # The budget of 100 could hold both sections at once; the boundary is what stops it.
    assert len(chunks) == 2
    assert {c.section_path for c in chunks} == {"1 Method", "2 Results"}
    for produced in chunks:
        assert not ("a0" in produced.content and "b0" in produced.content)


def test_references_are_dropped_by_default() -> None:
    config = RetrievalConfig(chunk_tokens=50, contextual_headers=False)
    body = " ".join(words(6, "a") for _ in range(2))
    refs = " ".join(words(6, "r") for _ in range(2))
    document = build_document([(body, 1, 10.0, False), (refs, 2, 10.0, False)])
    boundary = document.text.index(refs)
    sections = SectionMap(
        sections=(
            section("1 Method", 0, boundary),
            section("References", boundary, len(document.text), references=True),
        ),
        used_fallback=False,
    )

    kept = chunk_document(document, sections, paper_title="P", config=config, counter=COUNTER)
    assert {c.section_path for c in kept} == {"1 Method"}

    retained = chunk_document(
        document,
        sections,
        paper_title="P",
        config=config.model_copy(update={"drop_references": False}),
        counter=COUNTER,
    )
    assert "References" in {c.section_path for c in retained}


def test_figure_only_pages_are_dropped_by_default() -> None:
    config = RetrievalConfig(chunk_tokens=50, contextual_headers=False)
    body = " ".join(words(6, "a") for _ in range(2))
    caption = words(6, "cap")
    document = build_document(
        [(body, 1, 10.0, False), (caption, 2, 10.0, False)], figure_only_pages=frozenset({2})
    )
    chunks = chunk_document(
        document,
        single_section(len(document.text)),
        paper_title="P",
        config=config,
        counter=COUNTER,
    )
    assert "cap0" not in " ".join(c.content for c in chunks)


# --- contextual headers -----------------------------------------------------


def test_header_is_in_embed_input_but_never_in_content() -> None:
    config = RetrievalConfig(chunk_tokens=60, contextual_headers=True)
    chunks = chunk(" ".join(words(6, "a") for _ in range(4)), config)
    produced = chunks[0]
    assert produced.embed_input.startswith("A Paper — 1 Method\n\n")
    assert "A Paper" not in produced.content


def test_header_reduces_the_content_budget() -> None:
    # Stated plainly because it is a real confound in the Phase 7 headers arm: the
    # header is charged against the same budget as the content.
    text = " ".join(words(6, f"s{i}_") for i in range(12))
    without = chunk(text, RetrievalConfig(chunk_tokens=30, contextual_headers=False))
    with_header = chunk(text, RetrievalConfig(chunk_tokens=30, contextual_headers=True))
    # The 5-token header is charged against the same 30-token budget, leaving 25 for
    # content: five sentences fit without it, only four with it.
    assert COUNTER.count(without[0].content) == 30
    assert COUNTER.count(with_header[0].content) == 24


# --- record integrity -------------------------------------------------------


def test_ordinals_are_contiguous_and_start_at_zero() -> None:
    config = RetrievalConfig(chunk_tokens=20, chunk_overlap_pct=0.0, contextual_headers=False)
    chunks = chunk(" ".join(words(6, f"s{i}_") for i in range(9)), config)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_char_offsets_are_ordered_and_within_the_document() -> None:
    config = RetrievalConfig(chunk_tokens=20, chunk_overlap_pct=0.0, contextual_headers=False)
    text = " ".join(words(6, f"s{i}_") for i in range(9))
    document = build_document([(text, 1, 10.0, False)])
    for produced in chunk(text, config):
        assert 0 <= produced.char_start < produced.char_end <= len(document.text)


def test_identical_input_produces_identical_digests() -> None:
    # Ingestion has to be reproducible or the embedding cache and every committed eval
    # result silently stop referring to the same chunks.
    config = RetrievalConfig(chunk_tokens=20, contextual_headers=False)
    text = " ".join(words(6, f"s{i}_") for i in range(9))
    assert [c.content_sha256 for c in chunk(text, config)] == [
        c.content_sha256 for c in chunk(text, config)
    ]


def test_empty_document_produces_no_chunks() -> None:
    config = RetrievalConfig()
    document = build_document([("", 1, 10.0, False)])
    assert (
        chunk_document(
            document,
            single_section(0),
            paper_title="P",
            config=config,
            counter=COUNTER,
        )
        == []
    )
