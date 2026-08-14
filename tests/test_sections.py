from dataclasses import replace
from itertools import pairwise

import pytest

from conftest import build_document
from rag.ingest.sections import detect_sections

BODY = 10.0
HEADING = 12.0


def test_builds_a_hierarchical_section_path() -> None:
    document = build_document(
        [
            ("1 Introduction", 1, HEADING, True),
            ("We study retrieval over scientific papers in this work.", 1, BODY, False),
            ("3 Method", 1, HEADING, True),
            ("Our approach has two stages.", 1, BODY, False),
            ("3.2 Training", 2, HEADING, True),
            ("We train for ten epochs on eight GPUs.", 2, BODY, False),
        ]
    )
    result = detect_sections(document)
    assert not result.used_fallback
    assert [s.path for s in result.sections] == [
        "1 Introduction",
        "3 Method",
        "3 Method > 3.2 Training",
    ]


def test_subsection_pops_back_to_the_right_level() -> None:
    document = build_document(
        [
            ("2 Method", 1, HEADING, True),
            ("Body text for the method section here.", 1, BODY, False),
            ("2.1 Data", 1, HEADING, True),
            ("Body text about the data used here.", 1, BODY, False),
            ("3 Results", 2, HEADING, True),
            ("Body text about results obtained here.", 2, BODY, False),
        ]
    )
    paths = [s.path for s in detect_sections(document).sections]
    # 3 Results is a top-level section: it must not inherit 2.1 Data as a parent.
    assert paths[-1] == "3 Results"


def test_detects_unnumbered_headings_by_name() -> None:
    document = build_document(
        [
            ("Abstract", 1, HEADING, True),
            ("We present a system for retrieval over papers.", 1, BODY, False),
            ("Introduction", 1, HEADING, True),
            ("Retrieval augmented generation is widely used now.", 1, BODY, False),
        ]
    )
    result = detect_sections(document)
    assert [s.path for s in result.sections] == ["Abstract", "Introduction"]


def test_body_text_is_not_mistaken_for_a_heading() -> None:
    # Body font, and too long -- both signals must reject it.
    document = build_document(
        [
            ("1 Introduction", 1, HEADING, True),
            (
                "3.2 shows that the model improves substantially over the baseline "
                "across every dataset we evaluated in this study.",
                1,
                BODY,
                False,
            ),
            ("2 Method", 1, HEADING, True),
            ("We describe the approach below.", 1, BODY, False),
        ]
    )
    assert [s.path for s in detect_sections(document).sections] == ["1 Introduction", "2 Method"]


def test_bold_emphasis_is_not_a_heading() -> None:
    # Short and bold, but matches no heading pattern, so typography alone must not
    # promote it. Requiring both signals is what keeps precision usable.
    document = build_document(
        [
            ("1 Introduction", 1, HEADING, True),
            ("Note that", 1, BODY, True),
            ("2 Method", 1, HEADING, True),
            ("We describe the approach below.", 1, BODY, False),
        ]
    )
    assert [s.path for s in detect_sections(document).sections] == ["1 Introduction", "2 Method"]


@pytest.mark.parametrize(
    "line",
    [
        # Found in real output: a displayed equation whose leading capital satisfies the
        # numbering pattern. Promoting it invents a boundary mid-derivation.
        "N X1N)⊺∈RD×D is the sample covariance matrix.",
        "M = ∑x∈B 2ℓ′ xA(x) + ηH⊺H",
        "3.2 shows the model improves over the baseline.",
    ],
)
def test_math_and_prose_are_not_promoted_to_headings(line: str) -> None:
    document = build_document(
        [
            ("1 Introduction", 1, HEADING, True),
            (line, 1, HEADING, True),  # heading-sized and bold: typography alone passes
            ("2 Method", 1, HEADING, True),
            ("We describe the approach below.", 1, BODY, False),
        ]
    )
    assert [s.path for s in detect_sections(document).sections] == ["1 Introduction", "2 Method"]


@pytest.mark.parametrize(
    "line", ["3.2 Training", "A.1 Extra Results", "6.5. Estimation Error", "Related Work"]
)
def test_genuine_headings_survive_the_title_guards(line: str) -> None:
    document = build_document(
        [
            ("1 Introduction", 1, HEADING, True),
            ("Some introductory text goes here.", 1, BODY, False),
            (line, 1, HEADING, True),
            ("Some more body text goes here.", 1, BODY, False),
        ]
    )
    assert any(line in s.path for s in detect_sections(document).sections)


def test_falls_back_to_a_single_body_section() -> None:
    # One honest section beats a handful of spurious ones: the chunker treats section
    # boundaries as hard limits, so a false boundary permanently severs related text.
    document = build_document(
        [
            ("Some text without any detectable structure at all.", 1, BODY, False),
            ("More text of the same undifferentiated kind here.", 1, BODY, False),
        ]
    )
    result = detect_sections(document)
    assert result.used_fallback
    assert [s.path for s in result.sections] == ["Body"]
    assert result.sections[0].char_start == 0
    assert result.sections[0].char_end == len(document.text)


def test_text_before_the_first_heading_becomes_frontmatter() -> None:
    # The title block and abstract are among the most retrievable text in a paper and
    # must not be dropped just because they precede the first numbered heading.
    document = build_document(
        [
            ("Attention Is All You Need", 1, 16.0, True),
            ("We propose a new simple network architecture.", 1, BODY, False),
            ("1 Introduction", 1, HEADING, True),
            ("Recurrent networks have long been established.", 1, BODY, False),
            ("2 Method", 1, HEADING, True),
            ("The Transformer follows this overall architecture.", 1, BODY, False),
        ]
    )
    result = detect_sections(document)
    assert result.sections[0].path == "Frontmatter"
    assert result.sections[0].char_start == 0
    assert (
        "We propose a new simple network architecture."
        in (document.text[result.sections[0].char_start : result.sections[0].char_end])
    )


def test_sections_tile_the_document_without_gaps() -> None:
    document = build_document(
        [
            ("1 Introduction", 1, HEADING, True),
            ("Some introductory text goes here.", 1, BODY, False),
            ("2 Method", 1, HEADING, True),
            ("Some method text goes here too.", 1, BODY, False),
        ]
    )
    sections = detect_sections(document).sections
    for earlier, later in pairwise(sections):
        assert earlier.char_end == later.char_start
    assert sections[-1].char_end == len(document.text)


def test_references_section_is_flagged_and_droppable() -> None:
    document = build_document(
        [
            ("1 Introduction", 1, HEADING, True),
            ("Some introductory text goes here.", 1, BODY, False),
            ("References", 2, HEADING, True),
            ("Smith et al. A paper title. In Proceedings, 2020.", 2, BODY, False),
        ]
    )
    result = detect_sections(document)
    assert [s.is_references for s in result.sections] == [False, True]
    assert [s.path for s in result.body_sections(drop_references=True)] == ["1 Introduction"]
    assert len(result.body_sections(drop_references=False)) == 2


def test_numbered_references_heading_is_also_flagged() -> None:
    document = build_document(
        [
            ("1 Introduction", 1, HEADING, True),
            ("Some introductory text goes here.", 1, BODY, False),
            ("5 References", 2, HEADING, True),
            ("Smith et al. A paper title. In Proceedings, 2020.", 2, BODY, False),
        ]
    )
    result = detect_sections(document)
    assert result.sections[-1].is_references


def test_headings_split_across_spans_are_still_detected() -> None:
    # A heading routinely arrives as several spans -- the number, the space and the
    # title often carry different font records -- so detection has to regroup spans into
    # lines before testing them. Testing spans individually finds nothing.
    document = build_document(
        [
            ("1 Introduction", 1, HEADING, True),
            ("Some introductory text goes here.", 1, BODY, False),
            ("2 Method", 1, HEADING, True),
            ("Some method text goes here too.", 1, BODY, False),
        ]
    )
    intact = detect_sections(document)
    assert [s.path for s in intact.sections] == ["1 Introduction", "2 Method"]

    heading, *rest = document.spans
    fragmented = replace(
        document,
        spans=(
            replace(heading, text="1 ", char_end=heading.char_start + 2),
            replace(heading, text="Introduction", char_start=heading.char_start + 2),
            *rest,
        ),
    )
    assert [s.path for s in detect_sections(fragmented).sections] == [
        "1 Introduction",
        "2 Method",
    ]
