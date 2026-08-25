"""Pure-function tests for app.core.sections - no network, no LLM cost."""
from app.core.sections import split_sections, trim_by_priority

# ── split_sections ────────────────────────────────────────────


def test_split_sections_no_marker_returns_whole_text_as_preamble():
    assert split_sections("just some text, no headings at all") == [
        ("", "just some text, no headings at all")
    ]


def test_split_sections_empty_text_returns_nothing():
    assert split_sections("") == []


def test_split_sections_captures_preamble_before_first_heading():
    text = "lead-in paragraph\n## Methods\nwe did things"
    sections = split_sections(text)
    assert sections[0] == ("", "lead-in paragraph\n")
    assert sections[1] == ("Methods", "## Methods\nwe did things")


def test_split_sections_no_preamble_when_text_starts_with_a_heading():
    text = "## Abstract\nsummary text\n## Results\nthe numbers"
    sections = split_sections(text)
    assert sections == [
        ("Abstract", "## Abstract\nsummary text\n"),
        ("Results", "## Results\nthe numbers"),
    ]


def test_split_sections_untitled_content_absorbed_into_preceding_section():
    # Mirrors a JATS <sec> with no <title>: its paragraphs carry no marker of
    # their own and must not be dropped or treated as a new section.
    text = "## Methods\nhow we did it\nan unlabelled subsection's text\n## Results\nfindings"
    sections = split_sections(text)
    assert sections[0][0] == "Methods"
    assert "an unlabelled subsection's text" in sections[0][1]
    assert sections[1] == ("Results", "## Results\nfindings")


def test_split_sections_reconstructs_original_text():
    text = "lead\n## A\nbody a\n## B\nbody b"
    sections = split_sections(text)
    assert "".join(body for _title, body in sections) == text


# ── trim_by_priority ──────────────────────────────────────────


def test_trim_by_priority_returns_short_text_unchanged():
    text = "## Results\nshort"
    assert trim_by_priority(text, max_chars=1000) == text


def test_trim_by_priority_first_section_starts_the_output():
    text = "## Abstract\nshort intro\n## Results\n" + ("r" * 5000)
    trimmed = trim_by_priority(text, max_chars=1000, priority=("result",))
    assert trimmed.startswith("## Abstract\nshort intro")


def test_trim_by_priority_first_section_is_bounded_not_unconditional():
    # A pathologically long first section (a real HTML page can put a long
    # lead block before its first recognised heading) must not consume the
    # entire budget and starve every section ranked after it.
    text = "## Overview\n" + ("x" * 20000) + "\n## Results\n" + ("y" * 500)
    trimmed = trim_by_priority(text, max_chars=2000, priority=("result",))
    assert "## Results" in trimmed
    assert "y" * 100 in trimmed


def test_trim_by_priority_drops_skipped_sections():
    text = (
        "## Abstract\nintro\n"
        "## References\n" + ("r" * 2000) + "\n"
        "## Results\n" + ("f" * 2000)
    )
    trimmed = trim_by_priority(
        text, max_chars=200, priority=("result",), skip=("reference",),
    )
    assert "References" not in trimmed
    assert "r" * 100 not in trimmed


def test_trim_by_priority_prefers_higher_priority_section():
    text = (
        "## Abstract\nintro\n"
        "## Introduction\n" + ("i" * 2000) + "\n"
        "## Results\n" + ("r" * 2000)
    )
    trimmed = trim_by_priority(
        text, max_chars=300, priority=("result", "introduction"),
    )
    results_start = trimmed.find("## Results")
    intro_start = trimmed.find("## Introduction")
    assert results_start != -1
    # Results is ranked first, so under a tight budget it should get at
    # least as much room as Introduction.
    results_len = len(trimmed) - results_start
    intro_len = (results_start - intro_start) if intro_start != -1 else 0
    assert results_len >= intro_len


def test_trim_by_priority_fair_share_does_not_starve_lower_priority_section():
    # A single oversized top-priority section must not consume the entire
    # remaining budget and leave nothing for the next one.
    text = (
        "## Abstract\nintro\n"
        "## Results\n" + ("r" * 10000) + "\n"
        "## Discussion\n" + ("d" * 500)
    )
    trimmed = trim_by_priority(
        text, max_chars=2000, priority=("result", "discussion"),
    )
    assert "## Discussion" in trimmed
    assert "d" * 100 in trimmed


def test_trim_by_priority_score_fn_ranks_by_score_not_by_name():
    text = "## A\n" + ("x" * 10) + "\n## B\n" + ("y" * 2000)
    trimmed = trim_by_priority(
        text, max_chars=100, score_fn=lambda body: len(body),
    )
    # B scores higher (longer body) so should be preferred over A once the
    # unconditionally-kept first section ("" preamble, empty here) is gone.
    assert "## B" in trimmed


def test_trim_by_priority_partial_tail_slice_when_room_remains():
    text = "## Abstract\nintro\n## Results\n" + ("r" * 1000)
    trimmed = trim_by_priority(text, max_chars=50, priority=("result",))
    assert trimmed.startswith("## Abstract\nintro")
    assert len(trimmed) <= 50 + len("## Abstract\nintro\n")
