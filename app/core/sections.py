"""Section-aware trimming for content that carries ``## Heading`` (or similar)
markers.

A character budget forces a choice about which part of a long document a
model actually sees. Slicing from the start treats every section as equally
worth keeping, which for structured content it is not: an RCT's Results
routinely matters more than its Methods, and a source's own section markers
already say which is which. This module turns those markers into a priority-
ordered fill instead of a blind prefix.

Fitting under the budget still means some content is left out - a fixed
character count can't hold an entire long document - so the fill is
fair-share rather than winner-take-all: every surviving section gets an equal
slice of what's left after the always-kept first section, and only the
leftover from sections that needed less than their share flows to higher-
priority sections that wanted more. A single oversized top-priority section
can consume its own share and the slack, but never the whole budget, so a
long Results section cannot silently reduce Discussion to nothing.

Used by :func:`app.sources.pubmed._fetch_pmc_fulltext` for JATS ``## Heading``
markers and by :func:`app.sources.wikipedia.WikipediaProvider._smart_trim` for
MediaWiki's ``== Heading ==`` convention - one algorithm, two marker shapes
and two priority policies (a fixed biomedical section order here, a factual-
density score there).
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence

# `^` in MULTILINE mode matches the start of each line without consuming its
# own leading newline, so a section's slice runs from its own `##` through to
# (not including) the next one - no accumulated blank lines when sections are
# dropped or reordered.
DEFAULT_MARKER = re.compile(r"(?m)^##\s+(.+?)\s*$")

# Priority order for biomedical full text: the sections `finding_direction`
# actually depends on, ranked above the sections that only set up for them.
# Matched as a case-insensitive substring of the heading, so "Results and
# Discussion", "2. Methods" and "Patients and Methods" all match correctly
# despite journals never agreeing on exact wording.
BIOMED_PRIORITY: tuple[str, ...] = (
    "result", "finding", "outcome", "efficacy",
    "discussion", "conclusion",
    "method", "material",
    "introduction", "background",
)

# Sections that never bear on whether a source supports or contradicts a
# claim - administrative boilerplate a priority fill would otherwise spend
# budget on ahead of a truncated Discussion section.
BIOMED_SKIP: tuple[str, ...] = (
    "reference", "bibliography", "acknowledg", "conflict of interest",
    "funding", "author contribution", "supplementary", "data availability",
)


def split_sections(
    text: str, marker: re.Pattern[str] | str = DEFAULT_MARKER,
) -> list[tuple[str, str]]:
    """Split *text* into ``(title, section_text)`` pairs on lines matching *marker*.

    *section_text* runs from its own heading line up to (not including) the
    next one, so concatenating every returned pair's second element
    reconstructs *text* exactly. Content
    before the first heading - a lead-in paragraph, or the whole text when no
    heading matches at all - comes back as one pair with title ``""``.

    Content between one heading and the next belongs entirely to the
    preceding heading. A JATS `<sec>` with no `<title>` therefore contributes
    no heading of its own and is absorbed into whatever section came before
    it, rather than being dropped or left unscored.
    """
    if isinstance(marker, str):
        marker = re.compile(marker, re.MULTILINE)
    matches = list(marker.finditer(text))
    if not matches:
        return [("", text)] if text else []

    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        sections.append(("", text[: matches[0].start()]))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((m.group(1).strip(), text[m.start() : end]))
    return sections


def trim_by_priority(
    text: str,
    max_chars: int,
    *,
    marker: re.Pattern[str] | str = DEFAULT_MARKER,
    priority: Sequence[str] = (),
    skip: Sequence[str] = (),
    score_fn: Callable[[str], float] | None = None,
    always_keep_first: bool = True,
) -> str:
    """Trim *text* to *max_chars*, filling by section priority.

    Sections matching *skip* are dropped outright. Everything left is ranked
    - either by *priority* (a name list, matched case-insensitively as a
    substring of the heading) or by *score_fn* over section text, when
    priority isn't expressible as a fixed name order - with the first section
    ranked ahead of all of them when *always_keep_first* is set, since it is
    the one every caller of this function already trusts most (an abstract,
    a Wikipedia intro). Ranked sections are then filled fair-share: the
    budget is split evenly across all of them, and only the slack left by a
    section shorter than its share is handed to higher-ranked sections that
    wanted more. This bounds every section, first included, to at most its
    share plus whatever slack ranks ahead of it earns - so one long section,
    first or not, can claim extra room but never the entire budget outright
    and starve everything ranked below it.

    Returns *text* unchanged if it already fits within *max_chars*.
    """
    if len(text) <= max_chars:
        return text

    sections = split_sections(text, marker)
    if not sections:
        return text[:max_chars]

    def _matches(title: str, names: Sequence[str]) -> bool:
        lowered = title.lower()
        return any(name in lowered for name in names)

    if always_keep_first and sections:
        first, rest = sections[0], sections[1:]
    else:
        first, rest = None, sections

    kept = [(title, body) for title, body in rest if not _matches(title, skip)]

    def _priority_rank(title: str) -> int:
        lowered = title.lower()
        for rank, name in enumerate(priority):
            if name in lowered:
                return rank
        return len(priority)

    if score_fn is not None:
        kept.sort(key=lambda item: score_fn(item[1]), reverse=True)
    else:
        kept.sort(key=lambda item: _priority_rank(item[0]))

    ordered = ([first] if first is not None else []) + kept
    if not ordered:
        return ""

    share = max_chars // len(ordered)
    allocations = [min(len(body), share) for _title, body in ordered]
    slack = max_chars - sum(allocations)
    for i, (_title, body) in enumerate(ordered):
        if slack <= 0:
            break
        wants = len(body) - allocations[i]
        if wants <= 0:
            continue
        grant = min(wants, slack)
        allocations[i] += grant
        slack -= grant

    parts = [
        body[:allocation]
        for (_title, body), allocation in zip(ordered, allocations, strict=True)
        if allocation > 0
    ]
    # No separator needed: each section already carries its own leading
    # heading line and the whitespace preceding it (see split_sections), so
    # joining with "" reconstructs the same layout the source had rather than
    # inserting a blank line between every kept section.
    return "".join(parts)
