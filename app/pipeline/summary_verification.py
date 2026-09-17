"""Grounding check for `synthesise`'s free-text `global_summary`, the other
half of what claim_verification.py checks for `key_findings`.

`key_findings` and `global_summary` are sibling outputs of the same
`synthesise` call, not parent and child - both cite source indices directly
(`_SYNTHESIS_SYSTEM` requires a bracketed `[n]` after every substantive
statement in the overview, the same as a claim's `source_indices`), so
verifying `key_findings` against its sources says nothing about whether the
prose overview stays within what those sources establish. A synthesis can
smooth a disagreement into confident prose, or generalise one finding into a
broader statement than its source made, without ever touching a claim.

This reuses claim_verification.verify_claims rather than re-implementing
entailment checking: `_SYNTHESIS_SYSTEM`'s citation requirement already
gives the overview the same shape a claim has - text plus the source indices
it rests on - once split into pieces at their citation markers. split_into_
chunks does that split mechanically, on the `[n]` markers themselves rather
than sentence punctuation: the text since the last citation (or paragraph
start) up to and including the next `[n]`/`[n][m]` cluster is one chunk,
covering exactly what that citation is claimed to back, including a
citation that trails two short sentences at once - a sentence-first split on
`.`/`?`/`!` would misread the first of those as uncited. Text left over
after a paragraph's last citation (or a paragraph with none at all) becomes
its own chunk with no source indices - see _is_substantive below for what
happens to it.

verify_summary turns each cited chunk into a throwaway Claim and hands the
list straight to verify_claims - no separate TypeSafe integration, just a
different source of (text, source_indices) pairs. A chunk with no citation
at all never reaches TypeSafe: `_SYNTHESIS_SYSTEM` calls citing every
substantive statement mandatory, so a long uncited chunk is flagged directly
from the split, no model call needed to know a rule was broken.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.config import Settings
from app.core.job_stats import JobStats
from app.pipeline import claim_verification
from app.pipeline.claim_verification import ClaimVerdict
from app.schemas import ArticleSummary, Claim

logger = logging.getLogger(__name__)

# One or more adjacent bracketed indices - "[0]" or "[1][3]" - the exact
# citation shape _SYNTHESIS_SYSTEM asks the model to produce. A little
# whitespace tolerance between adjacent brackets ("[1] [3]") costs nothing
# and the model doesn't always follow the no-space example to the letter.
# The optional trailing punctuation mark absorbs a sentence's own closing
# ".", "," etc immediately after the citation ("...1.2% [0].") into the
# boundary itself - otherwise that lone character would spawn its own
# meaningless trailing chunk with no citation of its own.
_CITATION_CLUSTER_RE = re.compile(r"(?:\[\d+\]\s*)+[.,;:!?]?")

# A chunk with no citation shorter than this (in words, after stripping
# light Markdown emphasis) is treated as connective tissue - "Inoltre,"
# "However," a lone caveat clause - not a substantive, checkable statement.
# Flagging every short fragment would bury the real findings in noise.
_MIN_FLAG_WORDS = 6

# Deliberately lower than claim_verification.REJECT_CONFIDENCE (0.85): that
# constant gates dropping a claim outright, a hard-to-reverse action: this
# one only gates surfacing a flag for a human to look at, which is cheap to
# get wrong in the direction of a false positive. Tuned against a real
# TypeSafe run (see ts_test5 in the working notes), not guessed: two genuine
# overreaches - a claim generalising past what its source established, and a
# claim citing a source addressing a different aspect entirely - verified as
# "says_nothing" at confidence 0.75-0.76, well below claim_verification's
# 0.85 bar, while the two genuinely well-cited chunks in the same run scored
# 0.99-1.0. Reusing 0.85 here would have silently let both real overreaches
# through unflagged.
FLAG_CONFIDENCE = 0.6


@dataclass
class SummaryChunk:
    """One piece of `global_summary`, split at its own citation markers.

    *source_indices* is empty for a chunk with no citation attached - either
    trivial connective text, or a genuine violation of the "every
    substantive statement is cited" rule, distinguished by _is_substantive.
    """

    text: str
    source_indices: list[int]


def _parse_cluster(cluster: str) -> list[int]:
    return [int(n) for n in re.findall(r"\d+", cluster)]


def _split_paragraph(paragraph: str) -> list[SummaryChunk]:
    chunks: list[SummaryChunk] = []
    pos = 0
    for match in _CITATION_CLUSTER_RE.finditer(paragraph):
        text = paragraph[pos:match.start()].strip()
        if text:
            chunks.append(SummaryChunk(text=text, source_indices=_parse_cluster(match.group())))
        pos = match.end()
    trailing = paragraph[pos:].strip()
    if trailing:
        chunks.append(SummaryChunk(text=trailing, source_indices=[]))
    return chunks


def split_into_chunks(global_summary: str) -> list[SummaryChunk]:
    """Split `global_summary` into citation-bounded chunks, paragraph by
    paragraph so an uncited tail in one paragraph never merges with the next
    paragraph's opening text into a single, wrongly-attributed chunk.
    """
    chunks: list[SummaryChunk] = []
    for paragraph in re.split(r"\n\s*\n", global_summary):
        # Internal single newlines (a wrapped line, not a paragraph break)
        # would otherwise split a citation from the text right before it if
        # a line happens to break there.
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if paragraph:
            chunks.extend(_split_paragraph(paragraph))
    return chunks


def _is_substantive(text: str) -> bool:
    return len(re.sub(r"[*_]", "", text).split()) >= _MIN_FLAG_WORDS


def _snippet(text: str, limit: int = 140) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _uncited_flag(text: str, language: str) -> str:
    snippet = _snippet(text)
    if language == "it":
        return f'Questa affermazione della sintesi non cita alcuna fonte: "{snippet}"'
    return f'This statement in the overview carries no source citation: "{snippet}"'


def _contradicted_flag(text: str, language: str) -> str:
    snippet = _snippet(text)
    if language == "it":
        return f'Questa affermazione sembra in contraddizione con la fonte citata: "{snippet}"'
    return f'This statement appears to contradict its cited source: "{snippet}"'


def _unsupported_flag(text: str, language: str) -> str:
    snippet = _snippet(text)
    if language == "it":
        return f'La fonte citata non sembra affrontare quanto affermato qui: "{snippet}"'
    return f'The cited source does not appear to address what this statement says: "{snippet}"'


def verify_summary(
    global_summary: str,
    articles: list[ArticleSummary],
    settings: Settings,
    summary_language: str = "it",
    job_stats: JobStats | None = None,
) -> list[str]:
    """Flag passages of `global_summary` that carry no citation, or whose
    citation doesn't hold up against the source it points to.

    Returns human-readable flags in *summary_language*, not a structured
    verdict list - unlike apply_verdicts, there is no claim object here to
    edit: rewriting a fragment out of otherwise-flowing prose without
    breaking it is a different, harder problem this does not attempt. A
    flag is a signal for review, not an automatic edit.
    """
    if not global_summary.strip():
        return []

    allowed = set(range(len(articles)))
    flags: list[str] = []
    cited_claims: list[Claim] = []
    cited_texts: list[str] = []

    for chunk in split_into_chunks(global_summary):
        indices = [i for i in chunk.source_indices if i in allowed]
        if not indices:
            if _is_substantive(chunk.text):
                flags.append(_uncited_flag(chunk.text, summary_language))
            continue
        cited_claims.append(Claim(text=chunk.text, source_indices=indices))
        cited_texts.append(chunk.text)

    if not cited_claims:
        return flags

    verdicts = claim_verification.verify_claims(
        cited_claims, articles, settings, job_stats=job_stats,
    )
    by_chunk: dict[int, list[ClaimVerdict]] = {}
    for verdict in verdicts:
        by_chunk.setdefault(verdict.claim_index, []).append(verdict)

    for c_index, text in enumerate(cited_texts):
        real = [v for v in by_chunk.get(c_index, []) if v.relation != "error"]
        if not real:
            continue
        if any(
            v.relation == "contradicts" and (v.confidence or 0.0) >= FLAG_CONFIDENCE
            for v in real
        ):
            flags.append(_contradicted_flag(text, summary_language))
        elif all(
            v.relation == "says_nothing" and (v.confidence or 0.0) >= FLAG_CONFIDENCE
            for v in real
        ):
            flags.append(_unsupported_flag(text, summary_language))

    return flags
