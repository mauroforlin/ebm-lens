"""Reading the evidence: per-source appraisal, stance judgment, and the
grounded synthesis.

Three LLM passes turn a ranked list of papers into something a person can act
on. All three are written to fail safe, because a wrong summary of medical
evidence is worse than no summary:

* :func:`summarise_sources` reads each shortlisted source and returns a short
  summary, a relevance score, and an appraisal - what design the study used,
  who it studied, and the distinct findings (if any, up to
  ``MAX_FINDINGS_PER_SOURCE``) that bear on the topic, each quoted verbatim
  as its own ``evidence_quote``. Most sources yield one finding; a source
  reporting more than one result bearing on the topic (an efficacy outcome
  and a separate safety outcome, say) is not forced to pick just one. The
  relevance score is the pipeline's only judgement made after actually
  reading the content, and the final ranking leans on it; the design is a
  reading of the full text, so it overrides the pool-wide heuristic in
  :mod:`app.pipeline.evidence_grade`.
* :func:`judge_directions` decides, separately and per finding, whether that
  finding supports or contradicts the topic's claim - graded
  ``strongly_contradicts`` through ``strongly_supports``, plus ``mixed`` and
  a deterministic ``no_evidence`` for findings with nothing to judge (this
  includes every finding of a source with nothing worth reporting, since such
  a source yields no findings at all). Splitting this out of
  ``summarise_sources`` and grading it on a scale rather than a flat
  supports/contradicts/neutral label follows paper-qa's ``contracrow``
  setting (github.com/Future-House/paper-qa): evidence-gathering and verdict
  are different tasks, and "the source doesn't address this" deserves its own
  bucket rather than being folded into whichever label a forced choice
  reaches for. Judging per finding rather than per source also means a
  source with one supporting and one contradicting result doesn't get
  smoothed into a single, wrong-either-way verdict.
* :func:`synthesise` writes the overview, using **only** sources that cleared
  the relevance bar. Feeding it everything retrieved would let marginal
  sources contribute claims to a text the user reads as a conclusion.

The overview comes back as claims rather than prose alone. Each claim names
the articles it rests on, by index, and every index is checked against the
articles actually being returned - a claim that cites nothing real is dropped
before the user sees it. Prose can assert anything; a claim that has to point
at its sources can be checked, and the reader can follow the pointer.

Conflicts are reported rather than resolved. Roughly half of post-2010
biomedical papers disagree with something else in their own literature, and an
overview that averages two opposing findings reads exactly like settled
science. When the sources disagree, the disagreement is the finding.

When too little relevant evidence exists, all three passes say so plainly
rather than producing confident prose out of weak input. Relevance is not the
same test as directness, though: a source can be topically on-topic - same
drug, same organ, same disease family - while studying a different
population or intervention than the one actually asked about. The summariser
judges each source's ``directness`` against the topic's PICO for exactly
this, and the synthesiser is not trusted to honour it unassisted: a claim
resting only on non-direct sources is clamped to "weak" in code, and the
overview is prefixed with an explicit flag when nothing cited is a direct
match.

Validated claim-level attribution follows the attribution-evaluation
literature (arXiv:2605.06635, arXiv:2408.04568); refusing over answering from
weak evidence follows arXiv:2409.11242; reporting conflicts rather than
averaging them follows *Contradictions in Context* (arXiv:2511.06668).
"""
from __future__ import annotations

import json
import logging

from app.config import Settings
from app.core.job_stats import JobStats
from app.core.llm_client import generate_json, generate_with_tools
from app.core.parallel import run_parallel
from app.core.sections import (
    BIOMED_PRIORITY,
    BIOMED_SKIP,
    GUIDELINE_PRIORITY,
    split_sections,
    trim_by_priority,
)
from app.pipeline import evidence_grade
from app.schemas import PICO, ArticleSummary, Claim, Disagreement
from app.sources.base import SourceResult
from app.sources.content_extractor import fetch_full_text

logger = logging.getLogger(__name__)

# Sources below this relevance never reach the synthesis prompt. Set high
# enough to clear the score band genuinely on-topic sources land in, above
# the score a source merely sharing the right subject tends to get without
# actually testing or reporting anything - see relevance_score's rule in
# _SUMMARISE_SYSTEM for that distinction.
MIN_SYNTHESIS_RELEVANCE = 0.65

# Distinct findings a single source may contribute. Most sources yield one;
# this exists for the source that genuinely tests more than one outcome
# bearing on the topic (efficacy and safety, two subgroups, ...) - not to
# invite exhaustive extraction of everything a paper reports.
MAX_FINDINGS_PER_SOURCE = 3

# Chars of each source shown to the summariser up front, before any
# read_full_text call.
_MAX_SOURCE_CHARS = 4000

# Chars returned by a read_full_text call, once the model has decided the
# excerpt above isn't enough. Larger than _MAX_SOURCE_CHARS because this is
# meant to actually reach the sections the excerpt couldn't, and capped to
# match generate_with_tools' own _MAX_TOOL_RESULT_CHARS backstop in
# llm_client.py - going higher would just have that backstop re-truncate it
# blindly, the two must move together. Filled by section priority (see
# trim_by_priority) rather than a raw prefix, so the budget goes to
# Results/Discussion (or, for a guideline, Recommendations - see
# content_extractor.py's own priority choice at fetch time) instead of
# whichever section the source happens to put first.
_MAX_FULL_TEXT_CHARS = 12000

# Upper bound on distinct sources actually fetched (network I/O) within one
# appraisal batch - read_full_text and read_section share this budget and
# its cache, so asking for a section of an already-fetched source never
# counts against it a second time. Generous - there is no per-call cost
# pressure here - but still a bound, because the upstream sources this hits
# (NCBI, EMA, ...) have their own rate limits. Never actually binds today
# since a batch has at most _SUMMARISE_BATCH_SIZE sources to read, but stays
# as the real ceiling should that constant grow.
_MAX_FULL_TEXT_READS = 12

_MAX_APPRAISAL_TOOL_ROUNDS = 6

# Sources appraised per summarise_sources call, and how many such calls run
# at once. Smaller batches mean each evidence_quote/directness judgment - see
# _SUMMARISE_SYSTEM's rules on both - competes with fewer other sources'
# content for the model's attention than one call covering the whole
# shortlist would; running batches concurrently keeps wall-clock latency
# close to the single-call version despite the extra round trips.
_SUMMARISE_BATCH_SIZE = 5
_SUMMARISE_MAX_WORKERS = 6

_TIER_LABELS = {1: "institutional", 2: "curated", 3: "general"}

# Sources per judge_directions call, run concurrently.
_STANCE_MAX_WORKERS = 8

# The graded labels judge_directions' stance call is allowed to choose
# between - three grades of strength on each side, plus "mixed" for a source
# that itself reports opposite effects across subgroups or analyses.
# "no_evidence" is deliberately not in this set: it is assigned
# deterministically before the call ever happens (see judge_directions), not
# guessed by the model.
_STANCE_LABELS = frozenset({
    "strongly_contradicts", "contradicts", "weakly_contradicts",
    "weakly_supports", "supports", "strongly_supports",
    "mixed",
})

# The full vocabulary a source's finding_direction can carry once
# judge_directions has run. Anything else the model returns is read as
# "no_evidence" rather than trusted.
_DIRECTIONS = _STANCE_LABELS | {"no_evidence"}

_STRENGTHS = frozenset({"strong", "moderate", "weak"})

# How closely a source's own population/intervention match the topic's PICO,
# as opposed to merely sharing a topic or a drug/organ/keyword. Anything else
# the model returns is read as "unclear". Only "direct" exempts a claim from
# the strength clamp in _read_claims - "unclear" is treated the same as
# "adjacent" there, so an unparseable answer fails safe rather than silently
# granting full strength.
_DIRECTNESS = frozenset({"direct", "adjacent", "unclear"})


def _language_instruction(summary_language: str) -> str:
    return "ITALIAN (italiano)" if summary_language == "it" else "ENGLISH"


def low_evidence_message(summary_language: str = "it") -> str:
    if summary_language == "it":
        return (
            "Le fonti trovate sono poco rilevanti per questo tema, quindi il quadro "
            "resta debole. Per un riassunto utile servono fonti piu mirate "
            "(istituzionali o accademiche)."
        )
    return (
        "The collected sources are not relevant enough for this topic, so the "
        "current picture is weak. A useful summary needs more targeted "
        "institutional or academic sources."
    )


def _no_direct_evidence_prefix(summary_language: str = "it") -> str:
    """Flag that follows when every cited source is adjacent, not direct.

    A source can be topically relevant - same drug, same organ, same disease
    family - without studying the population or intervention the question
    actually named. That is real evidence discovery work and worth keeping,
    but the overview must say so before the reader takes an extrapolation for
    a direct answer.
    """
    if summary_language == "it":
        return (
            "Nessuna fonte affronta direttamente la domanda posta: quanto segue "
            "e' estrapolato da fonti su popolazioni o condizioni correlate ma "
            "diverse. "
        )
    return (
        "No source directly addresses the question as asked: what follows is "
        "extrapolated from sources on related but different populations or "
        "conditions. "
    )


_SUMMARISE_SYSTEM = """\
You are a research assistant appraising source articles for a user \
investigating a specific medical/clinical topic. For each source you produce \
a concise summary, a relevance score, a short critical appraisal, and the \
distinct findings - up to {max_findings} - that source itself reports bearing \
on the topic, each with the exact sentence that reports it. You do NOT decide \
whether a finding supports or contradicts the topic's claim - a separate pass \
does that from the quote you give it, so your job is to find and quote the \
evidence precisely, not to render a verdict on it.

Each source starts with an excerpt - an abstract, a label section, a search \
snippet. That is enough for most sources. When it is not, two tools can read \
further: read_section fetches one or more named sections in full - use it \
directly when you already know you need the numbers in Results, or the \
authors' own interpretation in Discussion; read_full_text fetches a broad, \
automatically-prioritised excerpt instead, for when the source needs a closer \
look but you don't yet know which section has it. A working human reviewer \
does not open every paper's PDF, only the ones whose abstract does not \
answer the question; appraise the same way - do not call either tool \
reflexively for every source, and do not call one for a source whose excerpt \
already gives you what you need. When you are done reading whatever you \
needed to read, call submit_appraisals once with every source's appraisal.

A source INDEXED AS guideline is a special case: its starting excerpt is \
usually just a scope-and-purpose preamble, not a summary of what it actually \
recommends - unlike a study abstract, a thin excerpt there does NOT mean the \
source has nothing to report. Read further before scoring it low or leaving \
findings empty.

Each item in the appraisals array has:
{{
  "index": <int, matching the source index>,
  "summary": "<Ultra-brief scan summary in {language}: short and plain-language, \
focused only on the main takeaway for this topic.>",
  "relevance_score": <float 0.0-1.0, how directly relevant this source is to \
the user's topic. 1.0 = perfectly on-topic, 0.0 = completely unrelated>,
  "study_design": "<one of: meta_analysis, systematic_review, guideline, rct, \
controlled_trial, cohort, case_control, cross_sectional, case_report, \
narrative_review, preclinical, drug_label, trial_registry, surveillance, \
commentary, encyclopedic, unknown>",
  "population": "<who or what was studied, in {language}, a few words; \
empty string if the source does not say>",
  "findings": [
    {{
      "text": "<ONE claim this source itself reports data or a result for, \
in {language}, one sentence>",
      "evidence_quote": "<the exact sentence(s) from the source's own text \
that report this finding, copied VERBATIM - not paraphrased, not translated, \
not summarised>"
    }}
  ],
  "directness": "<one of: direct, adjacent, unclear - judged against the \
PICO given below, when one is given>"
}}

Rules:
- Prioritise readability and speed: this is a quick-glance summary, not a review.
- Keep only the core idea; omit secondary details and long context.
- Mention at most one concrete detail (number/date/entity) only if critical.
- ALL prose fields MUST be written in {language}, EXCEPT evidence_quote, \
which stays in the source's own original language and wording since it must \
be checkable against the source text. study_design and directness are ids: \
return them in English, exactly as listed.
- relevance_score should reflect how DIRECTLY USEFUL this source is for \
understanding the topic, not just whether it mentions related terms. A source \
can share the same drug, disease or general subject as the topic while \
reporting nothing that actually bears on it (e.g. background context, a \
different question in the same paper, a citation with no data behind it) - \
score that low, even though it is "about" the right subject. Being on-topic \
and being evidence are different questions; this field grades the second one.
- study_design: report what the source ACTUALLY IS, not what it discusses. A \
review of randomised trials is narrative_review or systematic_review, not rct. \
Use "unknown" when the source does not make its design clear - do not guess.
- findings is EMPTY (not an item with empty strings) when the source is \
topically on-topic but never tests or reports anything that speaks to it - do \
not add an entry just because the source is relevant reading. Most sources \
yield exactly ONE finding: add a second or third ONLY when the source reports \
genuinely distinct results bearing on the topic (e.g. an efficacy outcome AND \
a separate safety outcome, or results in two different subgroups) - never by \
splitting one result into several entries, and never to pad the list. Each \
finding must be backed by data or a result the source actually reports, not \
an inference from it merely discussing the same drug, disease or subject area.
- List findings in DESCENDING order of importance to the topic - the result \
that most directly and decisively bears on the topic's claim goes first. If \
the source genuinely reports more than {max_findings} distinct results \
bearing on the topic, keep only the {max_findings} most important and drop \
the rest - never truncate by whichever order they happened to come to mind; \
decide importance first, then list only that many.
- Every finding needs its own evidence_quote pointing at the specific sentence \
that makes it; a finding without a matching quote is treated downstream as if \
neither had been given, so an approximate quote is worse than an empty finding.
- evidence_quote is read by a separate pass with no access to the source \
itself, only to what you copy here - if you paraphrase, summarise, or \
translate it instead of quoting exactly, that pass is judging words the \
source never wrote. When you are unsure whether a sentence counts as \
"the" result, quote it anyway rather than smoothing it into the finding's \
paraphrase; a slightly-too-generous quote costs nothing, a missing one means \
the source's own result never gets judged at all.
- directness is about the SAME facts as the findings, judged more strictly: a \
source that treats a different population, a different disease mechanism, or \
a different formulation than the one asked about is "adjacent" even when it \
shares a drug, organ or keyword with the topic - do not mark a source \
"direct" just because it is topically relevant. Use "unclear" only when the \
topic gives no specific enough population/intervention to judge against.
- If a source has very little content or is mostly irrelevant, give it a \
low relevance_score and a very short summary noting limited utility.
"""


# Publication types (see europe_pmc.py's PUB_TYPE query and pubmed.py's
# guideline[pt] filter) that mark a record as a clinical practice guideline
# rather than a primary study - the provider's own fact, not a text guess.
# Used to pick read_full_text's section-priority list: a guideline's
# recommendations live under headings BIOMED_PRIORITY doesn't recognise.
_GUIDELINE_PUB_TYPES = frozenset({
    "guideline", "practice guideline", "consensus development conference",
})


def _is_guideline(result: SourceResult) -> bool:
    return any(
        isinstance(t, str) and t.strip().lower() in _GUIDELINE_PUB_TYPES
        for t in result.publication_types
    )


def _pico_block(pico: PICO | None) -> str:
    """Render the PICO as the directness yardstick, when the topic has one.

    Only emitted when the PICO is specific enough to judge against - an
    unspecified comparator or population would make "adjacent" a guess rather
    than a reading.
    """
    if pico is None or not pico.is_specified():
        return ""
    lines = ["PICO (judge \"directness\" against these fields specifically):"]
    if pico.population:
        lines.append(f"  Population: {pico.population}")
    if pico.intervention:
        lines.append(f"  Intervention: {pico.intervention}")
    if pico.comparison:
        lines.append(f"  Comparison: {pico.comparison}")
    if pico.outcome:
        lines.append(f"  Outcome: {pico.outcome}")
    return "\n".join(lines) + "\n\n"


_READ_FULL_TEXT_TOOL = {
    "type": "function",
    "function": {
        "name": "read_full_text",
        "description": (
            "Fetch a broad excerpt of one source's full page text, when its "
            "starting excerpt is too thin to judge study design, directness "
            "or the key finding with confidence but you aren't yet sure which "
            "section has what you need. Automatically weighted toward the "
            "sections most likely to matter (Results, Discussion) - use "
            "read_section instead once you know exactly which section to "
            "read in full. Only works for sources on a trusted allowlist of "
            "regulator/publisher domains - returns a message saying so "
            "otherwise, in which case appraise from the excerpt you already have."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "The source's [index] as given in the prompt.",
                },
            },
            "required": ["index"],
        },
    },
}

_READ_SECTION_TOOL = {
    "type": "function",
    "function": {
        "name": "read_section",
        "description": (
            "Fetch the complete, untrimmed text of one or more named "
            "sections from a source (e.g. \"results\", \"discussion\", "
            "\"methods\" for a study; \"recommendation\", \"evidence to "
            "decision\", \"rationale\" for a clinical guideline - check the "
            "source's INDEXED AS line) - matched case-insensitively as a "
            "substring of the source's own section headings, so \"results\" "
            "also matches \"Results and Discussion\". Use this instead of read_full_text "
            "when you already know exactly what you need: the precise "
            "numbers, subgroup results or stated direction of an effect "
            "usually live entirely in Results or Discussion, and this "
            "returns that section whole rather than a budget-limited excerpt "
            "of the whole paper. Asking for more than one or two sections at "
            "once risks the same length limit read_full_text has, so prefer "
            "the fewest sections that answer what you need. Same allowlist "
            "as read_full_text; a miss returns the source's actual section "
            "headings so you can retry with the right name."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "The source's [index] as given in the prompt.",
                },
                "sections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "One or more section names to fetch in full, e.g. "
                        "[\"results\"] or [\"results\", \"discussion\"]."
                    ),
                },
            },
            "required": ["index", "sections"],
        },
    },
}

_SUBMIT_APPRAISALS_TOOL_NAME = "submit_appraisals"

_SUBMIT_APPRAISALS_TOOL = {
    "type": "function",
    "function": {
        "name": _SUBMIT_APPRAISALS_TOOL_NAME,
        "description": (
            "Submit the final appraisal for every source, once you are done "
            "reading whatever you needed to read."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "appraisals": {
                    "type": "array",
                    "description": "One item per source.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "summary": {"type": "string"},
                            "relevance_score": {"type": "number"},
                            "study_design": {"type": "string"},
                            "population": {"type": "string"},
                            "findings": {
                                "type": "array",
                                "maxItems": MAX_FINDINGS_PER_SOURCE,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "text": {"type": "string"},
                                        "evidence_quote": {"type": "string"},
                                    },
                                    "required": ["text", "evidence_quote"],
                                },
                            },
                            "directness": {"type": "string"},
                        },
                        "required": ["index", "summary", "relevance_score"],
                    },
                },
            },
            "required": ["appraisals"],
        },
    },
}


def summarise_sources(
    topic: str,
    results: list[SourceResult],
    settings: Settings,
    summary_language: str = "it",
    job_stats: JobStats | None = None,
    pico: PICO | None = None,
) -> dict[int, dict]:
    """Summarise and appraise every shortlisted source, in parallel batches.

    Returns ``{source_index: {...}}``. A source missing from the result
    simply gets no summary - the same is true of every source in a batch
    whose call failed outright, since one batch's failure must not cost the
    others theirs.

    Batching keeps each judgment's context narrow: with
    ``_SUMMARISE_BATCH_SIZE`` sources per call rather than the whole
    shortlist, reading source 12 doesn't compete against nineteen others'
    content for the model's attention. Running batches concurrently keeps
    wall-clock latency close to a single call covering the whole shortlist
    despite the extra round trips. See :func:`_summarise_batch` for the
    per-batch mechanics - excerpt-first reading, the bounded
    ``read_full_text``/``read_section`` tools.

    This appraises each source's own evidence but does not judge whether it
    supports or contradicts the topic - see :func:`judge_directions` for
    that, run separately once every batch here has returned.

    Passing *pico* gives the model a concrete population/intervention to
    judge each source's ``directness`` against, rather than leaving it to
    infer specificity from the topic string alone.
    """
    if not results:
        return {}

    batches = [
        list(range(start, min(start + _SUMMARISE_BATCH_SIZE, len(results))))
        for start in range(0, len(results), _SUMMARISE_BATCH_SIZE)
    ]

    summaries: dict[int, dict] = {}
    for batch_result in run_parallel(
        lambda batch: _summarise_batch(
            topic, results, batch, settings, summary_language, job_stats, pico,
        ),
        batches, _SUMMARISE_MAX_WORKERS,
    ):
        summaries.update(batch_result)
    return summaries


def _summarise_batch(
    topic: str,
    results: list[SourceResult],
    batch_indices: list[int],
    settings: Settings,
    summary_language: str,
    job_stats: JobStats | None,
    pico: PICO | None,
) -> dict[int, dict]:
    """Appraise one batch of sources in a single tool-call loop.

    *results* is the full shortlist; *batch_indices* is the subset this call
    is responsible for. Indices in the prompt and in ``submit_appraisals``
    are the real (global) positions in *results*, not batch-local ones - so a
    claim later citing ``[7]`` means the same source no matter which batch
    appraised it, and ``read_full_text``'s in-place mutation of
    ``results[index].content`` lands on the one list every batch and the
    caller share, safely, since concurrent batches never touch the same index.
    """
    allowed = set(batch_indices)

    def _block(index: int, result: SourceResult) -> str:
        content = (result.content or result.snippet or "")[:_MAX_SOURCE_CHARS]
        block = (
            f"[{index}] TITLE: {result.title}\n"
            f"    URL: {result.url}\n"
            f"    SOURCE: {result.source_type}\n"
        )
        # The provider's own publication types are the strongest statement
        # available about what the source is, so the reader is shown them
        # rather than left to infer the design from the prose alone.
        if result.publication_types:
            block += f"    INDEXED AS: {', '.join(result.publication_types[:4])}\n"
        if result.publication_date:
            block += f"    DATE: {result.publication_date}\n"
        block += f"    CONTENT: {content}\n"
        return block

    prompt = (
        f"TOPIC: {topic}\n\n"
        + _pico_block(pico)
        + f"SOURCES TO APPRAISE ({len(batch_indices)} of {len(results)} total "
        "in this run - the indices below are not necessarily contiguous "
        "from 0, use them exactly as given):\n\n"
        + "\n".join(_block(i, results[i]) for i in batch_indices)
    )

    reads_used = 0
    # None means "fetched and came back empty" (allowlist miss or failed
    # fetch) - cached too, so a second tool call on the same index doesn't
    # retry a fetch already known to fail, and doesn't count against
    # _MAX_FULL_TEXT_READS twice.
    fetched: dict[int, str | None] = {}

    def _ensure_fetched(index: int) -> str | None:
        # Untrusted page text (see content_extractor.py) enters the tool loop
        # right here - it becomes results[index].content, returned below as
        # an ordinary tool result the model reads.
        nonlocal reads_used
        if index in fetched:
            return fetched[index]
        if reads_used >= _MAX_FULL_TEXT_READS:
            return None
        reads_used += 1
        result = results[index]
        # PubMed's abstract page cannot be scraped (cookie-challenge on a
        # plain GET) - its full text, when it has any, lives in PMC instead.
        if result.source_type == "pubmed":
            from app.sources.pubmed import fetch_full_text_by_url
            text = fetch_full_text_by_url(result.url)
        else:
            text = fetch_full_text(result)
        fetched[index] = text or None
        if text:
            results[index].content = text
        return fetched[index]

    def _read_full_text(args: dict) -> str:
        index = args.get("index")
        if not isinstance(index, int) or index not in allowed:
            return "invalid index"
        text = _ensure_fetched(index)
        if text is None:
            if index in fetched:
                return (
                    "not available - this source is not on the fetchable "
                    "allowlist, or the fetch failed. Appraise from the "
                    "excerpt you already have."
                )
            return "read budget exhausted for this run - appraise from the excerpt"
        priority = GUIDELINE_PRIORITY if _is_guideline(results[index]) else BIOMED_PRIORITY
        return trim_by_priority(
            text, _MAX_FULL_TEXT_CHARS,
            priority=priority, skip=BIOMED_SKIP,
        )

    def _read_section(args: dict) -> str:
        index = args.get("index")
        if not isinstance(index, int) or index not in allowed:
            return "invalid index"
        wanted = [
            name.strip().lower() for name in (args.get("sections") or [])
            if isinstance(name, str) and name.strip()
        ]
        if not wanted:
            return "sections must be a non-empty list of section names"
        text = _ensure_fetched(index)
        if text is None:
            if index in fetched:
                return (
                    "not available - this source is not on the fetchable "
                    "allowlist, or the fetch failed. Appraise from the "
                    "excerpt you already have."
                )
            return "read budget exhausted for this run - appraise from the excerpt"
        sections = split_sections(text)
        matched = [
            (title, body) for title, body in sections
            if any(name in title.lower() for name in wanted)
        ]
        if not matched:
            headings = ", ".join(title for title, _body in sections if title)
            return (
                f"no section matching {wanted} found. This source's actual "
                f"section headings: {headings or '(none detected)'}"
            )
        return "".join(body for _title, body in matched)

    handlers = {
        "read_full_text": _read_full_text,
        "read_section": _read_section,
        _SUBMIT_APPRAISALS_TOOL_NAME: lambda args: json.dumps(
            {"accepted": True, "count": len(args.get("appraisals") or [])}
        ),
    }

    try:
        _, invocations = generate_with_tools(
            settings=settings,
            prompt=prompt,
            system_instruction=_SUMMARISE_SYSTEM.format(
                language=_language_instruction(summary_language),
                max_findings=MAX_FINDINGS_PER_SOURCE,
            ),
            tools=[_READ_FULL_TEXT_TOOL, _READ_SECTION_TOOL, _SUBMIT_APPRAISALS_TOOL],
            tool_handlers=handlers,
            temperature=0.15,
            purpose="related_articles_summarize",
            job_stats=job_stats,
            max_tool_rounds=_MAX_APPRAISAL_TOOL_ROUNDS,
            final_tool=_SUBMIT_APPRAISALS_TOOL_NAME,
        )
    except Exception as exc:
        logger.warning("Source appraisal batch failed: %s", exc)
        return {}

    submission = next(
        (inv for inv in reversed(invocations) if inv.name == _SUBMIT_APPRAISALS_TOOL_NAME),
        None,
    )
    if submission is None:
        return {}

    raw = submission.arguments.get("appraisals")
    if not isinstance(raw, list):
        return {}

    summaries: dict[int, dict] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        if isinstance(index, int) and index in allowed:
            # Enforced here, not just on read in the orchestrator: judge_directions
            # runs one stance call per finding on whatever this dict holds, so a
            # model that ignores the "up to MAX_FINDINGS_PER_SOURCE" instruction
            # would otherwise burn calls on findings no caller ever keeps. Keeps
            # the prefix, not a random subset, because _SUMMARISE_SYSTEM asks for
            # findings in descending importance order - truncating the tail is
            # meant to drop the least important ones, not an arbitrary slice.
            findings = item.get("findings")
            if isinstance(findings, list) and len(findings) > MAX_FINDINGS_PER_SOURCE:
                item["findings"] = findings[:MAX_FINDINGS_PER_SOURCE]
            summaries[index] = item
    return summaries


def clamp_relevance(value: object) -> float:
    try:
        return min(1.0, max(0.0, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def read_direction(value: object) -> str:
    candidate = value.strip().lower() if isinstance(value, str) else ""
    return candidate if candidate in _DIRECTIONS else "no_evidence"


def collapse_direction(direction: str) -> str:
    """Collapse the graded scale to SciFact's SUPPORT/CONTRADICT/NOINFO.

    SciFact's annotators never graded strength, so every ``*_supports``
    label maps to SUPPORT and every ``*_contradicts`` label to CONTRADICT
    regardless of grade; ``mixed`` has no SciFact counterpart (no claim in
    the corpus has both SUPPORT and CONTRADICT evidence within one document)
    and collapses to NOINFO alongside ``no_evidence`` and anything
    unrecognised. The canonical collapse, so eval/stance_eval.py grades
    against the same rule production actually uses instead of re-deriving it.
    """
    if direction.endswith("supports"):
        return "SUPPORT"
    if direction.endswith("contradicts"):
        return "CONTRADICT"
    return "NOINFO"


def read_directness(value: object) -> str:
    candidate = value.strip().lower() if isinstance(value, str) else ""
    return candidate if candidate in _DIRECTNESS else "unclear"


def is_quote_grounded(quote: str, content: str) -> bool:
    """Whether *quote* appears verbatim (whitespace/case folded) in *content*.

    The one mechanical check standing between an ``evidence_quote`` the model
    claims to have copied and one it actually did - anything downstream that
    treats a quote as trustworthy (stance judgment, and the synthesis prompt
    via :func:`_source_lines`) must pass through this first.
    """
    if not quote or not content:
        return False
    norm = lambda s: " ".join(s.lower().split())  # noqa: E731
    return norm(quote) in norm(content)


def appraise_design(
    summary: dict,
    fallback_design: str,
    fallback_score: float,
) -> tuple[str, float]:
    """Resolve a source's design, preferring the reading of its full text.

    Returns ``(label, score)``. The summariser has read the content and the
    provider's own publication types, so its answer wins; where it declines to
    name a design, the pool-wide detection stands.
    """
    design_id = evidence_grade.normalise_design(summary.get("study_design"))
    if design_id is None or design_id == "unknown":
        return fallback_design, fallback_score
    return evidence_grade.design_label(design_id), evidence_grade.evidence_score(design_id)


_STANCE_SYSTEM = """\
You are judging whether ONE source's own finding supports or contradicts a \
specific topic claim. You are given the topic, the source's key finding, and \
the exact sentence(s) it reports that finding in (its evidence quote) - \
decide how the source's own result relates to what the topic specifically \
claims, based only on the quote.

Return a JSON object:
{
  "reasoning": "<one or two sentences: what does the evidence quote actually \
say, and how does that compare to what the topic claims? Write this BEFORE \
choosing the direction below - it is where the comparison happens, not a \
summary written after the fact.>",
  "direction": "<one of: strongly_contradicts, contradicts, weakly_contradicts, \
weakly_supports, supports, strongly_supports, mixed - never anything else>"
}

Rules:
- A null result, "no significant difference", or an effect running the \
OPPOSITE way from what the topic claims belongs on the CONTRADICTS side, \
never the supports side - the source addressed the claim and its answer \
leaned no.
- The three grades on each side are about STRENGTH, not confidence: \
"strongly_*" is a large, consistent effect, or one backed by a higher-tier \
design (meta-analysis, large RCT); "weakly_*" is a real but small, \
borderline, or single-subgroup effect. Use the plain "contradicts"/"supports" \
grade when nothing pushes toward either extreme.
- Use "mixed" only when the evidence quote ITSELF reports opposite effects \
across different subgroups or analyses within the same source - not merely \
because you are unsure which side it falls on. When unsure, re-read the \
quote and pick a side; "mixed" is for evidence that is genuinely two-sided, \
not for your own uncertainty.
- Getting the SIDE backwards - reading a source's "no" as "yes" - is the \
single worst error possible here, worse than picking the wrong strength. \
When torn between contradicts and supports, reread the evidence quote once \
more before deciding; do not guess toward supports.
- Judge only what the evidence quote itself states, never the source's \
overall tone or whether it is generally about a beneficial/harmful subject - \
a study of a beneficial drug can still report a contradicting result for the \
ONE claim being asked about.
"""


def judge_directions(
    topic: str,
    results: list[SourceResult],
    appraisals: dict[int, dict],
    settings: Settings,
    job_stats: JobStats | None = None,
) -> dict[int, list[str]]:
    """Judge each appraised finding's direction, one call per finding.

    Split out of :func:`summarise_sources` and graded on the scale
    :data:`_STANCE_LABELS` rather than a flat supports/contradicts/neutral
    label, following paper-qa's ``contracrow`` setting (github.com/Future-
    House/paper-qa): evidence-gathering and the support/contradict verdict
    are different tasks, and grading strength gives "the source found
    nothing" its own place on the scale instead of forcing it into whichever
    of three labels a forced choice reaches for.

    A source can report more than one finding (see ``MAX_FINDINGS_PER_SOURCE``),
    so this judges each finding on its own - a source with an efficacy result
    and a safety result can have one supporting and one contradicting, and
    collapsing that into a single source-level verdict would lose exactly the
    distinction the multi-finding schema exists to keep.

    Two gates skip the LLM call entirely and assign a finding ``"no_evidence"``
    directly, judge-free:

    - An empty ``text`` - ``_SUMMARISE_SYSTEM`` already declines to fill one
      in when the source doesn't test the topic's claim, so there is nothing
      here to judge a direction on.
    - A quote that doesn't check out: ``evidence_quote`` not appearing
      (loosely - whitespace/case folded) in the source's own fetched content,
      or missing outright. ``_SUMMARISE_SYSTEM`` asks for it verbatim so it
      can be verified; a quote that fails that check might as well not have
      been given, since trusting it would mean judging words the source may
      never have written.

    A third gate applies at the source level - a ``relevance_score`` at or
    below ``MIN_SYNTHESIS_RELEVANCE`` marks every one of that source's
    findings ``"no_evidence"``, since such a source never reaches
    :func:`synthesise` anyway.

    Every finding that clears all gates gets its own call, run in parallel.
    Returns ``{source_index: [direction, ...]}``, one entry per finding in
    that source's ``findings`` list, same order.
    """
    content_by_index = {index: result.content or "" for index, result in enumerate(results)}

    to_judge: dict[tuple[int, int], dict] = {}
    directions: dict[int, list[str]] = {}
    for index, appraisal in appraisals.items():
        findings = appraisal.get("findings")
        findings = findings if isinstance(findings, list) else []
        relevant_enough = clamp_relevance(appraisal.get("relevance_score")) > MIN_SYNTHESIS_RELEVANCE
        content = content_by_index.get(index, "")

        slots: list[str] = ["no_evidence"] * len(findings)
        for f_index, finding in enumerate(findings):
            if not isinstance(finding, dict):
                continue
            quote = finding.get("evidence_quote")
            quote = quote.strip() if isinstance(quote, str) else ""
            if finding.get("text") and relevant_enough and is_quote_grounded(quote, content):
                to_judge[(index, f_index)] = finding
        directions[index] = slots

    if not to_judge:
        return directions

    def _judge_one(key: tuple[int, int]) -> tuple[tuple[int, int], str]:
        finding = to_judge[key]
        prompt = (
            f"TOPIC: {topic}\n\n"
            f"SOURCE'S FINDING: {finding.get('text', '')}\n"
            f"EVIDENCE QUOTE: {finding.get('evidence_quote', '')}\n"
        )
        try:
            raw = generate_json(
                settings=settings, prompt=prompt, system_instruction=_STANCE_SYSTEM,
                temperature=0.1, purpose="related_articles_stance", job_stats=job_stats,
            )
        except Exception as exc:
            logger.warning("Stance judgment failed for source %d finding %d: %s", key[0], key[1], exc)
            return key, "no_evidence"
        direction = raw.get("direction") if isinstance(raw, dict) else None
        return key, direction if direction in _STANCE_LABELS else "no_evidence"

    for (index, f_index), direction in run_parallel(_judge_one, list(to_judge), _STANCE_MAX_WORKERS):
        directions[index][f_index] = direction
    return directions


_SYNTHESIS_SYSTEM = """\
You are an evidence synthesis writer working from a set of numbered,
titled sources. Every statement you make must be traceable to them.

A source may list more than one "finding:" line - genuinely distinct results
it reports (e.g. an efficacy finding and a separate safety finding), each
possibly pointing the same or a different direction. Treat them as separate
facts about that source, not as one blurred claim. Each finding may carry its
own "quote:" line beneath it - the source's own sentence(s), copied verbatim,
verified to actually appear in that source's text. Where present, it is
ground truth for what that specific finding says, tighter than the finding's
own paraphrase. When you write a claim citing a source for one of its
findings, what you write must be something that finding's quote itself
supports - do not go beyond what the quote states, and if the finding text
and its quote seem to say different things, trust the quote.

Given a topic and the appraised sources, produce ONE thorough, well-organized
overview plus the claims and conflicts behind it.

Return a JSON object:
{{
  "global_summary": "<standalone overview in {language}. Cite sources inline \
with their bracketed index, e.g. [0] or [1][3], for every substantive \
statement - this is mandatory and never optional. Where it helps the reader \
follow the argument, you may ALSO name the source in prose (its trial name, \
if the TITLE given to you shows a distinctive one, e.g. 'the STEP-1 trial \
[2]', or otherwise a short paraphrase of its title/design+population, e.g. \
'a 2021 cohort study of adolescents with X [4]') - the bracketed index stays \
mandatory either way, since that is what gets verified; the name is a \
readability aid on top of it, never a replacement. Never invent an author, \
trial acronym or name that is not evident in the TITLE or content you were \
given. Formatting: this is meant to be scanned, not just read start to \
finish, so use light Markdown to help the eye - blank lines between \
paragraphs that cover different sub-points, and *italics* sparingly for a \
study/trial name or an explicit caveat. \
**Bold is for two things only: (1) a concrete, neutral anchor - the \
statistic itself (the OR/RR/HR/CI/p-value) or a distinguishing fact (an \
unusual sample size, a specific subgroup, a dosing detail); (2) a neutral \
observation about the SHAPE of the evidence - that it is mixed, that it \
depends on a subgroup, dose, or population, that one design disagrees with \
another - never which side is right, only that the picture is not uniform \
(e.g. "**contrasting results**", "**effect only with daily dosing**", \
"**no effect in older adults, but present in younger ones**"). Bold is \
NEVER for a clause that states which way a finding points on its own** \
("has shown a significant reduction", "found no significant effect", \
"supports the treatment", and equivalents are NEVER bold, even when they \
are the sentence's main point - stating a direction is fine in plain text, \
just never emphasised). This is not a style preference: bolding a verdict \
clause makes it look like the answer, and when the topic's sources \
disagree that is exactly the false impression of settled science this \
synthesiser exists to avoid; bolding that the picture is mixed does the \
opposite; a number is always safe. If nothing in a paragraph fits either \
of the two allowed uses, leave it unbolded rather than bolding the \
interpretation instead. Most sentences should carry no markup at all. No \
headings, no bullet lists, no markdown tables - those break the plain \
overview this renders into.>",
  "key_findings": [
    {{
      "text": "<one claim, in {language}, one sentence>",
      "source_indices": [<indices of the sources that support this claim>],
      "strength": "<strong|moderate|weak>"
    }}
  ],
  "disagreements": [
    {{
      "issue": "<what the sources disagree about, in {language}>",
      "positions": ["<position A, in {language}>", "<position B>"],
      "source_indices": [<indices of the sources on either side>]
    }}
  ],
  "evidence_gaps": ["<what these sources do NOT establish, in {language}>"]
}}

Output requirements:
1. The overview should give the reader real context, not just a verdict: for
   each major point, say what was actually studied - design, population,
   rough size or duration when the sources state it - and what it found,
   before or alongside the takeaway. Several short paragraphs are expected
   when the sources support them. Length must come ONLY from more grounded
   detail already present in the sources, never from restating the same
   point, hedging boilerplate, or generic framing sentences that carry no
   citation. If a sentence could be deleted without losing anything a source
   actually said, delete it. When several sources report a closely similar
   result (same direction, similar magnitude), report them TOGETHER in one
   sentence carrying all their indices, rather than one near-identical
   sentence per source - a wall of "meta-analysis X found Y (stat)" repeated
   for every source is not depth, it is noise. Give a sentence of its own
   only to a source that adds something the others do not: a different
   population, a notably larger/smaller effect, a subgroup or dosing finding.
2. Focus on: what is happening, why it matters, where the evidence points,
   and - since this is a synthesis, not a list of abstracts - how the
   sources relate to each other (agreement, disagreement, one study's design
   being stronger than another's, one population differing from another).
3. Keep language precise and direct; avoid jargon unless unavoidable, and
   briefly gloss a technical term the first time it appears if doing so aids
   a non-specialist reader.
4. Include concrete details (numbers, dates, entities, study names) whenever
   the sources actually state them - these are what make the overview
   substantive rather than vague, so include them rather than paraphrasing
   them away. Never state a number, date or name the sources do not give you.
5. Do NOT invent facts and do NOT introduce information absent from the input.
   A longer overview must not mean a less careful one: every added sentence
   still needs its own citation.
6. Write all prose in {language}.
7. EVERY key_finding must carry at least one source index, and every index
   must be one of the indices listed in the input. Never cite an index that
   was not given to you. A claim you cannot attribute must not be made.
8. "strength" reflects the EVIDENCE behind the claim, not your confidence in
   it: several higher-tier designs agreeing is strong; one small or
   low-tier source is weak. Each source's design is given to you - use it.
   Each source is also marked (direct) or (adjacent): adjacent means it
   studies a related-but-different population or intervention, not the one
   asked about. A claim resting only on adjacent sources is capped at "weak"
   no matter the design, and its text must say the evidence is indirect
   (e.g. "in adults" / "in the related condition X" / "extrapolated from")
   rather than stating it as settled for the population actually asked about.
9. disagreements: report where the sources genuinely conflict - opposite
   directions of effect, incompatible recommendations, findings a later or
   stronger study overturns. Do NOT smooth a conflict into a single claim,
   and do NOT invent one where the sources simply address different things.
   Return an empty list when they agree.
10. evidence_gaps: name what a reader might reasonably expect these sources to
   settle and they do not - an untested population, an absent comparison, an
   outcome nobody measured. Return an empty list if nothing stands out.
"""

def _source_lines(
    articles: list[ArticleSummary],
    relevant_indices: list[int],
) -> str:
    """Render the sources for the synthesis prompt, keeping their real indices.

    The index shown is the article's position in the response, not its
    position among the relevant subset. Citations therefore point at what the
    user is actually looking at, and validation is a bounds check rather than
    a translation step.
    """
    lines = []
    for index in relevant_indices:
        article = articles[index]
        attributes = [article.source_type, _TIER_LABELS.get(article.reliability_tier, "unknown")]
        if article.study_design:
            attributes.append(article.study_design)
        if article.publication_date:
            attributes.append(article.publication_date[:7])
        if article.directness in ("direct", "adjacent"):
            attributes.append(article.directness)

        title = article.title.strip() if article.title else ""
        title_part = f' "{title}"' if title else ""
        header = f"[{index}]{title_part} ({', '.join(attributes)})"

        if not article.findings:
            # No finding cleared the bar for this source - fall back to the
            # scan summary so the source is still readable, unattributable to
            # any specific claim.
            lines.append(f"{header} {article.full_summary}")
            if article.population:
                lines.append(f"      population: {article.population}")
            continue

        lines.append(header)
        if article.population:
            lines.append(f"      population: {article.population}")
        for finding in article.findings:
            direction = finding.finding_direction
            direction_part = f" (direction: {direction})" if direction and direction != "no_evidence" else ""
            lines.append(f"      finding: {finding.text}{direction_part}")
            if finding.evidence_quote:
                lines.append(f'      quote: "{finding.evidence_quote}"')
    return "\n".join(lines)


def _valid_indices(raw: object, allowed: set[int]) -> list[int]:
    # TODO: this only checks that a cited index exists among the shortlisted
    # sources, not that the cited source actually supports the claim citing
    # it. Verifying that would need a separate entailment check between claim
    # text and source content, not just index membership.
    if not isinstance(raw, list):
        return []
    seen: list[int] = []
    for value in raw:
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value in allowed and value not in seen:
            seen.append(value)
    return seen


def _read_claims(
    raw: object, allowed: set[int], directness_by_index: dict[int, str],
) -> list[Claim]:
    """Parse the model's claims, dropping any that cite nothing verifiable.

    A claim's *stated* strength is trusted only when at least one of its
    cited sources is a direct match for the topic's PICO; otherwise it is
    clamped to "weak" here, mechanically, regardless of what the model wrote.
    The system prompt already asks for this, but a prompt is advice and this
    is enforcement - the same reasoning that makes citation indices checked
    rather than trusted (see the module docstring).
    """
    if not isinstance(raw, list):
        return []

    claims: list[Claim] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        indices = _valid_indices(item.get("source_indices"), allowed)
        if not indices:
            logger.debug("Dropping unattributed claim: %s", text[:80])
            continue
        strength = item.get("strength")
        strength = strength if strength in _STRENGTHS else "moderate"
        if not any(directness_by_index.get(i) == "direct" for i in indices):
            strength = "weak"
        claims.append(Claim(
            text=text.strip(),
            source_indices=indices,
            strength=strength,
        ))
    return claims


def _read_disagreements(raw: object, allowed: set[int]) -> list[Disagreement]:
    if not isinstance(raw, list):
        return []

    conflicts: list[Disagreement] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        issue = item.get("issue")
        if not isinstance(issue, str) or not issue.strip():
            continue
        indices = _valid_indices(item.get("source_indices"), allowed)
        if len(indices) < 2:
            # A conflict needs at least two sources to be a conflict.
            continue
        positions = [
            p.strip() for p in (item.get("positions") or [])
            if isinstance(p, str) and p.strip()
        ]
        conflicts.append(Disagreement(
            issue=issue.strip(), positions=positions[:4], source_indices=indices,
        ))
    return conflicts


def synthesise(
    topic: str,
    articles: list[ArticleSummary],
    settings: Settings,
    summary_language: str = "it",
    job_stats: JobStats | None = None,
) -> tuple[str, list[Claim], list[Disagreement], list[str]]:
    """Write the grounded overview from the articles that cleared the bar.

    Returns ``(global_summary, key_findings, disagreements, evidence_gaps)``.
    On failure the summary is empty and the lists are empty: a caller that
    shows nothing is in a better position than one shown claims from a call
    that did not complete.

    Clearing the relevance bar is not the same as answering the question: a
    source can be topically relevant while only studying a related-but-
    different population or intervention. Each article carries a
    ``directness`` verdict from :func:`summarise_sources` for exactly this -
    a claim resting only on non-direct sources is clamped to "weak" here, and
    the overview is prefixed with an explicit "no direct evidence" flag when
    every cited source is at best adjacent.
    """
    relevant_indices = [
        index for index, article in enumerate(articles)
        if article.relevance_score > MIN_SYNTHESIS_RELEVANCE
    ]
    if not relevant_indices:
        return low_evidence_message(summary_language), [], [], []

    prompt = (
        f"TOPIC: {topic}\n\nSOURCES:\n"
        + _source_lines(articles, relevant_indices)
    )

    try:
        raw = generate_json(
            settings=settings,
            prompt=prompt,
            system_instruction=_SYNTHESIS_SYSTEM.format(
                language=_language_instruction(summary_language),
            ),
            temperature=0.2,
            purpose="related_articles_synthesis",
            job_stats=job_stats,
        )
    except Exception as exc:
        logger.warning("Synthesis failed: %s", exc)
        return "", [], [], []

    if not isinstance(raw, dict):
        return "", [], [], []

    allowed = set(relevant_indices)
    directness_by_index = {i: articles[i].directness for i in relevant_indices}
    claims = _read_claims(raw.get("key_findings"), allowed, directness_by_index)
    conflicts = _read_disagreements(raw.get("disagreements"), allowed)
    gaps = [
        g.strip() for g in (raw.get("evidence_gaps") or [])
        if isinstance(g, str) and g.strip()
    ][:5]

    summary = raw.get("global_summary")
    summary = summary.strip() if isinstance(summary, str) else ""
    if summary and "direct" not in directness_by_index.values():
        # Every cited source is adjacent at best: the overview reads as a
        # direct answer unless it says up front that it is not one.
        summary = _no_direct_evidence_prefix(summary_language) + summary
    return summary, claims, conflicts, gaps
