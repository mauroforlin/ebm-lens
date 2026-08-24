"""Reading the evidence: per-source appraisal and the grounded synthesis.

Two LLM passes turn a ranked list of papers into something a person can act
on. Both are written to fail safe, because a wrong summary of medical
evidence is worse than no summary:

* :func:`summarise_sources` reads each shortlisted source and returns a short
  summary, a relevance score, and an appraisal - what design the study used,
  who it studied, what it found and which way that cuts for the topic. The
  relevance score is the pipeline's only judgement made after actually reading
  the content, and the final ranking leans on it; the design is a reading of
  the full text, so it overrides the pool-wide heuristic in
  :mod:`app.pipeline.evidence_grade`.
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

When too little relevant evidence exists, both passes say so plainly rather
than producing confident prose out of weak input. Relevance is not the same
test as directness, though: a source can be topically on-topic - same drug,
same organ, same disease family - while studying a different population or
intervention than the one actually asked about. The summariser judges each
source's ``directness`` against the topic's PICO for exactly this, and the
synthesiser is not trusted to honour it unassisted: a claim resting only on
non-direct sources is clamped to "weak" in code, and the overview is
prefixed with an explicit flag when nothing cited is a direct match.

Validated claim-level attribution follows the attribution-evaluation
literature (arXiv:2605.06635, arXiv:2408.04568); refusing over answering from
weak evidence follows arXiv:2409.11242; reporting conflicts rather than
averaging them follows *Contradictions in Context* (arXiv:2511.06668).
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.config import Settings
from app.core.job_stats import JobStats
from app.core.llm_client import generate_json, generate_with_tools
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

# Chars of each source shown to the summariser up front, before any
# read_full_text call.
_MAX_SOURCE_CHARS = 4000

# Upper bound on read_full_text calls within one appraisal batch. Generous -
# there is no per-call cost pressure here - but still a bound, because the
# upstream sources this hits (NCBI, EMA, ...) have their own rate limits.
# Never actually binds today since a batch has at most _SUMMARISE_BATCH_SIZE
# sources to read, but stays as the real ceiling should that constant grow.
_MAX_FULL_TEXT_READS = 12

_MAX_APPRAISAL_TOOL_ROUNDS = 6

# Sources appraised per summarise_sources call, and how many such calls run
# at once. Smaller batches mean each finding_direction/directness judgment -
# see _SUMMARISE_SYSTEM's rules on both - competes with fewer other sources'
# content for the model's attention than one call covering the whole
# shortlist would; running batches concurrently keeps wall-clock latency
# close to the single-call version despite the extra round trips.
_SUMMARISE_BATCH_SIZE = 5
_SUMMARISE_MAX_WORKERS = 6

_TIER_LABELS = {1: "institutional", 2: "curated", 3: "general"}

# Directions a source's finding can take on the topic. Anything else the model
# returns is read as "neutral" rather than trusted.
_DIRECTIONS = frozenset({"supports", "contradicts", "mixed", "neutral"})

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
    """Say that the evidence was too weak, instead of synthesising anyway."""
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


# ══════════════════════════════════════════════════════════════
#  Per-source appraisal
# ══════════════════════════════════════════════════════════════

_SUMMARISE_SYSTEM = """\
You are a research assistant appraising source articles for a user \
investigating a specific medical/clinical topic. For each source you produce \
a concise summary, a relevance score, and a short critical appraisal.

Each source starts with an excerpt - an abstract, a label section, a search \
snippet. That is enough for most sources. When it is not - the excerpt is \
too thin to tell what the study actually did, or you need the methods or \
results to judge study_design or directness with confidence rather than \
guessing - call read_full_text with that source's index before appraising \
it. A working human reviewer does not open every paper's PDF, only the ones \
whose abstract does not answer the question; appraise the same way. Do not \
call it reflexively for every source, and do not call it for a source whose \
excerpt already gives you what you need. When you are done reading whatever \
you needed to read, call submit_appraisals once with every source's \
appraisal.

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
  "key_finding": "<the ONE claim this source itself reports data or a result \
for, in {language}, one sentence; empty string if the source reports nothing \
that bears on the topic - discussing the same drug, disease or general \
subject is not enough on its own, see rules below>",
  "finding_direction": "<one of: supports, contradicts, mixed, neutral - \
compare what the TOPIC specifically claims (its direction, its asserted \
effect) against what the source's own result actually shows, then label the \
comparison - see rules below for what counts as which>",
  "directness": "<one of: direct, adjacent, unclear - judged against the \
PICO given below, when one is given>"
}}

Rules:
- Prioritise readability and speed: this is a quick-glance summary, not a review.
- Keep only the core idea; omit secondary details and long context.
- Mention at most one concrete detail (number/date/entity) only if critical.
- ALL prose fields MUST be written in {language}. study_design, \
finding_direction and directness are ids: return them in English, exactly as \
listed.
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
- key_finding must be a claim the source itself makes, backed by data or a \
result it actually reports - not an inference from the source merely \
discussing the same drug, disease or subject area. When a source is topically \
on-topic but never tests or reports anything that speaks to the topic, leave \
key_finding EMPTY and finding_direction "neutral" - do not fill it in just \
because the source is relevant reading. Only mark it non-empty when you could \
point to the specific sentence or data point in the source that makes the claim.
- finding_direction: a null result, "no significant difference", or an effect \
running the OPPOSITE way from what the topic claims is "contradicts" - never \
"supports", and not "neutral" either, since the source did address the claim \
and its answer was no. Reserve "neutral" for a source that does not test or \
speak to the topic's specific claim at all (see key_finding above). Do not \
judge direction from the source's overall tone or from it being about a \
generally beneficial/harmful topic - a study of a beneficial drug can still \
report a null or negative result for the ONE claim being asked about, and \
that is "contradicts". Getting this backwards - reading a source's "no" as \
"yes" - is the single worst error possible here, worse than an empty or \
"unclear" answer; when genuinely torn between "supports" and "contradicts", \
re-read the specific result sentence before deciding, do not guess toward \
"supports".
- directness is about the SAME facts as key_finding, judged more strictly: a \
source that treats a different population, a different disease mechanism, or \
a different formulation than the one asked about is "adjacent" even when it \
shares a drug, organ or keyword with the topic - do not mark a source \
"direct" just because it is topically relevant. Use "unclear" only when the \
topic gives no specific enough population/intervention to judge against.
- If a source has very little content or is mostly irrelevant, give it a \
low relevance_score and a very short summary noting limited utility.
"""


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
            "Fetch the full page text for one source, when its excerpt is too "
            "thin to judge study design, directness or the key finding with "
            "confidence. Only works for sources on a trusted allowlist of "
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
                            "key_finding": {"type": "string"},
                            "finding_direction": {"type": "string"},
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

    The old version put every source in one call: with a 20-30 source
    shortlist, that meant the model was judging source 12's
    ``finding_direction`` - the single highest-stakes field here, and the
    one call that reads "the source found nothing" as "the source agrees"
    is the worst error this pipeline can make - with the content of the
    other 19 still in view. Splitting the shortlist into batches of
    ``_SUMMARISE_BATCH_SIZE`` gives each judgment a narrower, less
    distracting context; running the batches concurrently keeps the wall-clock
    cost close to the single big call this replaces rather than multiplying
    it by the batch count. See :func:`_summarise_batch` for the per-batch
    mechanics, which are otherwise unchanged from the single-call version -
    same excerpt-first reading, same bounded ``read_full_text`` tool.

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
    # Manual lifecycle, not `with ThreadPoolExecutor(...)`: as elsewhere in
    # the pipeline (see app.pipeline.agentic), a bare `with` calls
    # shutdown(wait=True) on exit, which would block on every batch even
    # after some have already failed.
    pool = ThreadPoolExecutor(max_workers=min(_SUMMARISE_MAX_WORKERS, len(batches)))
    try:
        futures = {
            pool.submit(
                _summarise_batch, topic, results, batch, settings,
                summary_language, job_stats, pico,
            ): batch
            for batch in batches
        }
        for future in as_completed(futures):
            try:
                summaries.update(future.result())
            except Exception as exc:
                logger.warning("Source appraisal batch failed: %s", exc)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
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
    already_read: set[int] = set()

    def _read_full_text(args: dict) -> str:
        nonlocal reads_used
        index = args.get("index")
        if not isinstance(index, int) or index not in allowed:
            return "invalid index"
        if index in already_read:
            return "already fetched - appraise from the full text already given to you"
        if reads_used >= _MAX_FULL_TEXT_READS:
            return "read budget exhausted for this run - appraise from the excerpt"
        reads_used += 1
        already_read.add(index)
        result = results[index]
        # PubMed's abstract page cannot be scraped (cookie-challenge on a
        # plain GET) - its full text, when it has any, lives in PMC instead.
        if result.source_type == "pubmed":
            from app.sources.pubmed import fetch_full_text_by_url
            text = fetch_full_text_by_url(result.url)
        else:
            text = fetch_full_text(result)
        if not text:
            return (
                "not available - this source is not on the fetchable allowlist, "
                "or the fetch failed. Appraise from the excerpt you already have."
            )
        results[index].content = text
        return text[:_MAX_SOURCE_CHARS]

    handlers = {
        "read_full_text": _read_full_text,
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
            ),
            tools=[_READ_FULL_TEXT_TOOL, _SUBMIT_APPRAISALS_TOOL],
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
            summaries[index] = item
    return summaries


def clamp_relevance(value: object) -> float:
    """Coerce an LLM-supplied relevance score into 0.0-1.0, defaulting to 0.0."""
    try:
        return min(1.0, max(0.0, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def read_direction(value: object) -> str:
    """Coerce a finding direction to a known value, defaulting to neutral."""
    candidate = value.strip().lower() if isinstance(value, str) else ""
    return candidate if candidate in _DIRECTIONS else "neutral"


def read_directness(value: object) -> str:
    """Coerce a directness verdict to a known value, defaulting to unclear."""
    candidate = value.strip().lower() if isinstance(value, str) else ""
    return candidate if candidate in _DIRECTNESS else "unclear"


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


# ══════════════════════════════════════════════════════════════
#  Global synthesis
# ══════════════════════════════════════════════════════════════

_SYNTHESIS_SYSTEM = """\
You are an evidence synthesis writer working from a set of numbered,
titled sources. Every statement you make must be traceable to them.

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
        if article.finding_direction and article.finding_direction != "neutral":
            attributes.append(f"direction: {article.finding_direction}")
        if article.directness in ("direct", "adjacent"):
            attributes.append(article.directness)

        body = article.key_finding or article.full_summary
        title = article.title.strip() if article.title else ""
        title_part = f' "{title}"' if title else ""
        lines.append(f"[{index}]{title_part} ({', '.join(attributes)}) {body}")
        if article.population:
            lines.append(f"      population: {article.population}")
    return "\n".join(lines)


def _valid_indices(raw: object, allowed: set[int]) -> list[int]:
    """Keep only the source indices that name a source actually in the answer."""
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
    """Parse reported conflicts, keeping only those pointing at real sources."""
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
