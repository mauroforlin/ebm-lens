"""Study design detection and the evidence hierarchy.

Relevance ranking answers *is this paper about the topic*. In clinical
medicine that is only half the question: a case report and a Cochrane
meta-analysis on the same subject are equally on-topic and are not equally
informative. Evidence-based medicine has ranked study designs for decades,
and this module turns that ranking into a signal the ranker can weigh.

Three sources can say what a paper's design is, and they are consulted in
order of how much they know:

1. the **provider**, where it is definitional - a DailyMed record is a
   regulatory drug label whatever its title says, a bioRxiv record is a
   preprint, a ClinicalTrials.gov record is a registration;
2. the **publication types the provider assigns**. PubMed's come from NLM's
   own indexers, who read the paper: ``Randomized Controlled Trial`` there is
   a statement of fact, not an inference, and it outranks anything read off
   the wording;
3. the **title and abstract**, read by one batched LLM call across every
   record that reaches this stage without a provider fact or publication type
   to lean on. Grading runs on the shortlist a search has already narrowed to,
   not the raw discovery pool, so one request covers the whole batch cheaply -
   a reading of what the sentence actually says, rather than a pattern hoping
   "randomised" only appears in trials and never in reviews of them.

A record the model leaves unplaced, or that reaches this with the LLM call
unavailable, falls back to :func:`detect_design`'s text-pattern reading rather
than losing a score outright. Above all three,
:func:`app.pipeline.synthesis.summarise_sources` returns a design read out of
the full text for every source that reaches the answer, and the orchestrator
prefers that reading over anything here.

Unrecognised designs get a neutral score rather than a low one. A wrong guess
demotes real evidence, so ambiguity resolves to "no opinion", never to
"weak".

The scores are an ordering, not GRADE. GRADE grades a body of evidence for a
specific question, weighing risk of bias, imprecision and indirectness that no
title-level heuristic can see. What this gives the ranker is the design tier -
the first and largest of GRADE's inputs - and the response reports the design
by name so the user grades the rest themselves.

The ordering follows the OCEBM Levels of Evidence and GRADE (Guyatt et al.,
BMJ 2008); the publication-type vocabulary is NLM's own, as assigned to
MEDLINE records.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from app.config import Settings
from app.core.job_stats import JobStats
from app.core.llm_client import generate_json
from app.pipeline.dedup import dedup_key
from app.sources.base import SourceResult

logger = logging.getLogger(__name__)


#
# Scores are relative positions in [0, 1], not probabilities. The gaps encode
# the judgements that matter: synthesised evidence sits clearly above any one
# study, randomisation clearly above observation, and anything that has not
# been through peer review or is not about humans sits below both.

DESIGN_SCORES: dict[str, float] = {
    "meta_analysis": 1.00,
    "systematic_review": 0.95,
    "guideline": 0.92,
    "rct": 0.82,
    "drug_label": 0.78,        # regulatory, authoritative for labelling facts
    "controlled_trial": 0.72,  # non-randomised or unspecified allocation
    "cohort": 0.62,
    "trial_registry": 0.55,    # registered, results may not exist yet
    "case_control": 0.52,
    "narrative_review": 0.48,
    "cross_sectional": 0.42,
    "unknown": 0.40,           # neutral: no opinion, not a penalty
    "surveillance": 0.38,      # spontaneous-report databases (FAERS)
    "preclinical": 0.30,       # in vitro, animal, computational
    "case_report": 0.25,
    "commentary": 0.20,   # editorial, letter, comment: opinion, not data
    "encyclopedic": 0.15,
}

# Preprints are a design-independent discount: a preprinted RCT is still an
# RCT, and still has not been reviewed. Applied multiplicatively so it scales
# with what the design was worth in the first place.
_PREPRINT_FACTOR = 0.75

# Human-readable labels, returned to the client alongside each article.
DESIGN_LABELS: dict[str, str] = {
    "meta_analysis": "meta-analysis",
    "systematic_review": "systematic review",
    "guideline": "clinical guideline",
    "rct": "randomised controlled trial",
    "drug_label": "regulatory drug label",
    "controlled_trial": "controlled trial",
    "cohort": "cohort study",
    "trial_registry": "registered trial",
    "case_control": "case-control study",
    "narrative_review": "review",
    "cross_sectional": "cross-sectional study",
    "surveillance": "adverse-event surveillance",
    "preclinical": "preclinical study",
    "case_report": "case report",
    "commentary": "editorial or commentary",
    "encyclopedic": "encyclopedic",
    "unknown": "",
}

# Providers whose records have one design by construction. Checked before any
# text matching: the title of a DailyMed label may well mention a trial.
_PROVIDER_DESIGNS: dict[str, str] = {
    "dailymed": "drug_label",
    "openfda": "drug_label",
    "rxnav": "drug_label",
    "ema": "drug_label",
    "clinicaltrials": "trial_registry",
    "wikipedia": "encyclopedic",
    "who_gho": "surveillance",
    "chembl": "preclinical",
    "open_targets": "preclinical",
}

# NLM publication types, lowercased, mapped onto the hierarchy. A record
# usually carries several ("Journal Article" plus "Meta-Analysis" plus
# "Review"); the highest-scoring match wins, which is also the most specific
# one - a meta-analysis tagged as a review is a meta-analysis. Types that say
# nothing about design ("Journal Article", "Research Support") are absent, so
# they simply do not vote.
_PUB_TYPE_DESIGNS: dict[str, str] = {
    "meta-analysis": "meta_analysis",
    "systematic review": "systematic_review",
    "practice guideline": "guideline",
    "guideline": "guideline",
    "consensus development conference": "guideline",
    "randomized controlled trial": "rct",
    "controlled clinical trial": "controlled_trial",
    "clinical trial": "controlled_trial",
    "clinical trial, phase iii": "rct",
    "clinical trial, phase ii": "controlled_trial",
    "pragmatic clinical trial": "controlled_trial",
    "equivalence trial": "controlled_trial",
    "adaptive clinical trial": "controlled_trial",
    "observational study": "cohort",
    "multicenter study": "cohort",
    "comparative study": "cohort",
    "twin study": "cohort",
    "case reports": "case_report",
    "review": "narrative_review",
    "editorial": "commentary",
    "letter": "commentary",
    "comment": "commentary",
    "news": "commentary",
    "published erratum": "commentary",
}


def _design_from_publication_types(publication_types: Sequence[str]) -> str | None:
    matched = []
    for raw in publication_types or []:
        if not isinstance(raw, str):
            continue
        pub_type = raw.strip().lower()
        design = _PUB_TYPE_DESIGNS.get(pub_type) or normalise_design(pub_type)
        if design and design != "unknown":
            matched.append(design)
    if not matched:
        return None
    return max(matched, key=lambda d: DESIGN_SCORES.get(d, 0.0))


# Ordered most-specific first: "systematic review and meta-analysis" must
# match as a meta-analysis, and a "post hoc analysis of a randomised trial"
# must not be read as a fresh RCT before the RCT pattern is even reached.
_DESIGN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("meta_analysis", re.compile(
        r"\bmeta[- ]?analy(?:sis|ses|tic)\b|\bnetwork meta\b|\bpooled analysis\b", re.I)),
    ("systematic_review", re.compile(
        r"\bsystematic review\b|\bcochrane review\b|\bscoping review\b"
        r"|\bumbrella review\b|\bevidence synthesis\b|\brapid review\b", re.I)),
    ("guideline", re.compile(
        r"\b(?:clinical |practice |treatment )?guidelines?\b|\bconsensus statement\b"
        r"|\bposition (?:paper|statement)\b|\brecommendations? (?:of|from|for) the\b"
        r"|\bexpert consensus\b", re.I)),
    ("rct", re.compile(
        r"\brandomi[sz]ed\b|\brandomi[sz]ation\b|\bdouble[- ]blind\b"
        r"|\bplacebo[- ]controlled\b|\bphase (?:ii|iii|2|3) trial\b|\brct\b", re.I)),
    ("controlled_trial", re.compile(
        r"\bclinical trial\b|\bcontrolled trial\b|\bopen[- ]label trial\b"
        r"|\bcrossover (?:trial|study)\b|\bnon[- ]inferiority\b", re.I)),
    ("cohort", re.compile(
        r"\bcohort\b|\bprospective (?:study|analysis|follow)\b"
        r"|\blongitudinal (?:study|analysis)\b|\bregistry[- ]based\b", re.I)),
    ("case_control", re.compile(r"\bcase[- ]control\b|\bnested case\b", re.I)),
    ("cross_sectional", re.compile(
        r"\bcross[- ]sectional\b|\bsurvey (?:of|among|study)\b|\bprevalence (?:of|study)\b"
        r"|\bnhanes\b", re.I)),
    ("case_report", re.compile(
        r"\bcase report\b|\bcase series\b|\ba case of\b|\bwe report a\b", re.I)),
    ("preclinical", re.compile(
        r"\bin vitro\b|\bin vivo\b|\bmouse\b|\bmurine\b|\brats?\b|\bzebrafish\b"
        r"|\bcell line\b|\bmolecular docking\b|\bknockout\b|\bxenograft\b", re.I)),
    ("narrative_review", re.compile(
        r"\breview\b|\boverview of\b|\bstate of the art\b|\bcurrent perspectives\b"
        r"|\bnarrative synthesis\b", re.I)),
)

_PREPRINT_RE = re.compile(r"\bpreprint\b|\bbiorxiv\b|\bmedrxiv\b|\bnot .{0,20}peer[- ]reviewed\b", re.I)

# Chars of the record examined. The design is named in the title or the first
# lines of the abstract; reading further mostly finds designs the paper is
# discussing rather than the one it used.
_SCAN_CHARS = 700


def _scan_text(result: SourceResult) -> str:
    body = result.content or result.snippet or ""
    return f"{result.title}. {body}"[:_SCAN_CHARS]


def detect_design(result: SourceResult) -> tuple[str, bool]:
    """Return ``(design_id, is_preprint)`` for one candidate, with no LLM call.

    The provider wins over the text where it is definitive, because it is a
    fact about the record rather than an inference from its wording. This is
    the reading used where a pool is too large or too early in the pipeline to
    have earned an LLM call yet, and it is :func:`grade_results`'s own
    fallback for records its batched call can't place.
    """
    source = (result.source_type or "").strip().lower()
    is_preprint = source == "biorxiv" or bool(_PREPRINT_RE.search(_scan_text(result)))

    fixed = _PROVIDER_DESIGNS.get(source)
    if fixed:
        return fixed, is_preprint

    indexed = _design_from_publication_types(result.publication_types)
    if indexed:
        return indexed, is_preprint

    text = _scan_text(result)
    for design, pattern in _DESIGN_PATTERNS:
        if pattern.search(text):
            return design, is_preprint

    return "unknown", is_preprint


def evidence_score(design: str, is_preprint: bool = False) -> float:
    score = DESIGN_SCORES.get(design, DESIGN_SCORES["unknown"])
    return score * _PREPRINT_FACTOR if is_preprint else score


def design_label(design: str, is_preprint: bool = False) -> str:
    label = DESIGN_LABELS.get(design, "")
    if is_preprint:
        return f"{label} (preprint)" if label else "preprint"
    return label


def normalise_design(design: str | None) -> str | None:
    """Coerce an externally supplied design name to a known id, or None.

    The summariser is asked for one of :data:`DESIGN_SCORES`' keys and mostly
    obliges, but it answers in prose often enough ("randomized controlled
    trial") that accepting only exact ids would throw away a reading of the
    full text over a hyphen.
    """
    if not isinstance(design, str):
        return None
    candidate = design.strip().lower().removesuffix("(preprint)").strip()
    candidate = candidate.replace("-", "_").replace(" ", "_")
    if candidate in DESIGN_SCORES:
        return candidate
    for design_id, label in DESIGN_LABELS.items():
        if not label:
            continue
        normalised = label.lower().replace("-", "_").replace(" ", "_")
        if candidate in (normalised, normalised.replace("randomised", "randomized")):
            return design_id
    return None


# Ids the batched call may return - exactly the designs :func:`detect_design`
# would otherwise have to guess from title/abstract wording. Provider-fact
# designs (drug_label, trial_registry, surveillance) are excluded: a record
# reaches this call only after those checks have already failed to place it.
_TEXT_DESIGNS = frozenset({
    "meta_analysis", "systematic_review", "guideline", "rct", "controlled_trial",
    "cohort", "case_control", "cross_sectional", "case_report", "preclinical",
    "narrative_review", "commentary", "unknown",
})

_DESIGN_BATCH_SYSTEM = """\
You are a clinical study design classifier. For each numbered record (its
title plus the opening of its abstract), name the study design using ONLY
these ids:

meta_analysis, systematic_review, guideline, rct, controlled_trial, cohort,
case_control, cross_sectional, case_report, preclinical, narrative_review,
commentary, unknown

Read what the record says about ITS OWN design, not what it discusses: a
review about randomised trials is systematic_review or narrative_review, not
rct. A negated or hedged mention ("was not randomised", "trials such as X may
show...") does not count as that design. Use "unknown" whenever the record
does not clearly state its own design - guessing from the topic is worse than
abstaining.

Return a JSON object: {"designs": {"<record index>": "<design id>", ...}},
exactly one entry per record given.
"""


def _llm_batch_designs(
    pending: Sequence[tuple[str, SourceResult]],
    settings: Settings,
    job_stats: JobStats | None,
) -> dict[str, str]:
    """Classify design for records with no provider fact or publication type.

    One request over the whole batch, not one per record: at the shortlist
    size this runs on, a single call is negligible next to the summarisation
    stage that follows it. A record the model omits or answers outside
    :data:`_TEXT_DESIGNS` for is simply left out of the result, for the caller
    to fall back on.
    """
    prompt = "RECORDS:\n\n" + "\n\n".join(
        f"{i}. {_scan_text(result)}" for i, (_key, result) in enumerate(pending)
    )
    try:
        raw = generate_json(
            settings=settings,
            prompt=prompt,
            system_instruction=_DESIGN_BATCH_SYSTEM,
            temperature=0.0,
            purpose="related_articles_design_grade",
            job_stats=job_stats,
        )
    except Exception as exc:
        logger.warning("Design batch classification failed, falling back to text patterns: %s", exc)
        return {}

    answers = raw.get("designs") if isinstance(raw, dict) else None
    if not isinstance(answers, dict):
        return {}

    resolved: dict[str, str] = {}
    for i, (key, _result) in enumerate(pending):
        design = answers.get(str(i))
        if design in _TEXT_DESIGNS:
            resolved[key] = design
    return resolved


def grade_results(
    results: Sequence[SourceResult],
    settings: Settings | None = None,
    job_stats: JobStats | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """Grade a whole candidate pool.

    Provider facts and publication types are read straight off each record.
    Whatever is left goes to one batched LLM call (skipped, falling straight
    to :func:`detect_design`, when *settings* is not supplied); a record the
    call fails to place still gets that same text-pattern fallback rather than
    an empty score.

    Returns ``(score_map, label_map)``, both keyed by dedup key: the score
    feeds the ranker, the label is shown to the user so the ranking can be
    argued with.
    """
    is_preprint: dict[str, bool] = {}
    design: dict[str, str] = {}
    pending: list[tuple[str, SourceResult]] = []

    for result in results:
        key = dedup_key(result)
        source = (result.source_type or "").strip().lower()
        is_preprint[key] = source == "biorxiv" or bool(_PREPRINT_RE.search(_scan_text(result)))

        fixed = _PROVIDER_DESIGNS.get(source) or _design_from_publication_types(result.publication_types)
        if fixed:
            design[key] = fixed
        else:
            pending.append((key, result))

    if pending and settings is not None:
        design.update(_llm_batch_designs(pending, settings, job_stats))

    scores: dict[str, float] = {}
    labels: dict[str, str] = {}
    for result in results:
        key = dedup_key(result)
        if key not in design:
            design[key], _ = detect_design(result)
        scores[key] = evidence_score(design[key], is_preprint[key])
        labels[key] = design_label(design[key], is_preprint[key])
    return scores, labels


# Designs strong enough that a body of evidence containing them is worth
# calling strong, rather than merely large.
_HIGH_TIER = frozenset({"meta_analysis", "systematic_review", "guideline", "rct"})


def evidence_profile(results: Sequence[SourceResult]) -> dict[str, object]:
    """Describe the *shape* of a result set's evidence, for the response.

    Ten case reports and ten randomised trials answer a question with very
    different authority while looking alike in a list. This is the number that
    tells the user which of the two they are reading, so it is reported even
    though nothing ranks on it.
    """
    if not results:
        return {"designs": {}, "high_tier_share": 0.0, "preprint_share": 0.0}

    designs: dict[str, int] = {}
    high_tier = 0
    preprints = 0
    for result in results:
        design, is_preprint = detect_design(result)
        label = DESIGN_LABELS.get(design) or "unclassified"
        designs[label] = designs.get(label, 0) + 1
        if design in _HIGH_TIER and not is_preprint:
            high_tier += 1
        if is_preprint:
            preprints += 1

    total = len(results)
    return {
        "designs": dict(sorted(designs.items(), key=lambda kv: -kv[1])),
        "high_tier_share": round(high_tier / total, 2),
        "preprint_share": round(preprints / total, 2),
    }
