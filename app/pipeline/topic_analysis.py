"""Turning a user's sentence into something searchable.

Two LLM passes run before any database is touched, because both change what
gets searched rather than merely how results are ranked:

* :func:`analyse_topic` fixes the domain and writes the query variants. Its
  output decides which of the twelve providers are even plausible, and its
  keywords are what disambiguate a polysemous topic.
* :func:`decompose_topic` detects a *composite* topic - one carrying several
  genuinely independent research axes. Searched as a single string, such a
  topic returns whatever its most-published axis is and silently drops the
  rest; searched per axis, each gets its own evidence.

Both degrade to a usable default when the LLM fails, so neither can take the
pipeline down with it.

:func:`analyse_topic` also puts the topic in PICO form (Richardson et al.,
*The Well-Built Clinical Question*, 1995), the frame biomedical databases are
indexed to answer. Rewriting a free-text clinical question that way before
retrieving follows PICOs-RAG (arXiv:2510.23998).
"""
from __future__ import annotations

import logging

from app.config import Settings
from app.core.job_stats import JobStats
from app.core.llm_client import generate_json
from app.schemas import PICO, DomainContext, TopicSpec

logger = logging.getLogger(__name__)

_MAX_FACETS = 5


_DOMAIN_SYSTEM = """\
You are a medical/clinical domain classifier and search query optimizer.
Given a topic or claim, analyse it and return a JSON object:

{
  "domain": "<one of: medicine, veterinary_medicine, generic>",
  "subdomain": "<short specialism, e.g. 'pharmacology', 'cardiology', 'oncology'>",
  "topic_summary": "<one-sentence summary>",
  "keywords_en": ["2-4 English keywords for search"],
  "keywords_it": ["2-4 Italian keywords for search"],
  "search_query": "<optimised 5-8 word English search query>",
  "search_query_it": "<optimised 5-8 word Italian search query>",
  "conceptual_query": "<BROADER English fallback query, 3-5 words>",
  "conceptual_query_it": "<Italian equivalent of conceptual_query>",
  "subject": "<main entity or concept>",
  "topic_type": "<assertion, definition, statistic, dosage, drug, drug_classification, contraindication, drug_interaction, adverse_effect, diagnostic_criteria, epidemiology, reference_value>",
  "pico": {
    "population": "<who/what the question is about, '' if not specified>",
    "intervention": "<the drug, exposure, test or procedure, '' if none>",
    "comparison": "<what it is compared against, '' if none stated>",
    "outcome": "<the endpoint asked about, '' if none stated>",
    "study_designs": ["designs that would best answer this, e.g. 'randomized controlled trial'"]
  }
}

Rules:
- Use "veterinary_medicine" when the topic discusses animal health/breeding/anatomy.
- Use "medicine" for human medicine, pharmacology, clinical research, public health.
- Use "generic" only when the topic is clearly NOT medical/clinical - this
  system is built for medical/clinical research discovery, so default to
  "medicine" for anything ambiguous with a plausible health angle.
- search_query must be encyclopedic, precise, suitable for biomedical databases.
- search_query_it must be optimised for Italian sources (Wikipedia IT).
- conceptual_query is a DELIBERATELY BROADER framing, used only if the precise
  queries return too little. Drop the narrowest qualifier and name the general
  mechanism, disease or drug class instead. It must NOT repeat search_query.
- keywords must help constrain searches to the correct domain.
- pico is how evidence-based medicine frames a searchable clinical question,
  and biomedical databases are indexed to answer it. Fill ONLY the elements
  the topic actually states. NEVER invent a comparison or an outcome the user
  did not ask about: a fabricated element sends the search after literature
  the user has no interest in. For a non-clinical topic (a definition, an
  anatomy question) leave every element empty.
- Use the field's standard terminology in the pico elements, not the user's
  wording: "myocardial infarction", not "heart attack".
"""


def analyse_topic(
    topic: str,
    settings: Settings,
    domain_hint: str | None = None,
    job_stats: JobStats | None = None,
) -> tuple[DomainContext, TopicSpec]:
    """Classify *topic* and build the search spec every provider works from.

    Search queries are always generated in English (the academic standard for
    biomedical databases) with an Italian variant for Wikipedia IT.

    On LLM failure this returns a spec built from the raw topic text: worse
    targeting, but the search still runs.
    """
    hint = f"\nHint from the user: {domain_hint}" if domain_hint else ""
    prompt = (
        "Analyse this topic and classify its domain. Generate optimised search "
        "queries. Search queries MUST be in ENGLISH (the academic standard). "
        f"Also generate an Italian variant for Wikipedia IT.{hint}\n\n"
        f"TOPIC: {topic}\n\n"
        "Return the JSON object as described in your instructions."
    )

    try:
        raw = generate_json(
            settings=settings,
            prompt=prompt,
            system_instruction=_DOMAIN_SYSTEM,
            temperature=0.05,
            purpose="related_articles_domain",
            job_stats=job_stats,
        )
    except Exception as exc:
        logger.warning("Domain detection failed, using defaults: %s", exc)
        raw = {}
    if not isinstance(raw, dict):
        raw = {}

    domain = raw.get("domain") or domain_hint or "medicine"
    keywords_en = raw.get("keywords_en") or []
    keywords_it = raw.get("keywords_it") or []

    context = DomainContext(
        domain=domain,
        subdomain=raw.get("subdomain", ""),
        topic=raw.get("topic_summary", topic),
        keywords_en=keywords_en,
        keywords_it=keywords_it,
    )

    spec = TopicSpec(
        text=topic,
        topic_type=raw.get("topic_type", "assertion"),
        domain=domain,
        subject=raw.get("subject", ""),
        search_query=raw.get("search_query", topic),
        search_query_it=raw.get("search_query_it", topic),
        conceptual_query=raw.get("conceptual_query"),
        conceptual_query_it=raw.get("conceptual_query_it"),
        alternative_queries=keywords_en,
        alternative_queries_it=keywords_it,
        pico=_build_pico(raw.get("pico")),
    )

    return context, spec


def _build_pico(raw: object) -> PICO | None:
    """Build a :class:`PICO` from the analyser's output, or None if unusable.

    An under-specified PICO is discarded rather than kept: a P/I/C/O carrying
    only a population tells the query builder nothing the topic string did not
    already say, and would still cost a weight in the ranker.
    """
    if not isinstance(raw, dict):
        return None

    def _text(key: str) -> str:
        value = raw.get(key)
        return value.strip() if isinstance(value, str) else ""

    designs = [d.strip() for d in (raw.get("study_designs") or []) if isinstance(d, str) and d.strip()]
    pico = PICO(
        population=_text("population"),
        intervention=_text("intervention"),
        comparison=_text("comparison"),
        outcome=_text("outcome"),
        study_designs=designs[:3],
    )
    return pico if pico.is_specified() else None


_DECOMPOSE_SYSTEM = """\
You are a research decomposition engine for a medical/biomedical search
pipeline. Given a complex research topic, determine whether it contains
multiple DISTINCT research axes that would benefit from independent
parallel search.

Return a JSON object:
{
  "is_composite": true/false,
  "confidence": 0.0-1.0,
  "sub_claims": [
    {
      "facet_label": "short label (3-6 words, e.g. 'ctDNA methylation in glioblastoma')",
      "search_query": "optimised 5-8 word English query for this facet",
      "search_query_it": "Italian equivalent for Wikipedia IT",
      "keywords_en": ["2-3 keywords"],
      "topic_type": "assertion|definition|statistic|procedure"
    }
  ],
  "bridge_query": "English query targeting the INTERSECTION of all sub-topics",
  "bridge_query_it": "Italian equivalent of bridge_query",
  "bridge_concepts": ["terms shared across sub-topics (e.g. 'glioblastoma')"],
  "enrichment_keywords": ["extra keywords useful even if topic is NOT composite"]
}

Rules:
- Set is_composite=true ONLY when there are 2+ GENUINELY DISTINCT research axes
  that have their own independent literature. Examples:
  YES: "CAR-T therapy + focused ultrasound BBB disruption + ctDNA monitoring in GBM"
       (each axis has its own body of research)
  NO:  "aspirin effects on cardiovascular health" (one axis, multi-keyword)
  NO:  "pathogenesis and treatment of type 2 diabetes" (two aspects of ONE topic)
  NO:  "CRISPR-Cas9 gene editing mechanism" (one concept, even if technical)
- Each sub_claim must be independently searchable in PubMed/Europe PMC
- Max 5 sub_claims (beyond that, the topic needs reformulation)
- bridge_query targets papers that INTEGRATE or REVIEW multiple sub-topics
- bridge_concepts are shared anchoring terms (disease name, drug class, etc.)
- ALWAYS populate enrichment_keywords - even for simple topics, provide 2-3
  synonyms or related terms useful for search diversification
- Sub-claims should preserve enough context to be meaningful alone
  (include the disease/system name, don't just say "focused ultrasound")
"""


def decompose_topic(
    topic: str,
    domain: str,
    settings: Settings,
    job_stats: JobStats | None = None,
) -> dict | None:
    """Analyse *topic* for independent research axes.

    Returns the decomposition dict - whose ``is_composite`` flag the caller
    must check - or None if the analysis failed or produced too few axes to
    be worth fanning out. Even a non-composite result is worth returning: it
    carries ``enrichment_keywords`` that diversify a simple search.
    """
    prompt = (
        f"TOPIC: {topic}\n"
        f"DOMAIN: {domain}\n\n"
        "Analyse this topic and determine if it needs decomposition into "
        "independent research axes. Return the JSON object."
    )

    try:
        raw = generate_json(
            settings=settings,
            prompt=prompt,
            system_instruction=_DECOMPOSE_SYSTEM,
            temperature=0.05,
            purpose="related_articles_decompose",
            job_stats=job_stats,
        )
    except Exception as exc:
        logger.warning("Topic decomposition failed: %s - treating as simple", exc)
        return None

    if not isinstance(raw, dict):
        return None
    if not raw.get("is_composite", False):
        return raw

    sub_claims = raw.get("sub_claims", [])
    if not isinstance(sub_claims, list) or len(sub_claims) < 2:
        return None

    raw["sub_claims"] = sub_claims[:_MAX_FACETS]
    return raw


def is_composite(decomposition: dict | None) -> bool:
    return (
        decomposition is not None
        and bool(decomposition.get("is_composite"))
        and len(decomposition.get("sub_claims", [])) >= 2
    )
