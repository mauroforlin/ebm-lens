"""Pydantic schemas for the evidence-discovery API.

Two families of models live here:

* **Internal** - :class:`DomainContext` and :class:`TopicSpec` describe *what
  to search for*. They are built once from the user's topic by the domain
  detector and then handed to every provider-facing component.
* **Wire** - the request/response models for ``POST /related-articles``.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DomainContext(BaseModel):
    """Domain framing used to steer queries towards the right literature.

    Populated by the domain-detection LLM pass. Its keywords are appended to
    provider queries so that, say, "resistance" lands on antimicrobial
    resistance rather than electrical resistance.
    """

    domain: str = Field(
        "generic",
        description="High-level domain: medicine, veterinary_medicine, generic",
    )
    subdomain: str = Field("", description="More specific area, e.g. 'pharmacology', 'cardiology'")
    species: str = Field("", description="Target species if veterinary (e.g. 'canine', 'equine', 'human', '')")
    topic: str = Field("", description="Short free-text topic summary")
    keywords_en: list[str] = Field(default_factory=list, description="2-4 English keywords to append to search queries")
    keywords_it: list[str] = Field(default_factory=list, description="2-4 Italian keywords to append to search queries")


class PICO(BaseModel):
    """The clinical question behind a topic, in evidence-based medicine's own frame.

    PICO - Population, Intervention, Comparison, Outcome - is how a clinician
    is taught to turn a question into a search, and it is what biomedical
    databases are indexed to answer. A free-text topic hides its elements:
    "does semaglutide reduce heart attacks in obese patients without diabetes"
    is one string to a search engine and four separable constraints to a
    librarian. Naming them lets each one be searched, weighted and reported
    on separately - and lets the response say which element the evidence
    actually addressed, when it addressed only some.

    Every field is optional: most topics are not fully specified clinical
    questions, and an invented comparator is worse than an absent one.
    """

    population: str = Field("", description="Who or what the question is about")
    intervention: str = Field("", description="The exposure, drug, test or procedure")
    comparison: str = Field("", description="What it is compared against, if anything")
    outcome: str = Field("", description="The endpoint the question is about")
    study_designs: list[str] = Field(
        default_factory=list,
        description="Designs that would answer this question best (e.g. 'randomized controlled trial')",
    )

    def terms(self) -> list[str]:
        """The non-empty PICO elements, for query building and lexical scoring."""
        return [t for t in (self.population, self.intervention, self.comparison, self.outcome) if t.strip()]

    def is_specified(self) -> bool:
        """True when enough elements are present to be worth searching separately."""
        return bool(self.intervention.strip()) and bool(
            self.outcome.strip() or self.population.strip()
        )


class TopicSpec(BaseModel):
    """A single search target: the topic plus every query variant for it.

    One of these is built per search axis - the main topic, each facet of a
    composite topic, and each query variant the discovery loop issues.
    Providers never see the raw user string; they see the variant of this
    spec that suits their query language.
    """

    text: str = Field(..., description="The topic text this spec searches for")
    topic_type: str = Field(
        "assertion",
        description="assertion, definition, statistic, dosage, drug, "
                    "drug_classification, contraindication, drug_interaction, "
                    "adverse_effect, diagnostic_criteria, epidemiology, reference_value",
    )
    domain: str = Field(
        "generic",
        description="Topic domain: medicine, veterinary_medicine, generic",
    )
    subject: str = Field("", description="The main entity the topic is about")
    value: str | None = Field(None, description="Specific value involved (e.g. '500 mg')")

    search_query: str | None = Field(None, description="Optimised English search query")
    search_query_it: str | None = Field(None, description="Italian search query for IT providers")
    alternative_queries: list[str] = Field(default_factory=list, description="Backup English queries")
    alternative_queries_it: list[str] = Field(default_factory=list, description="Backup Italian queries")
    conceptual_query: str | None = Field(
        None,
        description="Deliberately broader English query, used only when the "
                    "precise queries return too little to work with",
    )
    conceptual_query_it: str | None = Field(None, description="Italian equivalent of conceptual_query")

    pico: PICO | None = Field(
        None,
        description="The topic's PICO decomposition, when it is a clinical question",
    )


class RelatedArticlesRequest(BaseModel):
    """Body for POST /related-articles."""

    topic: str = Field(..., min_length=3, max_length=2000, description="The topic to find related articles for")
    domain_hint: str | None = Field(
        None,
        description="Optional domain hint: medicine, veterinary_medicine, generic",
    )
    max_sources: int = Field(10, gt=0, le=30, description="Maximum number of articles to return")
    summary_language: str = Field("it", description="Language for the generated summaries: 'it' or 'en'. Search is always conducted in English.")


class Finding(BaseModel):
    """One distinct result a source itself reports that bears on the topic.

    Most sources yield exactly one; a source testing more than one outcome
    relevant to the topic (e.g. an efficacy result and a safety result) can
    yield up to a few, rather than forcing the appraiser to keep only its
    single best pick and silently drop the rest.
    """

    text: str = Field(..., description="The finding, in the requested summary language")
    evidence_quote: str = Field(
        "",
        description="Verbatim sentence(s) backing text, in the source's own "
                    "original language and wording. Empty when the quote could "
                    "not be verified against the source's own fetched content.",
    )
    finding_direction: str = Field(
        "",
        description="Graded direction of THIS finding against the topic's claim: "
                    "strongly_contradicts, contradicts, weakly_contradicts, "
                    "no_evidence, weakly_supports, supports, strongly_supports, "
                    "or mixed",
    )


class ArticleSummary(BaseModel):
    """A single related article with LLM-generated summary."""

    url: str = Field(..., description="Source URL")
    title: str = Field("", description="Article/page title")
    source_type: str = Field("", description="Provider id: pubmed, europe_pmc, wikipedia, etc.")
    snippet: str = Field("", description="Original snippet from the provider")
    full_summary: str = Field("", description="LLM-generated summary of the article content")
    relevance_score: float = Field(0.0, ge=0.0, le=1.0, description="LLM-assigned relevance to the topic (0-1)")
    reliability_tier: int = Field(3, ge=1, le=3, description="1=institutional, 2=curated, 3=general")
    publication_date: str | None = Field(None, description="Publication date if available")
    citation_count: int = Field(0, description="Times-cited count if available (academic authority)")
    domain: str = Field("generic", description="Domain of the source content")

    study_design: str = Field(
        "",
        description="Detected study design (meta-analysis, RCT, cohort, case report, ...)",
    )
    evidence_level: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Position of the study design on the evidence hierarchy (0-1)",
    )
    findings: list[Finding] = Field(
        default_factory=list,
        description="Distinct results this source reports bearing on the topic, "
                    "each with its own verbatim quote and stance. Empty when the "
                    "source reports nothing that bears on the topic.",
    )
    population: str = Field("", description="Who the source studied, if stated")
    directness: str = Field(
        "unclear",
        description="Whether the source's own population/intervention match the "
                    "topic's PICO (direct), study a related-but-different one "
                    "(adjacent), or the topic has no PICO specific enough to judge "
                    "(unclear). Distinct from relevance_score: a source can be "
                    "topically relevant while only offering extrapolated evidence.",
    )


class Claim(BaseModel):
    """One statement in the overview, with the sources that support it.

    In an unattributed block of prose a sentence the model invented looks
    exactly like a sentence three papers agreed on. Emitting the overview as
    claims, each carrying the indices of the articles it rests on, makes the
    check mechanical: indices are validated against the returned articles, and
    a claim citing nothing real is dropped before the user sees it.
    """

    text: str = Field(..., description="The claim, in the requested summary language")
    source_indices: list[int] = Field(
        default_factory=list,
        description="0-based indices into `articles` supporting this claim",
    )
    strength: str = Field(
        "moderate",
        description="How well the cited evidence supports the claim: strong, moderate, weak",
    )


class Disagreement(BaseModel):
    """A point the retrieved sources do not agree on.

    Roughly half of post-2010 biomedical papers conflict with something else
    in their own literature, and a synthesis that averages them reads as
    settled science. Conflicts are surfaced instead of resolved: the sources
    on each side are named so the user can go and look.
    """

    issue: str = Field(..., description="What the sources disagree about")
    positions: list[str] = Field(default_factory=list, description="The competing positions")
    source_indices: list[int] = Field(
        default_factory=list, description="Articles involved in the disagreement",
    )


class RelatedArticlesResponse(BaseModel):
    """Full synchronous result of a related-articles search."""

    status: str = "completed"
    topic: str = ""
    domain_detected: str = Field("generic", description="Auto-detected domain of the topic")
    articles: list[ArticleSummary] = Field(default_factory=list)
    global_summary: str = Field("", description="LLM-generated synthesis across all sources")
    total_sources_consulted: int = Field(0, description="Total evidence items found before filtering")
    duration_seconds: float = Field(0.0, description="Pipeline wall-clock time")
    job_stats: dict[str, Any] = Field(default_factory=dict, description="Cost/usage statistics for this request")
    error: str | None = None

    key_findings: list[Claim] = Field(
        default_factory=list,
        description="The overview as individual claims, each citing the articles supporting it",
    )
    disagreements: list[Disagreement] = Field(
        default_factory=list,
        description="Points the retrieved sources conflict on, with the sources on each side",
    )
    evidence_profile: dict[str, Any] = Field(
        default_factory=dict,
        description="Shape of the returned evidence: design mix, high-tier share, preprint share",
    )
    pico: PICO | None = Field(
        None, description="The clinical question the search was built from, when one was extracted",
    )
    evidence_gaps: list[str] = Field(
        default_factory=list,
        description="What the retrieved evidence does not cover, stated plainly",
    )
