"""The discovery pipeline, end to end.

``run_related_articles`` takes a topic and returns ranked, summarised evidence.
It runs synchronously inside the HTTP request: no job queue, no task id, no
polling. A search takes tens of seconds and the client waits for it.

The shape of the run:

1. **Analyse** the topic - domain, query variants, vocabulary.
2. **Discover** candidates through the multi-round agentic loop: a research
   brief, several rounds of reformulated queries, and relevance-seeded
   citation expansion. Composite topics - ones spanning several distinct
   research axes - get their per-axis queries folded into the same run.
3. **Score** candidates on retrieval signals - semantic similarity, lexical
   overlap, rank fusion, study design - and keep a shortlist. This is the cost
   gate: summarisation is the expensive step, so only the shortlist gets one.
4. **Read** the shortlist - fetch full text, then summarise and appraise each
   source, and separately judge whether each source's own finding supports
   or contradicts the topic.
5. **Select** the final set under the ranking policy.
6. **Synthesise** an overview from the sources that cleared the relevance bar,
   as claims that each cite the articles they rest on.

Every stage records its own timing and cost in :class:`JobStats`, returned to
the caller as ``job_stats``, and reports its progress through an optional
:class:`~app.core.events.Emitter` (see that module) that
``POST /api/related-articles/stream`` uses to stream real stage updates over
SSE while this same blocking call runs in a worker thread.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from app.config import Settings, get_settings
from app.core.events import NULL_EMITTER, Emitter
from app.core.job_stats import JobStats
from app.pipeline import evidence_grade, lexical, ranking, selection, topic_analysis
from app.pipeline.dedup import dedup_key, deduplicate
from app.pipeline.ranking import citation_score, tier_score
from app.pipeline.relevance import (
    concept_and_offtopic_maps,
    embed_topic_and_candidates,
    llm_rerank_map,
)
from app.pipeline.selection import SignalMaps
from app.pipeline.synthesis import (
    appraise_design,
    clamp_relevance,
    judge_directions,
    read_direction,
    read_directness,
    summarise_sources,
    synthesise,
)
from app.schemas import (
    PICO,
    ArticleSummary,
    DomainContext,
    RelatedArticlesResponse,
    TopicSpec,
)
from app.sources.base import SourceResult
from app.sources.content_extractor import enrich_with_full_content

logger = logging.getLogger(__name__)


_MAX_CONTENT_FETCHES = 8       # full-text fetches per run
_SHORTLIST_BUFFER = 6          # extra candidates summarised beyond max_sources
_RERANK_CANDIDATE_CAP = 60     # wider rerank pool for the discovery loop's larger candidate set
_EMBED_SAFETY_CAP = 180        # bound embedding cost on very large pools
_RELEVANCE_FLOOR = 0.45        # drop articles below this: fewer, but on-topic


@dataclass
class _Discovery:
    """The candidate pool plus whatever the discovery loop learned about it."""

    evidence: list[SourceResult]
    # The research brief's vocabulary signals.
    must_include: list[str] = field(default_factory=list)
    negative_terms: list[str] = field(default_factory=list)


@contextmanager
def _stage(stats: JobStats, name: str) -> Iterator[None]:
    started = time.monotonic()
    try:
        yield
    finally:
        stats.record_stage(name, (time.monotonic() - started) * 1000)


def _discover(
    topic: str,
    spec: TopicSpec,
    context: DomainContext,
    settings: Settings,
    stats: JobStats,
    emitter: Emitter = NULL_EMITTER,
) -> _Discovery:
    from app.pipeline.agentic import run_agentic_discovery

    evidence, brief = run_agentic_discovery(
        topic=topic, spec=spec, context=context, settings=settings, stats=stats,
        emitter=emitter,
    )

    def _terms(key: str) -> list[str]:
        return [t for t in (brief.get(key) or []) if isinstance(t, str) and t.strip()]

    return _Discovery(
        evidence=evidence,
        must_include=_terms("must_include_concepts"),
        negative_terms=_terms("negative_terms"),
    )


def _authority_order(result: SourceResult) -> float:
    """Cheap pre-embedding sort key, used only to cap an oversized pool.

    Study design counts alongside citations and provider tier, because this
    cut happens before anything reads the candidates: on citations alone a
    well-cited case report survives a cut that discards a recent
    meta-analysis, and no later stage gets the chance to notice.
    """
    design, is_preprint = evidence_grade.detect_design(result)
    return (
        citation_score(result.citation_count)
        + tier_score(result.reliability_tier)
        + evidence_grade.evidence_score(design, is_preprint)
    )


def _vocabulary(spec: TopicSpec, discovery: _Discovery) -> list[str]:
    """The run's technical vocabulary, for lexical scoring.

    Whatever the run has already worked out about the topic in the field's own
    words: the PICO elements, the research brief's must-include concepts.
    These discriminate between candidates far better than the user's phrasing
    does, because they are the terms only the on-topic papers will contain.
    """
    terms: list[str] = []
    if spec.pico:
        terms.extend(spec.pico.terms())
    terms.extend(discovery.must_include)
    return terms


def _build_signals(
    topic: str,
    spec: TopicSpec,
    discovery: _Discovery,
    settings: Settings,
    stats: JobStats,
) -> SignalMaps:
    """Score the candidate pool, trimming it to what is affordable to rank.

    On a frontier topic the on-topic papers are recent and barely cited, so
    the pool is embedded first and cut on semantic similarity rather than
    authority - an authority cut would discard them before anything looked
    at their content.
    """
    evidence = discovery.evidence

    if len(evidence) > _EMBED_SAFETY_CAP:
        evidence.sort(key=_authority_order, reverse=True)
        del evidence[_EMBED_SAFETY_CAP:]

    topic_vector, vectors = embed_topic_and_candidates(
        topic, evidence, settings, job_stats=stats,
        model_override=settings.embedding_model_rerank,
    )
    cosine_raw = {
        key: ranking.cosine_similarity(topic_vector, vector)
        for key, vector in vectors.items()
    }
    evidence.sort(key=lambda r: cosine_raw.get(dedup_key(r), 0.0), reverse=True)
    del evidence[_RERANK_CANDIDATE_CAP:]

    # Min-max over the surviving pool, same scope and reason as bm25/rrf below:
    # raw cosine for this embedding model sits in a narrow, pool-dependent band
    # (e.g. 0.7-0.85 for a thematically tight pool), so its nominal weight in
    # PRIOR_WEIGHTS/FINAL_WEIGHTS bought far less spread than bm25's, which is
    # always stretched to fill [0, 1]. The sort/cut above already happened on
    # the raw values - normalising is order-preserving, so the cut is unaffected.
    cosine = ranking.normalize_minmax(
        {dedup_key(r): cosine_raw.get(dedup_key(r), 0.0) for r in evidence}
    )

    concept, offtopic = concept_and_offtopic_maps(
        evidence, discovery.must_include, discovery.negative_terms,
    )
    # Not its own `_stage`: this whole function already runs inside the
    # caller's "rerank" stage, and a nested timer here would double-count
    # this call's wall time into stage_timings_ms (once under "rerank",
    # once under its own name) without adding information - the LLM
    # call's own latency is already recorded per-call under
    # job_stats.llm.by_purpose["related_articles_rerank"].
    rerank = llm_rerank_map(topic, evidence, settings, job_stats=stats)

    signals = SignalMaps(
        cosine=cosine,
        concept=concept,
        offtopic=offtopic,
        rerank=rerank,
        vectors=vectors,
    )

    signals.rrf = ranking.normalize_minmax(
        ranking.reciprocal_rank_fusion(selection.provider_ranked_lists(evidence))
    )

    # Both of these are computed last, over the pool as it will actually be
    # ranked: BM25's discriminative power comes from IDF over the surviving
    # candidates, and grading a candidate that has already been cut is work
    # nobody reads. Neither costs an API call.
    signals.bm25 = lexical.bm25_scores(
        lexical.query_terms_for(topic, _vocabulary(spec, discovery)), evidence,
    )
    signals.evidence, signals.design_labels = evidence_grade.grade_results(
        evidence, settings, job_stats=stats,
    )

    return signals


def _summarise_shortlist(
    topic: str,
    shortlist: list[SourceResult],
    signals: SignalMaps,
    domain: str,
    settings: Settings,
    stats: JobStats,
    summary_language: str,
    pico: PICO | None,
) -> list[ArticleSummary]:
    with _stage(stats, "content_enrichment"):
        enrich_with_full_content(
            shortlist, settings=settings, max_fetches=_MAX_CONTENT_FETCHES,
        )

    with _stage(stats, "summarization"):
        summaries = summarise_sources(
            topic, shortlist, settings,
            summary_language=summary_language, job_stats=stats, pico=pico,
        )

    with _stage(stats, "stance"):
        directions = judge_directions(
            topic, shortlist, summaries, settings, job_stats=stats,
        )

    articles = []
    for index, result in enumerate(shortlist):
        summary = summaries.get(index, {})
        key = dedup_key(result)

        # The summariser has read the full text and the provider's own
        # publication types, so its reading of the design supersedes the
        # pool-wide detection; where it declines to name one, that stands.
        design, level = appraise_design(
            summary,
            signals.design_labels.get(key, ""),
            signals.evidence.get(key, 0.0),
        )

        articles.append(ArticleSummary(
            url=result.url,
            title=result.title,
            source_type=result.source_type,
            snippet=result.snippet[:500] if result.snippet else "",
            full_summary=summary.get("summary", ""),
            relevance_score=clamp_relevance(summary.get("relevance_score", 0.0)),
            reliability_tier=result.reliability_tier,
            publication_date=result.publication_date or None,
            citation_count=result.citation_count,
            domain=domain,
            study_design=design,
            evidence_level=level,
            key_finding=_text_field(summary, "key_finding"),
            finding_direction=read_direction(directions.get(index)),
            population=_text_field(summary, "population"),
            directness=read_directness(summary.get("directness")),
        ))
    return articles


def _text_field(summary: dict, key: str, limit: int = 400) -> str:
    value = summary.get(key)
    return value.strip()[:limit] if isinstance(value, str) else ""


def _as_results(articles: list[ArticleSummary]) -> list[SourceResult]:
    """View the returned articles as source records, for evidence profiling.

    The profile answers "what kind of evidence is this answer made of", so it
    has to be computed over the articles that survived selection rather than
    the pool they came from. Each carries a design already appraised against
    its full text, which is passed through as a publication type so the
    profile reads that rather than re-deriving it from the summary prose.
    """
    return [
        SourceResult(
            title=a.title,
            url=a.url,
            snippet=a.snippet,
            source_type=a.source_type,
            reliability_tier=a.reliability_tier,
            publication_date=a.publication_date or "",
            citation_count=a.citation_count,
            publication_types=[a.study_design] if a.study_design else [],
        )
        for a in articles
    ]


def run_related_articles(
    *,
    topic: str,
    domain_hint: str | None = None,
    max_sources: int = 10,
    summary_language: str = "it",
    emitter: Emitter = NULL_EMITTER,
) -> RelatedArticlesResponse:
    """Run the discovery pipeline for *topic* and return the full result.

    Never raises: a failure anywhere returns a response with ``status ==
    "failed"`` and the error message, so the caller always gets the timing and
    cost accounting for the work that was done.

    *emitter* receives a `progress` event as each of the six stages starts or
    finishes, with a human-readable message carrying real numbers from that
    stage (candidates found, sources shortlisted, and so on) rather than a
    fixed caption - see `app/core/events.py` for who reads these. Defaults to
    a no-op, so the plain (non-streaming) call path is unaffected.
    """
    settings = get_settings()
    stats = JobStats()
    started = time.monotonic()

    def _elapsed() -> float:
        return round(time.monotonic() - started, 2)

    try:
        # 1. Analyse the topic.
        emitter.emit("domain", "Analysing the topic…")
        with _stage(stats, "domain_detection"):
            context, spec = topic_analysis.analyse_topic(
                topic, settings, domain_hint, job_stats=stats,
            )
        domain = context.domain
        emitter.emit("domain", f"Domain detected: {domain}")

        # 2. Discover candidates.
        discovery = _discover(topic, spec, context, settings, stats, emitter=emitter)

        discovery.evidence = deduplicate(discovery.evidence)
        total_consulted = len(discovery.evidence)
        emitter.emit(
            "discovery",
            f"Discovery complete: {total_consulted} unique candidates after dedup",
        )
        if not discovery.evidence:
            return RelatedArticlesResponse(
                status="completed",
                topic=topic,
                domain_detected=domain,
                duration_seconds=_elapsed(),
                job_stats=stats.to_dict(),
            )

        # 3. Score candidates and shortlist what is worth summarising.
        emitter.emit("rerank", f"Scoring and reranking {total_consulted} candidates…")
        with _stage(stats, "rerank"):
            signals = _build_signals(topic, spec, discovery, settings, stats)
            # A wider shortlist so the relevance floor still leaves enough
            # on-topic articles to reach max_sources.
            shortlist = selection.shortlist(
                discovery.evidence, signals, 2 * max_sources + _SHORTLIST_BUFFER,
            )
        emitter.emit("rerank", f"Shortlisted {len(shortlist)} sources to read in full")

        # 4. Read the shortlist.
        emitter.emit("read", f"Fetching and summarising {len(shortlist)} sources…")
        articles = _summarise_shortlist(
            topic, shortlist, signals, domain,
            settings, stats, summary_language, spec.pico,
        )
        emitter.emit("read", f"Summarised {len(articles)} sources")

        # 5. Select the final set.
        emitter.emit("select", "Selecting the final set…")
        articles = selection.select_articles(
            articles,
            signals,
            medical_domain=selection.is_medical_domain(domain),
            max_sources=max_sources,
            relevance_floor=_RELEVANCE_FLOOR,
        )
        emitter.emit("select", f"{len(articles)} articles selected")

        # 6. Synthesise the overview, grounded in the articles being returned.
        emitter.emit("synthesis", "Writing the synthesis…")
        with _stage(stats, "global_synthesis"):
            global_summary, key_findings, disagreements, evidence_gaps = synthesise(
                topic, articles, settings,
                summary_language=summary_language,
                job_stats=stats,
            )
        emitter.emit(
            "synthesis",
            f"Synthesis complete: {len(key_findings)} findings, {len(disagreements)} conflicts",
        )

        logger.info(
            "done: %d articles in %.1fs (domain=%s, %d claims, %d conflicts)",
            len(articles), time.monotonic() - started, domain,
            len(key_findings), len(disagreements),
        )

        return RelatedArticlesResponse(
            status="completed",
            topic=topic,
            domain_detected=domain,
            articles=articles,
            global_summary=global_summary,
            total_sources_consulted=total_consulted,
            duration_seconds=_elapsed(),
            job_stats=stats.to_dict(),
            key_findings=key_findings,
            disagreements=disagreements,
            evidence_gaps=evidence_gaps,
            evidence_profile=evidence_grade.evidence_profile(
                _as_results(articles),
            ),
            pico=spec.pico,
        )

    except Exception as exc:
        logger.exception("Discovery pipeline failed")
        return RelatedArticlesResponse(
            status="failed",
            topic=topic,
            duration_seconds=_elapsed(),
            job_stats=stats.to_dict(),
            error=str(exc),
        )
