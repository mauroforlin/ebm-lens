"""Ranking and final selection: which articles the user actually gets back.

Two rankings happen, for different reasons.

The **prior** ranks the raw candidate pool on retrieval signals alone, before
any summarisation. Its job is purely economic: summarising is the expensive
step, so only a shortlist earns one.

The **final** ranking runs after summarisation, when the LLM's own relevance
judgement is available, and applies the policies that decide the answer:

* a **Wikipedia cap** on medical topics - Wikipedia is a fine starting point
  and poor clinical evidence, and left uncapped it wins on semantic
  similarity because it is written in the topic's own vocabulary;
* a **relevance floor** - returning six on-topic papers beats padding to ten
  with four that merely share words;
* **MMR diversification** - without it the top slots fill with near-duplicate
  papers on the pool's single strongest sub-aspect.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.pipeline import ranking
from app.pipeline.dedup import dedup_key, identity_key
from app.schemas import ArticleSummary
from app.sources.base import SourceResult

# Wikipedia's semantic-similarity advantage on medical topics is real but
# misleading, so it is penalised in scoring and capped in selection.
_MEDICAL_WIKIPEDIA_PENALTY = 0.25
_MEDICAL_MAX_WIKIPEDIA_ARTICLES = 1

# MMR trade-off (Carbonell & Goldstein, SIGIR 1998): how much relevance to give
# up per unit of redundancy avoided. Higher keeps the ranking closer to pure
# relevance.
_MMR_LAMBDA = 0.72

_MEDICAL_DOMAINS = frozenset({"medicine", "veterinary_medicine"})


@dataclass
class SignalMaps:
    """Every per-candidate signal available for a run, keyed by identity.

    Signals are optional because they are best-effort: an embedding or rerank
    call that fails leaves its map empty, and the corresponding weight simply
    contributes nothing rather than breaking the ranking.
    """

    cosine: dict[str, float] = field(default_factory=dict)
    bm25: dict[str, float] = field(default_factory=dict)
    rrf: dict[str, float] = field(default_factory=dict)
    # Position of each candidate's study design on the evidence hierarchy.
    evidence: dict[str, float] = field(default_factory=dict)
    # Human-readable design name per candidate, carried through to the client.
    design_labels: dict[str, str] = field(default_factory=dict)
    concept: dict[str, float] = field(default_factory=dict)
    offtopic: dict[str, float] = field(default_factory=dict)
    rerank: dict[str, float] = field(default_factory=dict)
    # Candidate embeddings, reused for MMR diversification.
    vectors: dict[str, list[float]] = field(default_factory=dict)

    def _signals_for(self, key: str) -> dict[str, float]:
        return {
            "cosine": self.cosine.get(key, 0.0),
            "bm25": self.bm25.get(key, 0.0),
            "rrf": self.rrf.get(key, 0.0),
            "evidence": self.evidence.get(key, 0.0),
            "concept": self.concept.get(key, 0.0),
            "offtopic": self.offtopic.get(key, 0.0),
            "rerank": self.rerank.get(key, 0.0),
        }


def is_medical_domain(domain: str | None) -> bool:
    return (domain or "").strip().lower() in _MEDICAL_DOMAINS


def provider_ranked_lists(results: list[SourceResult]) -> list[list[str]]:
    """Group candidate keys by provider, preserving each provider's own order.

    Reciprocal rank fusion needs one ranked list per retrieval system; a paper
    several providers independently ranked highly is a stronger result than
    one a single provider loved.
    """
    lists: dict[str, list[str]] = {}
    for result in results:
        lists.setdefault(result.source_type, []).append(dedup_key(result))
    return list(lists.values())


# ══════════════════════════════════════════════════════════════
#  Prior ranking (pre-summarisation)
# ══════════════════════════════════════════════════════════════


def candidate_prior(result: SourceResult, signals: SignalMaps) -> float:
    """Retrieval-only score, used to pick which candidates get summarised."""
    key = dedup_key(result)
    built = ranking.build_signals(
        result,
        cosine=signals.cosine.get(key, 0.0),
        bm25=signals.bm25.get(key, 0.0),
        rrf=signals.rrf.get(key, 0.0),
        evidence=signals.evidence.get(key, 0.0),
        concept=signals.concept.get(key, 0.0),
        offtopic=signals.offtopic.get(key, 0.0),
        rerank=signals.rerank.get(key, 0.0),
    )
    return ranking.weighted_score(built, ranking.PRIOR_WEIGHTS)


def shortlist(
    results: list[SourceResult],
    signals: SignalMaps,
    limit: int,
) -> list[SourceResult]:
    """Return the *limit* best candidates by prior score, best first."""
    ordered = sorted(results, key=lambda r: candidate_prior(r, signals), reverse=True)
    return ordered[:limit]


# ══════════════════════════════════════════════════════════════
#  Final ranking (post-summarisation)
# ══════════════════════════════════════════════════════════════


def final_score(
    article: ArticleSummary,
    signals: SignalMaps,
    *,
    medical_domain: bool,
) -> float:
    """Rank a summarised article on relevance, semantics and authority."""
    key = identity_key(article.url, article.title)
    built = signals._signals_for(key)
    built.update({
        "relevance": article.relevance_score,
        "citation": ranking.citation_score(article.citation_count),
        "recency": ranking.recency_score(article.publication_date or ""),
        "tier": ranking.tier_score(article.reliability_tier),
    })
    # The article's own evidence level wins over the candidate-pool map: by
    # this point the summariser has read the full text and may have corrected
    # a design the title-level heuristic guessed wrong.
    if article.evidence_level > 0.0:
        built["evidence"] = article.evidence_level
    score = ranking.weighted_score(built, ranking.FINAL_WEIGHTS)

    if medical_domain and article.source_type.strip().lower() == "wikipedia":
        score -= _MEDICAL_WIKIPEDIA_PENALTY
    return score


def _apply_wikipedia_cap(
    ordered: list[ArticleSummary],
    *,
    medical_domain: bool,
    max_sources: int,
) -> list[ArticleSummary]:
    """Take the top *max_sources*, admitting only one Wikipedia article first.

    Surplus Wikipedia articles are not dropped, only deferred: if the pool is
    thin they still fill the remaining slots, which beats returning fewer
    results than asked for.
    """
    if not medical_domain:
        return ordered[:max_sources]

    selected: list[ArticleSummary] = []
    deferred: list[ArticleSummary] = []
    wikipedia_used = 0

    for article in ordered:
        if len(selected) >= max_sources:
            break
        if article.source_type.strip().lower() == "wikipedia":
            if wikipedia_used < _MEDICAL_MAX_WIKIPEDIA_ARTICLES:
                selected.append(article)
                wikipedia_used += 1
            else:
                deferred.append(article)
            continue
        selected.append(article)

    for article in deferred:
        if len(selected) >= max_sources:
            break
        selected.append(article)

    return selected


def _mmr_order(
    articles: list[ArticleSummary],
    signals: SignalMaps,
    scores: dict[int, float],
) -> list[ArticleSummary]:
    """Greedy Maximal Marginal Relevance ordering over *articles*.

    Each pick trades its own relevance against its similarity to what is
    already selected, so the result set spans the topic's sub-aspects instead
    of stacking near-duplicates of its single strongest one.
    """
    ordered: list[ArticleSummary] = []
    remaining = list(articles)

    while remaining:
        if not ordered:
            best = max(remaining, key=lambda a: scores[id(a)])
        else:
            best = max(remaining, key=lambda a: _mmr_score(a, ordered, signals, scores))
        ordered.append(best)
        remaining.remove(best)

    return ordered


def _mmr_score(
    article: ArticleSummary,
    selected: list[ArticleSummary],
    signals: SignalMaps,
    scores: dict[int, float],
) -> float:
    vector = signals.vectors.get(identity_key(article.url, article.title))
    if not vector:
        return scores[id(article)]  # no vector: judge on relevance alone

    max_similarity = 0.0
    for chosen in selected:
        other = signals.vectors.get(identity_key(chosen.url, chosen.title))
        if other:
            max_similarity = max(max_similarity, ranking.cosine_similarity(vector, other))

    return _MMR_LAMBDA * scores[id(article)] - (1.0 - _MMR_LAMBDA) * max_similarity


def select_articles(
    articles: list[ArticleSummary],
    signals: SignalMaps,
    *,
    medical_domain: bool,
    max_sources: int,
    relevance_floor: float = 0.0,
) -> list[ArticleSummary]:
    """Rank and select the articles to return.

    With a *relevance_floor* set, articles the summariser judged off-topic are
    dropped before ranking - so the answer can be shorter than *max_sources*.
    The single best article always survives the floor: "nothing found" is a
    worse answer than one marginal result the user can judge themselves.
    """
    if relevance_floor > 0.0:
        above_floor = [a for a in articles if a.relevance_score >= relevance_floor]
        articles = above_floor or sorted(
            articles, key=lambda a: a.relevance_score, reverse=True,
        )[:1]

    if not articles:
        return []

    scores = {
        id(a): final_score(a, signals, medical_domain=medical_domain)
        for a in articles
    }

    if signals.vectors:
        ordered = _mmr_order(articles, signals, scores)
    else:
        ordered = sorted(articles, key=lambda a: (-scores[id(a)], a.reliability_tier))

    return _apply_wikipedia_cap(
        ordered, medical_domain=medical_domain, max_sources=max_sources,
    )
