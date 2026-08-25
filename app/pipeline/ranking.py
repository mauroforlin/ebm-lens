"""Ranking utilities for related-articles discovery.

Pure functions - no I/O, no DB.  They combine several relevance and
authority signals into a single score used to rank candidate sources:

* **cosine**    - semantic similarity between topic and source (embeddings)
* **bm25**      - lexical overlap, IDF-weighted over the candidate pool itself
* **rrf**       - reciprocal rank fusion across the provider result lists
* **evidence**  - position of the study design on the evidence hierarchy
* **citation**  - log-scaled times-cited count (academic authority)
* **recency**   - exponential decay on publication year
* **tier**      - provider reliability tier (1=gold, 2=strong, 3=general)
* **relevance** - LLM-assigned relevance (only available post-summarisation)

``cosine`` and ``bm25`` are deliberately kept as separate signals rather than
merged into one "content" score: they fail differently - the embedding misses
the one rare term the question turns on, the lexical score misses a paper that
never uses the user's words - and keeping both means a candidate has to be
wrong in both ways to be ranked down. ``evidence`` is the only signal that is
not about relevance at all: it asks how much a paper of that design is worth
believing, which is the question relevance ranking cannot answer.

Two weight profiles are exposed: ``PRIOR_WEIGHTS`` ranks candidates
*before* the expensive LLM pass (so we only summarise the shortlist), and
``FINAL_WEIGHTS`` ranks the survivors *after* the LLM relevance score is in.
Citation authority is heavily downweighted in both - polysemous high-citation
noise should not float to the top on authority alone; semantic relevance
dominates instead.
"""
from __future__ import annotations

import math
import re
from collections.abc import Sequence
from datetime import datetime, timezone

from app.sources.base import SourceResult

_RRF_K = 60                 # RRF constant from Cormack et al., SIGIR 2009
_CITATION_CAP = 1000        # citations at/above this saturate the signal to 1.0
_RECENCY_HALF_LIFE = 8.0    # years; a paper this old scores 0.5 on recency
_RECENCY_UNKNOWN = 0.5      # neutral score when no date is available
_TIER_SCORES = {1: 1.0, 2: 0.6, 3: 0.2}

# Weights used to shortlist candidates BEFORE the LLM summarisation pass.
#
# Three signals beyond the retrieval basics (cosine/bm25/rrf/evidence):
#   * concept  - fraction of the research brief's must_include_concepts present
#                in the candidate (rewards genuine technical relevance, not
#                surface terms)
#   * offtopic - intensity of negative_terms matches (an OFF-topic homonym
#                signal; carries a NEGATIVE weight so it suppresses the wrong sense)
#   * rerank   - an LLM cross-encoder-style relevance score over title+abstract
PRIOR_WEIGHTS: dict[str, float] = {
    "rerank": 0.35,   # LLM reads title+abstract → more precise on polysemy
    "cosine": 0.20,   # semantic support signal
    "bm25": 0.12,
    "rrf": 0.09,
    "concept": 0.09,
    "evidence": 0.08,
    "citation": 0.02,
    "recency": 0.02,
    "tier": 0.01,
    "offtopic": -0.30,
}

# Weights used to rank the shortlist AFTER the LLM relevance score is known.
FINAL_WEIGHTS: dict[str, float] = {
    "relevance": 0.36,
    "cosine": 0.19,
    "evidence": 0.12,
    "rerank": 0.11,
    "bm25": 0.08,
    "concept": 0.06,
    "recency": 0.04,
    "citation": 0.02,
    "tier": 0.02,
    "offtopic": -0.30,
}


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    k: int = _RRF_K,
) -> dict[str, float]:
    """Fuse several ranked lists of ids into one RRF score per id.

    Each input list is ordered best-first.  An id's RRF score is
    ``sum(1 / (k + rank))`` over every list it appears in (rank 0-based).
    Ids absent from a list contribute nothing for that list.
    """
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, key in enumerate(lst):
            if key:
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return scores


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors; 0.0 on empty, zero or mismatched input.

    A length mismatch means two different embedding models produced the
    vectors. Truncating to the shorter one would return a plausible-looking
    number from an invalid comparison, so it scores 0.0 instead.
    """
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def citation_score(count: int) -> float:
    if count <= 0:
        return 0.0
    return min(1.0, math.log1p(count) / math.log1p(_CITATION_CAP))


def _extract_year(date_str: str) -> int | None:
    match = re.search(r"(19|20)\d{2}", date_str or "")
    return int(match.group()) if match else None


def recency_score(date_str: str) -> float:
    year = _extract_year(date_str)
    if year is None:
        return _RECENCY_UNKNOWN
    age = max(0, datetime.now(timezone.utc).year - year)
    return 0.5 ** (age / _RECENCY_HALF_LIFE)


def tier_score(tier: int) -> float:
    return _TIER_SCORES.get(tier, 0.2)


def _term_in_text(term: str, text_lower: str) -> bool:
    """Word-boundary-aware containment test for a (possibly multi-word) term."""
    term = (term or "").strip().lower()
    if not term:
        return False
    # Multi-word phrases: simple substring is precise enough.
    if " " in term:
        return term in text_lower
    return re.search(rf"\b{re.escape(term)}\b", text_lower) is not None


def concept_match_score(text: str, concepts: Sequence[str]) -> float:
    """Fraction of *concepts* (must-include vocabulary) present in *text* (pro).

    Rewards candidates that actually use the field's core technical vocabulary,
    not just papers that happen to be semantically close on the surface.
    """
    if not concepts:
        return 0.0
    text_lower = (text or "").lower()
    if not text_lower:
        return 0.0
    hits = sum(1 for c in concepts if _term_in_text(c, text_lower))
    return hits / len(concepts)


def offtopic_score(text: str, negative_terms: Sequence[str]) -> float:
    """Off-topic-homonym intensity in [0, 1] from *negative_terms* (pro).

    Saturates quickly: a single strong negative-term hit already flags the
    wrong sense, so two matches reach the ceiling.
    """
    if not negative_terms:
        return 0.0
    text_lower = (text or "").lower()
    if not text_lower:
        return 0.0
    hits = sum(1 for t in negative_terms if _term_in_text(t, text_lower))
    return min(1.0, hits / 2.0)


def normalize_minmax(values: dict[str, float]) -> dict[str, float]:
    """Min-max normalise a dict of scores to [0, 1]; flat input maps to 0.0."""
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi <= lo:
        return dict.fromkeys(values, 0.0)
    span = hi - lo
    return {k: (v - lo) / span for k, v in values.items()}


def build_signals(
    result: SourceResult,
    *,
    cosine: float,
    rrf: float,
    bm25: float = 0.0,
    evidence: float = 0.0,
    relevance: float = 0.0,
    concept: float = 0.0,
    offtopic: float = 0.0,
    rerank: float = 0.0,
) -> dict[str, float]:
    """Assemble the normalised signal dict for one candidate.

    Every argument defaults to 0.0 for the same reason a failed embedding
    leaves cosine at 0.0: a signal that could not be computed (or wasn't
    asked for) contributes nothing rather than breaking the score.
    """
    return {
        "cosine": cosine,
        "bm25": bm25,
        "rrf": rrf,
        "evidence": evidence,
        "citation": citation_score(result.citation_count),
        "recency": recency_score(result.publication_date),
        "tier": tier_score(result.reliability_tier),
        "relevance": relevance,
        "concept": concept,
        "offtopic": offtopic,
        "rerank": rerank,
    }


def weighted_score(signals: dict[str, float], weights: dict[str, float]) -> float:
    return sum(weight * signals.get(key, 0.0) for key, weight in weights.items())
