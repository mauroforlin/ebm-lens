"""Relevance signals: turning a candidate pool into per-candidate scores.

Retrieval gets papers that share words with the topic. Deciding which of them
are *about* the topic is a separate problem, and one signal is not enough for
it - each of these is blind in a way the others are not:

* **cosine** (embeddings) reads meaning rather than keywords, but a bi-encoder
  compresses a paper into one vector and regularly rates a famous paper about
  a different sense of the same words as highly similar.
* **concept / off-topic** term matching is literal and cheap, and catches
  exactly that: the research brief names the vocabulary an on-topic paper must
  use, and the vocabulary that betrays the wrong sense of a polysemous topic.
* **LLM rerank** reads title and abstract together and judges subject matter
  directly. It is the most accurate and the most expensive, so it only runs
  over an already-bounded pool, in a few chunked calls rather than one.

Every function here is best-effort: a failed embedding or rerank call returns
an empty map, the corresponding weight contributes nothing, and ranking
continues on the signals that did arrive.

The reranker is the pointwise form of the LLM-as-reranker result (Sun et al.,
*Is ChatGPT Good at Search?*, EMNLP 2023). That paper ranks by asking for a
permutation of the list, which is stronger on close calls; scoring each
candidate independently is weaker there and degrades gracefully, since a
malformed or partial answer costs the scores it omitted rather than the whole
ordering - one chunk failing loses that chunk, not the whole rerank.
"""
from __future__ import annotations

import logging

from app.config import Settings
from app.core.embeddings import embed_texts
from app.core.job_stats import JobStats
from app.core.llm_client import generate_json
from app.pipeline import ranking
from app.pipeline.dedup import dedup_key
from app.sources.base import SourceResult

logger = logging.getLogger(__name__)

# Chars of each candidate fed to the embedding model. Title plus roughly an
# abstract: enough to characterise the paper, short enough to keep a 60-item
# pool inside one batch.
_EMBED_CHARS = 1000

# Candidates sent to the LLM reranker in total, across all chunked calls
# (token guard). Must be >= orchestrator._PRO_RERANK_CANDIDATE_CAP: a lower
# cap here would silently truncate the pool orchestrator.py already bounded,
# leaving the tail candidates with no rerank score (defaulting to 0.0 in
# ranking) even though they survived every earlier cut.
_RERANK_INPUT_CAP = 60
# Chars of each candidate shown to the reranker. Matches the source providers'
# own abstract cap (``content[:4000]`` - see e.g. europe_pmc.py, pubmed.py),
# so this is "the whole abstract" for the near-totality of records rather than
# a further truncation - this is the signal the docstring above calls the
# most accurate one, and it was previously seeing 200 chars against cosine's
# 1000 and BM25's 1200.
_RERANK_TEXT_CHARS = 4000


def _candidate_text(result: SourceResult) -> str:
    body = result.content or result.snippet or ""
    return f"{result.title}. {body}"[:_EMBED_CHARS]


def embed_topic_and_candidates(
    topic: str,
    results: list[SourceResult],
    settings: Settings,
    job_stats: JobStats | None = None,
    model_override: str | None = None,
) -> tuple[list[float], dict[str, list[float]]]:
    """Embed the topic and every candidate in one batch.

    Returns ``(topic_vector, {dedup_key: vector})``, or ``([], {})`` if the
    embedding call fails. Candidate vectors are returned rather than consumed
    so the caller can reuse them for MMR diversification without paying for a
    second embedding pass.
    """
    if not results:
        return [], {}

    texts = [topic] + [_candidate_text(r) for r in results]
    try:
        vectors = embed_texts(
            texts, settings, job_stats=job_stats, model_override=model_override,
        )
    except Exception as exc:
        logger.warning("Rerank embedding failed (cosine disabled): %s", exc)
        return [], {}

    if len(vectors) != len(texts):
        logger.warning(
            "Embedding returned %d vectors for %d texts - disabling cosine",
            len(vectors), len(texts),
        )
        return [], {}

    return vectors[0], {dedup_key(r): vectors[i + 1] for i, r in enumerate(results)}


def cosine_map(
    topic: str,
    results: list[SourceResult],
    settings: Settings,
    job_stats: JobStats | None = None,
    model_override: str | None = None,
) -> dict[str, float]:
    topic_vector, vectors = embed_topic_and_candidates(
        topic, results, settings, job_stats=job_stats, model_override=model_override,
    )
    if not vectors:
        return {}
    return {
        key: ranking.cosine_similarity(topic_vector, vector)
        for key, vector in vectors.items()
    }


def concept_and_offtopic_maps(
    results: list[SourceResult],
    must_include: list[str],
    negative_terms: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Score candidates against the brief's required and disqualifying vocabulary.

    *must_include* are the technical terms an on-topic paper in this field will
    actually use; *negative_terms* name the wrong sense of a polysemous topic.
    The off-topic map carries a negative weight in the ranking profile, so it
    suppresses rather than merely fails to reward.
    """
    concepts: dict[str, float] = {}
    offtopic: dict[str, float] = {}
    if not must_include and not negative_terms:
        return concepts, offtopic

    for result in results:
        text = f"{result.title}. {result.content or result.snippet or ''}"
        key = dedup_key(result)
        if must_include:
            concepts[key] = ranking.concept_match_score(text, must_include)
        if negative_terms:
            offtopic[key] = ranking.offtopic_score(text, negative_terms)

    return concepts, offtopic


_RERANK_SYSTEM = """\
You are a precise relevance grader for a medical/biomedical literature search.
Given a TOPIC and a numbered list of candidate papers (title + abstract),
score how directly each paper is ABOUT the topic - in the intended clinical
sense - on a 0.0-1.0 scale.

Return a JSON array: [{"index": <int>, "score": <float 0.0-1.0>}]

Scoring, against these fixed criteria - not relative to the other papers in
this batch, since the same paper may be scored in a different batch and the
number must mean the same thing both times:
- 1.0 = squarely on-topic, a core paper for this exact subject.
- 0.5 = related/adjacent but not centrally about the topic.
- 0.0 = wrong sense, homonym, or unrelated despite shared words.
Judge the SUBJECT MATTER, not citation count or prestige. Be strict about
polysemy: a highly-cited paper on a different meaning of the words scores low.
"""

# Candidates per LLM call. A full abstract (_RERANK_TEXT_CHARS) times a
# 60-wide batch puts on the order of 25-30k tokens of candidate text in front
# of the model in one shot; multi-item judgment inside a single long context
# is known to degrade for items that land in the middle of it (Liu et al.,
# "Lost in the Middle", TACL 2024), and the degradation grows with the
# context, so reading more per candidate makes a wide single-call batch worse
# off, not better. Splitting into narrower calls keeps each one inside the
# range grading stays reliable in, at the cost of a repeated system prompt.
_RERANK_CHUNK_SIZE = 15


def llm_rerank_map(
    topic: str,
    results: list[SourceResult],
    settings: Settings,
    job_stats: JobStats | None = None,
) -> dict[str, float]:
    """Score candidates by reading them, in ``_RERANK_CHUNK_SIZE``-wide calls.

    This is the highest-leverage precision lever in the pipeline: it catches
    the polysemous off-topic noise that survives cosine similarity, because it
    judges the paper rather than the distance between two vectors.

    Each chunk is an independent call, so a score is only comparable across
    chunks to the extent the rubric is absolute rather than batch-relative -
    the system prompt says so explicitly, but this is a real limit of
    chunking a pointwise scorer, not something splitting the calls fixes on
    its own. One chunk failing loses only that chunk's scores.
    """
    # FIXME: no anchoring between chunks - nothing verifies that the absolute
    # rubric actually scores the same paper the same way in two different
    # calls. Lifting this means replacing independent-chunk pointwise scoring
    # with sliding-window listwise reranking (RankGPT/RankVicuna-style:
    # overlapping windows, the model reorders each window instead of scoring
    # items in isolation, and top candidates propagate forward across
    # windows), trading the current parallelism for cross-item comparability.
    if not results:
        return {}

    subset = results[:_RERANK_INPUT_CAP]
    scores: dict[str, float] = {}
    for start in range(0, len(subset), _RERANK_CHUNK_SIZE):
        chunk = subset[start:start + _RERANK_CHUNK_SIZE]
        scores.update(_score_chunk(topic, chunk, settings, job_stats))
    return scores


def _score_chunk(
    topic: str,
    chunk: list[SourceResult],
    settings: Settings,
    job_stats: JobStats | None,
) -> dict[str, float]:
    lines = [
        # content is the abstract; snippet is provider metadata (authors,
        # journal, pub types) for most sources - content must come first or
        # the model never sees the abstract at all for those providers.
        f"[{i}] {r.title} - {(r.content or r.snippet or '')[:_RERANK_TEXT_CHARS]}"
        for i, r in enumerate(chunk)
    ]
    prompt = (
        f"TOPIC: {topic}\n\nCANDIDATES:\n" + "\n".join(lines)
        + "\n\nScore every candidate as JSON."
    )

    try:
        raw = generate_json(
            settings=settings,
            prompt=prompt,
            system_instruction=_RERANK_SYSTEM,
            temperature=0.0,
            purpose="related_articles_rerank",
            job_stats=job_stats,
        )
    except Exception as exc:
        logger.warning("LLM rerank failed for one chunk (non-fatal): %s", exc)
        return {}

    if not isinstance(raw, list):
        return {}

    scores: dict[str, float] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        if not isinstance(index, int) or not 0 <= index < len(chunk):
            continue
        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        scores[dedup_key(chunk[index])] = min(1.0, max(0.0, score))

    return scores
