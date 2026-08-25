"""BM25 lexical scoring over the candidate pool.

A bi-encoder compresses a paper into a single vector, so a rare but decisive
term - a drug's INN, a trial acronym, a gene symbol - contributes about as
much to the similarity as any other word in the abstract. That is exactly the
term a clinical question turns on.

BM25 is blind in the opposite direction: it cannot read meaning at all, but it
weights a term by how rare it is in the pool, so a candidate that actually
contains ``semaglutide`` outranks one that is merely *about diabetes drugs*.
Fusing the two is the standard hybrid-retrieval result, and it holds even when
the dense model is strong.

The corpus is the candidate pool itself, not a global index. IDF computed over
~40-180 already-retrieved papers is what makes the signal discriminative here:
every candidate mentions the topic's common words, so those get an IDF near
zero and the score is decided by the terms that only some of them carry.

Pure Python, no index, no dependency: the pool is small and the whole scoring
pass costs less than a millisecond.

Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and
Beyond* (2009) for the scoring function; Lv & Zhai, *Lower-Bounding Term
Frequency Normalization* (CIKM 2011) for the non-negative IDF. That the
lexical leg still earns its place next to a strong encoder is the current
question rather than a settled one - see arXiv:2605.10848 and
arXiv:2606.04194 - which is why this is a separate weighted signal that can
be turned down rather than a step every candidate must pass.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

from app.pipeline.dedup import dedup_key
from app.sources.base import SourceResult

_K1 = 1.2    # term-frequency saturation; the standard BM25 default
_B = 0.75    # length normalisation; the standard BM25 default

# Chars of each candidate that form its document text. Title plus abstract:
# the same window the embedding sees, so the two signals disagree about
# meaning rather than about how much of the paper they read.
_DOC_CHARS = 1200

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]{1,}")

# Words that appear in almost every biomedical abstract. IDF would damp them
# anyway; dropping them keeps short queries from being dominated by structure
# words when the pool happens to be small.
_STOPWORD_TEXT = """
a an and are as at be been but by for from has have how in into is it its of on
or that the their there these this to was were what when which who why will with
study studies patient patients effect effects result results conclusion
conclusions background methods objective purpose using used use based
"""
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def tokenize(text: str) -> list[str]:
    """Lowercase, split on word boundaries, drop stopwords and bare numbers.

    Hyphens are kept inside tokens: ``non-small-cell`` and ``anti-pd-1`` are
    single terms in this literature, and splitting them turns a precise match
    into three vague ones.
    """
    return [
        token for token in _TOKEN_RE.findall((text or "").lower())
        if token not in _STOPWORDS and not token.isdigit()
    ]


def _document_text(result: SourceResult) -> str:
    body = result.content or result.snippet or ""
    return f"{result.title} {result.title} {body}"[:_DOC_CHARS]


def bm25_scores(
    query_terms: Sequence[str],
    results: Sequence[SourceResult],
) -> dict[str, float]:
    """Score every candidate against *query_terms*, min-max normalised to [0, 1].

    Returns ``{dedup_key: score}``. An empty query or pool returns ``{}`` so
    the caller's weight contributes nothing, matching how every other signal
    here fails.

    Scores are normalised because BM25 is unbounded and its scale depends on
    the pool: the ranker mixes it with signals already in [0, 1], and an
    unnormalised term would silently dominate or vanish depending on how many
    candidates came back.
    """
    terms = [t for t in (tokenize(" ".join(query_terms)) if query_terms else []) if t]
    if not terms or not results:
        return {}

    docs: list[tuple[str, Counter[str], int]] = []
    for result in results:
        tokens = tokenize(_document_text(result))
        docs.append((dedup_key(result), Counter(tokens), len(tokens)))

    n_docs = len(docs)
    avg_len = sum(length for _, _, length in docs) / n_docs or 1.0

    # Document frequency, restricted to the query's terms: nothing else can
    # affect the score, and the pool is scanned once either way.
    doc_freq: Counter[str] = Counter()
    unique_terms = set(terms)
    for _, counts, _ in docs:
        for term in unique_terms:
            if counts[term]:
                doc_freq[term] += 1

    # BM25+ style IDF floor. Classic BM25 IDF goes negative for a term present
    # in more than half the pool, which would let a *matching* candidate score
    # below a non-matching one - nonsense here, where the pool is a single
    # topic's results and common terms are expected.
    idf = {
        term: max(0.0, math.log(1.0 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5)))
        for term in unique_terms
    }

    raw: dict[str, float] = {}
    for key, counts, length in docs:
        score = 0.0
        norm = _K1 * (1.0 - _B + _B * length / avg_len)
        for term in terms:
            freq = counts[term]
            if freq:
                score += idf[term] * (freq * (_K1 + 1.0)) / (freq + norm)
        raw[key] = score

    top = max(raw.values(), default=0.0)
    if top <= 0.0:
        return dict.fromkeys(raw, 0.0)
    return {key: score / top for key, score in raw.items()}


def query_terms_for(topic: str, *extra: Iterable[str] | None) -> list[str]:
    """Assemble the BM25 query: the topic plus any extra vocabulary available.

    The extra terms are whatever the run already knows - PICO elements, the
    brief's must-include concepts, a facet's keywords. They matter more than
    the topic string, because they are the field's real terminology rather
    than the user's phrasing, so they are repeated once to weight them up
    without needing a separate per-term weight vector.
    """
    parts = [topic or ""]
    for group in extra:
        for term in group or []:
            if isinstance(term, str) and term.strip():
                parts.extend([term, term])
    return parts
