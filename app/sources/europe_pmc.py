"""Europe PMC provider - free, no API key required.

Superset of PubMed with 47 M+ articles, 10 M+ full-text articles, and
6.5 M open-access papers.  Also indexes Cochrane reviews, Agricola,
patents, and NICE guidelines.

API docs: https://europepmc.org/RestfulWebService
Rate limit: EBI fair-use (~10 req/s recommended).
"""
from __future__ import annotations

import logging
import re
import threading
import time

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import SourceProvider, SourceResult, build_headers

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_TIMEOUT = 12
def _headers() -> dict[str, str]:
    """Outbound headers, built per call so settings stay lazily loaded."""
    return build_headers(accept="application/json,application/xml;q=0.9,*/*;q=0.8")

_GUIDELINE_PUB_TYPES = (
    'PUB_TYPE:"guideline" OR PUB_TYPE:"practice guideline" '
    'OR PUB_TYPE:"consensus development conference"'
)

# Rate limiter: ~6 req/s to respect EBI fair-use policy
_rate_lock = threading.Lock()
_last_request_time = 0.0

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "to",
    "for", "with", "on", "at", "from", "by", "about", "as", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "out", "off", "over", "under", "again", "further", "then", "once",
    "that", "this", "these", "those", "what", "which", "who", "whom",
    "when", "where", "why", "how", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "no", "not", "only",
    "same", "so", "than", "too", "very", "and", "but", "or", "if",
    "it", "its", "they", "their", "them", "he", "she", "his", "her",
})


def _rate_limit() -> None:
    global _last_request_time
    with _rate_lock:
        now = time.monotonic()
        if now - _last_request_time < 0.17:
            time.sleep(0.17 - (now - _last_request_time))
        _last_request_time = time.monotonic()


def _strip_html(text: str) -> str:
    """Strip markup from structured abstracts (e.g. ``<h4>Background</h4>``)."""
    clean = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", clean).strip()


def _simplify_query(query: str) -> str:
    """Strip stopwords and keep max 8 biomedical keywords."""
    clean = re.sub(r"[\"'()[\]{}<>]", " ", query)
    tokens = clean.split()
    keywords = [t for t in tokens if t.lower() not in _STOPWORDS and len(t) > 2]
    return " ".join(keywords[:8])


class EuropePMCProvider(SourceProvider):
    """Search Europe PMC for biomedical literature (free, no API key)."""

    source_type = "europe_pmc"

    def __init__(
        self,
        *,
        prefer_reviews: bool = False,
        recent_only: bool = False,
    ):
        self._prefer_reviews = prefer_reviews
        self._recent_only = recent_only

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
    )
    def search(self, query: str, max_results: int = 3) -> list[SourceResult]:
        if not query or len(query) < 3:
            return []

        simplified = _simplify_query(query)
        if not simplified:
            return []

        # Build advanced query parts
        adv_parts: list[str] = []

        # Restrict to MEDLINE/PubMed source for higher quality
        adv_parts.append("SRC:MED")

        if self._prefer_reviews:
            adv_parts.append('(PUB_TYPE:"review" OR PUB_TYPE:"systematic review" '
                             'OR PUB_TYPE:"meta-analysis")')

        if self._recent_only:
            import datetime as _dt
            min_year = _dt.date.today().year - 5
            adv_parts.append(f"FIRST_PDATE:[{min_year} TO *]")

        enhanced = (
            f"({simplified}) AND {' AND '.join(adv_parts)}" if adv_parts else simplified
        )

        results = self._search_relevance_and_citations(enhanced, max_results)

        # Fallback: drop filters if no results
        if not results:
            results = self._search_relevance_and_citations(simplified, max_results)

        # Fallback: broaden query if still no results
        if not results and len(simplified.split()) > 3:
            shorter = " ".join(simplified.split()[:4])
            results = self._search_relevance_and_citations(shorter, max_results)

        return results[:max_results]

    def _search_relevance_and_citations(
        self, query: str, max_results: int,
    ) -> list[SourceResult]:
        """Fill the result set by relevance first, citation authority second -
        or, on a review/foundational query, a guaranteed share of both.

        On a broad, high-volume query Europe PMC's default relevance sort
        skews toward the newest papers, so a landmark paper - or the paper the
        field actually cites for the mechanism it names - can rank outside the
        page we fetch even though it's indexed. A second query sorted by
        citation count recovers it.

        For most queries that recovery only fills whatever slots relevance
        left unfilled: a query specific enough that relevance already fills
        ``max_results`` has nothing to gain from a highly-cited paper that
        merely matches its keywords, and letting it displace a relevance
        match would cost more precision than the recovery is worth. A
        review/foundational query (``prefer_reviews``) is the exception: the
        paper most worth surfacing there is often an older landmark whose
        keyword-relevance rank undersells it, so a third of the slots go to
        citation authority regardless of whether relevance alone already
        filled the page - trimming its weakest matches to make room rather
        than never trying citation authority at all.
        """
        by_relevance = self._search_core(query, max_results, use_synonyms=True)

        merged: list[SourceResult] = []
        seen: set[str] = set()
        for r in by_relevance:
            key = r.url.rstrip("/").lower()
            if key and key not in seen:
                seen.add(key)
                merged.append(r)

        reserved = max(1, max_results // 3) if self._prefer_reviews else 0
        slots_open = max_results - len(merged)

        if slots_open > 0 or reserved > 0:
            by_citations = self._search_core(
                query, max_results, use_synonyms=True, sort="CITED desc",
            )

            trim = max(0, reserved - slots_open)
            if trim:
                merged = merged[: max(0, len(merged) - trim)]
                seen = {r.url.rstrip("/").lower() for r in merged}

            for r in by_citations:
                if len(merged) >= max_results:
                    break
                key = r.url.rstrip("/").lower()
                if key and key not in seen:
                    seen.add(key)
                    merged.append(r)

        return merged

    def search_guidelines(self, topic: str, max_results: int = 5) -> list[SourceResult]:
        """Search Europe PMC for clinical practice guidelines on *topic*.

        Reuses the shared request/parse path (``_search_core``) with a
        guideline publication-type filter.
        """
        topic = (topic or "").strip()
        if not topic:
            return []

        query = f"({topic}) AND ({_GUIDELINE_PUB_TYPES})"
        try:
            return self._search_core(query, max_results, use_synonyms=False)[:max_results]
        except Exception as exc:
            logger.debug("Europe PMC guideline search failed: %s", exc)
            return []

    def _search_core(
        self,
        query: str,
        max_results: int,
        *,
        use_synonyms: bool = False,
        sort: str | None = None,
    ) -> list[SourceResult]:
        _rate_limit()

        params = {
            "query": query,
            "format": "json",
            "resultType": "core",  # includes abstract, MeSH, metadata
            "pageSize": min(max_results * 2, 10),
        }
        if use_synonyms:
            params["synonym"] = "true"
        if sort:
            params["sort"] = sort

        try:
            resp = httpx.get(_SEARCH_URL, params=params, headers=_headers(), timeout=_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.debug("Europe PMC search failed: %s", exc)
            raise

        hits = data.get("resultList", {}).get("result", [])
        if not hits:
            return []

        results: list[SourceResult] = []
        for hit in hits:
            try:
                result = self._parse_hit(hit)
                if result:
                    results.append(result)
            except Exception as exc:
                logger.debug("Europe PMC parse error: %s", exc)

        return results

    def _parse_hit(self, hit: dict) -> SourceResult | None:
        title = hit.get("title", "").strip()
        if not title:
            return None

        pmid = hit.get("pmid", "")
        pmcid = hit.get("pmcid", "")
        doi = hit.get("doi", "")

        if pmcid:
            url = f"https://europepmc.org/article/PMC/{pmcid}"
        elif pmid:
            url = f"https://europepmc.org/article/MED/{pmid}"
        elif doi:
            url = f"https://doi.org/{doi}"
        else:
            return None

        abstract = _strip_html(hit.get("abstractText", "") or "")
        authors = hit.get("authorString", "") or ""
        journal = hit.get("journalTitle", "") or ""
        pub_date = hit.get("firstPublicationDate", "") or ""
        cited_by = hit.get("citedByCount", 0) or 0
        is_oa = hit.get("isOpenAccess", "N") == "Y"

        snippet_parts: list[str] = []
        if authors:
            snippet_parts.append(f"Authors: {authors[:150]}")
        if journal:
            snippet_parts.append(f"Journal: {journal}")
        if pub_date:
            snippet_parts.append(f"Published: {pub_date}")
        if cited_by:
            snippet_parts.append(f"Cited by: {cited_by}")
        if is_oa:
            snippet_parts.append("Open Access: Yes")
        snippet = " | ".join(snippet_parts)

        # Content = abstract. Full text, when it earns its cost, is fetched
        # later for the shortlist only (content_extractor.py) - not here for
        # every candidate the search happens to turn up.
        content = abstract

        return SourceResult(
            title=title,
            url=url,
            snippet=snippet[:300],
            content=content[:4000],
            source_type=self.source_type,
            reliability_tier=1,
            publication_date=pub_date,
            citation_count=int(cited_by or 0),
            language="en",
        )
