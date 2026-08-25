"""Citation-graph expansion - find *truly* related papers for free.

Keyword search alone misses papers that don't share surface terms with the
query.  Citation-graph neighbours fill that gap: given a seed paper, the
Semantic Scholar Recommendations API and OpenAlex ``related_works`` return
papers that are semantically/bibliographically close.

Both APIs are free and need no key, so this adds recall at **zero cost**.
All network calls are best-effort: any failure returns an empty list and the
pipeline carries on with the keyword results it already has.

Note: this module calls the Semantic Scholar / OpenAlex HTTP APIs directly -
it does not depend on any ``SourceProvider`` class, so it works from PubMed /
Europe PMC / bioRxiv seed papers even though this project does not ship
OpenAlex/Semantic Scholar as standalone search providers.
"""
from __future__ import annotations

import logging
import re

import httpx

from app.sources.base import SourceResult, build_headers, contact_email, extract_doi

logger = logging.getLogger(__name__)

_S2_REC_URL = "https://api.semanticscholar.org/recommendations/v1/papers/forpaper"
_S2_GRAPH_URL = "https://api.semanticscholar.org/graph/v1/paper"
_OPENALEX_URL = "https://api.openalex.org/works"
_TIMEOUT = 12


def _headers() -> dict[str, str]:
    """Outbound headers, built per call so settings load lazily."""
    return build_headers(accept="application/json")


# Defaults keep the expansion cheap and bounded.
_MAX_SEEDS = 2          # expand from at most this many seed papers
_PER_SEED = 5           # candidates requested per seed
_S2_FIELDS = "title,abstract,url,year,citationCount,externalIds"
# Nested field selector for the graph references/citations endpoints.
_S2_GRAPH_FIELDS = (
    "title,abstract,url,year,citationCount,externalIds"
)


def _seed_paper_id(result: SourceResult) -> str | None:
    url = (result.url or "").lower()
    if not url:
        return None

    m = re.search(r"semanticscholar\.org/paper/([0-9a-f]{40})", url)
    if m:
        return m.group(1)

    m = re.search(r"arxiv\.org/abs/([\w.\-]+?)(?:v\d+)?/?$", url)
    if m:
        return f"ARXIV:{m.group(1)}"

    m = re.search(r"(?:pubmed\.ncbi\.nlm\.nih\.gov|/med/)/?(\d{4,9})", url)
    if m:
        return f"PMID:{m.group(1)}"

    doi = extract_doi(url)
    if doi:
        return f"DOI:{doi}"

    return None


def _is_paper(result: SourceResult) -> bool:
    return result.source_type in {
        "pubmed", "europe_pmc", "biorxiv",
    }


def _s2_recommendations(paper_id: str, limit: int) -> list[SourceResult]:
    try:
        resp = httpx.get(
            f"{_S2_REC_URL}/{paper_id}",
            params={"fields": _S2_FIELDS, "limit": limit},
            headers=_headers(),
            timeout=_TIMEOUT,
        )
        if resp.status_code != 200:
            return []
        papers = resp.json().get("recommendedPapers", []) or []
    except Exception as exc:
        logger.debug("S2 recommendations failed for %s: %s", paper_id, exc)
        return []

    return [r for r in (_paper_to_result(p) for p in papers) if r is not None]


def _s2_graph_neighbours(paper_id: str, direction: str, limit: int) -> list[SourceResult]:
    """Backward (``references``) or forward (``citations``) graph traversal.

    Backward references surface the FOUNDATIONAL works a paper builds on
    (where the seminal papers live); forward citations surface the recent
    extensions. Both are free on the Semantic Scholar graph API.
    """
    if direction not in ("references", "citations"):
        return []
    nested = "citedPaper" if direction == "references" else "citingPaper"
    try:
        resp = httpx.get(
            f"{_S2_GRAPH_URL}/{paper_id}/{direction}",
            params={"fields": _S2_GRAPH_FIELDS, "limit": limit},
            headers=_headers(),
            timeout=_TIMEOUT,
        )
        if resp.status_code != 200:
            return []
        rows = resp.json().get("data", []) or []
    except Exception as exc:
        logger.debug("S2 %s failed for %s: %s", direction, paper_id, exc)
        return []

    papers = [row.get(nested) or {} for row in rows]
    return [r for r in (_paper_to_result(p) for p in papers) if r is not None]


def _openalex_related(doi: str, limit: int) -> list[SourceResult]:
    try:
        meta = httpx.get(
            f"{_OPENALEX_URL}/https://doi.org/{doi}",
            params={"select": "related_works", "mailto": contact_email()},
            headers=_headers(),
            timeout=_TIMEOUT,
        )
        if meta.status_code != 200:
            return []
        related_ids = meta.json().get("related_works", []) or []
        if not related_ids:
            return []
        short_ids = [rid.rsplit("/", 1)[-1] for rid in related_ids[:limit]]

        resp = httpx.get(
            _OPENALEX_URL,
            params={
                "filter": f"openalex_id:{'|'.join(short_ids)}",
                "select": "title,doi,publication_year,cited_by_count",
                "per_page": limit,
                "mailto": contact_email(),
            },
            headers=_headers(),
            timeout=_TIMEOUT,
        )
        if resp.status_code != 200:
            return []
        works = resp.json().get("results", []) or []
    except Exception as exc:
        logger.debug("OpenAlex related failed for %s: %s", doi, exc)
        return []

    return [r for r in (_work_to_result(w) for w in works) if r is not None]


def _paper_to_result(paper: dict) -> SourceResult | None:
    title = paper.get("title") or ""
    if not title:
        return None
    url = paper.get("url") or ""
    ext = paper.get("externalIds") or {}
    if not url and ext.get("DOI"):
        url = f"https://doi.org/{ext['DOI']}"
    if not url:
        return None

    year = paper.get("year")
    abstract = paper.get("abstract") or ""
    cited = int(paper.get("citationCount") or 0)
    content = f"Title: {title}"
    if year:
        content += f"\nYear: {year}"
    if cited:
        content += f"\nCitations: {cited}"
    if abstract:
        content += f"\n\nAbstract: {abstract}"

    return SourceResult(
        title=title,
        url=url,
        snippet=(abstract[:300] if abstract else title),
        content=content[:4000],
        source_type="semantic_scholar",
        reliability_tier=2,
        publication_date=str(year) if year else "",
        citation_count=cited,
        language="en",
    )


def _work_to_result(work: dict) -> SourceResult | None:
    title = work.get("title") or ""
    doi = work.get("doi") or ""
    if not title or not doi:
        return None
    year = work.get("publication_year")
    cited = int(work.get("cited_by_count") or 0)
    content = f"Title: {title}"
    if year:
        content += f"\nYear: {year}"
    if cited:
        content += f"\nCitations: {cited}"

    return SourceResult(
        title=title,
        url=doi if doi.startswith("http") else f"https://doi.org/{doi}",
        snippet=title,
        content=content,
        source_type="openalex",
        reliability_tier=2,
        publication_date=str(year) if year else "",
        citation_count=cited,
    )


def expand_with_citation_graph(
    seeds: list[SourceResult],
    *,
    max_seeds: int = _MAX_SEEDS,
    per_seed: int = _PER_SEED,
    order_by_citations: bool = True,
    bidirectional: bool = False,
) -> list[SourceResult]:
    """Return citation-graph neighbours of the most central seed papers.

    Seeds are the keyword-search results.  By default we expand from the few
    with the highest citation count (most central in the graph).  When
    *order_by_citations* is False the input order is preserved instead - the
    agentic-discovery path relies on this to seed from the most *relevant*
    papers (pre-sorted by semantic similarity) rather than the most cited,
    which avoids amplifying high-citation off-topic noise.  De-duplication
    against the existing evidence is left to the caller.

    When *bidirectional* is True (pro path) the expansion also walks the
    citation graph in both directions: BACKWARD references (the foundational
    works a relevant paper builds on - where seminal papers hide) and FORWARD
    citations (recent papers extending it). This is the highest-recall, still
    free, route to the canonical literature a keyword search misses.
    """
    papers = [s for s in seeds if _is_paper(s) and _seed_paper_id(s)]
    if not papers:
        return []

    if order_by_citations:
        papers.sort(key=lambda s: s.citation_count, reverse=True)

    expanded: list[SourceResult] = []
    seen: set[str] = {s.url.rstrip("/").lower() for s in seeds if s.url}
    used_openalex = False

    for seed in papers[:max_seeds]:
        pid = _seed_paper_id(seed)
        if not pid:
            continue

        recs = _s2_recommendations(pid, per_seed)

        # Bidirectional graph traversal (pro): foundational refs + extensions.
        if bidirectional:
            recs += _s2_graph_neighbours(pid, "references", per_seed)
            recs += _s2_graph_neighbours(pid, "citations", per_seed)

        # OpenAlex related from the single top DOI seed (one extra hop).
        if not used_openalex:
            doi = extract_doi(seed.url)
            if doi:
                recs += _openalex_related(doi, per_seed)
                used_openalex = True

        for r in recs:
            key = r.url.rstrip("/").lower()
            if key and key not in seen:
                seen.add(key)
                expanded.append(r)

    if expanded:
        logger.info(
            "Citation-graph expansion: +%d related papers from %d seeds (bidirectional=%s)",
            len(expanded), min(len(papers), max_seeds), bidirectional,
        )
    return expanded
