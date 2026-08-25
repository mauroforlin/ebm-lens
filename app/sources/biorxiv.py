"""bioRxiv / medRxiv preprint provider.

Frontier life-sciences research lives in preprints months before it reaches
peer-reviewed journals, so a discovery pipeline that only queries PubMed /
Europe PMC misses the newest work.  bioRxiv (biology) and medRxiv (health
sciences) both register DOIs under the Cold Spring Harbor prefix ``10.1101``,
and Crossref offers free, keyword-searchable metadata for them - so we query
Crossref rather than the bioRxiv content API (which is date/DOI-addressed
only and not keyword-searchable).

Crossref's own ``prefix:10.1101`` filter does **not** work for this: it
matches the Cold Spring Harbor *journals* (Learning & Memory, CSH Protocols)
and almost none of the preprints. Preprints are instead reached by their
Crossref work type, ``posted-content``, and narrowed to these two servers
here - which is why the provider over-fetches and filters client-side.

Free API, no auth required.
"""
from __future__ import annotations

import logging
import re

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import SourceProvider, SourceResult, build_headers, contact_email

logger = logging.getLogger(__name__)

_API_BASE = "https://api.crossref.org/works"
_PREPRINT_TYPE = "posted-content"  # Crossref work type for preprints
_DOI_PREFIX = "10.1101/"           # Cold Spring Harbor → bioRxiv + medRxiv
_SERVERS = frozenset({"biorxiv", "medrxiv"})
# posted-content spans every preprint server (SSRN, Research Square, Qeios...),
# and roughly half of any result page is not bioRxiv/medRxiv - so ask for
# several times what we need and keep the ones that are.
_OVERFETCH = 4
_MAX_ROWS = 60
_TIMEOUT = 15
def _headers() -> dict[str, str]:
    """Outbound headers, built per call so settings stay lazily loaded."""
    return build_headers(accept="application/json")


def _server_name(item: dict) -> str:
    institutions = item.get("institution") or []
    if institutions and isinstance(institutions[0], dict):
        return institutions[0].get("name") or ""
    containers = item.get("container-title") or []
    return containers[0] if containers else ""


def _is_target_server(item: dict) -> bool:
    """True when *item* is a bioRxiv or medRxiv preprint.

    Matched on the server name, falling back to the DOI prefix: Crossref
    reports the institution for most preprints but not all of them.
    """
    if _server_name(item).strip().lower() in _SERVERS:
        return True
    return (item.get("DOI") or "").lower().startswith(_DOI_PREFIX)


class BiorxivProvider(SourceProvider):
    """Search bioRxiv + medRxiv preprints via Crossref (prefix 10.1101).

    Preprints are tier-3 (not peer-reviewed) but invaluable for frontier
    topics where the canonical work has not yet been formally published.
    """

    source_type = "biorxiv"

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
    )
    def search(self, query: str, max_results: int = 5) -> list[SourceResult]:
        if not query:
            return []

        params = {
            # Bibliographic search matches title/abstract/author. The generic
            # "query" field also searches references and funder metadata,
            # which returns papers that merely cite something on the topic.
            "query.bibliographic": query,
            "filter": f"type:{_PREPRINT_TYPE}",
            "rows": min(max_results * _OVERFETCH, _MAX_ROWS),
            "mailto": contact_email(),
            "sort": "relevance",
        }

        with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
            response = client.get(_API_BASE, params=params, headers=_headers())
            response.raise_for_status()
            data = response.json()

        items = data.get("message", {}).get("items", [])
        results: list[SourceResult] = []

        for item in items:
            if len(results) >= max_results:
                break
            if not _is_target_server(item):
                continue

            titles = item.get("title") or []
            title = titles[0] if titles else ""
            if not title:
                continue

            doi = item.get("DOI", "")
            url = item.get("URL", "")
            if not url and doi:
                url = f"https://doi.org/{doi}"
            if not url:
                continue

            abstract = item.get("abstract") or ""
            if abstract:
                abstract = re.sub(r"<[^>]+>", "", abstract).strip()

            snippet = abstract[:300] if abstract else title

            # Which server it came from matters clinically: medRxiv carries
            # health-sciences work, bioRxiv basic biology.
            server = _server_name(item) or "bioRxiv/medRxiv"

            parts: list[str] = [f"Title: {title}", f"Source: {server} (preprint)"]

            authors = item.get("author") or []
            author_names: list[str] = []
            for a in authors[:5]:
                name = f"{a.get('given', '')} {a.get('family', '')}".strip()
                if name:
                    author_names.append(name)
            if author_names:
                parts.append(f"Authors: {', '.join(author_names)}")

            # Preprints carry a "posted" date; fall back to published-*.
            pub_year = ""
            date_obj = (
                item.get("posted")
                or item.get("published-online")
                or item.get("published-print")
                or {}
            )
            date_parts = date_obj.get("date-parts", [[]])
            if date_parts and date_parts[0] and date_parts[0][0]:
                pub_year = str(date_parts[0][0])
                parts.append(f"Posted: {pub_year}")

            cited = item.get("is-referenced-by-count", 0)
            if cited:
                parts.append(f"Citations: {cited}")
            if doi:
                parts.append(f"DOI: {doi}")
            if abstract:
                parts.append(f"\nAbstract: {abstract}")

            content = "\n".join(parts)

            results.append(SourceResult(
                title=title,
                url=url,
                snippet=snippet,
                content=content[:4000],
                source_type=self.source_type,
                reliability_tier=3,  # preprint - not peer-reviewed
                publication_date=pub_year,
                citation_count=int(cited or 0),
                language="en",
            ))

        logger.debug("bioRxiv/medRxiv returned %d results for query: %s", len(results), query)
        return results
