"""Domain blocklist - the source-quality gate applied to every result.

Every current provider is a curated biomedical/scientific API (PubMed,
Europe PMC, ClinicalTrials.gov, ChEMBL, DailyMed, EMA, OpenFDA, RxNav,
WHO GHO, bioRxiv, Wikipedia) or the citation-graph expansion (Semantic
Scholar / OpenAlex), which resolves to DOI landing pages and abstract
mirrors rather than the open web. None of them can surface a social-media
or SEO-content-farm URL, so this list only needs to cover what the
citation graph actually produces: aggregators that mirror an abstract
without adding retrievable full-text content.

Kept deliberately as a static list rather than a heuristic: the set is
small, stable and easy to audit, and a reviewer can see exactly what the
pipeline refuses to cite.
"""
from __future__ import annotations

from urllib.parse import urlparse

# Domains whose content is never usable as evidence. Matching is on the
# registrable host and any subdomain of it.
BLOCKED_DOMAINS: frozenset[str] = frozenset({
    # Aggregators that mirror abstracts without adding retrievable content
    "core.ac.uk",
    "researchgate.net",   # paywalled abstracts - use PubMed/Europe PMC instead
    "academia.edu",
    "scribd.com",
    "slideshare.net",
})


def is_blocked_url(url: str) -> bool:
    """Return True when *url*'s host is blocked (exact host or a subdomain).

    A malformed or hostless URL is treated as *not* blocked: this is a
    quality filter, not a security boundary, and the ranker will discard a
    useless result anyway.
    """
    if not url:
        return False
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    if not host:
        return False
    # removeprefix, NOT lstrip: lstrip strips a *character set*, so
    # "who.int".lstrip("www.") would yield "ho.int" and silently miss.
    host = host.removeprefix("www.")
    return any(
        host == blocked or host.endswith("." + blocked)
        for blocked in BLOCKED_DOMAINS
    )


def filter_blocked(results: list) -> list:
    """Drop every result whose URL is blocked. Accepts any object with ``.url``."""
    return [r for r in results if not is_blocked_url(getattr(r, "url", ""))]
