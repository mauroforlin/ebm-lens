"""Domain blocklist - the source-quality gate applied to every result.

Evidence-based medicine is only as good as what feeds it. Several providers
(and the citation graph in particular) surface URLs that are technically
"about" the topic but worthless as clinical evidence: SEO health magazines,
user-generated Q&A, paywalled abstract mirrors, social media. This module is
the single place that decides what never reaches the ranker.

Kept deliberately as a static list rather than a heuristic: the set of
low-quality health publishers is small, stable and easy to audit, and a
reviewer can see exactly what the pipeline refuses to cite.
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
    # Social media / user-generated
    "instagram.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "tiktok.com",
    "reddit.com",
    "pinterest.com",
    "linkedin.com",
    "threads.net",
    "tumblr.com",
    "quora.com",
    "youtube.com",
    "medium.com",
    # Q&A / how-to farms
    "answers.yahoo.com",
    "answers.com",
    "wikihow.com",
    # Consumer health/wellness publishers (IT)
    "mypersonaltrainer.it",
    "my-personaltrainer.it",
    "medicitalia.it",
    "doctissimo.it",
    "donnamoderna.com",
    "starbene.it",
    "riza.it",
    "cure-naturali.it",
    "greenstyle.it",
    "benessereblog.it",
    "tantasalute.it",
    "viverepiusani.it",
    "ohga.it",
    "salutarmente.it",
    "greenme.it",
    "dilei.it",
    "alfemminile.com",
    "cosmopolitan.com",
    # Consumer health/wellness publishers (EN)
    "healthline.com",
    "webmd.com",
    "verywellhealth.com",
    "medicalnewstoday.com",
    "livestrong.com",
    "mindbodygreen.com",
    "self.com",
    "prevention.com",
    "health.com",
    "everydayhealth.com",
    # News aggregators - prefer the primary source
    "msn.com",
    "huffpost.com",
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
