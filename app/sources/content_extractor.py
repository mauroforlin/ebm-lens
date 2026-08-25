"""Content extractor - fetches full page text for evidence enrichment.

Some providers return an abstract-length snippet and little else. Summarising
a paper from two sentences produces a summary that says nothing, so before
the summarisation pass the shortlist gets its full page text fetched.

This is the one part of the pipeline that touches arbitrary web pages, so it
is deliberately fenced in:

- **Allowlist only** - fetches are limited to ``_FETCHABLE_DOMAINS``. Anything
  else keeps whatever the provider gave us. No open-ended crawling.
- **Budget** - at most ``max_fetches`` URLs per run.
- **Size cap** - ``_MAX_CONTENT_CHARS`` per page, so one enormous page cannot
  dominate the summariser's context window.
- **Timeout** - ``_FETCH_TIMEOUT`` seconds per request, fetched in parallel.

None of this makes the fetched text trustworthy. It is untrusted input: the
domain allowlist bounds who can supply it, not what it says - and for a
user-editable host like Wikipedia, "who" still means anyone with an edit
button. Nothing here sanitises the page's content, and this extracted text
goes on to reach the model as ordinary content in a pipeline where the model
has tools available (see synthesis.py).
"""
from __future__ import annotations

import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from app.config import Settings

from app.core.sections import BIOMED_PRIORITY, BIOMED_SKIP, trim_by_priority
from app.sources.base import SourceResult, build_headers

logger = logging.getLogger(__name__)


_MAX_FETCHES_PER_RUN = 3       # max URLs the blind pre-pass fetches per run
_MAX_CONTENT_CHARS = 6000      # max chars extracted per page
_FETCH_TIMEOUT = 6             # seconds per HTTP request
_MIN_USEFUL_CONTENT = 150      # skip pages shorter than this

# The only hosts this module will fetch a page from. Every entry is either a
# provider whose results we already trust, or a regulator/HTA body whose
# guidance is citable evidence in its own right. Subdomains match.
#
# Two domains that look like they belong here are deliberately absent:
# - pubmed.ncbi.nlm.nih.gov serves a JS cookie-challenge to a plain GET, never
#   the abstract page - fetching it returns 2000+ chars of challenge
#   boilerplate that reads as real content. PubMed sources go through the
#   PMC route instead (see synthesis.py's read_full_text handler).
# - clinicaltrials.gov's study page is a client-rendered SPA; a plain GET
#   returns only nav chrome, no trial data. ClinicalTrialsProvider's own
#   content, built from the JSON API, already has the full structured
#   registry entry - there is no more prose to fetch.
_FETCHABLE_DOMAINS: frozenset[str] = frozenset({
    # Providers already in the pipeline
    "en.wikipedia.org", "it.wikipedia.org",
    "ncbi.nlm.nih.gov",
    "europepmc.org",
    "dailymed.nlm.nih.gov",
    "who.int",
    # Regulators, HTA bodies and evidence syntheses
    "ema.europa.eu",
    "aifa.gov.it",
    "salute.gov.it",
    "epicentro.iss.it",
    "cochranelibrary.com", "cochrane.org",
    "nice.org.uk",
    "msdmanuals.com",
})


def _is_fetchable_url(url: str) -> bool:
    try:
        host = urlparse(url).hostname or ""
        # Exact host or any subdomain of an allowed one.
        return any(
            host == domain or host.endswith("." + domain)
            for domain in _FETCHABLE_DOMAINS
        )
    except Exception:
        return False


_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")

# A null byte cannot occur in real page text, so it survives get_text's
# whitespace collapse untouched - the marker is reinstated as a `## Heading`
# line afterwards, once collapsing can no longer flatten it away.
_HEADING_SENTINEL = "\x00"
_HEADING_MARKER_RE = re.compile(
    re.escape(_HEADING_SENTINEL) + r"(.*?)" + re.escape(_HEADING_SENTINEL)
)


def _mark_headings(soup) -> None:
    """Replace each heading tag with a sentinel-wrapped copy of its text.

    Gives HTML sources the same ``## Heading`` convention
    :func:`app.sources.pubmed._fetch_pmc_fulltext` uses for JATS section
    titles, so :func:`app.core.sections.trim_by_priority` can rank Cochrane,
    NICE and EMA pages by section the same way it ranks PMC full text.
    """
    for tag in soup.find_all(_HEADING_TAGS):
        heading_text = tag.get_text(" ", strip=True)
        if heading_text:
            tag.replace_with(f"{_HEADING_SENTINEL}{heading_text}{_HEADING_SENTINEL}")
        else:
            tag.decompose()


def _restore_heading_markers(text: str) -> str:
    return _HEADING_MARKER_RE.sub(r"\n## \1\n", text)


def _extract_text_from_html(html: str, url: str = "") -> str:
    """Extract readable text from HTML, preferring BeautifulSoup4."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        _mark_headings(soup)
        text = soup.get_text(separator=" ", strip=True)
        text = _restore_heading_markers(re.sub(r"\s+", " ", text)).strip()
        if text and len(text) >= _MIN_USEFUL_CONTENT:
            return trim_by_priority(
                text, _MAX_CONTENT_CHARS,
                priority=BIOMED_PRIORITY, skip=BIOMED_SKIP,
            )
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("bs4 extraction failed for %s: %s", url[:60], exc)

    # Fallback: basic HTML tag stripping, headings marked before the generic
    # strip removes the tags that would otherwise identify them.
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(
        r"<(h[1-6])[^>]*>(.*?)</\1>",
        lambda m: f"{_HEADING_SENTINEL}{re.sub(r'<[^>]+>', '', m.group(2))}{_HEADING_SENTINEL}",
        text, flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r"<[^>]+>", " ", text)
    text = _restore_heading_markers(re.sub(r"\s+", " ", text)).strip()

    if len(text) < _MIN_USEFUL_CONTENT:
        return ""
    return trim_by_priority(text, _MAX_CONTENT_CHARS, priority=BIOMED_PRIORITY, skip=BIOMED_SKIP)


@retry(wait=wait_exponential(multiplier=1, min=1, max=4), stop=stop_after_attempt(3), reraise=True)
def _fetch_one(url: str) -> tuple[str, str]:
    """Returns (extracted_text, content_hash)."""
    try:
        with httpx.Client(
            timeout=_FETCH_TIMEOUT,
            follow_redirects=True,
            headers=build_headers(
                accept="text/html,application/xhtml+xml",
                accept_language="it,en;q=0.9",
            ),
        ) as client:
            resp = client.get(url)
            if resp.status_code >= 400:
                logger.debug("Fetch failed %d for %s", resp.status_code, url[:80])
                return "", ""

            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                logger.debug("Skipping non-HTML content-type %s for %s", content_type, url[:80])
                return "", ""

            html = resp.text
            text = _extract_text_from_html(html, url)
            content_hash = hashlib.sha256(text.encode()).hexdigest()[:16] if text else ""
            return text, content_hash

    except httpx.TimeoutException as exc:
        logger.debug("Fetch timeout for %s", url[:80])
        raise exc
    except Exception as exc:
        logger.debug("Fetch error for %s: %s", url[:80], exc)
        raise


def fetch_full_text(result: SourceResult) -> str:
    """Fetch one result's page text on demand, outside the batch pre-pass.

    Same allowlist and size cap as :func:`enrich_with_full_content` - this is
    the single-URL primitive the summariser's ``read_full_text`` tool calls
    when it decides a specific source needs more than its current excerpt.
    Returns "" when the URL is not on the allowlist or the fetch yields
    nothing useful; the caller decides what that means for the model.
    """
    if not result.url or not _is_fetchable_url(result.url):
        return ""
    try:
        text, _ = _fetch_one(result.url)
    except Exception as exc:
        logger.debug("On-demand fetch failed for %s: %s", result.url[:80], exc)
        return ""
    return text


def enrich_with_full_content(
    results: list[SourceResult],
    settings: Settings | None = None,
    max_fetches: int = _MAX_FETCHES_PER_RUN,
) -> int:
    """Modifies *results* in-place; only fetches URLs still under ``_MIN_USEFUL_CONTENT``."""
    # Filter to results that need enrichment
    candidates = [
        r for r in results
        if r.url
        and len(r.content) < _MIN_USEFUL_CONTENT
        and _is_fetchable_url(r.url)
    ]

    if not candidates:
        return 0

    # Limit to budget
    to_fetch = candidates[:max_fetches]

    enriched = 0
    with ThreadPoolExecutor(max_workers=min(8, len(to_fetch))) as pool:
        futures = {
            pool.submit(_fetch_one, r.url): r
            for r in to_fetch
        }
        for fut in as_completed(futures, timeout=_FETCH_TIMEOUT * 2):
            result = futures[fut]
            try:
                text, content_hash = fut.result()
                if text and len(text) > len(result.content):
                    result.content = text
                    result.content_hash = content_hash
                    enriched += 1
                    logger.debug(
                        "Enriched %s with %d chars of content",
                        result.url[:60], len(text),
                    )
            except Exception as exc:
                logger.debug("Content enrichment failed for %s: %s", result.url[:60], exc)

    if enriched:
        logger.info(
            "Content enrichment: %d/%d results enriched with full text",
            enriched, len(to_fetch),
        )
    return enriched
