"""Wikipedia Action API provider - completely free, no API key required.

- Full article retrieval for deeper evidence.
- Disambiguation page detection → follows first relevant link.
- Section-level text extraction for targeted evidence.
"""
from __future__ import annotations

import contextlib
import logging
import re

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.sections import trim_by_priority
from app.sources.base import SourceProvider, SourceResult, user_agent

logger = logging.getLogger(__name__)

# Matches a level-2 "== Section ==" header from the Action API's
# explaintext extract. Deeper levels ("=== Subsection ===") start with a
# third "=" right where this pattern requires whitespace, so they fall
# through and stay attached to their parent section's body rather than
# splitting it further.
_SECTION_HEADING_RE = re.compile(r"(?m)^==\s+(.+?)\s*=+\s*$")

_SKIP_SECTIONS = (
    "voci correlate", "see also", "note",
    "references", "bibliography", "bibliografia",
    "collegamenti esterni", "external links", "altri progetti",
)


class WikipediaProvider(SourceProvider):
    """Search and retrieve article summaries from Wikipedia (IT + EN).

    Optimised for Italian-language queries: searches it.wikipedia.org first,
    then follows interlanguage links to en.wikipedia.org for richer content.
    Falls back to direct en.wiki search if the IT edition has no results.
    """

    source_type = "wikipedia"

    def __init__(self, languages: tuple[str, ...] = ("en",)) -> None:
        self.languages = languages
        # Shared HTTP client - avoids creating 12+ TCP connections per search
        self._client = httpx.Client(
            timeout=10,
            headers={"User-Agent": user_agent()},
            follow_redirects=True,
            limits=httpx.Limits(max_connections=6, max_keepalive_connections=4),
        )

    def __del__(self) -> None:
        # Interpreter shutdown can already have torn down what close() needs.
        with contextlib.suppress(Exception):
            self._client.close()

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
    def search(self, query: str, max_results: int = 3) -> list[SourceResult]:
        results: list[SourceResult] = []
        seen_titles: set[str] = set()

        has_it = "it" in self.languages
        has_en = "en" in self.languages

        # ── Phase 1: Search IT Wikipedia with the (Italian) query ──
        if has_it:
            it_titles = self._search_titles(query, "it", limit=max_results)
            # Fallback: full-text search when opensearch misses
            if not it_titles:
                it_titles = self._fulltext_search_titles(query, "it", limit=max_results)
            for title in it_titles:
                if len(results) >= max_results:
                    break
                key = f"it:{title.lower()}"
                if key in seen_titles:
                    continue
                seen_titles.add(key)

                article = self._get_extract(title, "it")
                if article:
                    results.append(article)

                # Follow interlanguage link → EN for richer/broader content
                if has_en and len(results) < max_results:
                    en_title = self._get_langlink(
                        title, from_lang="it", to_lang="en",
                    )
                    if en_title:
                        en_key = f"en:{en_title.lower()}"
                        if en_key not in seen_titles:
                            seen_titles.add(en_key)
                            en_article = self._get_extract(en_title, "en")
                            if en_article:
                                results.append(en_article)

        # ── Phase 2: Direct EN wiki search for remaining slots ──
        if has_en and len(results) < max_results:
            remaining = max_results - len(results)
            en_titles = self._search_titles(query, "en", limit=remaining)
            # Fallback: full-text search when opensearch misses
            if not en_titles:
                en_titles = self._fulltext_search_titles(query, "en", limit=remaining)
            for title in en_titles:
                if len(results) >= max_results:
                    break
                key = f"en:{title.lower()}"
                if key in seen_titles:
                    continue
                seen_titles.add(key)

                article = self._get_extract(title, "en")
                if article:
                    results.append(article)

        return results

    # ── internal ──────────────────────────────────────

    def _get_langlink(self, title: str, from_lang: str, to_lang: str) -> str | None:
        """Get the interlanguage link from one wiki edition to another."""
        url = f"https://{from_lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "prop": "langlinks",
            "lllang": to_lang,
            "lllimit": 1,
            "redirects": 1,
            "titles": title,
        }
        try:
            resp = self._client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return None
            page = next(iter(pages.values()))
            langlinks = page.get("langlinks", [])
            if langlinks:
                return langlinks[0].get("*")
            return None
        except Exception as exc:
            logger.debug(
                "Wikipedia langlink lookup failed (%s→%s/%s): %s",
                from_lang, to_lang, title, exc,
            )
            return None

    def _search_titles(self, query: str, lang: str, limit: int = 3) -> list[str]:
        """Use the MediaWiki Action API opensearch to find page titles."""
        url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "opensearch",
            "search": query,
            "limit": limit,
            "format": "json",
        }
        try:
            resp = self._client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            # opensearch returns [query, [titles], [descriptions], [urls]]
            return data[1] if len(data) > 1 else []
        except Exception as exc:
            logger.debug("Wikipedia title search failed (%s): %s", lang, exc)
            return []

    def _fulltext_search_titles(self, query: str, lang: str, limit: int = 3) -> list[str]:
        """Full-text content search via action=query&list=search.

        Unlike opensearch (which only matches title prefixes), this
        searches inside article body text - essential for conceptual
        queries like 'normal sperm motility dog' that don't match any
        article title prefix.
        """
        url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "srnamespace": 0,  # main namespace only
            "srprop": "snippet",
            "format": "json",
        }
        try:
            resp = self._client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            hits = data.get("query", {}).get("search", [])
            return [h["title"] for h in hits if "title" in h]
        except Exception as exc:
            logger.debug("Wikipedia fulltext search failed (%s): %s", lang, exc)
            return []

    def _get_extract(self, title: str, lang: str) -> SourceResult | None:
        """Fetch a full Wikipedia page extract via the Action API.

        Retrieves the full article text (not just intro) for richer
        evidence, then selects the most relevant sections.
        """
        url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts|info",
            "inprop": "url",
            "explaintext": 1,
            # No exintro flag - the full body, not just the lead section, so
            # _smart_trim below has whole sections to score and pick from.
            "redirects": 1,
            "titles": title,
        }
        try:
            resp = self._client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            pages = data.get("query", {}).get("pages", {})
            if not pages:
                return None

            page = next(iter(pages.values()))
            extract = (page.get("extract") or "").strip()
            page_url = page.get("canonicalurl") or ""
            display_title = page.get("title") or title

            if not extract:
                return None

            # Detect disambiguation pages (contain "può riferirsi a:" or "may refer to:")
            if WikipediaProvider._is_disambiguation(extract):
                logger.debug("Disambiguation page detected: %s", display_title)
                return None

            trimmed = WikipediaProvider._smart_trim(extract, max_chars=4000)

            return SourceResult(
                title=f"[Wikipedia {lang.upper()}] {display_title}",
                url=page_url or f"https://{lang}.wikipedia.org/wiki/{title}",
                snippet=trimmed[:300],
                content=trimmed,
                source_type="wikipedia",
                language=lang,
            )
        except Exception as exc:
            logger.debug("Wikipedia extract failed (%s/%s): %s", lang, title, exc)
            return None

    @staticmethod
    def _is_disambiguation(text: str) -> bool:
        """Detect if a page is a disambiguation page."""
        patterns = [
            "può riferirsi a:",
            "può indicare:",
            "may refer to:",
            "may also refer to:",
            "disambiguation",
        ]
        first_200 = text[:200].lower()
        return any(p in first_200 for p in patterns)

    @staticmethod
    def _smart_trim(text: str, max_chars: int = 4000) -> str:
        """Keep the intro and the most informative sections, within max_chars.

        Splits on the Action API's plain-text section headers and scores each
        surviving section by factual density - number, date and length
        signals - since an encyclopedia article's most citation-worthy
        content isn't reliably first in reading order the way an abstract is.
        """
        def _density(section: str) -> float:
            num_count = len(re.findall(r"\d+", section))
            date_count = len(re.findall(r"\d{4}", section))  # years are high-value
            length_bonus = min(len(section) / 500, 3.0)  # reward substantial sections
            return num_count * 0.5 + date_count * 1.5 + length_bonus + (1.0 if len(section) > 200 else 0.3)

        return trim_by_priority(
            text, max_chars,
            marker=_SECTION_HEADING_RE,
            skip=_SKIP_SECTIONS,
            score_fn=_density,
        )
