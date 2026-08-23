"""WHO Global Health Observatory (GHO) provider - free, no API key.

Authoritative global health statistics: life expectancy, disease
prevalence, mortality rates, vaccination coverage, health expenditure.

API docs: https://www.who.int/data/gho/info/gho-odata-api
Rate limit: No documented limit; use reasonable rate.
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

_API_BASE = "https://ghoapi.azureedge.net/api"
_TIMEOUT = 12
def _headers() -> dict[str, str]:
    """Outbound headers, built per call so settings stay lazily loaded."""
    return build_headers(accept="application/json")

# Rate limiter: ~4 req/s
_rate_lock = threading.Lock()
_last_request_time = 0.0

# Keywords that signal epidemiological/statistical queries
_STAT_KEYWORDS = frozenset({
    "mortality", "prevalence", "incidence", "morbidity", "rate",
    "life expectancy", "vaccination", "immunization", "coverage",
    "deaths", "cases", "population", "birth", "fertility",
    "obesity", "malnutrition", "sanitation", "drinking water",
    "mortalità", "prevalenza", "incidenza", "tasso",
    "aspettativa di vita", "vaccinazione", "copertura",
    "decessi", "casi", "popolazione", "nascita",
})


def _rate_limit() -> None:
    global _last_request_time
    with _rate_lock:
        now = time.monotonic()
        if now - _last_request_time < 0.25:
            time.sleep(0.25 - (now - _last_request_time))
        _last_request_time = time.monotonic()


# Grammatical filler, not indicator vocabulary - "rate"/"level" stay in since
# WHO indicator names are full of them ("mortality rate", "coverage level").
_QUERY_STOPWORDS = frozenset({
    "what", "is", "are", "was", "were", "the", "a", "an", "of", "in", "on",
    "at", "for", "to", "and", "or", "with", "by", "from", "about", "as",
    "how", "much", "many", "does", "do", "did", "can", "could", "would",
    "should", "will", "there", "this", "that", "which", "who",
    "che", "cosa", "qual", "quali", "il", "la", "lo", "gli", "le", "un",
    "una", "di", "per", "con", "da", "e", "sono", "quanto", "quanti",
})


def _extract_search_terms(query: str) -> str:
    """Extract key terms for GHO indicator search."""
    clean = re.sub(r"[\"'()[\]{}<>?!.,;:]", " ", query)
    clean = re.sub(r"\s+", " ", clean).strip()
    tokens = [t for t in clean.split() if t.lower() not in _QUERY_STOPWORDS]
    if not tokens:
        tokens = clean.split()
    return " ".join(tokens[:6])


class WHOGHOProvider(SourceProvider):
    """Search WHO GHO for global health statistics (free, no API key)."""

    source_type = "who_gho"

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
    )
    def search(self, query: str, max_results: int = 3) -> list[SourceResult]:
        if not query or len(query) < 3:
            return []

        terms = _extract_search_terms(query)
        if not terms:
            return []

        # Step 1: Find relevant indicators. Indicator names are terse and
        # don't share the query's vocabulary (a "mortality" question may
        # match an indicator that says "deaths"), so an AND of every term
        # is brittle - broaden progressively rather than giving up after one
        # combination that happens not to co-occur in any indicator name.
        indicators = self._search_indicators(terms)
        if not indicators:
            short = " ".join(terms.split()[:2])
            if short != terms:
                indicators = self._search_indicators(short)
        if not indicators:
            indicators = self._search_indicators(terms, operator="or")

        if not indicators:
            return []

        results: list[SourceResult] = []
        for ind in indicators[:max_results]:
            result = self._fetch_indicator_data(ind, query)
            if result:
                results.append(result)

        return results

    def _search_indicators(self, terms: str, *, operator: str = "and") -> list[dict]:
        """Search for GHO indicators matching the given terms."""
        _rate_limit()

        # Use OData $filter with contains() for each term
        filters: list[str] = []
        for term in terms.split():
            if len(term) > 2:
                # Escape single quotes
                safe = term.replace("'", "''")
                filters.append(f"contains(IndicatorName,'{safe}')")

        if not filters:
            return []

        filter_str = f" {operator} ".join(filters)
        try:
            resp = httpx.get(
                f"{_API_BASE}/Indicator",
                params={"$filter": filter_str},
                headers=_headers(),
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.debug("WHO GHO indicator search failed: %s", exc)
            # Fallback: try just the first term
            if len(filters) > 1:
                try:
                    resp = httpx.get(
                        f"{_API_BASE}/Indicator",
                        params={"$filter": filters[0]},
                        headers=_headers(),
                        timeout=_TIMEOUT,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                except Exception:
                    return []
            else:
                return []

        indicators = data.get("value", [])
        if operator == "or" and len(indicators) > 10:
            # An OR of terms like "rate"/"children" matches broadly - rank by
            # how many of the query's own terms the indicator name actually
            # contains, so the closest match surfaces instead of whichever
            # unrelated indicator the API happened to list first.
            query_terms = [f.split("'")[1].lower() for f in filters]

            def _term_overlap(ind: dict) -> int:
                name = (ind.get("IndicatorName") or "").lower()
                return sum(1 for t in query_terms if t in name)

            indicators = sorted(indicators, key=_term_overlap, reverse=True)

        return indicators[:10]

    def _fetch_indicator_data(
        self, indicator: dict, original_query: str
    ) -> SourceResult | None:
        """Fetch actual data points for an indicator."""
        code = indicator.get("IndicatorCode", "")
        name = indicator.get("IndicatorName", "")
        if not code or not name:
            return None

        _rate_limit()

        country_code = self._detect_country(original_query)

        params: dict[str, str] = {}
        if country_code:
            params["$filter"] = f"SpatialDim eq '{country_code}'"
            params["$orderby"] = "TimeDim desc"
            params["$top"] = "10"
        else:
            params["$orderby"] = "TimeDim desc"
            params["$top"] = "20"

        try:
            resp = httpx.get(
                f"{_API_BASE}/{code}",
                params=params,
                headers=_headers(),
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.debug("WHO GHO data fetch failed for %s: %s", code, exc)
            # Return indicator info without data
            return SourceResult(
                title=f"{name} - WHO GHO",
                url=f"https://www.who.int/data/gho/data/indicators/indicator-details/GHO/{code}",
                snippet=f"WHO Global Health Observatory indicator: {name}",
                content=f"Indicator: {name}\nCode: {code}\n(Data unavailable)",
                source_type=self.source_type,
                reliability_tier=1,
                language="en",
            )

        values = data.get("value", [])
        url = f"https://www.who.int/data/gho/data/indicators/indicator-details/GHO/{code}"

        content_parts: list[str] = [
            f"WHO Global Health Observatory\nIndicator: {name}\nCode: {code}\n",
        ]

        if not values:
            content_parts.append("No data available for the specified filters.")
        else:
            # Group by country for readability
            by_country: dict[str, list[dict]] = {}
            for v in values:
                country = v.get("SpatialDim", "GLOBAL")
                by_country.setdefault(country, []).append(v)

            for country, points in list(by_country.items())[:5]:
                content_parts.append(f"\n## {country}")
                for p in points[:5]:
                    year = p.get("TimeDim", "?")
                    val = p.get("NumericValue") or p.get("Value", "N/A")
                    dim1 = p.get("Dim1", "")
                    dim1_str = f" ({dim1})" if dim1 else ""
                    content_parts.append(f"  {year}{dim1_str}: {val}")

        snippet = f"WHO GHO: {name}"
        if values:
            latest = values[0]
            yr = latest.get("TimeDim", "")
            val = latest.get("NumericValue") or latest.get("Value", "")
            if yr and val:
                snippet += f" | Latest: {val} ({yr})"

        return SourceResult(
            title=f"{name} - WHO Global Health Observatory",
            url=url,
            snippet=snippet[:300],
            content="\n".join(content_parts)[:4000],
            source_type=self.source_type,
            reliability_tier=1,
            language="en",
        )

    def _detect_country(self, query: str) -> str:
        """Try to detect a country ISO code from the query."""
        query_lower = query.lower()
        # Common country mappings (expand as needed)
        country_map = {
            "italy": "ITA", "italia": "ITA", "italian": "ITA",
            "france": "FRA", "francia": "FRA", "french": "FRA",
            "germany": "DEU", "germania": "DEU", "german": "DEU",
            "spain": "ESP", "spagna": "ESP", "spanish": "ESP",
            "united states": "USA", "stati uniti": "USA", "usa": "USA",
            "united kingdom": "GBR", "regno unito": "GBR", "uk": "GBR",
            "china": "CHN", "cina": "CHN",
            "japan": "JPN", "giappone": "JPN",
            "brazil": "BRA", "brasile": "BRA",
            "india": "IND",
            "world": "", "global": "", "mondiale": "",
        }
        for name, code in country_map.items():
            if name in query_lower:
                return code
        return ""
