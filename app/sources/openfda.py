"""OpenFDA drug data provider - free, no API key required.

Accesses the FDA's open drug data including:
- Drug labeling (prescribing information, indications, contraindications,
  dosage, adverse reactions, mechanism of action)
- Drug adverse event reports (FAERS)
- Drug recalls and enforcement actions
- NDC directory (drug product info)

API docs: https://open.fda.gov/apis/
Rate limit: 240 requests/minute without key, 120K/day with key.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import SourceProvider, SourceResult, build_headers

logger = logging.getLogger(__name__)

_API_BASE = "https://api.fda.gov/drug"
_TIMEOUT = 12
def _headers() -> dict[str, str]:
    """Outbound headers, built per call so settings stay lazily loaded."""
    return build_headers(accept="application/json")

# Rate limiter: 4 requests/second to stay well under 240/min
_rate_lock = threading.Lock()
_last_request_time = 0.0


def _rate_limit():
    global _last_request_time
    with _rate_lock:
        now = time.monotonic()
        if now - _last_request_time < 0.25:
            time.sleep(0.25 - (now - _last_request_time))
        _last_request_time = time.monotonic()


class OpenFDAProvider(SourceProvider):
    """Search OpenFDA for drug information (labeling, adverse events).

    Primarily searches drug labeling (SPL) which contains the official
    prescribing information: indications, dosage, contraindications,
    warnings, mechanism of action, pharmacokinetics, adverse reactions.
    """

    source_type = "openfda"

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
    )
    def search(self, query: str, max_results: int = 3) -> list[SourceResult]:
        if not query or len(query) < 2:
            return []

        results: list[SourceResult] = []

        # Strategy 1: Search drug labeling (most informative)
        label_results = self._search_labels(query, max_results)
        results.extend(label_results)

        # Strategy 2: If query looks like a safety/adverse event query,
        # also search adverse events
        safety_keywords = {"adverse", "side effect", "reaction", "toxicity",
                          "overdose", "death", "contraindic", "interaction",
                          "effetto", "reazione", "tossicità"}
        if any(kw in query.lower() for kw in safety_keywords):
            ae_results = self._search_adverse_events(query, max(1, max_results - len(results)))
            results.extend(ae_results)

        return results[:max_results]

    def _search_labels(self, query: str, max_results: int) -> list[SourceResult]:
        _rate_limit()

        # BUG-02 workaround: match on any of the identifiers a drug could be
        # searched by. openFDA's Lucene parser treats "+" between clauses as
        # "this clause is required", not as a joiner, so joining with "+"
        # would silently make the brand-name clause mandatory and drop
        # generic-only hits. A plain space (Lucene's default OR) is what
        # actually broadens the match.
        clean = self._clean_drug_name(query)
        search_q = (
            f'openfda.generic_name:"{clean}" '
            f'openfda.brand_name:"{clean}" '
            f'openfda.substance_name:"{clean}"'
        )

        params: dict[str, Any] = {
            "search": search_q,
            "limit": min(max_results, 5),
        }
        if self._api_key:
            params["api_key"] = self._api_key

        used_free_text = False
        try:
            with httpx.Client(timeout=_TIMEOUT, headers=_headers()) as client:
                resp = client.get(f"{_API_BASE}/label.json", params=params)

                # If the identifier match finds nothing, fall back to a
                # free-text search across all label fields - much broader,
                # so its hits need to be checked against the drug name below
                # instead of trusted as-is.
                if resp.status_code == 404 or resp.status_code == 400:
                    used_free_text = True
                    params["search"] = query
                    resp = client.get(f"{_API_BASE}/label.json", params=params)

                if resp.status_code != 200:
                    return []

                data = resp.json()
        except Exception as exc:
            logger.debug("OpenFDA label search failed: %s", exc)
            return []

        results_data = data.get("results", [])

        # Free-text search matches the drug name occurring anywhere in the
        # label prose (e.g. as a comparator in another drug's clinical
        # trial section), so restrict its hits to labels actually
        # identified as this drug before trusting them.
        if used_free_text:
            clean_lower = clean.lower()

            def _is_this_drug(item: dict) -> bool:
                openfda = item.get("openfda", {})
                identifiers = (
                    openfda.get("generic_name", [])
                    + openfda.get("brand_name", [])
                    + openfda.get("substance_name", [])
                )
                return any(clean_lower in ident.lower() for ident in identifiers)

            matched = [item for item in results_data if _is_this_drug(item)]
            if matched:
                results_data = matched

        results: list[SourceResult] = []

        for item in results_data:
            try:
                result = self._parse_label(item)
                if result:
                    results.append(result)
            except Exception as exc:
                logger.debug("Failed to parse FDA label: %s", exc)

        return results

    def _search_adverse_events(self, query: str, max_results: int) -> list[SourceResult]:
        _rate_limit()

        clean = self._clean_drug_name(query)
        params: dict[str, Any] = {
            "search": f'patient.drug.openfda.generic_name:"{clean}"',
            "count": "patient.reaction.reactionmeddrapt.exact",
            "limit": 20,  # top 20 reported reactions
        }
        if self._api_key:
            params["api_key"] = self._api_key

        try:
            with httpx.Client(timeout=_TIMEOUT, headers=_headers()) as client:
                resp = client.get(f"{_API_BASE}/event.json", params=params)
                if resp.status_code != 200:
                    return []
                data = resp.json()
        except Exception as exc:
            logger.debug("OpenFDA adverse event search failed: %s", exc)
            return []

        ae_results = data.get("results", [])
        if not ae_results:
            return []

        ae_lines = []
        total_reports = 0
        for ae in ae_results[:15]:
            term = ae.get("term", "")
            count = ae.get("count", 0)
            total_reports += count
            ae_lines.append(f"  - {term}: {count:,} reports")

        content = (
            f"FDA Adverse Event Reports for {clean}\n"
            f"Top reported adverse reactions (total reports in pool):\n"
            + "\n".join(ae_lines)
            + "\n\nNote: FAERS data is spontaneous reporting - counts "
            "indicate report frequency, not incidence rates."
        )

        return [SourceResult(
            title=f"FDA Adverse Events - {clean}",
            url="https://open.fda.gov/apis/drug/event/",
            snippet=f"Top adverse events for {clean}: {', '.join(ae.get('term', '') for ae in ae_results[:5])}",
            content=content[:4000],
            source_type=self.source_type,
            reliability_tier=1,
        )]

    def _parse_label(self, item: dict) -> SourceResult | None:
        openfda = item.get("openfda", {})
        generic_names = openfda.get("generic_name", [])
        brand_names = openfda.get("brand_name", [])
        # Not every label carries the openfda linkage FDA derives after
        # indexing (generic_name/brand_name can both be empty on otherwise
        # complete records); substance_name is populated far more often and
        # is still the FDA's own structured identification of the drug,
        # not a guess parsed out of the label prose.
        substance_names = openfda.get("substance_name", [])

        name = (generic_names[0] if generic_names
                else brand_names[0] if brand_names
                else substance_names[0].title() if substance_names
                else "Unknown drug")

        # Build comprehensive content from label sections
        sections: list[str] = []
        sections.append(f"Drug: {name}")
        if brand_names:
            sections.append(f"Brand: {', '.join(brand_names[:3])}")

        # Key label sections with their FDA field names
        label_sections = [
            ("boxed_warning", "BOXED WARNING"),
            ("indications_and_usage", "INDICATIONS AND USAGE"),
            ("dosage_and_administration", "DOSAGE AND ADMINISTRATION"),
            ("contraindications", "CONTRAINDICATIONS"),
            ("warnings_and_cautions", "WARNINGS AND PRECAUTIONS"),
            ("warnings", "WARNINGS"),
            ("adverse_reactions", "ADVERSE REACTIONS"),
            ("drug_interactions", "DRUG INTERACTIONS"),
            ("mechanism_of_action", "MECHANISM OF ACTION"),
            ("pharmacodynamics", "PHARMACODYNAMICS"),
            ("pharmacokinetics", "PHARMACOKINETICS"),
            ("clinical_pharmacology", "CLINICAL PHARMACOLOGY"),
            ("overdosage", "OVERDOSAGE"),
            ("pregnancy", "PREGNANCY"),
            ("pediatric_use", "PEDIATRIC USE"),
            ("geriatric_use", "GERIATRIC USE"),
        ]

        for field_name, header in label_sections:
            value = item.get(field_name)
            if value:
                text = value[0] if isinstance(value, list) else str(value)
                if len(text) > 800:
                    text = text[:800] + "…"
                sections.append(f"\n--- {header} ---\n{text}")

        content = "\n".join(sections)

        set_id = item.get("set_id", "")
        url = (f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}"
               if set_id else "https://dailymed.nlm.nih.gov/dailymed/")

        snippet_parts = [name]
        if brand_names:
            snippet_parts.append(f"({', '.join(brand_names[:2])})")
        indications = item.get("indications_and_usage", [""])[0] if item.get("indications_and_usage") else ""
        if indications:
            snippet_parts.append(f"- {indications[:150]}")

        return SourceResult(
            title=f"FDA Label - {name}",
            url=url,
            snippet=" ".join(snippet_parts)[:300],
            content=content[:4000],
            source_type=self.source_type,
            reliability_tier=1,
        )

    @staticmethod
    def _clean_drug_name(query: str) -> str:
        # Remove common query suffixes
        query = re.sub(
            r"\b(dosage|dose|mechanism|interaction|side effect|adverse|"
            r"contraindication|indication|pharmacokinetics|ADME)\b",
            "", query, flags=re.IGNORECASE,
        )
        # Remove MeSH-style qualifiers
        query = re.sub(r'\[.*?\]', '', query)
        # Keep only meaningful tokens
        tokens = [t.strip() for t in query.split() if len(t.strip()) > 1]
        return " ".join(tokens[:3]).strip() or query.strip()
