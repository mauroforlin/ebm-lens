"""DailyMed provider - free, no API key required.

NIH/NLM service providing FDA-approved drug labeling (SPL documents).
Complementary to OpenFDA: better name-based search, versioned label
history, and structured section retrieval.

API docs: https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm
Rate limit: NLM fair-use (~5 req/s safe).
"""
from __future__ import annotations

import logging
import re
import threading
import time
from xml.etree import ElementTree

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import SourceProvider, SourceResult, build_headers

logger = logging.getLogger(__name__)

_API_BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
_TIMEOUT = 12
def _headers() -> dict[str, str]:
    """Outbound headers, built per call so settings stay lazily loaded."""
    return build_headers(accept="application/json")

# Rate limiter: ~4 req/s
_rate_lock = threading.Lock()
_last_request_time = 0.0

# SPL sections we care about (label field → human label)
_INTERESTING_SECTIONS = {
    "boxed_warning": "Boxed Warning",
    "indications_and_usage": "Indications and Usage",
    "dosage_and_administration": "Dosage and Administration",
    "contraindications": "Contraindications",
    "warnings_and_cautions": "Warnings and Precautions",
    "warnings": "Warnings",
    "adverse_reactions": "Adverse Reactions",
    "drug_interactions": "Drug Interactions",
    "mechanism_of_action": "Mechanism of Action",
    "pharmacodynamics": "Pharmacodynamics",
    "pharmacokinetics": "Pharmacokinetics",
    "clinical_pharmacology": "Clinical Pharmacology",
    "overdosage": "Overdosage",
    "description": "Description",
}

# The single-SPL detail route only speaks HL7 SPL XML, not JSON. Sections are
# matched by LOINC code (codeSystem 2.16.840.1.113883.6.1) - stable across
# every SPL - rather than by the free-text display name, which authors phrase
# inconsistently ("Warnings" vs "Warnings and Precautions", abbreviations, ...).
_SPL_NS = "urn:hl7-org:v3"
_NS = f"{{{_SPL_NS}}}"
_SECTION_CODES: dict[str, str] = {
    "34066-1": "boxed_warning",
    "34067-9": "indications_and_usage",
    "34068-7": "dosage_and_administration",
    "34070-3": "contraindications",
    "43685-7": "warnings_and_cautions",
    "34071-1": "warnings",
    "34084-4": "adverse_reactions",
    "34073-7": "drug_interactions",
    "43679-0": "mechanism_of_action",
    "43681-6": "pharmacodynamics",
    "43682-4": "pharmacokinetics",
    "34090-1": "clinical_pharmacology",
    "34088-5": "overdosage",
    "34089-3": "description",
}


def _rate_limit() -> None:
    global _last_request_time
    with _rate_lock:
        now = time.monotonic()
        if now - _last_request_time < 0.25:
            time.sleep(0.25 - (now - _last_request_time))
        _last_request_time = time.monotonic()


def _clean_drug_name(query: str) -> str:
    clean = re.sub(
        r"\b(what|is|the|drug|for|of|and|with|side|effects?|dose|dosage|"
        r"indication|contraindication|interaction|mechanism|action|"
        r"farmaco|indicazion[ei]|dosaggio|controindicazion[ei]|"
        r"interazion[ei]|meccanismo|azione|effett[oi]|collateral[ie])\b",
        " ", query, flags=re.IGNORECASE,
    )
    clean = re.sub(r"[\"'()[\]{}<>?!.,;:]", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    tokens = [t for t in clean.split() if len(t) > 2]
    return " ".join(tokens[:3])


class DailyMedProvider(SourceProvider):
    """Search DailyMed for FDA drug labeling (free, NLM)."""

    source_type = "dailymed"

    # DailyMed indexes one SPL per *labeler*, so a single-substance query
    # ("dabigatran") routinely returns several near-identical labels - the
    # same approved text filed separately by each generic manufacturer. This
    # pipeline asks DailyMed "does an FDA label exist and what does it say",
    # which one representative label answers; the rest would only occupy
    # result slots with content a reader has already seen once.
    _MAX_LABELS_PER_SUBSTANCE = 1

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
    )
    def search(self, query: str, max_results: int = 3) -> list[SourceResult]:
        if not query or len(query) < 2:
            return []

        drug_name = _clean_drug_name(query)
        if not drug_name:
            return []

        spls = self._search_spls(drug_name)
        if not spls:
            # Try just the first word (likely the drug name)
            first_word = drug_name.split()[0] if drug_name.split() else ""
            if first_word and first_word != drug_name:
                spls = self._search_spls(first_word)

        if not spls:
            return []

        cap = min(max_results, self._MAX_LABELS_PER_SUBSTANCE)
        results: list[SourceResult] = []
        for spl in spls[:cap]:
            result = self._fetch_spl_details(spl)
            if result:
                results.append(result)

        return results

    def _search_spls(self, drug_name: str) -> list[dict]:
        _rate_limit()
        try:
            resp = httpx.get(
                f"{_API_BASE}/spls.json",
                params={"drug_name": drug_name, "page": 1, "pagesize": 5},
                headers=_headers(),
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.debug("DailyMed SPL search failed: %s", exc)
            return []

        return data.get("data", [])

    def _fetch_spl_details(self, spl: dict) -> SourceResult | None:
        set_id = spl.get("setid", "")
        title = spl.get("title", "") or spl.get("spl_name", "")
        if not set_id or not title:
            return None

        url = f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}"

        published = spl.get("published_date", "") or ""
        products = spl.get("products", [])
        product_names: list[str] = []
        for p in products[:3] if isinstance(products, list) else []:
            pname = p.get("name", "") if isinstance(p, dict) else ""
            if pname:
                product_names.append(pname)

        snippet_parts = [f"FDA Drug Label: {title}"]
        if product_names:
            snippet_parts.append(f"Products: {', '.join(product_names)}")
        if published:
            snippet_parts.append(f"Published: {published}")

        content = self._fetch_spl_content(set_id, title)

        return SourceResult(
            title=title,
            url=url,
            snippet=" | ".join(snippet_parts)[:300],
            content=content[:4000],
            source_type=self.source_type,
            reliability_tier=1,
            publication_date=published,
            language="en",
        )

    def _fetch_spl_content(self, set_id: str, title: str) -> str:
        _rate_limit()
        try:
            # The XML detail route 406s on `Accept: application/xml` - it only
            # accepts a wildcard Accept, unlike every other DailyMed endpoint.
            resp = httpx.get(
                f"{_API_BASE}/spls/{set_id}.xml",
                headers=build_headers(accept="*/*"),
                timeout=_TIMEOUT,
            )
            if resp.status_code != 200:
                return f"FDA Drug Label: {title} (details unavailable)"
            root = ElementTree.fromstring(resp.content)
        except Exception:
            return f"FDA Drug Label: {title} (details unavailable)"

        sections: dict[str, str] = {}
        for section in root.iter(f"{_NS}section"):
            code_el = section.find(f"{_NS}code")
            if code_el is None:
                continue
            field = _SECTION_CODES.get(code_el.get("code", ""))
            if not field or field in sections:
                continue
            # Some sections (e.g. Dosage & Administration) carry no direct
            # <text>, only a condensed <excerpt><highlight><text> - the rest
            # lives in nested subsections under unrelated codes. `Element`
            # is falsy when childless, so an explicit None-check is required
            # here rather than `find(...) or find(...)`.
            text_el = section.find(f"{_NS}text")
            if text_el is None:
                text_el = section.find(f"{_NS}excerpt/{_NS}highlight/{_NS}text")
            if text_el is None:
                continue
            text = re.sub(r"\s+", " ", "".join(text_el.itertext())).strip()
            if text:
                sections[field] = text[:500]

        parts: list[str] = [f"FDA Drug Label: {title}\n"]
        for field, label in _INTERESTING_SECTIONS.items():
            text = sections.get(field)
            if text:
                parts.append(f"## {label}\n{text}\n")

        return "\n".join(parts)
