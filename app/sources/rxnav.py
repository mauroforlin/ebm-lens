"""RxNav / RxNorm provider - free, no API key required.

Official NIH/NLM drug naming authority.  Resolves drug names (generic ↔
brand) and provides drug classes.

Does not provide drug-drug interactions (BUG-04): NLM retired the interaction
API on 2 Jan 2024 when its DrugBank data license lapsed, and it now 404s
unconditionally - there is no workaround, the data source is gone.
Interaction/contraindication text instead comes from the FDA label sections
that openfda and dailymed already carry.

API docs: https://lhncbc.nlm.nih.gov/RxNav/APIs/
Rate limit: 20 requests/second.
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

_API_BASE = "https://rxnav.nlm.nih.gov/REST"
_TIMEOUT = 10
def _headers() -> dict[str, str]:
    """Outbound headers, built per call so settings stay lazily loaded."""
    return build_headers(accept="application/json")

# Rate limiter: stay at ~8 req/s (well under 20/s)
_rate_lock = threading.Lock()
_last_request_time = 0.0


def _rate_limit() -> None:
    global _last_request_time
    with _rate_lock:
        now = time.monotonic()
        if now - _last_request_time < 0.13:
            time.sleep(0.13 - (now - _last_request_time))
        _last_request_time = time.monotonic()


def _extract_drug_name(query: str) -> str:
    # Remove common filler
    clean = re.sub(
        r"\b(what|is|the|drug|interaction|between|of|and|with|for|"
        r"effects?|side|adverse|dose|dosage|indication|generic|brand|"
        r"farmaco|interazione|tra|effett[oi]|collateral[ie]|"
        r"indicazion[ei]|dosaggio)\b",
        " ", query, flags=re.IGNORECASE,
    )
    clean = re.sub(r"[\"'()[\]{}<>?!.,;:]", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    # Return top tokens (likely the drug name(s))
    tokens = [t for t in clean.split() if len(t) > 2]
    return " ".join(tokens[:4])


class RxNavProvider(SourceProvider):
    """Search RxNorm for drug identity and classification info (free, NLM)."""

    source_type = "rxnav"

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
    )
    def search(self, query: str, max_results: int = 3) -> list[SourceResult]:
        if not query or len(query) < 2:
            return []

        drug_name = _extract_drug_name(query)
        if not drug_name:
            return []

        return self._search_drug(drug_name)[:max_results]

    def _resolve_rxcui(self, name: str) -> str | None:
        _rate_limit()
        try:
            resp = httpx.get(
                f"{_API_BASE}/rxcui.json",
                params={"name": name, "search": 2},  # search=2 = normalized
                headers=_headers(),
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            ids = data.get("idGroup", {}).get("rxnormId", [])
            return ids[0] if ids else None
        except Exception as exc:
            logger.debug("RxNav rxcui resolve failed for '%s': %s", name, exc)
            return None

    def resolve_drug(self, name: str) -> dict:
        """Resolve a drug name (brand or generic) to its active molecule.

        Returns a dict with ``rxnorm_id``, ``ingredient``, ``molecule`` and
        ``brand_names``, or ``{"matched": False}`` when the name resolves to
        nothing.
        """
        name = (name or "").strip()
        if not name:
            return {"query": "", "matched": False, "error": "empty name"}

        rxcui = self._resolve_rxcui(name)
        if not rxcui:
            return {"query": name, "matched": False}

        ingredient = ""
        brands: list[str] = []
        for tty in ("IN", "BN"):
            try:
                _rate_limit()
                resp = httpx.get(
                    f"{_API_BASE}/rxcui/{rxcui}/related.json",
                    params={"tty": tty},
                    headers=_headers(),
                    timeout=_TIMEOUT,
                )
                if resp.status_code != 200:
                    continue
                for cg in resp.json().get("relatedGroup", {}).get("conceptGroup", []):
                    names = [
                        p.get("name", "")
                        for p in cg.get("conceptProperties", [])
                        if p.get("name")
                    ]
                    if not names:
                        continue
                    if tty == "IN":
                        ingredient = names[0]
                    else:
                        brands = names[:5]
            except Exception as exc:
                logger.debug("RxNav related lookup failed (%s): %s", tty, exc)

        return {
            "query": name,
            "matched": True,
            "rxnorm_id": rxcui,
            "ingredient": ingredient,
            "molecule": ingredient or name,
            "brand_names": brands,
        }

    def _search_drug(self, drug_name: str) -> list[SourceResult]:
        _rate_limit()
        try:
            resp = httpx.get(
                f"{_API_BASE}/drugs.json",
                params={"name": drug_name},
                headers=_headers(),
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.debug("RxNav drug search failed: %s", exc)
            return []

        groups = data.get("drugGroup", {})
        concept_groups = groups.get("conceptGroup", [])
        if not concept_groups:
            # Try approximate term
            return self._search_approximate(drug_name)

        results: list[SourceResult] = []
        for cg in concept_groups:
            tty = cg.get("tty", "")
            props = cg.get("conceptProperties", [])
            if not props:
                continue
            for prop in props[:2]:  # max 2 per type
                rxcui = prop.get("rxcui", "")
                name = prop.get("name", "")
                synonym = prop.get("synonym", "")
                if not name:
                    continue

                content_parts = [
                    f"Drug: {name}",
                    f"RxCUI: {rxcui}",
                    f"Term Type: {tty}",
                ]
                if synonym:
                    content_parts.append(f"Synonym: {synonym}")

                extra = self._get_drug_properties(rxcui)
                if extra:
                    content_parts.extend(extra)

                results.append(SourceResult(
                    title=f"{name} - RxNorm Drug Information",
                    url=f"https://mor.nlm.nih.gov/RxNav/search?searchBy=RXCUI&searchTerm={rxcui}",
                    snippet=f"RxNorm: {name} (RxCUI: {rxcui}, type: {tty})",
                    content="\n".join(content_parts)[:4000],
                    source_type=self.source_type,
                    reliability_tier=1,
                    language="en",
                ))

        return results

    def _search_approximate(self, name: str) -> list[SourceResult]:
        _rate_limit()
        try:
            resp = httpx.get(
                f"{_API_BASE}/approximateTerm.json",
                params={"term": name, "maxEntries": 3},
                headers=_headers(),
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []

        candidates = data.get("approximateGroup", {}).get("candidate", [])
        results: list[SourceResult] = []
        for c in candidates[:2]:
            rxcui = c.get("rxcui", "")
            score = c.get("score", "")
            name_val = c.get("name", "") or rxcui

            results.append(SourceResult(
                title=f"{name_val} - RxNorm (approximate match)",
                url=f"https://mor.nlm.nih.gov/RxNav/search?searchBy=RXCUI&searchTerm={rxcui}",
                snippet=f"Approximate match: {name_val} (score: {score})",
                content=f"Drug: {name_val}\nRxCUI: {rxcui}\nMatch score: {score}",
                source_type=self.source_type,
                reliability_tier=1,
                language="en",
            ))
        return results

    def _get_drug_properties(self, rxcui: str) -> list[str]:
        if not rxcui:
            return []
        _rate_limit()
        parts: list[str] = []

        try:
            # Fetch related concepts (ingredients, dose forms, etc.)
            resp = httpx.get(
                f"{_API_BASE}/rxcui/{rxcui}/allrelated.json",
                headers=_headers(),
                timeout=_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                groups = data.get("allRelatedGroup", {}).get("conceptGroup", [])
                for g in groups:
                    tty = g.get("tty", "")
                    props = g.get("conceptProperties", [])
                    if props and tty in ("IN", "BN", "DF", "DFG", "SCDC"):
                        names = [p.get("name", "") for p in props[:3]]
                        label = {
                            "IN": "Ingredient", "BN": "Brand Name",
                            "DF": "Dose Form", "DFG": "Dose Form Group",
                            "SCDC": "Strength",
                        }.get(tty, tty)
                        parts.append(f"{label}: {', '.join(names)}")
        except Exception:
            pass
        return parts
