"""Open Targets Platform provider - free, no API key required.

Drug–target–disease associations with scored evidence from 20+ data
sources.  Excellent for verifying "drug X treats disease Y",
mechanism-of-action claims, and adverse-event associations.

API docs: https://platform-docs.opentargets.org/data-access/graphql-api
Rate limit: No hard limit; ~10 req/s recommended.
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

_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
_TIMEOUT = 12
def _headers() -> dict[str, str]:
    """Outbound headers, built per call so settings stay lazily loaded."""
    return build_headers(accept="application/json", extra={"Content-Type": "application/json"})

# Rate limiter: ~6 req/s
_rate_lock = threading.Lock()
_last_request_time = 0.0


def _rate_limit() -> None:
    global _last_request_time
    with _rate_lock:
        now = time.monotonic()
        if now - _last_request_time < 0.17:
            time.sleep(0.17 - (now - _last_request_time))
        _last_request_time = time.monotonic()


def _extract_keywords(query: str) -> str:
    clean = re.sub(r"[\"'()[\]{}<>?!.,;:]", " ", query)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:200]


_SEARCH_QUERY = """
query Search($q: String!, $size: Int!) {
  search(queryString: $q, entityNames: ["drug", "disease", "target"], page: {index: 0, size: $size}) {
    total
    hits {
      id
      name
      entity
      description
    }
  }
}
"""

# BUG-05 workaround: this query's field names were updated to match a schema
# migration - the old names (maximumClinicalTrialPhase, hasBeenWithdrawn,
# maxPhaseForIndication, indications.rows.references) were dropped from the
# GraphQL schema and now error instead of returning null.
_DRUG_QUERY = """
query DrugInfo($chemblId: String!) {
  drug(chemblId: $chemblId) {
    id
    name
    drugType
    maximumClinicalStage
    drugWarnings {
      warningType
      description
      year
      country
    }
    description
    mechanismsOfAction {
      rows {
        mechanismOfAction
        targets {
          approvedName
          approvedSymbol
        }
        references {
          source
          urls
        }
      }
    }
    indications {
      count
      rows {
        disease {
          id
          name
        }
        maxClinicalStage
      }
    }
    adverseEvents {
      count
      rows {
        name
        count
        logLR
        meddraCode
      }
    }
  }
}
"""


class OpenTargetsProvider(SourceProvider):
    """Search Open Targets for drug–disease associations (free, GraphQL)."""

    source_type = "open_targets"

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
    )
    def search(self, query: str, max_results: int = 3) -> list[SourceResult]:
        if not query or len(query) < 3:
            return []

        keywords = _extract_keywords(query)
        if not keywords:
            return []

        # Step 1: Search for matching entities
        hits = self._search_entities(keywords, max_results * 2)
        if not hits:
            return []

        results: list[SourceResult] = []

        # Step 2: For drug hits, fetch detailed drug info
        drug_hits = [h for h in hits if h.get("entity") == "drug"]
        other_hits = [h for h in hits if h.get("entity") != "drug"]

        for hit in drug_hits[:max_results]:
            result = self._fetch_drug_details(hit)
            if result:
                results.append(result)

        # For non-drug hits, use the search result directly
        for hit in other_hits[:max(1, max_results - len(results))]:
            result = self._hit_to_result(hit)
            if result:
                results.append(result)

        return results[:max_results]

    def _gql(self, query: str, variables: dict) -> dict | None:
        _rate_limit()
        try:
            resp = httpx.post(
                _GRAPHQL_URL,
                json={"query": query, "variables": variables},
                timeout=_TIMEOUT,
                headers=_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("data")
        except Exception as exc:
            logger.debug("Open Targets GraphQL failed: %s", exc)
            return None

    def _search_entities(self, keywords: str, size: int) -> list[dict]:
        data = self._gql(_SEARCH_QUERY, {"q": keywords, "size": size})
        if not data:
            return []
        return data.get("search", {}).get("hits", [])

    def _fetch_drug_details(self, hit: dict) -> SourceResult | None:
        chembl_id = hit.get("id", "")
        name = hit.get("name", "")
        if not chembl_id:
            return None

        data = self._gql(_DRUG_QUERY, {"chemblId": chembl_id})
        if not data or not data.get("drug"):
            return self._hit_to_result(hit)

        drug = data["drug"]
        drug_name = drug.get("name", name)
        drug_type = drug.get("drugType", "")
        max_phase = drug.get("maximumClinicalStage", 0)
        warnings = drug.get("drugWarnings", []) or []
        description = drug.get("description", "") or ""

        url = f"https://platform.opentargets.org/drug/{chembl_id}"

        content_parts: list[str] = [
            f"Drug: {drug_name} ({chembl_id})",
            f"Type: {drug_type}",
            f"Max clinical trial phase: {max_phase}",
        ]
        for w in warnings[:3]:
            line = f"⚠ {w.get('warningType', 'Warning')}"
            if w.get("description"):
                line += f": {w['description']}"
            if w.get("year"):
                line += f" [{w['year']}]"
            content_parts.append(line)
        if description:
            content_parts.append(f"\nDescription: {description[:500]}")

        moa = drug.get("mechanismsOfAction", {})
        moa_rows = moa.get("rows", []) if moa else []
        if moa_rows:
            content_parts.append("\n## Mechanisms of Action")
            for row in moa_rows[:5]:
                mech = row.get("mechanismOfAction", "")
                targets = row.get("targets", [])
                target_names = [t.get("approvedName", "") for t in targets[:3]]
                line = f"- {mech}"
                if target_names:
                    line += f" (targets: {', '.join(target_names)})"
                content_parts.append(line)

        indications = drug.get("indications", {})
        ind_rows = indications.get("rows", []) if indications else []
        if ind_rows:
            content_parts.append(f"\n## Indications ({indications.get('count', 0)} total)")
            for row in ind_rows[:8]:
                disease = row.get("disease", {})
                dname = disease.get("name", "")
                phase = row.get("maxClinicalStage", 0)
                content_parts.append(f"- {dname} (phase {phase})")

        ae = drug.get("adverseEvents", {})
        ae_rows = ae.get("rows", []) if ae else []
        if ae_rows:
            content_parts.append(f"\n## Adverse Events ({ae.get('count', 0)} total)")
            for row in ae_rows[:8]:
                ae_name = row.get("name", "")
                ae_count = row.get("count", 0)
                log_lr = row.get("logLR", 0)
                content_parts.append(
                    f"- {ae_name} (reports: {ae_count}, logLR: {log_lr:.2f})"
                )

        snippet = f"Open Targets: {drug_name}"
        if moa_rows:
            snippet += f" | {len(moa_rows)} mechanisms"
        if ind_rows:
            snippet += f" | {len(ind_rows)} indications"

        return SourceResult(
            title=f"{drug_name} - Open Targets Platform",
            url=url,
            snippet=snippet[:300],
            content="\n".join(content_parts)[:4000],
            source_type=self.source_type,
            reliability_tier=2,
            language="en",
        )

    def _hit_to_result(self, hit: dict) -> SourceResult | None:
        entity_id = hit.get("id", "")
        name = hit.get("name", "")
        entity = hit.get("entity", "")
        description = hit.get("description", "") or ""

        if not name or not entity_id:
            return None

        entity_paths = {
            "drug": "drug",
            "disease": "disease",
            "target": "target",
        }
        path = entity_paths.get(entity, "search")
        url = f"https://platform.opentargets.org/{path}/{entity_id}"

        return SourceResult(
            title=f"{name} - Open Targets ({entity})",
            url=url,
            snippet=f"Open Targets {entity}: {name}",
            content=f"{entity.title()}: {name}\nID: {entity_id}\n\n{description}"[:4000],
            source_type=self.source_type,
            reliability_tier=2,
            language="en",
        )
