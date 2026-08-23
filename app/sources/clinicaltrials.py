"""ClinicalTrials.gov API v2 provider - free, no API key required.

Searches for clinical trials related to medical claims. Useful for:
- Verifying treatment efficacy claims
- Finding drug comparison studies (RCTs)
- Checking clinical trial phases and results
- Validating drug indication claims

API docs: https://clinicaltrials.gov/data-api/api
"""
from __future__ import annotations

import logging
import re

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import SourceProvider, SourceResult, build_headers

logger = logging.getLogger(__name__)

_API_BASE = "https://clinicaltrials.gov/api/v2/studies"
_TIMEOUT = 15
def _headers() -> dict[str, str]:
    """Outbound headers, built per call so settings stay lazily loaded."""
    return build_headers(accept="application/json")


class ClinicalTrialsProvider(SourceProvider):
    """Search ClinicalTrials.gov for clinical trial evidence."""

    source_type = "clinicaltrials"

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
    )
    def search(self, query: str, max_results: int = 3) -> list[SourceResult]:
        """Search ClinicalTrials.gov API v2.

        Prioritises completed trials with results. Extracts:
        - Study title & brief summary
        - Study type (interventional, observational)
        - Phase (1-4)
        - Enrollment count
        - Primary outcomes (if results posted)
        """
        if not query or len(query) < 3:
            return []

        # Clean query for API
        clean_q = self._clean_query(query)

        # Explicit relevance sort - the API's default order (no sort param)
        # is not relevance, it just happens to look plausible. Sorting by
        # edit recency instead would surface whatever trial was last touched
        # in the registry, regardless of how well it matches the query.
        params = {
            "query.term": clean_q,
            "pageSize": min(max_results * 2, 10),  # over-fetch to filter
            "sort": "@relevance",
        }

        try:
            with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
                resp = client.get(_API_BASE, params=params, headers=_headers())
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.debug("ClinicalTrials.gov search failed: %s", exc)
            raise

        studies = data.get("studies", [])
        if not studies:
            return []

        results: list[SourceResult] = []
        for study in studies:
            try:
                result = self._parse_study(study)
                if result:
                    results.append(result)
            except Exception as exc:
                logger.debug("Failed to parse study: %s", exc)

        # Prioritise completed studies with results
        results.sort(key=self._study_priority, reverse=True)
        return results[:max_results]

    def _clean_query(self, query: str) -> str:
        """Clean and optimise query for ClinicalTrials.gov."""
        # Remove overly specific syntax
        query = re.sub(r'["\[\]()]', ' ', query)
        query = re.sub(r'\s+', ' ', query).strip()
        # Keep max 8 terms for better recall
        tokens = query.split()[:8]
        return " ".join(tokens)

    def _parse_study(self, study: dict) -> SourceResult | None:
        """Parse a study JSON object into a SourceResult."""
        proto = study.get("protocolSection", {})
        if not proto:
            return None

        ident = proto.get("identificationModule", {})
        nct_id = ident.get("nctId", "")
        brief_title = ident.get("briefTitle", "")
        official_title = ident.get("officialTitle", "")

        desc = proto.get("descriptionModule", {})
        summary = desc.get("briefSummary", "")

        status_mod = proto.get("statusModule", {})
        overall_status = status_mod.get("overallStatus", "")
        start_date = status_mod.get("startDateStruct", {}).get("date", "")
        completion_date = status_mod.get("completionDateStruct", {}).get("date", "")

        design = proto.get("designModule", {})
        study_type = design.get("studyType", "")
        phases_list = design.get("phases", [])
        phase = ", ".join(phases_list) if phases_list else "N/A"
        enrollment = design.get("enrollmentInfo", {}).get("count", "N/A")

        conditions_mod = proto.get("conditionsModule", {})
        conditions = conditions_mod.get("conditions", [])

        interventions_mod = proto.get("armsInterventionsModule", {})
        interventions = interventions_mod.get("interventions", [])
        intervention_names = [
            i.get("name", "") for i in interventions if i.get("name")
        ]

        outcomes_mod = proto.get("outcomesModule", {})
        primary_outcomes = outcomes_mod.get("primaryOutcomes", [])
        outcome_measures = [
            o.get("measure", "") for o in primary_outcomes if o.get("measure")
        ]

        has_results = study.get("hasResults", False)

        content_parts = [
            f"Study: {official_title or brief_title}",
            f"NCT ID: {nct_id}",
            f"Status: {overall_status}",
            f"Type: {study_type}, Phase: {phase}",
            f"Enrollment: {enrollment}",
        ]
        if conditions:
            content_parts.append(f"Conditions: {', '.join(conditions[:5])}")
        if intervention_names:
            content_parts.append(f"Interventions: {', '.join(intervention_names[:5])}")
        if outcome_measures:
            content_parts.append(f"Primary outcomes: {'; '.join(outcome_measures[:3])}")
        if start_date:
            content_parts.append(f"Period: {start_date} → {completion_date or 'ongoing'}")
        if has_results:
            content_parts.append("⚡ Has posted results")

        content_parts.append(f"\nSummary: {summary}")

        url = f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else ""

        snippet_parts = [brief_title]
        if conditions:
            snippet_parts.append(f"({', '.join(conditions[:3])})")
        snippet_parts.append(f"[{overall_status}, {phase}]")

        pub_date = start_date or completion_date or ""

        return SourceResult(
            title=brief_title or official_title or nct_id,
            url=url,
            snippet=" ".join(snippet_parts)[:300],
            content="\n".join(content_parts)[:4000],
            source_type=self.source_type,
            reliability_tier=1,
            publication_date=pub_date,
            language="en",
        )

    @staticmethod
    def _study_priority(result: SourceResult) -> int:
        """Score for sorting: completed with results > completed > recruiting > other."""
        content = result.content.lower()
        score = 0
        if "has posted results" in content:
            score += 100
        if "completed" in content:
            score += 50
        elif "active" in content:
            score += 30
        elif "recruiting" in content:
            score += 20
        if "phase 3" in content or "phase3" in content:
            score += 15
        elif "phase 4" in content or "phase4" in content:
            score += 12
        elif "phase 2" in content or "phase2" in content:
            score += 8
        # Larger enrollment = more reliable
        try:
            import re
            m = re.search(r"Enrollment: (\d+)", result.content)
            if m:
                n = int(m.group(1))
                if n > 1000:
                    score += 20
                elif n > 100:
                    score += 10
        except Exception:
            pass
        return score
