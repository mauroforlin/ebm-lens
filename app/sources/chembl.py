


"""ChEMBL provider - free, no API key required.

EMBL-EBI curated bioactivity database.  Covers drug mechanism of action,
indications, warnings/withdrawals, and clinical phase data.

API docs: https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services
Rate limit: EBI fair-use (~10 req/s recommended).
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

_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"
_TIMEOUT = 12
def _headers() -> dict[str, str]:
    """Outbound headers, built per call so settings stay lazily loaded."""
    return build_headers(accept="application/json")

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


def _clean_drug_name(query: str) -> str:
    clean = re.sub(
        r"\b(what|is|the|drug|for|of|and|with|mechanism|action|"
        r"side|effects?|warning|withdrawal|indication|"
        r"farmaco|meccanismo|azione|effett[oi]|avvertenz[ae]|"
        r"indicazion[ei]|ritir[oa])\b",
        " ", query, flags=re.IGNORECASE,
    )
    clean = re.sub(r"[\"'()[\]{}<>?!.,;:]", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    tokens = [t for t in clean.split() if len(t) > 2]
    return " ".join(tokens[:3])


class ChEMBLProvider(SourceProvider):
    """Search ChEMBL for drug mechanism, indications, and warnings."""

    source_type = "chembl"

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

        # Step 1: Find molecule by name
        search_name = drug_name
        molecules = self._search_molecules(drug_name)
        if not molecules:
            # Try first word only
            first = drug_name.split()[0] if drug_name.split() else ""
            if first and first != drug_name:
                search_name = first
                molecules = self._search_molecules(first)

        if not molecules:
            return []

        # ChEMBL's search ranking often puts unnamed research compounds
        # (no pref_name, no max_phase) ahead of the approved drug being
        # looked up, so re-rank before truncating to max_results: named
        # molecules first, exact name matches first among those, then by
        # clinical phase (approved/later-phase drugs first).
        def _phase(m: dict) -> float:
            try:
                return float(m.get("max_phase"))
            except (TypeError, ValueError):
                return -1.0

        search_name_lower = search_name.lower()
        molecules = sorted(
            molecules,
            key=lambda m: (
                m.get("pref_name") is None,
                (m.get("pref_name") or "").lower() != search_name_lower,
                -_phase(m),
            ),
        )

        # Unnamed research compounds add no value once a real, named molecule
        # matched - padding the slice with them just displaces it with noise.
        # Only fall back to unnamed hits when nothing named came back at all.
        named = [m for m in molecules if m.get("pref_name")]
        selected = named if named else molecules

        results: list[SourceResult] = []
        for mol in selected[:max_results]:
            result = self._build_molecule_result(mol)
            if result:
                results.append(result)

        return results

    def _search_molecules(self, name: str) -> list[dict]:
        _rate_limit()
        try:
            resp = httpx.get(
                f"{_API_BASE}/molecule/search.json",
                params={"q": name, "limit": 5},
                headers=_headers(),
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.debug("ChEMBL molecule search failed: %s", exc)
            return []

        return data.get("molecules", [])

    def _build_molecule_result(self, mol: dict) -> SourceResult | None:
        chembl_id = mol.get("molecule_chembl_id", "")
        pref_name = mol.get("pref_name", "") or ""
        if not chembl_id:
            return None

        mol_type = mol.get("molecule_type", "")
        max_phase = mol.get("max_phase", 0)
        first_approval = mol.get("first_approval", "")
        withdrawn = mol.get("withdrawn_flag", False)
        oral = mol.get("oral", False)
        parenteral = mol.get("parenteral", False)
        topical = mol.get("topical", False)
        black_box = mol.get("black_box_warning", 0)

        url = f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_id}/"

        content_parts: list[str] = [
            f"Drug: {pref_name or chembl_id}",
            f"ChEMBL ID: {chembl_id}",
            f"Type: {mol_type}",
            f"Max clinical phase: {max_phase}",
        ]
        if first_approval:
            content_parts.append(f"First approval: {first_approval}")
        if withdrawn:
            content_parts.append("⚠ WITHDRAWN FROM MARKET")
        if black_box:
            content_parts.append("⚠ BLACK BOX WARNING")

        routes: list[str] = []
        if oral:
            routes.append("oral")
        if parenteral:
            routes.append("parenteral")
        if topical:
            routes.append("topical")
        if routes:
            content_parts.append(f"Routes: {', '.join(routes)}")

        mechanisms = self._fetch_mechanisms(chembl_id)
        if mechanisms:
            content_parts.append("\n## Mechanism of Action")
            content_parts.extend(mechanisms)

        indications = self._fetch_indications(chembl_id)
        if indications:
            content_parts.append("\n## Indications")
            content_parts.extend(indications)

        warnings = self._fetch_warnings(chembl_id)
        if warnings:
            content_parts.append("\n## Warnings / Withdrawals")
            content_parts.extend(warnings)

        snippet = f"ChEMBL: {pref_name or chembl_id} (phase {max_phase})"
        if withdrawn:
            snippet += " [WITHDRAWN]"
        if black_box:
            snippet += " [BLACK BOX]"

        return SourceResult(
            title=f"{pref_name or chembl_id} - ChEMBL Drug Report",
            url=url,
            snippet=snippet[:300],
            content="\n".join(content_parts)[:4000],
            source_type=self.source_type,
            reliability_tier=1,
            language="en",
        )

    def _fetch_mechanisms(self, chembl_id: str) -> list[str]:
        _rate_limit()
        try:
            resp = httpx.get(
                f"{_API_BASE}/mechanism.json",
                params={"molecule_chembl_id": chembl_id, "limit": 5},
                headers=_headers(),
                timeout=_TIMEOUT,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
        except Exception:
            return []

        rows = data.get("mechanisms", [])
        parts: list[str] = []
        for row in rows[:5]:
            mech = row.get("mechanism_of_action", "")
            action_type = row.get("action_type", "")
            target_name = row.get("target_chembl_id", "")
            line = f"- {mech}"
            if action_type:
                line += f" ({action_type})"
            if target_name:
                line += f" [target: {target_name}]"
            parts.append(line)
        return parts

    def _fetch_indications(self, chembl_id: str) -> list[str]:
        _rate_limit()
        try:
            resp = httpx.get(
                f"{_API_BASE}/drug_indication.json",
                params={"molecule_chembl_id": chembl_id, "limit": 10},
                headers=_headers(),
                timeout=_TIMEOUT,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
        except Exception:
            return []

        rows = data.get("drug_indications", [])
        parts: list[str] = []
        for row in rows[:8]:
            mesh_heading = row.get("mesh_heading", "")
            max_phase = row.get("max_phase_for_ind", "")
            efo_term = row.get("efo_term", "")
            name = mesh_heading or efo_term or "Unknown"
            parts.append(f"- {name} (phase {max_phase})")
        return parts

    def _fetch_warnings(self, chembl_id: str) -> list[str]:
        _rate_limit()
        try:
            resp = httpx.get(
                f"{_API_BASE}/drug_warning.json",
                params={"molecule_chembl_id": chembl_id, "limit": 5},
                headers=_headers(),
                timeout=_TIMEOUT,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
        except Exception:
            return []

        rows = data.get("drug_warnings", [])
        parts: list[str] = []
        for row in rows[:5]:
            warning_type = row.get("warning_type", "")
            warning_class = row.get("warning_class", "")
            warning_desc = row.get("warning_description", "")
            year = row.get("warning_year", "")
            country = row.get("warning_country", "")
            line = f"- {warning_type}"
            if warning_class:
                line += f" ({warning_class})"
            if warning_desc:
                line += f": {warning_desc}"
            if year:
                line += f" [{year}]"
            if country:
                line += f" [{country}]"
            parts.append(line)
        return parts
