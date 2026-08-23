"""EMA provider - free, no API key required.

The European Medicines Agency is the regulator for medicines authorised
centrally across the EU. Where OpenFDA and DailyMed say what the FDA
approved, this says what the European Commission approved: the EPAR record
for every centrally authorised human and veterinary medicine, with its
therapeutic indication, ATC code, authorisation status and the regulatory
flags that qualify it - orphan, conditional approval, exceptional
circumstances, additional monitoring, biosimilar, generic.

EMA publishes no query API. What it publishes is the whole medicines table
as a JSON file, refreshed twice a day at 06:00 and 18:00 Amsterdam time,
explicitly for automated consumption:

    https://www.ema.europa.eu/en/about-us/about-website/download-website-data-json-data-format

So this provider is not a client of a search engine, it *is* the search
engine: the file is fetched once, indexed in memory and queried locally.
Roughly 2,700 records and one download per day, after which a query costs no
network at all. The dataset is the agency's own public output and reusable
with attribution.

The first search after a restart pays for the download, so it runs under its
own timeout and a failure returns nothing rather than raising - a cold index
must cost a European regulatory record, never the whole run. A failed load is
remembered briefly, because retrying a 7 MB fetch on every query of a broken
network is how one dead source becomes a dead search.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field

import httpx

from .base import SourceProvider, SourceResult, build_headers

logger = logging.getLogger(__name__)

_DATASET_URL = (
    "https://www.ema.europa.eu/en/documents/report/"
    "medicines-output-medicines_json-report_en.json"
)
# The dataset is refreshed twice daily; a download costs seconds, so the index
# is held for a day rather than chased.
_INDEX_TTL_SECONDS = 24 * 3600
# After a failed load, how long to serve nothing before trying the network again.
_FAILURE_BACKOFF_SECONDS = 300
_DOWNLOAD_TIMEOUT = 30


def _headers() -> dict[str, str]:
    """Outbound headers, built per call so settings stay lazily loaded."""
    return build_headers(accept="application/json")


# ── Query and record matching ─────────────────────────────────

# Words that carry no discriminating power in a query against a drug
# register, in the two languages this tool is asked questions in.
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "what", "how", "does", "drug", "medicine",
    "medicinal", "product", "treatment", "therapy", "use", "used", "uses",
    "approved", "authorised", "authorized", "approval", "europe", "european",
    "eu", "ema", "che", "cosa", "come", "per", "con", "del", "della", "dei",
    "delle", "farmaco", "farmaci", "medicinale", "trattamento", "terapia",
    "indicazione", "indicazioni", "uso", "approvato", "autorizzato", "europa",
    "europea",
})

# An ATC code is a letter, two digits, then up to two letters and two digits
# (L04AC07). Recognised as a unit so it is matched as a code rather than
# tokenised into noise.
_ATC_PATTERN = re.compile(r"^[a-z]\d{2}[a-z]{0,2}\d{0,2}$")

# Below this a record is matched only by incidental words and is not evidence.
# One query term found in an indication scores 0.5, which is the weakest match
# worth returning: "glioblastoma" finds temozolomide that way and should.
_SCORE_FLOOR = 0.4


@dataclass(frozen=True)
class _Term:
    """One query term, with the matcher used against every indexed field.

    Matching is anchored at a word start rather than run as a bare substring:
    "cosa cura Ozempic" otherwise finds the veterinary medicine Di*cural*.
    Left-anchored only, so "temozolomide" still matches "Temozolomide Accord"
    and an inflected form still matches its stem.
    """

    text: str
    pattern: re.Pattern[str]
    is_atc: bool


def _terms(text: str) -> list[_Term]:
    """Split a query into the terms worth matching a drug register on."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [
        _Term(t, re.compile(rf"\b{re.escape(t)}"), bool(_ATC_PATTERN.match(t)))
        for t in tokens
        if len(t) > 2 and t not in _STOPWORDS
    ]


def _iso_date(value: str) -> str:
    """Convert EMA's ``dd/mm/yyyy`` into an ISO date, or return ''."""
    match = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", (value or "").strip())
    if not match:
        return ""
    day, month, year = match.groups()
    return f"{year}-{month}-{day}"


@dataclass
class _Medicine:
    """One EPAR record, reduced to the fields worth holding in memory."""

    name: str
    inn: str
    active_substance: str
    atc: str
    status: str
    category: str            # "Human" or "Veterinary"
    therapeutic_area: str
    pharmacotherapeutic_group: str
    indication: str
    holder: str
    product_number: str
    authorisation_date: str  # ISO
    last_updated: str        # ISO
    url: str
    flags: list[str] = field(default_factory=list)
    is_reference: bool = True   # neither generic nor biosimilar

    # Lowercased match surfaces, precomputed once at index build.
    _name_l: str = ""
    _substance_l: str = ""
    _atc_l: str = ""
    _area_l: str = ""
    _indication_l: str = ""

    def score(self, terms: list[_Term]) -> float:
        """How well this record answers a query, as a mean over its terms.

        The weights encode what a match on each field is worth: a medicine's
        own name or its INN identifies it, an ATC code classifies it exactly,
        a therapeutic area places it, and a word in the indication text is a
        hint. Dividing by the term count keeps a long query from outscoring a
        precise one.
        """
        if not terms:
            return 0.0

        total = 0.0
        for term in terms:
            if term.text == self._name_l:
                total += 6.0
            elif term.pattern.search(self._name_l):
                total += 3.0
            if term.pattern.search(self._substance_l):
                total += 4.0
            if term.is_atc and term.text in self._atc_l:
                total += 4.0
            if term.pattern.search(self._area_l):
                total += 1.5
            if term.pattern.search(self._indication_l):
                total += 0.5

        total /= len(terms)

        # A withdrawn or refused authorisation is a real regulatory fact and
        # stays searchable - it just should not outrank a medicine a patient
        # can actually be prescribed.
        if self.status != "Authorised":
            total *= 0.6
        return total


# ── Index ─────────────────────────────────────────────────────

_index_lock = threading.Lock()
_index: list[_Medicine] | None = None
_index_expires_at = 0.0
_retry_after = 0.0


def _build_record(row: dict) -> _Medicine | None:
    """Turn one dataset row into an indexed record, or None if unusable."""
    name = (row.get("name_of_medicine") or "").strip()
    url = (row.get("medicine_url") or "").strip()
    if not name or not url:
        return None

    inn = (row.get("international_non_proprietary_name_common_name") or "").strip()
    substance = (row.get("active_substance") or "").strip()
    atc = " ".join(filter(None, (
        (row.get("atc_code_human") or "").strip(),
        (row.get("atcvet_code_veterinary") or "").strip(),
    )))
    area = "; ".join(filter(None, (
        (row.get("therapeutic_area_mesh") or "").strip(),
        (row.get("species_veterinary") or "").strip(),
    )))
    group = " ".join(filter(None, (
        (row.get("pharmacotherapeutic_group_human") or "").strip(),
        (row.get("pharmacotherapeutic_group_veterinary") or "").strip(),
    )))
    indication = " ".join((row.get("therapeutic_indication") or "").split())[:3000]

    # The regulatory qualifiers, kept only where they are set: each one says
    # something about how much evidence the authorisation rests on.
    flags = [
        label
        for key, label in (
            ("orphan_medicine", "orphan medicine"),
            ("conditional_approval", "conditional approval"),
            ("exceptional_circumstances", "exceptional circumstances"),
            ("accelerated_assessment", "accelerated assessment"),
            ("additional_monitoring", "additional monitoring"),
            ("advanced_therapy", "advanced therapy (ATMP)"),
            ("prime_priority_medicine", "PRIME priority medicine"),
            ("generic", "generic"),
            ("biosimilar", "biosimilar"),
        )
        if str(row.get(key, "")).strip().lower() == "yes"
    ]

    return _Medicine(
        name=name,
        inn=inn,
        active_substance=substance,
        atc=atc,
        status=(row.get("medicine_status") or "").strip(),
        category=(row.get("category") or "").strip(),
        therapeutic_area=area,
        pharmacotherapeutic_group=group,
        indication=indication,
        holder=(row.get("marketing_authorisation_developer_applicant_holder") or "").strip(),
        product_number=(row.get("ema_product_number") or "").strip(),
        authorisation_date=_iso_date(row.get("marketing_authorisation_date", "")),
        last_updated=_iso_date(row.get("last_updated_date", "")),
        url=url,
        flags=flags,
        is_reference=not ({"generic", "biosimilar"} & set(flags)),
        _name_l=name.lower(),
        _substance_l=f"{inn} {substance}".lower(),
        _atc_l=atc.lower(),
        _area_l=f"{area} {group}".lower(),
        _indication_l=indication.lower(),
    )


def build_index(payload: dict) -> list[_Medicine]:
    """Build the searchable index from a parsed dataset payload."""
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    built = (_build_record(row) for row in rows if isinstance(row, dict))
    return [record for record in built if record is not None]


def _fetch_index() -> list[_Medicine] | None:
    """Return the medicines index, downloading and building it if stale."""
    global _index, _index_expires_at, _retry_after

    now = time.monotonic()
    if _index is not None and now < _index_expires_at:
        return _index

    with _index_lock:
        # Another thread may have loaded it while this one waited.
        now = time.monotonic()
        if _index is not None and now < _index_expires_at:
            return _index
        if now < _retry_after:
            return None

        try:
            resp = httpx.get(_DATASET_URL, headers=_headers(), timeout=_DOWNLOAD_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.debug("EMA dataset download failed: %s", exc)
            _retry_after = time.monotonic() + _FAILURE_BACKOFF_SECONDS
            return None

        records = build_index(payload)
        if not records:
            logger.warning("EMA dataset parsed to no records - schema may have changed")
            _retry_after = time.monotonic() + _FAILURE_BACKOFF_SECONDS
            return None

        _index = records
        _index_expires_at = time.monotonic() + _INDEX_TTL_SECONDS
        _retry_after = 0.0
        logger.debug("EMA index built: %d medicines", len(records))
        return _index


def reset_index() -> None:
    """Drop the cached index. Tests use this; nothing in the app does."""
    global _index, _index_expires_at, _retry_after
    with _index_lock:
        _index = None
        _index_expires_at = 0.0
        _retry_after = 0.0


# ── Rendering ─────────────────────────────────────────────────


def _render_content(record: _Medicine) -> str:
    """Format one EPAR record as the text the summariser will read."""
    parts = [f"EMA European public assessment report (EPAR): {record.name}"]

    def add(label: str, value: str) -> None:
        if value:
            parts.append(f"{label}: {value}")

    add("Authorisation status", record.status)
    add("Type", "veterinary medicine" if record.category == "Veterinary" else "human medicine")
    add("Active substance", record.active_substance or record.inn)
    add("INN / common name", record.inn)
    add("ATC code", record.atc)
    add("Pharmacotherapeutic group", record.pharmacotherapeutic_group)
    add("Therapeutic area", record.therapeutic_area)
    add("Marketing authorisation holder", record.holder)
    add("Marketing authorisation date", record.authorisation_date)
    add("EMA product number", record.product_number)
    add("Regulatory status", ", ".join(record.flags))

    if record.indication:
        parts.append(f"\n--- THERAPEUTIC INDICATION ---\n{record.indication}")

    return "\n".join(parts)


def _to_result(record: _Medicine, source_type: str) -> SourceResult:
    """Convert an indexed record into a SourceResult."""
    substance = record.inn or record.active_substance
    title_bits = [f"EMA EPAR - {record.name}"]
    if substance and substance.lower() != record.name.lower():
        title_bits.append(f"({substance})")

    snippet_bits = [record.status or "status unknown"]
    if record.atc:
        snippet_bits.append(f"ATC {record.atc}")
    if record.therapeutic_area:
        snippet_bits.append(record.therapeutic_area)
    elif record.pharmacotherapeutic_group:
        snippet_bits.append(record.pharmacotherapeutic_group)
    if record.flags:
        snippet_bits.append(", ".join(record.flags))

    return SourceResult(
        title=" ".join(title_bits),
        url=record.url,
        snippet=" | ".join(snippet_bits)[:300],
        content=_render_content(record)[:4000],
        source_type=source_type,
        reliability_tier=1,
        publication_date=record.authorisation_date or record.last_updated,
        language="en",
        publication_types=["European public assessment report"],
    )


class EMAProvider(SourceProvider):
    """Search the EMA medicines register (free, no API key)."""

    source_type = "ema"

    def search(self, query: str, max_results: int = 3) -> list[SourceResult]:
        """Rank the EPAR index against *query* and return the best records.

        Not wrapped in the retry the network-bound providers use: the only
        fallible step is the dataset download, and re-attempting a 7 MB fetch
        inside a shared search budget would spend the budget rather than
        rescue the query. :func:`_fetch_index` handles its own failure.
        """
        if not query or len(query.strip()) < 3:
            return []

        terms = _terms(query)
        if not terms:
            return []

        index = _fetch_index()
        if not index:
            return []

        scored = [
            (score, record)
            for record in index
            if (score := record.score(terms)) >= _SCORE_FLOOR
        ]
        if not scored:
            return []

        # Equal scores are the norm - four biosimilars of one INN match a
        # query for that INN identically. Prefer the authorised one, then the
        # reference product over its copies, then the most recent record.
        scored.sort(
            key=lambda pair: (
                -pair[0],
                pair[1].status != "Authorised",
                not pair[1].is_reference,
                _newest_first(pair[1].last_updated),
            )
        )

        return [_to_result(record, self.source_type) for _, record in scored[:max_results]]


def _newest_first(iso: str) -> int:
    """Order ISO dates newest-first, undated last, inside an ascending sort."""
    digits = iso.replace("-", "")
    return -int(digits) if digits.isdigit() else 0
