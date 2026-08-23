"""Upstream reachability + response-shape contracts, one per database.

These bypass our provider classes entirely. A failure here means the public
API moved, changed shape, or is down - check the vendor's status page and
changelog before touching ``app/sources/``.

Every test asserts the *specific fields the provider reads*, so a silent
schema change (a renamed GraphQL field, a dropped JSON key) fails loudly
here instead of degrading into empty results in production.
"""
from __future__ import annotations

import json
import time
from xml.etree import ElementTree

import httpx
import pytest

from tests.conftest import skip_if_throttled

pytestmark = [pytest.mark.live]

UA = {"User-Agent": "EBM-Lens-tests/1.0 (mailto:tests@example.com)"}
JSON = {**UA, "Accept": "application/json"}
TIMEOUT = 30


def _get(url: str, params: dict | None = None, headers: dict | None = None) -> httpx.Response:
    return httpx.get(
        url, params=params, headers=headers or JSON, timeout=TIMEOUT, follow_redirects=True
    )


# ── NCBI E-utilities (PubMed + PMC) ───────────────────────────


def test_pubmed_esearch_responds():
    r = _get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        {"db": "pubmed", "term": "semaglutide", "retmode": "json", "retmax": 5},
    )
    r.raise_for_status()
    ids = r.json()["esearchresult"]["idlist"]
    assert ids, "esearch returned no PMIDs for a term with thousands of hits"


def test_pubmed_efetch_returns_parsable_xml():
    r = _get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        {"db": "pubmed", "id": "31422062", "rettype": "abstract", "retmode": "xml"},
    )
    r.raise_for_status()
    root = ElementTree.fromstring(r.content)
    article = next(root.iter("PubmedArticle"), None)
    assert article is not None, "efetch XML no longer contains <PubmedArticle>"
    assert article.find(".//ArticleTitle") is not None
    assert article.find(".//AbstractText") is not None


def test_elink_still_distinguishes_pmc_linknames():
    """ELink pubmed->pmc returns two different link sets.

    ``pubmed_pmc`` is *this* article in PMC. ``pubmed_pmc_refs`` is the
    hundreds of articles that CITE it. Conflating them attaches a stranger's
    full text to a paper - see BUG-01.
    """
    # Anonymous ELink is flaky: it drops connections and emits malformed JSON
    # under load. Two attempts, then skip - this is a canary, not a gate.
    r = None
    for attempt in range(2):
        try:
            r = _get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi",
                {"dbfrom": "pubmed", "db": "pmc", "id": "31422062", "retmode": "json"},
            )
            break
        except httpx.HTTPError as exc:
            if attempt:
                pytest.skip(f"ELink unreachable ({exc!r}) - known NCBI flakiness")
            time.sleep(2.0)
    assert r is not None
    skip_if_throttled(r, "NCBI ELink")
    r.raise_for_status()
    try:
        # NCBI's elink JSON sometimes carries raw control characters, which
        # strict json.loads rejects. The provider swallows that and returns {},
        # so full-text enrichment silently no-ops - see the audit's OBS-3.
        payload = json.loads(r.text, strict=False)
    except json.JSONDecodeError as exc:
        pytest.skip(f"ELink returned malformed JSON ({exc}) - known NCBI flakiness")
    linknames = {
        db.get("linkname")
        for ls in payload.get("linksets", [])
        for db in ls.get("linksetdbs", [])
    }
    if not linknames:
        # NCBI answers with an empty linkset when it throttles an anonymous
        # caller; that is not evidence about the link names.
        pytest.skip("ELink returned no linksetdbs (NCBI throttling)")
    assert "pubmed_pmc_refs" in linknames, (
        "the citing-articles link set is gone; BUG-01's failure mode may have "
        "changed shape - re-check app/sources/pubmed.py::_pmids_to_pmcids"
    )


def test_pmc_efetch_returns_fulltext_body():
    r = _get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        {"db": "pmc", "id": "PMC5334499", "rettype": "xml", "retmode": "xml"},
    )
    r.raise_for_status()
    root = ElementTree.fromstring(r.content)
    assert root.find(".//body") is not None, "PMC efetch no longer returns a <body>"


# ── Europe PMC ────────────────────────────────────────────────


def test_europe_pmc_search_responds():
    r = _get(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        {
            "query": "(semaglutide) AND SRC:MED",
            "format": "json",
            "resultType": "core",
            "pageSize": 5,
        },
    )
    r.raise_for_status()
    hits = r.json()["resultList"]["result"]
    assert hits
    for field in ("title", "source"):
        assert field in hits[0], f"Europe PMC dropped the {field!r} field"
    assert any(h.get("pmid") or h.get("pmcid") or h.get("doi") for h in hits), (
        "no hit carries an identifier we can build a URL from"
    )


def test_europe_pmc_fulltext_xml_available_for_oa():
    """The OA full-text endpoint the provider upgrades content with."""
    r = _get(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        {
            "query": "diabetes AND SRC:MED AND OPEN_ACCESS:Y",
            "format": "json",
            "resultType": "core",
            "pageSize": 5,
        },
    )
    r.raise_for_status()
    pmcids = [h["pmcid"] for h in r.json()["resultList"]["result"] if h.get("pmcid")]
    assert pmcids, "no open-access hit carried a PMCID"

    # EBI answers 406 intermittently under load, and not every OPEN_ACCESS:Y
    # record actually has an XML body deposited - so try the whole page and
    # require only that one of them serves full text.
    codes: dict[str, int] = {}
    for pmcid in pmcids:
        time.sleep(1.0)  # fullTextXML throttles harder than /search does
        ft = _get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML")
        codes[pmcid] = ft.status_code
        if ft.status_code == 200:
            assert b"<body" in ft.content, f"{pmcid}: full-text XML has no <body>"
            return
    if set(codes.values()) == {406}:
        pytest.skip(f"Europe PMC returned 406 for every OA candidate (throttling): {codes}")
    pytest.fail(f"no OA article served full text: {codes}")


# ── ClinicalTrials.gov v2 ─────────────────────────────────────


def test_clinicaltrials_v2_responds_and_keeps_schema():
    r = _get(
        "https://clinicaltrials.gov/api/v2/studies",
        {"query.term": "semaglutide", "pageSize": 5},
    )
    r.raise_for_status()
    studies = r.json()["studies"]
    assert studies
    proto = studies[0]["protocolSection"]
    for module in ("identificationModule", "statusModule", "designModule"):
        assert module in proto, f"v2 schema lost {module}"
    assert proto["identificationModule"].get("nctId", "").startswith("NCT")


def test_clinicaltrials_relevance_sort_is_supported():
    """``@relevance`` is what the provider *should* be sorting by - BUG-06."""
    r = _get(
        "https://clinicaltrials.gov/api/v2/studies",
        {
            "query.term": "semaglutide cardiovascular outcomes",
            "pageSize": 5,
            "sort": "@relevance",
        },
    )
    assert r.status_code == 200, f"@relevance rejected: {r.status_code} {r.text[:200]}"
    assert r.json()["studies"]


# ── openFDA ───────────────────────────────────────────────────


def test_openfda_label_field_query_responds():
    r = _get(
        "https://api.fda.gov/drug/label.json",
        {"search": 'openfda.generic_name:"metformin"', "limit": 3},
    )
    r.raise_for_status()
    results = r.json()["results"]
    assert results
    assert "openfda" in results[0]
    assert results[0].get("indications_and_usage"), "label lost indications_and_usage"


def test_openfda_or_syntax_is_broader_than_plus_syntax():
    """The provider joins its two clauses with ``+``.

    openFDA reads a leading ``+`` as "this clause is REQUIRED", not as OR, so
    the brand-name clause silently becomes mandatory. Documented as BUG-02.
    """

    def total(search: str) -> int:
        r = _get("https://api.fda.gov/drug/label.json", {"search": search, "limit": 1})
        return r.json()["meta"]["results"]["total"] if r.status_code == 200 else 0

    plus = total('openfda.generic_name:"semaglutide"+openfda.brand_name:"semaglutide"')
    or_ = total('openfda.generic_name:"semaglutide" OR openfda.brand_name:"semaglutide"')
    assert or_ > plus, (
        f"openFDA changed how it parses '+'; BUG-02 may be obsolete (plus={plus}, or={or_})"
    )


def test_openfda_faers_count_endpoint_responds():
    r = _get(
        "https://api.fda.gov/drug/event.json",
        {
            "search": 'patient.drug.openfda.generic_name:"metformin"',
            "count": "patient.reaction.reactionmeddrapt.exact",
            "limit": 5,
        },
    )
    r.raise_for_status()
    rows = r.json()["results"]
    assert rows and "term" in rows[0] and "count" in rows[0]


# ── DailyMed ──────────────────────────────────────────────────


def test_dailymed_spl_search_responds():
    r = _get(
        "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json",
        {"drug_name": "metformin", "page": 1, "pagesize": 3},
    )
    r.raise_for_status()
    data = r.json()["data"]
    assert data
    assert {"setid", "title"} <= set(data[0]), "DailyMed search schema changed"


def test_dailymed_label_text_is_only_available_as_xml():
    """``/spls/{setid}.json`` 415s; the label body lives in the XML - BUG-03."""
    search = _get(
        "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json",
        {"drug_name": "metformin", "page": 1, "pagesize": 1},
    )
    setid = search.json()["data"][0]["setid"]
    base = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls"

    as_json = _get(f"{base}/{setid}.json")
    as_xml = _get(f"{base}/{setid}.xml", headers={**UA, "Accept": "*/*"})

    assert as_xml.status_code == 200, f"the XML label endpoint broke: {as_xml.status_code}"
    assert len(as_xml.content) > 10_000, "XML label suspiciously small"
    if as_json.status_code == 200:
        pytest.fail("DailyMed now serves label JSON - BUG-03 can be fixed the easy way")


# ── EMA ───────────────────────────────────────────────────────

_EMA_DATASET_URL = (
    "https://www.ema.europa.eu/en/documents/report/"
    "medicines-output-medicines_json-report_en.json"
)


def test_ema_dataset_download_responds_and_keeps_schema():
    r = _get(_EMA_DATASET_URL)
    r.raise_for_status()
    rows = r.json()["data"]
    assert rows, "EMA medicines dataset has no rows"
    row = rows[0]
    for field in ("name_of_medicine", "medicine_url", "medicine_status"):
        assert field in row, f"EMA dataset schema lost {field!r}"


# ── RxNav / RxNorm ────────────────────────────────────────────


def test_rxnav_rxcui_lookup_responds():
    r = _get("https://rxnav.nlm.nih.gov/REST/rxcui.json", {"name": "warfarin", "search": 2})
    r.raise_for_status()
    assert r.json()["idGroup"]["rxnormId"], "RxNorm cannot resolve 'warfarin'"


def test_rxnav_drugs_endpoint_responds():
    r = _get("https://rxnav.nlm.nih.gov/REST/drugs.json", {"name": "warfarin"})
    r.raise_for_status()
    groups = r.json()["drugGroup"]["conceptGroup"]
    assert any(g.get("conceptProperties") for g in groups)


def test_rxnav_interaction_api_is_still_retired():
    """NLM switched the drug-interaction API off on 2 Jan 2024 - BUG-04.

    This test exists so the day it comes back (or a replacement is wired in)
    the suite says so, instead of the provider quietly returning nothing.
    """
    r = _get("https://rxnav.nlm.nih.gov/REST/interaction/interaction.json", {"rxcui": "11289"})
    assert r.status_code == 404, (
        f"the RxNav interaction API answered {r.status_code} - it may be back; "
        "consider re-adding interaction search to app/sources/rxnav.py"
    )


# ── Open Targets ──────────────────────────────────────────────

_OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"
_OT_HEADERS = {**JSON, "Content-Type": "application/json"}


def _gql(query: str, variables: dict) -> dict:
    r = httpx.post(
        _OT_URL, json={"query": query, "variables": variables},
        headers=_OT_HEADERS, timeout=TIMEOUT,
    )
    return r.json()


def test_open_targets_search_responds():
    body = _gql(
        """
        query Search($q: String!) {
          search(queryString: $q, entityNames: ["drug","disease","target"],
                 page: {index: 0, size: 5}) { hits { id name entity description } }
        }""",
        {"q": "semaglutide"},
    )
    assert "errors" not in body, body.get("errors")
    hits = body["data"]["search"]["hits"]
    assert hits and any(h["entity"] == "drug" for h in hits)


def test_open_targets_drug_query_matches_current_schema():
    """The field names the provider's detail query must use - BUG-05.

    ``maximumClinicalTrialPhase``, ``hasBeenWithdrawn``, ``maxPhaseForIndication``
    and ``indications.rows.references`` no longer exist on this schema; the
    query below is the corrected one.
    """
    body = _gql(
        """
        query DrugInfo($chemblId: String!) {
          drug(chemblId: $chemblId) {
            id name drugType maximumClinicalStage description
            drugWarnings { warningType description year country }
            mechanismsOfAction { rows { mechanismOfAction actionType targetName } }
            indications { count rows { maxClinicalStage disease { id name } } }
            adverseEvents { count rows { name count logLR meddraCode } }
          }
        }""",
        {"chemblId": "CHEMBL2108724"},
    )
    assert "errors" not in body, body.get("errors")
    drug = body["data"]["drug"]
    assert drug["name"].lower() == "semaglutide"
    assert drug["mechanismsOfAction"]["rows"], "no mechanism of action returned"
    assert drug["indications"]["count"] > 0


# ── ChEMBL ────────────────────────────────────────────────────


def test_chembl_search_responds():
    r = _get(
        "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json",
        {"q": "semaglutide", "limit": 5},
    )
    r.raise_for_status()
    assert r.json()["molecules"]


def test_chembl_search_ranks_the_real_drug_below_noise():
    """``molecule/search`` puts unnamed research compounds first - BUG-07.

    The named, approved molecule is in the response, just not in the first
    few rows, which is exactly what the provider slices.
    """
    r = _get(
        "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json",
        {"q": "semaglutide", "limit": 10},
    )
    mols = r.json()["molecules"]
    named = [m for m in mols if m.get("pref_name")]
    assert named, "ChEMBL returned no named molecule for 'semaglutide' at all"
    assert any(m["pref_name"].lower() == "semaglutide" for m in named)
    if mols[0].get("pref_name"):
        pytest.fail("ChEMBL now ranks the named molecule first - BUG-07 may be moot")


def test_chembl_detail_endpoints_respond():
    for endpoint, key in (
        ("mechanism", "mechanisms"),
        ("drug_indication", "drug_indications"),
        ("drug_warning", "drug_warnings"),
    ):
        r = _get(
            f"https://www.ebi.ac.uk/chembl/api/data/{endpoint}.json",
            {"molecule_chembl_id": "CHEMBL2108724", "limit": 5},
        )
        r.raise_for_status()
        assert key in r.json(), f"ChEMBL {endpoint} response lost the {key!r} key"


# ── WHO GHO ───────────────────────────────────────────────────

_GHO = "https://ghoapi.azureedge.net/api"


def test_who_gho_indicator_search_responds():
    r = _get(f"{_GHO}/Indicator", {"$filter": "contains(IndicatorName,'life expectancy')"})
    r.raise_for_status()
    values = r.json()["value"]
    assert values and "IndicatorCode" in values[0]


def test_who_gho_contains_filter_is_case_insensitive():
    """Confirms the AND-of-every-token filter, not casing, is what fails."""

    def n(f: str) -> int:
        return len(_get(f"{_GHO}/Indicator", {"$filter": f}).json()["value"])

    assert n("contains(IndicatorName,'mortality')") == n("contains(IndicatorName,'Mortality')")


def test_who_gho_data_endpoint_responds():
    r = _get(f"{_GHO}/WHOSIS_000001", {"$top": "5", "$orderby": "TimeDim desc"})
    r.raise_for_status()
    rows = r.json()["value"]
    assert rows and "TimeDim" in rows[0] and "SpatialDim" in rows[0]


# ── Crossref (bioRxiv / medRxiv) ──────────────────────────────


def test_crossref_posted_content_still_yields_biorxiv():
    r = _get(
        "https://api.crossref.org/works",
        {
            "query.bibliographic": "glioblastoma CAR-T",
            "filter": "type:posted-content",
            "rows": 20,
            "mailto": "tests@example.com",
            "sort": "relevance",
        },
    )
    r.raise_for_status()
    items = r.json()["message"]["items"]
    assert items
    on_target = [i for i in items if (i.get("DOI") or "").startswith("10.1101/")]
    assert on_target, (
        "no bioRxiv/medRxiv DOIs in 20 posted-content hits - the 4x over-fetch "
        "in biorxiv.py is no longer enough"
    )
    assert "posted" in items[0], "Crossref dropped the 'posted' date on preprints"


# ── Wikipedia ─────────────────────────────────────────────────


@pytest.mark.parametrize("lang", ["en", "it"])
def test_wikipedia_action_api_responds(lang: str):
    r = _get(
        f"https://{lang}.wikipedia.org/w/api.php",
        {
            "action": "query",
            "format": "json",
            "prop": "extracts|info",
            "inprop": "url",
            "explaintext": 1,
            "redirects": 1,
            "titles": "Diabetes",
        },
    )
    r.raise_for_status()
    page = next(iter(r.json()["query"]["pages"].values()))
    assert page.get("extract"), f"{lang}.wikipedia returned no extract"


def test_wikipedia_fulltext_search_responds():
    r = _get(
        "https://en.wikipedia.org/w/api.php",
        {
            "action": "query",
            "list": "search",
            "srsearch": "GLP-1 receptor agonist",
            "srlimit": 3,
            "srnamespace": 0,
            "format": "json",
        },
    )
    r.raise_for_status()
    assert r.json()["query"]["search"]


# ── Citation graph: Semantic Scholar + OpenAlex ───────────────


def test_semantic_scholar_graph_responds():
    r = _get(
        "https://api.semanticscholar.org/graph/v1/paper/PMID:31422062/references",
        {"fields": "title,abstract,url,year,citationCount,externalIds", "limit": 5},
    )
    if r.status_code == 429:
        pytest.skip("Semantic Scholar rate-limited this unauthenticated call")
    r.raise_for_status()
    assert r.json()["data"]


def test_openalex_related_works_responds():
    doi = "10.1016/S2213-8587(19)30249-9"
    meta = _get(
        f"https://api.openalex.org/works/https://doi.org/{doi}",
        {"select": "related_works", "mailto": "tests@example.com"},
    )
    meta.raise_for_status()
    related = meta.json()["related_works"]
    assert related
    ids = [rid.rsplit("/", 1)[-1] for rid in related[:5]]
    works = _get(
        "https://api.openalex.org/works",
        {
            "filter": f"openalex_id:{'|'.join(ids)}",
            "select": "title,doi,publication_year,cited_by_count",
            "per_page": 5,
            "mailto": "tests@example.com",
        },
    )
    works.raise_for_status()
    assert works.json()["results"], "the openalex_id filter stopped resolving"
