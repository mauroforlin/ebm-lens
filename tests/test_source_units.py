"""Offline tests for the pure logic in ``app/sources`` - no network.

These run in well under a second, so they belong in any pre-commit or CI
gate: ``pytest -m "not live"``.
"""
from __future__ import annotations

import pytest

from app.sources.base import SourceResult, build_headers, extract_doi, user_agent
from app.sources.biorxiv import _is_target_server, _server_name
from app.sources.blocklist import filter_blocked, is_blocked_url
from app.sources.chembl import _clean_drug_name as chembl_clean
from app.sources.citation_expander import _is_paper, _seed_paper_id
from app.sources.content_extractor import _extract_text_from_html, _is_fetchable_url
from app.sources.dailymed import _clean_drug_name as dailymed_clean
from app.sources.ema import _iso_date as ema_iso_date
from app.sources.ema import _newest_first, _terms, build_index
from app.sources.pubmed import _broaden_query, _simplify_query, build_pubmed_query
from app.sources.rxnav import _extract_drug_name
from app.sources.wikipedia import WikipediaProvider

# ── base ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://doi.org/10.1016/S2213-8587(19)30249-9", "10.1016/s2213-8587(19)30249-9"),
        ("https://www.nature.com/articles/10.1038/nm.4001", "10.1038/nm.4001"),
        ("https://pubmed.ncbi.nlm.nih.gov/31422062/", None),
        ("", None),
    ],
)
def test_extract_doi(url, expected):
    assert extract_doi(url) == expected


def test_user_agent_carries_a_contact_address():
    """NCBI, Crossref and OpenAlex route identified clients through the polite pool."""
    ua = user_agent()
    assert ua.startswith("EBM Lens/")
    assert "mailto:" in ua


def test_build_headers_always_identifies_the_client():
    headers = build_headers(accept="application/json", extra={"X-Test": "1"})
    assert headers["User-Agent"] == user_agent()
    assert headers["Accept"] == "application/json"
    assert headers["X-Test"] == "1"


# ── PubMed query construction ─────────────────────────────────


def test_pubmed_query_uses_majr_for_the_primary_mesh_term():
    q = build_pubmed_query("metformin in diabetes", mesh_terms=["Metformin", "Diabetes Mellitus"])
    assert '"Metformin"[Majr]' in q
    assert '"Diabetes Mellitus"[MeSH Terms]' in q


def test_pubmed_query_passes_through_a_query_that_already_has_field_tags():
    q = build_pubmed_query('"Aspirin"[MeSH Terms] AND stroke[tiab]')
    assert q.startswith('"Aspirin"[MeSH Terms]')


def test_pubmed_query_appends_the_publication_type_filter():
    q = build_pubmed_query("statins", evidence_type="systematic_review")
    assert "systematic review[pt]" in q and "meta-analysis[pt]" in q


def test_pubmed_query_builds_drug_and_disease_blocks():
    q = build_pubmed_query("dosing", drug_names=["metformin"], diseases=["type 2 diabetes"])
    assert '"metformin"[tiab]' in q
    assert '"type 2 diabetes"[tiab]' in q


def test_pubmed_fallback_queries_strip_field_tags_and_shorten():
    tagged = '("Metformin"[Majr]) AND (diabetes[tiab] AND glycaemic[tiab])'
    assert "[" not in _simplify_query(tagged)
    assert len(_broaden_query(tagged).split()) <= 4


# ── Provider-specific query cleaners ──────────────────────────


@pytest.mark.parametrize(
    ("cleaner", "query", "must_keep"),
    [
        (dailymed_clean, "what is the dosage of metformin", "metformin"),
        (dailymed_clean, "qual e il dosaggio di warfarin", "warfarin"),
        (chembl_clean, "mechanism of action of aspirin", "aspirin"),
        (_extract_drug_name, "interaction between warfarin and amiodarone", "warfarin"),
    ],
)
def test_drug_name_cleaners_keep_the_drug(cleaner, query, must_keep):
    assert must_keep in cleaner(query).lower()


# ── EMA: query terms, dataset parsing, scoring ───────────────
#
# EMA has no query API - the whole dataset is indexed locally and scored
# against terms extracted from the query - so that logic is exercised here
# without a network call.


def test_ema_terms_strips_stopwords_and_short_tokens():
    terms = {t.text for t in _terms("what is the dosage of Ozempic in the EU")}
    assert "ozempic" in terms and "dosage" in terms
    assert "what" not in terms and "the" not in terms and "eu" not in terms
    assert "of" not in terms and "in" not in terms  # length <= 2


def test_ema_terms_recognises_atc_codes_as_a_unit():
    terms = _terms("L04AC07 rheumatoid arthritis")
    atc = next(t for t in terms if t.is_atc)
    assert atc.text == "l04ac07"


def test_ema_term_pattern_is_word_boundary_anchored():
    """Regression: 'cosa cura Ozempic' used to match the veterinary medicine
    Dicural via the bare substring 'cura' inside its name.
    """
    terms = _terms("cosa cura Ozempic")
    cura = next(t for t in terms if t.text == "cura")
    assert cura.pattern.search("dicural") is None
    assert cura.pattern.search("a medicine that cura diabetes") is not None


@pytest.mark.parametrize(
    ("value", "expected"),
    [("15/03/2021", "2021-03-15"), ("not a date", ""), ("", "")],
)
def test_ema_iso_date(value, expected):
    assert ema_iso_date(value) == expected


def test_ema_newest_first_orders_dates_descending_and_undated_last():
    dates = ["2020-01-01", "2023-06-15", "", "2019-12-31"]
    assert sorted(dates, key=_newest_first) == [
        "2023-06-15", "2020-01-01", "2019-12-31", "",
    ]


_EMA_ROW = {
    "name_of_medicine": "Ozempic",
    "medicine_url": "https://www.ema.europa.eu/en/medicines/human/EPAR/ozempic",
    "international_non_proprietary_name_common_name": "semaglutide",
    "active_substance": "semaglutide",
    "atc_code_human": "A10BJ06",
    "medicine_status": "Authorised",
    "category": "Human",
    "therapeutic_area_mesh": "Diabetes Mellitus, Type 2",
    "pharmacotherapeutic_group_human": "Drugs used in diabetes",
    "therapeutic_indication": "Treatment of type 2 diabetes mellitus.",
    "marketing_authorisation_developer_applicant_holder": "Novo Nordisk A/S",
    "ema_product_number": "EMEA/H/C/004174",
    "marketing_authorisation_date": "08/02/2018",
    "last_updated_date": "01/01/2024",
}


def test_ema_build_index_parses_a_row():
    [record] = build_index({"data": [_EMA_ROW]})
    assert record.name == "Ozempic"
    assert record.atc == "A10BJ06"
    assert record.authorisation_date == "2018-02-08"
    assert record.is_reference


def test_ema_build_index_drops_rows_missing_name_or_url():
    bad = dict(_EMA_ROW, name_of_medicine="")
    assert build_index({"data": [bad]}) == []


def test_ema_build_index_ignores_a_payload_with_no_data_list():
    assert build_index({}) == []
    assert build_index({"data": "not a list"}) == []


def test_ema_build_index_flags_generics_as_not_reference():
    row = dict(_EMA_ROW, generic="Yes")
    [record] = build_index({"data": [row]})
    assert not record.is_reference
    assert "generic" in record.flags


def test_ema_score_weighs_name_match_above_indication_only_match():
    [record] = build_index({"data": [_EMA_ROW]})
    name_match = record.score(_terms("Ozempic"))
    indication_only = record.score(_terms("diabetes"))
    assert name_match > indication_only > 0


def test_ema_score_is_zero_for_terms_matching_nothing():
    [record] = build_index({"data": [_EMA_ROW]})
    assert record.score(_terms("aspirin headache")) == 0.0


def test_ema_score_penalises_a_non_authorised_status():
    withdrawn_row = dict(_EMA_ROW, medicine_status="Withdrawn")
    [authorised] = build_index({"data": [_EMA_ROW]})
    [withdrawn] = build_index({"data": [withdrawn_row]})
    terms = _terms("Ozempic")
    assert withdrawn.score(terms) < authorised.score(terms)


# ── Blocklist ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        "https://www.healthline.com/nutrition/x",
        "https://researchgate.net/publication/1",
        "https://sub.webmd.com/a",
        "https://www.mypersonaltrainer.it/salute/x.html",
    ],
)
def test_blocked_domains_are_rejected(url):
    assert is_blocked_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://pubmed.ncbi.nlm.nih.gov/31422062/",
        "https://www.who.int/data/gho",
        "https://clinicaltrials.gov/study/NCT01720446",
        "https://notwebmd.com/a",  # substring, not a subdomain
    ],
)
def test_legitimate_domains_pass(url):
    assert not is_blocked_url(url)


def test_blocklist_does_not_mangle_hosts_when_stripping_www():
    """``lstrip('www.')`` would turn who.int into ho.int - a silent miss."""
    assert not is_blocked_url("https://www.who.int/x")


def test_filter_blocked_drops_only_the_blocked_ones():
    keep = SourceResult(title="a", url="https://pubmed.ncbi.nlm.nih.gov/1/", snippet="", content="")
    drop = SourceResult(title="b", url="https://www.healthline.com/x", snippet="", content="")
    assert filter_blocked([keep, drop]) == [keep]


# ── Content extractor fencing ─────────────────────────────────


@pytest.mark.parametrize(
    ("url", "fetchable"),
    [
        ("https://en.wikipedia.org/wiki/Aspirin", True),
        ("https://www.ema.europa.eu/en/medicines/x", True),
        ("https://sub.who.int/page", True),
        ("https://example.com/anything", False),
        ("https://evil-who.int/page", False),
    ],
)
def test_content_extractor_allowlist(url, fetchable):
    assert _is_fetchable_url(url) is fetchable


def test_html_extraction_drops_script_and_nav():
    html = (
        "<html><head><style>.a{color:red}</style></head><body>"
        "<nav>menu menu menu</nav><script>var x=1;</script>"
        "<p>" + "Metformin lowers hepatic glucose production. " * 8 + "</p>"
        "</body></html>"
    )
    text = _extract_text_from_html(html)
    assert "Metformin lowers hepatic glucose production." in text
    assert "var x" not in text and "color:red" not in text


def test_html_extraction_rejects_pages_that_are_too_short():
    assert _extract_text_from_html("<html><body><p>hi</p></body></html>") == ""


# ── bioRxiv server filtering ──────────────────────────────────


def test_biorxiv_recognises_its_servers():
    assert _is_target_server({"institution": [{"name": "bioRxiv"}], "DOI": "10.1101/1"})
    assert _is_target_server({"DOI": "10.1101/2024.01.01.123456"})  # no institution
    assert not _is_target_server({"institution": [{"name": "Research Square"}],
                                  "DOI": "10.21203/rs.3.rs-1"})


def test_biorxiv_server_name_falls_back_to_container_title():
    assert _server_name({"container-title": ["medRxiv"]}) == "medRxiv"


# ── Citation-graph seed derivation ────────────────────────────


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://pubmed.ncbi.nlm.nih.gov/31422062/", "PMID:31422062"),
        ("https://europepmc.org/article/MED/31422062", "PMID:31422062"),
        ("https://doi.org/10.1101/2024.01.01.123456", "DOI:10.1101/2024.01.01.123456"),
        ("https://example.com/no-id", None),
    ],
)
def test_seed_paper_id(url, expected):
    assert _seed_paper_id(SourceResult(title="t", url=url, snippet="", content="")) == expected


def test_only_paper_sources_seed_the_citation_graph():
    """Expanding from an FDA label or a WHO indicator makes no sense."""
    def r(source_type: str) -> SourceResult:
        return SourceResult(title="t", url="https://x/", snippet="", content="",
                            source_type=source_type)

    assert _is_paper(r("pubmed")) and _is_paper(r("europe_pmc")) and _is_paper(r("biorxiv"))
    assert not _is_paper(r("openfda")) and not _is_paper(r("who_gho"))


# ── Wikipedia helpers ─────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    ["Mercury may refer to: the planet...", "Argento può riferirsi a: il metallo..."],
)
def test_disambiguation_detection(text):
    assert WikipediaProvider._is_disambiguation(text)


def test_smart_trim_keeps_the_intro_and_respects_the_budget():
    intro = "Intro paragraph about the disease. " * 10
    refs = "\n== References ==\n" + "citation " * 200
    facts = "\n== Epidemiology ==\n" + "In 2019 the prevalence was 9.3 percent. " * 30
    trimmed = WikipediaProvider._smart_trim(intro + refs + facts, max_chars=1500)
    assert trimmed.startswith("Intro paragraph")
    assert len(trimmed) <= 1600  # budget + the partial-section allowance
    assert "== References ==" not in trimmed


# ── SourceResult contract ─────────────────────────────────────


def test_source_result_defaults_are_conservative():
    """An unlabelled result must not claim gold-standard reliability."""
    r = SourceResult(title="t", url="https://x/", snippet="s", content="c")
    assert r.reliability_tier == 3
    assert r.citation_count == 0
    assert r.language == ""


# ── PubMed: PMID -> PMCID resolution ──────────────────────────
#
# Driven with a recorded ELink payload rather than a live call: the live
# version passes for the wrong reason whenever NCBI throttles.

# Real shape of elink.fcgi?dbfrom=pubmed&db=pmc&id=31422062 (truncated).
# The paper itself is NOT in PMC, so the only link set returned is
# `pubmed_pmc_refs` - the 746 PMC articles that cite it.
_ELINK_ONLY_CITING = {
    "linksets": [
        {
            "dbfrom": "pubmed",
            "ids": ["31422062"],
            "linksetdbs": [
                {
                    "dbto": "pmc",
                    "linkname": "pubmed_pmc_refs",
                    "links": ["13479717", "13471553", "13466521"],
                }
            ],
        }
    ]
}

# A PMID that IS in PMC returns the `pubmed_pmc` link set as well.
_ELINK_WITH_SELF = {
    "linksets": [
        {
            "dbfrom": "pubmed",
            "ids": ["28210224"],
            "linksetdbs": [
                {"dbto": "pmc", "linkname": "pubmed_pmc", "links": ["5334499"]},
                {"dbto": "pmc", "linkname": "pubmed_pmc_refs", "links": ["9999999"]},
            ],
        }
    ]
}


class _StubResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _StubClient:
    def __init__(self, payload: dict):
        self._payload = payload

    def get(self, *_args, **_kwargs) -> _StubResponse:
        return _StubResponse(self._payload)


@pytest.fixture
def elink(monkeypatch):
    """Feed ``_pmids_to_pmcids`` a canned ELink response."""
    from app.sources import pubmed as pubmed_mod

    def _install(payload: dict):
        monkeypatch.setattr(pubmed_mod, "_get_client", lambda: _StubClient(payload))
        monkeypatch.setattr(pubmed_mod, "_rate_limit", lambda: None)
        return pubmed_mod._pmids_to_pmcids

    return _install


def test_pmid_to_pmcid_ignores_the_citing_articles_linkset(elink):
    resolve = elink(_ELINK_ONLY_CITING)
    assert resolve(["31422062"]) == {}, (
        "a PMID with no PMC record of its own resolved to a PMCID - that is a "
        "citing article, and its full text is not this paper's"
    )


def test_pmid_to_pmcid_resolves_the_paper_when_it_is_in_pmc(elink):
    resolve = elink(_ELINK_WITH_SELF)
    assert resolve(["28210224"])["28210224"] == "5334499", (
        "the paper's own PMC record was overwritten by an article citing it"
    )


# ── Configuration wiring ──────────────────────────────────────


def test_ncbi_api_key_is_read_from_settings():
    import inspect

    from app.config import Settings
    from app.sources import pubmed as pubmed_mod

    assert "ncbi_api_key" in Settings.model_fields, "config no longer declares the key"
    source = inspect.getsource(pubmed_mod)
    assert "get_settings" in source, (
        "pubmed.py never consults Settings, so NCBI_API_KEY from .env cannot reach it"
    )


def test_openfda_api_key_can_be_configured():
    from app.config import Settings

    assert "openfda_api_key" in Settings.model_fields, (
        "no OPENFDA_API_KEY setting exists, so the 1,000/day anonymous cap is permanent"
    )
