"""End-to-end checks of every ``SourceProvider``, against the live APIs.

A failure here with the matching endpoint test still green means **our
adapter is broken**, not the database.

Each provider gets three kinds of assertion, in increasing strictness:

1. *responds*      - returns at least one well-formed result
2. *substantive*   - the content is long enough to summarise from
3. *on topic*      - the result is actually about what we asked for

Known upstream defects are tracked as ``BUG-nn`` ids, cited at the fix or
workaround in the matching ``app/sources/*.py`` provider - see
``tests/conftest.py`` for how those tests are structured.
"""
from __future__ import annotations

import httpx
import pytest

from app.sources.biorxiv import BiorxivProvider
from app.sources.chembl import ChEMBLProvider
from app.sources.clinicaltrials import ClinicalTrialsProvider
from app.sources.dailymed import DailyMedProvider
from app.sources.ema import EMAProvider
from app.sources.ema import reset_index as reset_ema_index
from app.sources.europe_pmc import EuropePMCProvider
from app.sources.open_targets import OpenTargetsProvider
from app.sources.openfda import OpenFDAProvider
from app.sources.pubmed import PubMedProvider
from app.sources.rxnav import RxNavProvider
from app.sources.who_gho import WHOGHOProvider
from app.sources.wikipedia import WikipediaProvider
from tests.conftest import (
    COMMON_DRUG,
    DRUG,
    EPI_TOPIC,
    ITALIAN_TOPIC,
    LITERATURE_TOPIC,
    PREPRINT_TOPIC,
    TRIAL_TOPIC,
    assert_mentions,
    assert_substantive,
    assert_wellformed,
    search_or_skip,
    skip_if_throttled,
)

pytestmark = [pytest.mark.live]


# ══════════════════════════════════════════════════════════════
#  PubMed
# ══════════════════════════════════════════════════════════════


def test_pubmed_responds():
    results = search_or_skip(PubMedProvider(), LITERATURE_TOPIC, max_results=3)
    assert_wellformed(results, source_type="pubmed", min_results=2)
    assert_substantive(results, min_chars=300, source_type="pubmed")
    assert all(r.publication_date for r in results), "PubMed result without a date"


def test_pubmed_prefers_reviews_when_asked():
    results = search_or_skip(
        PubMedProvider(prefer_reviews=True),
        "GLP-1 receptor agonists type 2 diabetes",
    )
    assert_wellformed(results, source_type="pubmed")
    blob = " ".join(r.content.lower() for r in results)
    assert "review" in blob or "meta-analys" in blob, (
        "prefer_reviews=True returned nothing labelled as a review or meta-analysis"
    )


# BUG-01 (full text of the wrong paper) is checked deterministically, offline,
# in test_source_units.py::test_pmid_to_pmcid_ignores_the_citing_articles_linkset -
# driving it live makes it pass for the wrong reason whenever ELink throttles.


# ══════════════════════════════════════════════════════════════
#  Europe PMC
# ══════════════════════════════════════════════════════════════


def test_europe_pmc_responds():
    results = search_or_skip(EuropePMCProvider(), LITERATURE_TOPIC, max_results=3)
    assert_wellformed(results, source_type="europe_pmc", min_results=2)
    assert_substantive(results, min_chars=300, source_type="europe_pmc")


def test_europe_pmc_guideline_search_responds():
    # search_guidelines swallows its own exceptions, so an empty list here can
    # mean either "no guidelines" or "EBI throttled us" - probe first.
    probe = httpx.get(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        params={"query": "diabetes", "format": "json", "pageSize": 1},
        headers={"User-Agent": "EBM-Lens-tests/1.0"},
        timeout=30,
    )
    skip_if_throttled(probe, "Europe PMC")
    results = EuropePMCProvider().search_guidelines("type 2 diabetes management", 3)
    assert_wellformed(results, source_type="europe_pmc")


def test_europe_pmc_content_is_plain_text():
    results = search_or_skip(EuropePMCProvider(), LITERATURE_TOPIC, max_results=3)
    tagged = [r for r in results if "<" in r.content and ">" in r.content]
    assert not tagged, f"HTML markup in content, e.g. {tagged[0].content[:120]!r}"


# ══════════════════════════════════════════════════════════════
#  ClinicalTrials.gov
# ══════════════════════════════════════════════════════════════


def test_clinicaltrials_responds():
    results = search_or_skip(ClinicalTrialsProvider(), TRIAL_TOPIC, max_results=3)
    assert_wellformed(results, source_type="clinicaltrials", min_results=2)
    assert_substantive(results, min_chars=200, source_type="clinicaltrials")
    assert all("/study/NCT" in r.url for r in results), "malformed NCT URL"


def test_clinicaltrials_finds_the_landmark_trials():
    relevance = httpx.get(
        "https://clinicaltrials.gov/api/v2/studies",
        params={"query.term": TRIAL_TOPIC, "pageSize": 10, "sort": "@relevance"},
        headers={"User-Agent": "EBM-Lens-tests/1.0", "Accept": "application/json"},
        timeout=30,
    )
    relevance.raise_for_status()
    expected = {
        s["protocolSection"]["identificationModule"]["nctId"]
        for s in relevance.json()["studies"]
    }
    got = {r.url.rsplit("/", 1)[-1] for r in ClinicalTrialsProvider().search(TRIAL_TOPIC, 5)}
    assert expected & got, (
        f"none of the 10 most relevant trials came back; provider returned {sorted(got)}, "
        f"relevance-ranked are {sorted(expected)}"
    )


# ══════════════════════════════════════════════════════════════
#  openFDA
# ══════════════════════════════════════════════════════════════


def test_openfda_responds_for_a_common_drug():
    results = search_or_skip(OpenFDAProvider(), COMMON_DRUG, max_results=3)
    assert_wellformed(results, source_type="openfda", min_results=1)
    assert_substantive(results, min_chars=500, source_type="openfda")
    assert_mentions(results, COMMON_DRUG, source_type="openfda")


def test_openfda_adverse_event_path_responds():
    results = search_or_skip(OpenFDAProvider(), f"{COMMON_DRUG} adverse reactions", max_results=3)
    assert_wellformed(results, source_type="openfda")
    assert any("adverse" in r.title.lower() or "ADVERSE" in r.content for r in results), (
        "a safety-worded query returned no adverse-event content"
    )


def test_openfda_returns_the_label_of_the_drug_asked_for():
    results = search_or_skip(OpenFDAProvider(), DRUG, max_results=3)
    assert results, f"no openFDA label at all for {DRUG!r}"
    wrong = [r for r in results if DRUG.lower() not in r.title.lower()]
    assert not wrong, (
        f"openFDA returned labels for other drugs: {[r.title for r in wrong]}"
    )


def test_openfda_results_are_tier_one():
    results = search_or_skip(OpenFDAProvider(), COMMON_DRUG, max_results=2)
    assert results and all(r.reliability_tier == 1 for r in results), (
        f"tiers: {[r.reliability_tier for r in results]}"
    )


# ══════════════════════════════════════════════════════════════
#  DailyMed
# ══════════════════════════════════════════════════════════════


def test_dailymed_responds():
    results = search_or_skip(DailyMedProvider(), COMMON_DRUG, max_results=3)
    assert_wellformed(results, source_type="dailymed", min_results=1)
    assert all("setid=" in r.url for r in results), "malformed DailyMed URL"


def test_dailymed_returns_actual_label_text():
    results = search_or_skip(DailyMedProvider(), COMMON_DRUG, max_results=2)
    assert results
    assert_substantive(results, min_chars=800, source_type="dailymed")
    assert not any("details unavailable" in r.content for r in results)


# ══════════════════════════════════════════════════════════════
#  EMA
# ══════════════════════════════════════════════════════════════


def test_ema_responds():
    results = search_or_skip(EMAProvider(), DRUG, max_results=3)
    assert_wellformed(results, source_type="ema", min_results=1)
    assert all(r.reliability_tier == 1 for r in results), (
        "an EPAR record is the agency's own output - it must stay tier 1"
    )


def test_ema_returns_actual_epar_text():
    results = search_or_skip(EMAProvider(), DRUG, max_results=2)
    assert results
    assert_substantive(results, min_chars=200, source_type="ema")
    assert_mentions(results, DRUG, source_type="ema")


def test_ema_word_boundary_rejects_the_substring_false_positive():
    """Regression: 'cosa cura Ozempic' used to match the veterinary medicine
    Dicural via the bare substring 'cura' - see app/sources/ema.py::_Term.
    """
    results = search_or_skip(EMAProvider(), "cosa cura Ozempic", max_results=5)
    assert results
    assert all("dicural" not in r.title.lower() for r in results), (
        "the word-boundary fix regressed - Dicural leaked back into an Ozempic query"
    )


def test_ema_index_is_cached_after_the_first_query():
    """The dataset has no query API, so the whole index is downloaded once and
    held for a day - a second query must cost no network at all.
    """
    import time

    reset_ema_index()
    provider = EMAProvider()
    first = search_or_skip(provider, DRUG, max_results=1)
    assert first, "cold index load returned nothing"

    start = time.monotonic()
    second = provider.search(DRUG, max_results=1)
    elapsed = time.monotonic() - start
    assert second
    assert elapsed < 1.0, f"second query took {elapsed:.2f}s - the index was not cached"


# ══════════════════════════════════════════════════════════════
#  RxNav / RxNorm
# ══════════════════════════════════════════════════════════════


def test_rxnav_responds():
    results = search_or_skip(RxNavProvider(), "warfarin", max_results=3)
    assert_wellformed(results, source_type="rxnav", min_results=1)
    assert_mentions(results, "warfarin", source_type="rxnav")


def test_rxnav_resolves_a_brand_name_to_its_molecule():
    """The brand->molecule hop the searcher relies on before querying anything."""
    resolved = RxNavProvider().resolve_drug("Coumadin")
    assert resolved.get("matched"), f"RxNorm could not resolve Coumadin: {resolved}"
    assert "warfarin" in (resolved.get("molecule") or "").lower(), resolved


def test_rxnav_interaction_query_returns_plain_drug_info_not_interactions():
    """NLM retired the interaction API on 2024-01-02 - it is not coming back.

    An interaction-worded query now deliberately returns identity/RxCUI
    records, same as any other query; interaction text comes from openfda
    and dailymed label sections instead (see app/sources/rxnav.py docstring).
    """
    results = search_or_skip(RxNavProvider(), "warfarin drug interaction", max_results=3)
    assert results, "no drug-info result for an interaction-worded query"
    assert not any("interaction" in r.title.lower() for r in results), (
        f"got an 'interaction' result - a working replacement may have been "
        f"wired in without updating this test: {[r.title[:60] for r in results]}"
    )


# ══════════════════════════════════════════════════════════════
#  Open Targets
# ══════════════════════════════════════════════════════════════


def test_open_targets_responds():
    results = search_or_skip(OpenTargetsProvider(), DRUG, max_results=3)
    assert_wellformed(results, source_type="open_targets", min_results=1)
    assert_mentions(results, DRUG, source_type="open_targets")


def test_open_targets_returns_mechanism_and_indications():
    results = search_or_skip(OpenTargetsProvider(), DRUG, max_results=2)
    assert results
    blob = " ".join(r.content for r in results)
    assert "Mechanisms of Action" in blob, "no mechanism-of-action block in content"
    assert "Indications" in blob, "no indications block in content"


# ══════════════════════════════════════════════════════════════
#  ChEMBL
# ══════════════════════════════════════════════════════════════


def test_chembl_responds():
    results = search_or_skip(ChEMBLProvider(), "aspirin", max_results=3)
    assert_wellformed(results, source_type="chembl", min_results=1)


def test_chembl_finds_the_named_approved_molecule():
    results = search_or_skip(ChEMBLProvider(), DRUG, max_results=3)
    assert results
    assert any(DRUG.lower() in r.title.lower() for r in results), (
        f"ChEMBL never returned {DRUG}; got {[r.title for r in results]}"
    )
    assert not any("Max clinical phase: None" in r.content for r in results)


# ══════════════════════════════════════════════════════════════
#  WHO GHO
# ══════════════════════════════════════════════════════════════


def test_who_gho_responds_for_an_indicator_phrase():
    results = search_or_skip(WHOGHOProvider(), EPI_TOPIC, max_results=3)
    assert_wellformed(results, source_type="who_gho", min_results=1)
    assert all("indicator-details" in r.url for r in results)


def test_who_gho_returns_actual_data_points():
    results = search_or_skip(WHOGHOProvider(), EPI_TOPIC, max_results=2)
    assert results
    assert not all("Data unavailable" in r.content for r in results), (
        "every indicator came back without data"
    )


def test_who_gho_survives_a_natural_language_question():
    results = search_or_skip(WHOGHOProvider(), "What is the measles mortality rate in children", max_results=3)
    assert results, "a plain epidemiological question returned nothing"


# ══════════════════════════════════════════════════════════════
#  bioRxiv / medRxiv (via Crossref)
# ══════════════════════════════════════════════════════════════


def test_biorxiv_responds():
    results = search_or_skip(BiorxivProvider(), PREPRINT_TOPIC, max_results=3)
    assert_wellformed(results, source_type="biorxiv", min_results=1)
    assert all(r.reliability_tier == 3 for r in results), (
        "preprints must stay tier 3 - they are not peer reviewed"
    )


def test_biorxiv_only_returns_biorxiv_or_medrxiv():
    """Crossref's posted-content type spans every preprint server."""
    results = search_or_skip(BiorxivProvider(), PREPRINT_TOPIC, max_results=5)
    off_server = [
        r for r in results
        if "10.1101" not in r.url.lower() and "biorxiv" not in r.content.lower()
        and "medrxiv" not in r.content.lower()
    ]
    assert not off_server, f"non-bioRxiv preprints leaked in: {[r.url for r in off_server]}"


# ══════════════════════════════════════════════════════════════
#  Wikipedia
# ══════════════════════════════════════════════════════════════


def test_wikipedia_responds_in_italian_and_english():
    results = WikipediaProvider(languages=("it", "en")).search(ITALIAN_TOPIC, max_results=3)
    assert_wellformed(results, source_type="wikipedia", min_results=2)
    assert_substantive(results, min_chars=500, source_type="wikipedia")
    langs = {"IT" if "[Wikipedia IT]" in r.title else "EN" for r in results}
    assert langs == {"IT", "EN"}, f"expected both editions, got {langs}"


def test_wikipedia_skips_disambiguation_pages():
    results = WikipediaProvider(languages=("en",)).search("Mercury", max_results=3)
    for r in results:
        assert "may refer to:" not in r.content[:200].lower()


def test_wikipedia_sets_the_language_field():
    results = WikipediaProvider(languages=("it", "en")).search(ITALIAN_TOPIC, max_results=2)
    assert results and all(r.language in {"it", "en"} for r in results), (
        f"languages: {[r.language for r in results]}"
    )


# ══════════════════════════════════════════════════════════════
#  Citation-graph expansion
# ══════════════════════════════════════════════════════════════


def test_citation_expansion_returns_neighbours():
    from app.sources.base import SourceResult
    from app.sources.citation_expander import expand_with_citation_graph

    seed = SourceResult(
        title="Cardiovascular outcomes with GLP-1 receptor agonists",
        url="https://pubmed.ncbi.nlm.nih.gov/31422062/",
        snippet="",
        content="",
        source_type="pubmed",
        citation_count=1000,
    )
    expanded = expand_with_citation_graph([seed], max_seeds=1, per_seed=5, bidirectional=True)
    if not expanded:
        pytest.skip("Semantic Scholar / OpenAlex returned nothing (often rate-limiting)")
    for r in expanded:
        assert r.title.strip()
        assert r.url.startswith("http")
        assert r.source_type in {"semantic_scholar", "openalex"}
