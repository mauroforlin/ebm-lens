"""Multi-source evidence search: fan out to the given providers, degrade gracefully.

Turning a topic into evidence means asking the *right* databases the *right*
way. PubMed wants MeSH-flavoured English; DailyMed wants a bare drug name and
returns nothing for a disease; Wikipedia is best asked in the user's language.
Sending one query to everything wastes most of the calls.

Which providers a topic searches is decided once per topic, upstream, by the
discovery loop's research brief (`app.pipeline.agentic._build_brief`) -
grounded in real RxNorm/PubMed lookups via the tools in `planner_tools.py`,
not reworked here on every query variant. This module runs that provider
list, then degrades if it came back thin:

1. **Query retries** - alternative queries, then a deliberately broader
   conceptual query, get another pass at the providers already in play.
2. **Rule-based routing** - a deterministic router adds providers from the
   topic type. The pipeline never depends on an LLM being available in order
   to return results.

Everything within a tier runs in parallel under a shared wall-clock budget,
and every provider result is cached in process (see :mod:`app.cache`).
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from typing import TYPE_CHECKING

from app.config import Settings
from app.core.cache import source_cache
from app.schemas import DomainContext, TopicSpec
from app.sources.base import SourceProvider, SourceResult
from app.sources.biorxiv import BiorxivProvider
from app.sources.blocklist import is_blocked_url
from app.sources.chembl import ChEMBLProvider
from app.sources.clinicaltrials import ClinicalTrialsProvider
from app.sources.dailymed import DailyMedProvider
from app.sources.ema import EMAProvider
from app.sources.europe_pmc import EuropePMCProvider
from app.sources.open_targets import OpenTargetsProvider
from app.sources.openfda import OpenFDAProvider
from app.sources.pubmed import PubMedProvider
from app.sources.rxnav import RxNavProvider
from app.sources.who_gho import WHOGHOProvider
from app.sources.wikipedia import WikipediaProvider

if TYPE_CHECKING:
    from app.core.job_stats import JobStats

logger = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────

_MIN_EVIDENCE = 3       # below this, the next retry tier is worth its latency
_MAX_PER_PROVIDER = 8   # results requested per provider per query

# The providers that actually carry papers/trials, as opposed to reference,
# regulatory or definitional lookups. The retry tiers below exist to reach
# these when the planned search came back thin - counting *any* provider
# toward `_MIN_EVIDENCE` lets a couple of quick Wikipedia/EMA/DailyMed hits
# satisfy the threshold and skip straight past them, leaving PubMed at
# whatever its single planned query returned.
_LITERATURE_PROVIDERS = frozenset({"pubmed", "europe_pmc", "biorxiv", "clinicaltrials"})

_MEDICAL_TOPIC_TYPES = frozenset({
    "drug", "dosage", "drug_classification", "mechanism_of_action",
    "contraindication", "drug_interaction", "adverse_effect",
    "diagnostic_criteria", "reference_value", "epidemiology",
    "statistic",  # anatomical/physiological statistics are medical
})
_MEDICAL_DOMAINS = frozenset({"medicine", "veterinary_medicine"})


def _is_medical(domain: str, spec: TopicSpec) -> bool:
    return domain in _MEDICAL_DOMAINS or spec.topic_type in _MEDICAL_TOPIC_TYPES


# ── Provider registry ─────────────────────────────────────────

_PROVIDER_CONSTRUCTORS: dict[str, type] = {
    "wikipedia": WikipediaProvider,
    "pubmed": PubMedProvider,
    "biorxiv": BiorxivProvider,
    "clinicaltrials": ClinicalTrialsProvider,
    "openfda": OpenFDAProvider,
    "europe_pmc": EuropePMCProvider,
    "rxnav": RxNavProvider,
    "dailymed": DailyMedProvider,
    "open_targets": OpenTargetsProvider,
    "who_gho": WHOGHOProvider,
    "chembl": ChEMBLProvider,
    "ema": EMAProvider,
}

PROVIDER_IDS: tuple[str, ...] = tuple(_PROVIDER_CONSTRUCTORS)


def _get_provider(
    pid: str,
    *,
    pubmed_kwargs: dict | None = None,
    europe_pmc_kwargs: dict | None = None,
) -> SourceProvider | None:
    """Instantiate a provider by its plan id, or None if the id is unknown.

    Two providers take per-topic construction options: PubMed (MeSH terms,
    publication-type filters) and Europe PMC (review/recency preference).
    """
    if pid == "wikipedia":
        return WikipediaProvider(languages=("it", "en"))
    if pid == "pubmed":
        return PubMedProvider(**(pubmed_kwargs or {}))
    if pid == "europe_pmc":
        return EuropePMCProvider(**(europe_pmc_kwargs or {}))
    cls = _PROVIDER_CONSTRUCTORS.get(pid)
    return cls() if cls else None


# ══════════════════════════════════════════════════════════════
#  1. PROVIDER SELECTION
# ══════════════════════════════════════════════════════════════

# The literature engines: on for every medical topic, unconditionally - the
# actual primary sources a medical question is answered from, not a floor to
# top up towards. Wikipedia is deliberately absent here: evidence_grade.py
# scores it lowest on the evidence hierarchy and selection.py caps it to one
# article in the final answer, so including it in a medical plan would only
# spend a query slot and pool-relevance budget on a source the ranking has
# already decided not to trust as evidence. It stays the sole provider for a
# non-medical topic, where there is nothing else to ask.
_MEDICAL_LITERATURE = ["pubmed", "europe_pmc", "clinicaltrials"]

# topic_type -> the providers that specific kind of question needs, added
# unconditionally whenever the type matches: a dosage question needs
# dailymed/ema regardless of what else the plan already has, and a topic
# whose type names none of these gets the literature baseline alone.
_TYPE_PROVIDERS: dict[str, list[str]] = {
    "contraindication": ["openfda", "dailymed", "rxnav"],
    "drug_interaction": ["openfda", "dailymed", "rxnav"],
    "dosage": ["dailymed", "ema"],
    "drug": ["dailymed", "ema"],
    "drug_classification": ["ema", "rxnav"],
    "adverse_effect": ["openfda", "dailymed"],
    "mechanism_of_action": ["open_targets", "chembl"],
    "epidemiology": ["who_gho"],
    "statistic": ["who_gho"],
}

# who_gho, ema, openfda and rxnav stay noisy even once topic_type is right,
# because the classifier tags "epidemiology"/"dosage"/"adverse_effect"/
# "drug_interaction" more liberally than any of the four actually warrant:
# a population-rate database, a drug regulator, an adverse-event register and
# a drug-interaction lookup each need the topic to name the thing they are
# records of, not just fall into a broad category. Gated on the topic's own
# text rather than a second LLM field, since a rate, a named medicine, a
# side-effect or an interaction is something the user's own wording either
# states or does not.
_EPIDEMIOLOGY_KEYWORDS = (
    "prevalence", "prevalent", "incidence", "mortality", "morbidity",
    "epidemiology", "epidemiological", "how common", "how widespread",
    "how many people", "frequency of", "number of cases", "per 100,000",
    "per capita", "global burden", "death rate", "case rate",
    "prevalenza", "incidenza", "mortalità", "diffusione",
    "quanto è comune", "quanto è diffus", "quante persone",
    "colpisce quante persone", "tasso di incidenza", "tasso di mortalità",
)
_REGULATORY_KEYWORDS = (
    "approv", "authoriz", "authoris", "regulatory", "marketing authorisation",
    "marketing authorization", "label", "ema", "european medicines agency",
    "withdrawn", "recall", "epar", "centrally authorised",
    "orphan drug", "orphan medicine", "biosimilar", "conditional marketing",
    "revoked", "suspended marketing", "autorizzazione", "approvazione",
    "immissione in commercio", "farmaco orfano",
)
_SAFETY_KEYWORDS = (
    "side effect", "side-effect", "adverse event", "adverse effect",
    "adverse reaction", "safety signal", "black box", "boxed warning",
    "recall", "faers", "reported cases of", "case reports of",
    "effetto collaterale", "effetti collaterali", "evento avverso",
    "eventi avversi", "reazione avversa", "reazioni avverse",
)
_INTERACTION_KEYWORDS = (
    "interact", "interaction", "contraindicat", "combined with",
    "co-administer", "coadminister", "concomitant", "taken with",
    "drug class", "classified as", "classification of",
    "interazione", "controindicat", "somministrato insieme",
    "classe farmacologica", "classificazione",
)


def _mentions_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(kw in lowered for kw in keywords)


def _drop_ungated(spec: TopicSpec, providers: set[str] | list[str]) -> set[str]:
    """Strip who_gho/ema/openfda/rxnav out of *providers* unless the topic's
    own text explicitly names what each one is a record of - a population
    rate, a named medicine's regulatory status, a reported side effect, or a
    drug interaction/classification.

    Applied to a provider set from any source, not just `_TYPE_PROVIDERS`'s
    own entries: the research brief's own prompt tells the model to propose
    these directly - who_gho "for epidemiology", ema "for regulatory
    status" - on exactly as loose a bar as topic_type itself, so a gate that
    only covered the deterministic list would leave that route wide open.
    """
    chosen = set(providers)
    text = spec.text or ""
    subject = spec.subject.strip()

    if "who_gho" in chosen and not _mentions_any(text, _EPIDEMIOLOGY_KEYWORDS):
        chosen.discard("who_gho")
    if "ema" in chosen and not (subject and _mentions_any(text, _REGULATORY_KEYWORDS)):
        chosen.discard("ema")
    if "openfda" in chosen and not (subject and _mentions_any(text, _SAFETY_KEYWORDS)):
        chosen.discard("openfda")
    if "rxnav" in chosen and not (subject and _mentions_any(text, _INTERACTION_KEYWORDS)):
        chosen.discard("rxnav")

    return chosen


def _gated_type_providers(spec: TopicSpec) -> list[str]:
    """`_TYPE_PROVIDERS` for *spec*'s topic_type, gated through `_drop_ungated`."""
    providers = _TYPE_PROVIDERS.get((spec.topic_type or "").strip().lower(), [])
    return list(_drop_ungated(spec, providers))


def select_providers(
    spec: TopicSpec,
    domain: str,
    brief_providers: list[str] | None = None,
) -> list[str]:
    """The provider ids a topic should search - one decision, one place.

    A non-medical topic searches Wikipedia plus whatever the caller proposed.
    A medical topic gets its literature baseline (`_MEDICAL_LITERATURE`) plus
    whatever `_TYPE_PROVIDERS` names for its topic_type, plus the caller's own
    proposal folded in rather than overridden. Every provider in the result
    earns its place by one of those two routes - none is added to reach a
    headcount.
    """
    if not _is_medical(domain, spec):
        chosen = {pid for pid in (brief_providers or []) if pid in PROVIDER_IDS}
        chosen.add("wikipedia")
        return [pid for pid in PROVIDER_IDS if pid in chosen]

    chosen = set(_MEDICAL_LITERATURE)
    chosen.update(
        pid for pid in (brief_providers or []) if pid in PROVIDER_IDS and pid != "wikipedia"
    )
    chosen.update(_TYPE_PROVIDERS.get((spec.topic_type or "").strip().lower(), []))
    chosen = _drop_ungated(spec, chosen)
    return [pid for pid in PROVIDER_IDS if pid in chosen]


def _select_providers_fallback(
    spec: TopicSpec,
    domain: str,
) -> tuple[list[SourceProvider], list[SourceProvider]]:
    """Return ``(base, supplementary)`` provider instances, chosen without an LLM.

    This is what keeps the tool working when the LLM is down, rate-limited or
    returning nonsense: worse query targeting, but always an answer. Packages
    `select_providers`'s decision as instantiated providers split into a base
    tier (literature, or Wikipedia alone for a non-medical topic) and a
    supplementary tier, so the caller can try base first and supp only if
    that still comes back thin.
    """
    pubmed_kwargs = _pubmed_kwargs(spec)
    epmc_kwargs = _europe_pmc_kwargs(pubmed_kwargs)
    ids = select_providers(spec, domain)

    def _instantiate(pid: str) -> SourceProvider | None:
        return _get_provider(pid, pubmed_kwargs=pubmed_kwargs, europe_pmc_kwargs=epmc_kwargs)

    base_ids = [pid for pid in ids if pid in {"pubmed", "europe_pmc", "wikipedia"}]
    supp_ids = [pid for pid in ids if pid not in base_ids]

    base = [p for p in (_instantiate(pid) for pid in base_ids) if p is not None]
    supp = [p for p in (_instantiate(pid) for pid in supp_ids) if p is not None]
    return base, supp


# ══════════════════════════════════════════════════════════════
#  2. PER-PROVIDER QUERY CONSTRUCTION
# ══════════════════════════════════════════════════════════════

# Topic types whose answer lives in a review or a textbook, not in the latest
# primary study - PubMed/Europe PMC are told to prefer review articles.
_FOUNDATIONAL_TYPES = frozenset({
    "assertion", "definition", "anatomy", "physiology",
    "mechanism_of_action", "pathophysiology", "classification",
    "drug_classification", "statistic", "reference_value",
    "diagnostic_criteria",
})
# Topic types where a ten-year-old figure is simply the wrong answer.
_RECENT_TYPES = frozenset({"epidemiology", "prevalence", "incidence"})

# Topic type → the publication type that actually answers it.
_TYPE_TO_EVIDENCE = {
    "dosage": "guideline",
    "drug_interaction": "systematic_review",
    "contraindication": "guideline",
    "adverse_effect": "systematic_review",
    "clinical_guideline": "guideline",
    "diagnostic_criteria": "guideline",
}


def _pubmed_kwargs(spec: TopicSpec) -> dict:
    """Build PubMed constructor kwargs from the topic spec."""
    kwargs: dict = {}

    mesh_terms: list[str] = []
    for query in spec.alternative_queries:
        for term in re.findall(r'"([^"]+)"\[MeSH Terms\]', query):
            if term not in mesh_terms:
                mesh_terms.append(term)
    if mesh_terms:
        kwargs["mesh_terms"] = mesh_terms

    drug_topic = spec.topic_type in (
        "dosage", "drug", "drug_classification", "drug_interaction",
    )
    if drug_topic and spec.subject.strip():
        kwargs["drug_names"] = [spec.subject.strip()]

    evidence_type = _TYPE_TO_EVIDENCE.get(spec.topic_type)
    if evidence_type:
        kwargs["evidence_type"] = evidence_type

    if spec.topic_type in _FOUNDATIONAL_TYPES:
        kwargs["prefer_reviews"] = True
    if spec.topic_type in _RECENT_TYPES:
        kwargs["recent_only"] = True

    return kwargs


def _europe_pmc_kwargs(pubmed_kwargs: dict) -> dict:
    """Europe PMC shares PubMed's review/recency preference, nothing else."""
    return {
        "prefer_reviews": pubmed_kwargs.get("prefer_reviews", False),
        "recent_only": pubmed_kwargs.get("recent_only", False),
    }


def _english_query(spec: TopicSpec) -> str:
    """The English query for *spec*, or a deterministic reconstruction of one.

    A specified PICO reconstructs a better query than the subject alone: its
    elements are already in the field's terminology and name the constraints
    a database can act on.
    """
    if spec.search_query:
        return spec.search_query.strip()
    if spec.pico and spec.pico.is_specified():
        return " ".join(spec.pico.terms())[:120]
    parts = [spec.subject, spec.value or ""]
    return " ".join(p for p in parts if p).strip() or spec.text[:60]


def _italian_query(spec: TopicSpec) -> str:
    """The Italian query for *spec*, or a deterministic reconstruction of one."""
    if spec.search_query_it:
        return spec.search_query_it.strip()

    parts = [p for p in (spec.subject, str(spec.value) if spec.value else "") if p]
    if parts:
        query = " ".join(parts)
        if len(query) >= 8:
            return query

    text = spec.text.strip()
    if len(text) > 80:
        text = text[:80].rsplit(" ", 1)[0]
    return text


def _with_keywords(query: str, keywords: list[str]) -> str:
    """Append domain keywords not already present in *query*.

    Disambiguation, cheaply: "resistance" plus "antimicrobial" retrieves a very
    different set than "resistance" alone.
    """
    if not keywords:
        return query
    lowered = query.lower()
    extras = [kw for kw in keywords if kw.lower() not in lowered]
    return f"{query} {' '.join(extras)}" if extras else query


_SUBJECT_ANCHORED_PROVIDERS = frozenset({"ema", "openfda", "dailymed"})


def _ensure_subject(spec: TopicSpec, pid: str, query: str) -> str:
    """Prepend the topic's named drug to *query* for a register that indexes
    by drug identity rather than free text.

    ema/openfda/dailymed match against a medicine's own name, INN or active
    substance - the discriminating term for them is the drug itself, which
    can sit anywhere in (or be paraphrased out of) an English sentence
    optimised for PubMed's free-text search rather than for a drug register.
    """
    if pid not in _SUBJECT_ANCHORED_PROVIDERS:
        return query
    subject = spec.subject.strip()
    if not subject or subject.lower() in query.lower():
        return query
    return f"{subject} {query}"


def _build_query_for_provider(
    spec: TopicSpec,
    provider: SourceProvider,
    context: DomainContext | None = None,
) -> str:
    """Generate a query in the language and style *provider* expects."""
    if provider.source_type == "wikipedia":
        return _with_keywords(
            _italian_query(spec), context.keywords_it if context else [],
        )

    query = _english_query(spec)
    if provider.source_type == "pubmed":
        mesh_query = next((q for q in spec.alternative_queries if "[MeSH" in q), None)
        if mesh_query:
            query = mesh_query
    return _with_keywords(query, context.keywords_en if context else [])


# ══════════════════════════════════════════════════════════════
#  3. CACHE LAYER (in-process, see app.cache)
# ══════════════════════════════════════════════════════════════


def _cache_key(query: str, source_type: str) -> str:
    """Hash a (provider, query) pair, normalising word order and punctuation.

    "aspirin and stroke" and "stroke aspirin" hit the same entry: different
    phrasings of one search should not each cost a round trip.
    """
    normalised = re.sub(r"[^\w\s]", "", query.lower())
    normalised = " ".join(sorted(normalised.split()))
    return hashlib.sha256(f"{source_type}:{normalised}".encode()).hexdigest()


def _serialise(result: SourceResult) -> dict:
    """Flatten a result for the cache, capping stored page content."""
    return {
        "title": result.title,
        "url": result.url,
        "snippet": result.snippet,
        "content": result.content[:4000],
        "source_type": result.source_type,
        "reliability_tier": result.reliability_tier,
        "publication_date": result.publication_date,
        "citation_count": result.citation_count,
        "language": result.language,
        "publication_types": list(result.publication_types),
    }


def _run_search(
    provider: SourceProvider,
    query: str,
    ttl_days: int,
    job_stats: JobStats | None = None,
    max_results: int = _MAX_PER_PROVIDER,
) -> list[SourceResult]:
    """Search one provider, serving from cache when possible."""
    key = _cache_key(query, provider.source_type)
    cached = source_cache.get(key)
    if cached is not None:
        logger.debug("Cache HIT  [%s] q='%s'", provider.source_type, query[:50])
        if job_stats:
            job_stats.record_provider_call(provider.source_type, cached=True)
        return [SourceResult(**r) for r in cached]

    logger.debug("Searching  [%s] q='%s'", provider.source_type, query[:50])
    results = provider.search(query, max_results=max_results)
    if job_stats:
        job_stats.record_provider_call(provider.source_type, cached=False)

    source_cache.set(key, [_serialise(r) for r in results], ttl_days * 86400)
    return results


# ══════════════════════════════════════════════════════════════
#  4. MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════


class _Collector:
    """Accumulates results across search tiers, deduplicating by URL.

    Providers overlap heavily (Europe PMC is a superset of PubMed) and the
    retry tiers deliberately re-query the same providers with new wording, so
    the same URL arrives repeatedly. Blocked domains are rejected on arrival
    rather than filtered at the end, so a tier's "did we get enough?" check
    counts only results that can actually be used.
    """

    def __init__(self) -> None:
        self.results: list[SourceResult] = []
        self._seen_urls: set[str] = set()

    def add(self, hits: list[SourceResult]) -> None:
        for result in hits:
            url = result.url
            if not url or url in self._seen_urls or is_blocked_url(url):
                continue
            self._seen_urls.add(url)
            self.results.append(result)

    def __len__(self) -> int:
        return len(self.results)


def search_evidence(
    *,
    spec: TopicSpec,
    domain: str,
    settings: Settings,
    context: DomainContext | None = None,
    job_stats: JobStats | None = None,
    max_per_provider: int | None = None,
    providers_override: list[str] | None = None,
) -> list[SourceResult]:
    """Search every relevant provider for *spec* and return merged evidence.

    Tiers run in order and each is skipped once ``_MIN_EVIDENCE`` results are
    in hand: given provider list → alternative queries → broader conceptual
    query → rule-based routing. All provider calls within a tier run
    concurrently under a single wall-clock budget (``SEARCH_TIMEOUT_SECONDS``),
    so a slow or hanging API costs latency but never the whole request.

    *providers_override* is the provider list to search, with the spec's own
    query. The discovery loop supplies one, grounded once per topic in its
    own research brief: it issues many query variants concurrently, so this
    keeps the grounding to one call per topic rather than one per variant.
    """
    from app.pipeline.evidence_cache import get_topic_evidence, save_topic_evidence

    ttl_days = settings.source_cache_ttl_days
    per_provider = max_per_provider or _MAX_PER_PROVIDER
    collector = _Collector()

    # ── Tier 0: whole-topic cache ──
    cached_evidence = get_topic_evidence(spec.text)
    if cached_evidence is not None:
        return [r for r in cached_evidence if not is_blocked_url(r.url)]

    # Planning runs once per topic upstream, in the research brief - this
    # placeholder just keeps the deadline set right before the provider I/O
    # below starts.
    deadline = 0.0

    def _time_left() -> float:
        return max(0.0, deadline - time.monotonic())

    def _search_one(provider: SourceProvider, query: str) -> list[SourceResult]:
        """Run one provider search, converting any failure into an empty result.

        A provider raising must never abort the run: the whole design is that
        twelve sources are consulted and some of them will be down.
        """
        try:
            return _run_search(
                provider, query, ttl_days,
                job_stats=job_stats, max_results=per_provider,
            )
        except Exception as exc:
            logger.debug(
                "Search failed [%s] q='%s': %s", provider.source_type, query[:50], exc,
            )
            if job_stats:
                job_stats.record_provider_call(provider.source_type, error=True)
            return []

    def _run_parallel(
        tasks: list[tuple[SourceProvider, str]], max_workers: int = 6,
    ) -> None:
        if not tasks or _time_left() <= 0:
            return
        # Not a context manager: `with ThreadPoolExecutor(...)` calls
        # shutdown(wait=True) on exit regardless of the as_completed timeout
        # below, so a single slow provider call would silently block this
        # function (and the whole request) until it finished anyway - the
        # budget existed in name only. shutdown(wait=False, cancel_futures=True)
        # returns as soon as the budget is up; still-running calls finish in
        # the background and their results are simply never collected.
        pool = ThreadPoolExecutor(max_workers=min(max_workers, len(tasks)))
        try:
            futures = [pool.submit(_search_one, prov, query) for prov, query in tasks]
            try:
                for fut in as_completed(futures, timeout=_time_left()):
                    collector.add(fut.result())
            except TimeoutError:
                logger.warning(
                    "Search budget exhausted (%ds)", settings.search_timeout_seconds,
                )
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    medical = _is_medical(domain, spec)
    pubmed_kwargs = _pubmed_kwargs(spec)
    epmc_kwargs = _europe_pmc_kwargs(pubmed_kwargs)

    def _task(pid: str, query: str) -> tuple[SourceProvider, str] | None:
        provider = _get_provider(
            pid, pubmed_kwargs=pubmed_kwargs, europe_pmc_kwargs=epmc_kwargs,
        )
        if provider is None:
            logger.debug("Skipping unknown provider '%s'", pid)
            return None
        return (provider, query)

    def _tasks(pids: list[str], query: str) -> list[tuple[SourceProvider, str]]:
        built = (_task(pid, query) for pid in pids)
        return [t for t in built if t is not None]

    def _needs_more(collector: _Collector) -> bool:
        """Whether the tier chain should still try another query.

        For medical topics this counts literature-provider hits only: a
        non-medical topic has no such distinction, but a medical one can
        clear `_MIN_EVIDENCE` on Wikipedia/EMA/DailyMed hits alone while
        PubMed contributed nothing beyond its single planned query - exactly
        the case these retry tiers exist to catch.
        """
        if not medical:
            return len(collector) < _MIN_EVIDENCE
        literature = sum(1 for r in collector.results if r.source_type in _LITERATURE_PROVIDERS)
        return literature < _MIN_EVIDENCE

    # ── Tier 1: the given provider list ──
    # The discovery loop supplies one, grounded once per topic in its own
    # research brief. A caller with no list of its own skips straight to the
    # retry/rule-based tiers below, the same degrade path an LLM outage takes.
    plan: list[dict[str, str]] | None = None
    if providers_override is not None:
        override_query = (spec.search_query or spec.text).strip()
        plan = [
            {"provider": pid, "query": _ensure_subject(spec, pid, override_query)}
            for pid in providers_override
        ]

    deadline = time.monotonic() + settings.search_timeout_seconds

    if plan is not None:
        planned: list[tuple[SourceProvider, str]] = []
        for entry in plan:
            task = _task(entry["provider"], entry["query"])
            if task is not None:
                planned.append(task)
        _run_parallel(planned)

        # The discovery loop gets its breadth from the many query variants it
        # runs itself, so the serial retry tiers below would only add latency
        # to work already happening in parallel elsewhere.
        if providers_override is not None:
            return collector.results

    # ── Tier 2: alternative queries ──
    retry_queries = spec.alternative_queries_it or spec.alternative_queries
    if _needs_more(collector) and retry_queries:
        logger.debug(
            "Primary search yielded %d results - trying %d alternative queries",
            len(collector), len(retry_queries),
        )
        if medical:
            retry_pids = ["pubmed", "europe_pmc"] + _gated_type_providers(spec)[:2]
        else:
            retry_pids = ["wikipedia"]
        alt_tasks: list[tuple[SourceProvider, str]] = []
        for query in retry_queries:
            alt_tasks.extend(_tasks(retry_pids, query))
        _run_parallel(alt_tasks, max_workers=4)

    # ── Tier 3: broader conceptual query ──
    if _needs_more(collector) and (spec.conceptual_query or spec.conceptual_query_it):
        conceptual_en = spec.conceptual_query
        conceptual_it = spec.conceptual_query_it or spec.conceptual_query
        logger.debug(
            "Trying broader conceptual query: '%s'",
            (conceptual_en or conceptual_it)[:60],
        )
        concept_tasks: list[tuple[SourceProvider, str]] = []
        if not medical and conceptual_it:
            concept_tasks.extend(_tasks(["wikipedia"], conceptual_it))
        if conceptual_en:
            concept_tasks.extend(_tasks(
                ["pubmed", "europe_pmc", "clinicaltrials"] if medical else ["pubmed"],
                conceptual_en,
            ))
        _run_parallel(concept_tasks, max_workers=3)

    # ── Tier 4: rule-based routing ──
    if _needs_more(collector):
        logger.debug(
            "Planned search yielded only %d results - supplementing with rules",
            len(collector),
        )
        base_providers, supp_providers = _select_providers_fallback(spec, domain)

        searched = {r.source_type for r in collector.results}
        _run_parallel([
            (provider, _build_query_for_provider(spec, provider, context))
            for provider in base_providers if provider.source_type not in searched
        ])

        if _needs_more(collector):
            searched = {r.source_type for r in collector.results}
            _run_parallel([
                (provider, _build_query_for_provider(spec, provider, context))
                for provider in supp_providers if provider.source_type not in searched
            ])

    if collector.results:
        save_topic_evidence(spec.text, collector.results, ttl_days)

    logger.debug(
        "Evidence search: %d results in %.1fs",
        len(collector), settings.search_timeout_seconds - _time_left(),
    )
    return collector.results
