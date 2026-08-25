"""Agentic multi-round discovery.

A frontier topic - one without a settled body of literature yet - fails a
one-shot search in two specific ways: the seminal works use terminology the
user's phrasing does not contain, and a highly-cited paper about a different
sense of the same words outranks everything genuinely on-topic.

This loop attacks both, reusing the same providers and citation graph:

1. **Research brief** - disambiguates the topic, names the vocabulary an
   on-topic paper will use and the wrong senses to suppress, writes several
   deliberately diverse query variants, and picks the providers this topic
   should search - grounded in real RxNorm/PubMed lookups rather than
   guessed, since a resolved drug name or a probed query costs one tool call
   here and saves the round-1 fan-out from spending on a guess.
2. **Round 1** - every variant fans out in parallel, with a wide per-provider
   cap.
3. **Relevance gate** - candidates are embedded and ranked, and only genuinely
   on-topic papers become citation-graph seeds. Seeding on the most-cited
   instead would expand *around the wrong paper*, multiplying the error.
4. **Citation expansion** - bidirectional from those seeds, running
   concurrently with the rounds below since it depends only on round 1.
5. **Adaptive rounds** - an LLM reads the best hits so far, extracts the terms
   the literature actually uses, and issues refined queries. This repeats
   while each round still surfaces enough new on-topic papers, and stops as
   soon as coverage saturates.

Everything is free-API: the extra recall costs a few more LLM calls, nothing
else.

Iterative retrieval does not uniformly beat one-shot (arXiv:2509.04820): it
pays off when the opening query needs refinement or the answer spans several
parts of the literature, and adds noise when one well-formed query already
suffices. Hence the stopping rule - the loop ends as soon as a round stops
adding on-topic work, rather than running its budget out.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from typing import TYPE_CHECKING

from app.config import Settings
from app.core.events import NULL_EMITTER, Emitter
from app.pipeline.dedup import dedup_key, deduplicate
from app.pipeline.relevance import cosine_map
from app.pipeline.searcher import search_evidence
from app.schemas import DomainContext, TopicSpec
from app.sources.base import SourceResult
from app.sources.citation_expander import expand_with_citation_graph

if TYPE_CHECKING:
    from app.core.job_stats import JobStats

logger = logging.getLogger(__name__)


_QUERY_VARIANTS = 5         # round-1 query variants kept from the brief
_MAX_PER_PROVIDER = 14      # per-provider results kept, wide enough for real recall
_ROUND_QUERIES = 3          # refined queries per reformulation round
_EXPANSION_SEEDS = 4        # citation-graph seeds (relevance-ordered)
_PER_SEED = 6               # neighbours requested per seed
_SEED_COSINE_FLOOR = 0.30   # min similarity to seed expansion or count as new
_REFORMULATE_HITS = 8       # top hits shown to the reformulator
_MAX_WORKERS = 12           # fire all query variants concurrently
_MAX_ROUNDS = 3             # round 1 + up to 2 reformulation rounds
_MIN_NEW_RELEVANT = 3       # stop when a round adds fewer new relevant papers
_FACET_QUERIES = 4          # max composite facet/bridge queries folded into round 1
_SEARCH_TIMEOUT = 90        # seconds for one round's fan-out
_REFORMULATE_TOOL_ROUNDS = 3  # probe budget per reformulation: try, adjust, submit
_BRIEF_TOOL_ROUNDS = 4        # probe budget for the brief: resolve names, probe, submit

_MEDICAL_DOMAINS = ("medicine", "veterinary_medicine")

# Fallback provider set per domain, used only when the brief fails to name
# any (see _build_brief's degrade path) - normally the brief picks providers
# per topic, not per domain.
_DISCOVERY_PROVIDERS: dict[str, list[str]] = {
    "medicine": ["pubmed", "europe_pmc", "biorxiv", "clinicaltrials"],
    "veterinary_medicine": ["pubmed", "europe_pmc", "biorxiv", "clinicaltrials"],
}
_DEFAULT_PROVIDERS = ["pubmed", "europe_pmc", "wikipedia"]


def _providers_for_topic(domain: str, spec: TopicSpec, brief: dict) -> list[str]:
    """Providers to search for this topic: the brief's choice, folded into what
    the topic_type structurally warrants.

    The brief (see _build_brief) names providers per topic - a drug-safety
    question gets openfda/dailymed/rxnav, an epidemiology one gets who_gho -
    and `searcher.select_providers` is the single place that turns that
    proposal, together with the topic_type, into the final list: the same
    decision whether the brief succeeded, named too little, or failed
    outright and left this function working from the domain-level fallback
    above.
    """
    from app.pipeline.searcher import select_providers

    chosen = [p for p in (brief.get("providers") or []) if isinstance(p, str)]
    if not chosen:
        chosen = list(_DISCOVERY_PROVIDERS.get((domain or "").strip().lower(), _DEFAULT_PROVIDERS))

    return select_providers(spec, domain, brief_providers=chosen)


_BRIEF_SYSTEM = """\
You are a senior research librarian planning a literature discovery for a
curious user, focused on medicine and clinical research. Given a TOPIC,
produce a research brief that will drive a multi-query search across
medical/biomedical databases, then submit it by calling submit_brief.

You have tools, and a query or provider choice you checked beats one you
guessed:
- resolve_drug(name) - a brand name retrieves far less than its molecule.
  Call it for every drug name in the topic before writing queries.
- probe_pubmed(query) - hit_count, PubMed's MeSH translation, unmatched
  phrases and sample titles. Probe a query you are unsure of before
  submitting it; a near-zero hit_count means the WORDING is wrong, not that
  the literature is missing.
- search_guidelines(topic) - real clinical practice guideline titles via
  Europe PMC, for treatment-standard/recommendation topics.
- submit_brief(...) - submit the finished brief. ALWAYS end with this call.

submit_brief fields:
  canonical_topic: one precise, disambiguated restatement of the topic
  query_variants: 4-6 DIVERSE English search queries, 4-8 words each
  query_variants_it: 1-2 Italian queries for Italian sources
  must_include_concepts: 3-6 technical terms an on-topic paper will use
  negative_terms: terms that indicate an OFF-topic homonym/wrong sense
  seminal_hints: named landmark trials, methods, drugs or acronyms, if any
  providers: the provider ids this topic should search - always include
    pubmed; add openfda/dailymed/rxnav/ema whenever the topic names a
    specific drug or concerns safety, interactions, dosage or regulatory
    status; clinicaltrials for treatment-efficacy questions; who_gho for
    epidemiology

Rules:
- DISAMBIGUATE aggressively. Identify the precise clinical/biomedical sense
  intended. List the wrong senses in negative_terms.
- query_variants MUST be genuinely diverse: cover synonyms, alternative
  framings, the proper clinical/technical name, and adjacent terminology -
  not five rewordings of the same phrase. They are the field's REAL vocabulary.
- seminal_hints: if you know the foundational trials/papers/drugs/guidelines
  for this exact topic, name them. These help retrieve the works a naive
  keyword search would miss. If unsure, leave the list empty rather than
  guessing unrelated names.
- Keep every query in the field's standard academic English.
"""


_BRIEF_JSON_FALLBACK_SYSTEM = """\
You are a senior research librarian planning a literature discovery for a
curious user, focused on medicine and clinical research. Given a TOPIC,
produce a research brief that will drive a multi-query search across
medical/biomedical databases.

Return a JSON object:
{
  "canonical_topic": "<one precise, disambiguated restatement of the topic>",
  "query_variants": ["4-6 DIVERSE English search queries, 4-8 words each"],
  "query_variants_it": ["1-2 Italian queries for Italian sources"],
  "must_include_concepts": ["3-6 technical terms an on-topic paper will use"],
  "negative_terms": ["terms that indicate an OFF-topic homonym/wrong sense"],
  "seminal_hints": ["named landmark trials, methods, drugs or acronyms, if any"],
  "providers": ["provider ids this topic should search, from: pubmed, "
                "europe_pmc, clinicaltrials, openfda, dailymed, rxnav, ema, "
                "who_gho, chembl, open_targets, biorxiv, wikipedia"]
}

Rules:
- DISAMBIGUATE aggressively. Identify the precise clinical/biomedical sense
  intended. List the wrong senses in negative_terms.
- query_variants MUST be genuinely diverse: cover synonyms, alternative
  framings, the proper clinical/technical name, and adjacent terminology -
  not five rewordings of the same phrase. They are the field's REAL vocabulary.
- seminal_hints: if you know the foundational trials/papers/drugs/guidelines
  for this exact topic, name them. These help retrieve the works a naive
  keyword search would miss. If unsure, leave the list empty rather than
  guessing unrelated names.
- providers: always include pubmed; add openfda/dailymed/rxnav/ema whenever
  the topic names a specific drug or concerns safety, interactions, dosage or
  regulatory status; clinicaltrials for treatment-efficacy questions; who_gho
  for epidemiology.
- Keep every query in the field's standard academic English.
"""


def _build_brief(
    topic: str,
    domain: str,
    settings: Settings,
    job_stats: JobStats | None,
) -> dict:
    """Tool-grounded brief: disambiguated queries, vocabulary and providers.

    A drug name resolved against RxNorm and a query probed against PubMed
    cost one tool round each, here, once per topic - and settle by fact what
    a single ungrounded generation would otherwise guess at before the
    round-1 fan-out spends anything on it. Degrades to a plain JSON
    generation on tool-loop failure, and from there to the raw topic.
    """
    from app.core.llm_client import generate_json, generate_with_tools
    from app.pipeline.planner_tools import SUBMIT_BRIEF_TOOL, brief_tools

    medical = domain in _MEDICAL_DOMAINS
    prompt = f"TOPIC: {topic}\nDOMAIN: {domain}\n\nProduce the research brief."

    raw: dict = {}
    try:
        schemas, handlers = brief_tools(medical=medical)
        _, invocations = generate_with_tools(
            settings=settings,
            prompt=prompt,
            system_instruction=_BRIEF_SYSTEM,
            tools=schemas,
            tool_handlers=handlers,
            temperature=0.2,
            purpose="related_articles_brief",
            job_stats=job_stats,
            max_tool_rounds=_BRIEF_TOOL_ROUNDS,
            final_tool=SUBMIT_BRIEF_TOOL,
        )
        submission = next(
            (inv for inv in reversed(invocations) if inv.name == SUBMIT_BRIEF_TOOL), None,
        )
        if submission is not None:
            raw = submission.arguments
    except Exception as exc:
        logger.warning("Tool-grounded brief failed (%s) - falling back to plain JSON", exc)

    if not isinstance(raw, dict) or not raw:
        try:
            raw = generate_json(
                settings=settings,
                prompt=f"TOPIC: {topic}\nDOMAIN: {domain}\n\nProduce the research brief as JSON.",
                system_instruction=_BRIEF_JSON_FALLBACK_SYSTEM,
                temperature=0.2,
                purpose="related_articles_brief",
                job_stats=job_stats,
            )
        except Exception as exc:
            logger.warning("Research brief failed, falling back to raw topic: %s", exc)
            raw = {}
    if not isinstance(raw, dict):
        raw = {}

    # The user's own phrasing is always searched, whatever the brief says.
    variants = [q for q in (raw.get("query_variants") or []) if isinstance(q, str) and q.strip()]
    if topic not in variants:
        variants = [topic, *variants]
    raw["query_variants"] = variants[:_QUERY_VARIANTS]
    raw["query_variants_it"] = [
        q for q in (raw.get("query_variants_it") or []) if isinstance(q, str) and q.strip()
    ][:2]
    return raw


def _spec_for_query(
    query: str,
    query_it: str | None,
    template: TopicSpec,
    keywords: list[str],
) -> TopicSpec:
    return TopicSpec(
        text=query,
        topic_type=template.topic_type,
        domain=template.domain,
        subject=template.subject,
        search_query=query,
        search_query_it=query_it or query,
        alternative_queries=list(keywords),
    )


def _parallel_search(
    specs: list[TopicSpec],
    domain: str,
    settings: Settings,
    context: DomainContext,
    stats: JobStats,
    providers: list[str],
) -> list[SourceResult]:
    if not specs:
        return []

    def _one(spec: TopicSpec) -> list[SourceResult]:
        return search_evidence(
            spec=spec,
            domain=domain,
            settings=settings,
            context=context,
            job_stats=stats,
            max_per_provider=_MAX_PER_PROVIDER,
            providers_override=providers,
        )

    merged: list[SourceResult] = []
    # Manual lifecycle, not `with ThreadPoolExecutor(...)`: its shutdown(wait=True)
    # on exit would block until every variant's search finished regardless of
    # the as_completed timeout, silently turning _SEARCH_TIMEOUT into a no-op
    # and letting one slow query inflate the whole round's latency.
    pool = ThreadPoolExecutor(max_workers=min(len(specs), _MAX_WORKERS))
    try:
        futures = [pool.submit(_one, spec) for spec in specs]
        try:
            for future in as_completed(futures, timeout=_SEARCH_TIMEOUT):
                try:
                    merged.extend(future.result())
                except Exception as exc:
                    logger.warning("Discovery query failed: %s", exc)
        except TimeoutError:
            logger.warning("Round search budget exhausted (%ds)", _SEARCH_TIMEOUT)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return merged


_REFORMULATE_SYSTEM = """\
You refine a medical/biomedical literature search.  You are shown the TOPIC
and the titles/snippets of the best papers found so far.  Identify the
precise terminology, drug/method names and sub-topics the real literature
uses, then emit NEW search queries that will surface the most relevant and
SEMINAL works the first pass likely missed (foundational trials, key drugs,
canonical terms).

You have tools, and a query you tested beats a query you guessed:
- probe_pubmed(query) - returns hit_count, PubMed's MeSH translation of the
  query, phrases it could not match, and sample titles. Probe every candidate
  query before submitting it. A hit_count near zero, or your key term listed
  in unmatched_phrases, means the WORDING is wrong - not that the literature
  is missing. Sample titles about the wrong subject mean the query is
  ambiguous: add a disambiguating term and probe again.
- resolve_drug(name) - a brand name retrieves far less than its molecule.
- submit_queries(queries, rationale) - submit the queries you settled on.
  ALWAYS end with this call.

You may probe several queries in one turn; they run in parallel.

Rules:
- Do NOT repeat the queries implied by the titles already found - go deeper:
  the proper names of drugs/trials/methods, adjacent concepts.
- Do NOT repeat any query in ALREADY ISSUED QUERIES - emit genuinely new angles.
- Stay strictly on the intended clinical/biomedical sense of the topic.
- If the found papers are clearly off-topic, instead probe and submit queries
  that re-anchor on the topic's core concept using its standard clinical name.
- Submit only queries that probed well. If nothing probes well, submit your
  best two anyway rather than nothing.
"""


def _reformulate(
    topic: str,
    canonical_topic: str,
    seminal_hints: list[str],
    top_hits: list[SourceResult],
    settings: Settings,
    job_stats: JobStats | None,
    issued_queries: list[str],
) -> list[str]:
    """Read the best hits so far and emit refined queries.

    This runs as a tool loop rather than a single generation, because what it
    produces is a guess about vocabulary and PubMed can settle that guess for
    free: the model proposes a phrasing, sees how PubMed expands it through
    MeSH and how much it retrieves, and submits the version that resolves. A
    query arriving here either has literature behind it or was rejected before
    a whole search round was spent on it.

    Returns ``[]`` on failure, which the caller reads as "no new angles" and
    stops on - the same signal an empty submission gives.
    """
    from app.core.llm_client import generate_with_tools
    from app.pipeline.planner_tools import SUBMIT_QUERIES_TOOL, reformulation_tools

    hit_lines = "\n".join(
        f"[{i}] {r.title} - {(r.content or r.snippet or '')[:160]}"
        for i, r in enumerate(top_hits)
    )
    avoid = ""
    if issued_queries:
        avoid = "ALREADY ISSUED QUERIES (do not repeat):\n" + "\n".join(
            f"- {q}" for q in issued_queries
        ) + "\n\n"

    prompt = (
        f"TOPIC: {topic}\n"
        f"CANONICAL TOPIC: {canonical_topic or topic}\n"
        f"KNOWN SEMINAL HINTS: {', '.join(seminal_hints) or '(none provided)'}\n\n"
        f"{avoid}"
        f"BEST PAPERS FOUND SO FAR:\n{hit_lines}\n\n"
        "Probe your candidate queries, then submit the ones worth running."
    )

    schemas, handlers = reformulation_tools()
    try:
        _, invocations = generate_with_tools(
            settings=settings,
            prompt=prompt,
            system_instruction=_REFORMULATE_SYSTEM,
            tools=schemas,
            tool_handlers=handlers,
            temperature=0.25,
            purpose="related_articles_reformulate",
            job_stats=job_stats,
            max_tool_rounds=_REFORMULATE_TOOL_ROUNDS,
            final_tool=SUBMIT_QUERIES_TOOL,
        )
    except Exception as exc:
        logger.warning("Reformulation failed (non-fatal): %s", exc)
        return []

    submission = next(
        (inv for inv in reversed(invocations) if inv.name == SUBMIT_QUERIES_TOOL), None,
    )
    if submission is None:
        logger.info("[pro] reformulator submitted no queries")
        return []

    queries = submission.arguments.get("queries")
    if not isinstance(queries, list):
        return []

    rationale = submission.arguments.get("rationale")
    if isinstance(rationale, str) and rationale.strip():
        logger.info("[pro] reformulation rationale: %s", rationale.strip()[:160])

    return [q.strip() for q in queries if isinstance(q, str) and q.strip()][:_ROUND_QUERIES]


def _fold_in_decomposition(
    topic: str,
    domain: str,
    settings: Settings,
    stats: JobStats,
    must_include: list[str],
) -> list[str]:
    """Fold composite-topic axes into the round-1 query set.

    A frontier topic spanning several research axes would otherwise have its
    weakest axis starved by the fan-out: the variants all drift towards
    whichever axis has the most literature. One query per axis, plus the
    intersection query, keeps each of them represented.

    Mutates *must_include* with the axes' shared vocabulary, and returns the
    extra queries.
    """
    from app.pipeline.topic_analysis import decompose_topic, is_composite

    try:
        decomposition = decompose_topic(topic, domain, settings, job_stats=stats)
    except Exception as exc:
        logger.debug("[pro] decomposition failed (non-fatal): %s", exc)
        return []
    if not isinstance(decomposition, dict):
        return []

    def _extend(terms) -> None:
        for term in terms or []:
            if isinstance(term, str) and term.strip() and term not in must_include:
                must_include.append(term.strip())

    _extend(decomposition.get("enrichment_keywords"))
    if not is_composite(decomposition):
        return []

    queries = [
        sub["search_query"]
        for sub in decomposition.get("sub_claims", []) or []
        if isinstance(sub, dict) and sub.get("search_query")
    ]
    bridge = decomposition.get("bridge_query")
    if isinstance(bridge, str) and bridge.strip():
        queries.append(bridge.strip())
    _extend(decomposition.get("bridge_concepts"))

    return queries[:_FACET_QUERIES]


def run_agentic_discovery(
    *,
    topic: str,
    spec: TopicSpec,
    context: DomainContext,
    settings: Settings,
    stats: JobStats,
    emitter: Emitter = NULL_EMITTER,
) -> tuple[list[SourceResult], dict]:
    """Run the agentic discovery loop.

    Returns ``(evidence, brief)``: the deduplicated candidate pool ready for
    the shared ranking tail, and the research brief - whose vocabulary the
    caller reuses as ranking signals.

    Emits a `discovery` progress event after round 1, after citation
    expansion and after every reformulation round, each carrying the same
    counts the matching `logger.info` call below logs - this is the
    longest-running and most eventful stage, so it is where a streaming
    client gets the most to show.
    """
    domain = context.domain

    def _cosine(results: list[SourceResult]) -> dict[str, float]:
        """Relevance gate, on the stronger rerank-tier embedding model."""
        return cosine_map(
            topic, results, settings,
            job_stats=stats, model_override=settings.embedding_model_rerank,
        )

    # 1. Research brief - also names the providers this topic should search.
    started = time.monotonic()
    brief = _build_brief(topic, domain, settings, stats)
    stats.record_stage("brief", (time.monotonic() - started) * 1000)

    providers = _providers_for_topic(domain, spec, brief)
    emitter.emit(
        "brief",
        f"Research brief ready: {len(brief.get('query_variants') or [])} query variants, "
        f"providers: {', '.join(providers)}",
    )

    variants = brief.get("query_variants") or [topic]
    variants_it = brief.get("query_variants_it") or []
    must_include = [c for c in (brief.get("must_include_concepts") or []) if isinstance(c, str)]
    seminal = [s for s in (brief.get("seminal_hints") or []) if isinstance(s, str)]

    # 1b. Composite axes folded into the same fan-out.
    started = time.monotonic()
    facet_queries = _fold_in_decomposition(topic, domain, settings, stats, must_include)
    stats.record_stage("pro_decomposition", (time.monotonic() - started) * 1000)

    # 2. Round 1: every variant, seminal probe and facet query in parallel.
    started = time.monotonic()
    round1_specs = [
        _spec_for_query(
            query,
            variants_it[i] if i < len(variants_it) else None,
            spec, must_include,
        )
        for i, query in enumerate(variants)
    ]
    for hint in seminal[:2]:
        round1_specs.append(_spec_for_query(hint, None, spec, must_include))
    for query in facet_queries:
        round1_specs.append(_spec_for_query(query, None, spec, must_include))

    round1 = deduplicate(
        _parallel_search(round1_specs, domain, settings, context, stats, providers)
    )
    stats.record_stage("pro_round1", (time.monotonic() - started) * 1000)
    logger.info(
        "[pro] round 1: %d candidates from %d queries (%d facets)",
        len(round1), len(round1_specs), len(facet_queries),
    )
    emitter.emit(
        "discovery",
        f"Round 1: {len(round1)} candidates from {len(round1_specs)} queries",
    )

    # 3. Relevance gate: seed expansion from on-topic papers, not popular ones.
    cosine = _cosine(round1)

    def _ranked(results: list[SourceResult]) -> list[SourceResult]:
        return sorted(results, key=lambda r: cosine.get(dedup_key(r), 0.0), reverse=True)

    ranked = _ranked(round1)
    seeds = [r for r in ranked if cosine.get(dedup_key(r), 0.0) >= _SEED_COSINE_FLOOR]
    seeds = seeds[:_EXPANSION_SEEDS] or ranked[:_EXPANSION_SEEDS]

    # 4. Citation expansion, concurrent with the rounds below: backward
    # references reach the foundational works, forward citations the recent
    # extensions, and neither depends on what round 2 finds.
    def _expand() -> list[SourceResult]:
        expansion_started = time.monotonic()
        found = expand_with_citation_graph(
            seeds,
            max_seeds=_EXPANSION_SEEDS,
            per_seed=_PER_SEED,
            order_by_citations=False,
            bidirectional=True,
        )
        stats.record_stage("pro_expansion", (time.monotonic() - expansion_started) * 1000)
        logger.info("[pro] citation expansion: +%d from %d seeds", len(found), len(seeds))
        emitter.emit(
            "discovery",
            f"Citation graph expansion: +{len(found)} candidates from {len(seeds)} seeds",
        )
        return found

    expansion_pool = ThreadPoolExecutor(max_workers=1)
    expansion = expansion_pool.submit(_expand)

    # 5. Adaptive reformulation rounds.
    evidence: list[SourceResult] = list(round1)
    seen = {dedup_key(r) for r in evidence}
    issued = list(variants)
    canonical = brief.get("canonical_topic", "")

    try:
        for round_index in range(2, _MAX_ROUNDS + 1):
            started = time.monotonic()
            emitter.emit("discovery", f"Round {round_index}: refining queries…")
            queries = _reformulate(
                topic, canonical, seminal, ranked[:_REFORMULATE_HITS],
                settings, stats, issued,
            )
            if not queries:
                logger.info("[pro] round %d: no new queries - stopping", round_index)
                break
            issued.extend(queries)

            found = _parallel_search(
                [_spec_for_query(q, None, spec, must_include) for q in queries],
                domain, settings, context, stats, providers,
            )
            round_cosine = _cosine(found)
            cosine.update(round_cosine)

            new_relevant = [
                r for r in found
                if dedup_key(r) not in seen
                and round_cosine.get(dedup_key(r), 0.0) >= _SEED_COSINE_FLOOR
            ]
            seen.update(dedup_key(r) for r in found)
            evidence.extend(found)
            ranked = _ranked(evidence)

            stats.record_stage(f"pro_round{round_index}", (time.monotonic() - started) * 1000)
            logger.info(
                "[pro] round %d: %d candidates from %d queries (%d new relevant)",
                round_index, len(found), len(queries), len(new_relevant),
            )
            emitter.emit(
                "discovery",
                f"Round {round_index}: {len(found)} candidates from {len(queries)} queries "
                f"({len(new_relevant)} new relevant)",
            )
            if len(new_relevant) < _MIN_NEW_RELEVANT:
                logger.info("[pro] coverage saturated at round %d - stopping", round_index)
                break
    finally:
        expanded = expansion.result()
        expansion_pool.shutdown(wait=False)

    pool = deduplicate(evidence + expanded)
    logger.info("[pro] merged candidate pool: %d", len(pool))
    return pool, brief
