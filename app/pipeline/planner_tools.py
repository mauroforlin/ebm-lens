"""Tools the model can call before committing to a search.

Writing a biomedical query is guesswork from the inside of a prompt. A brand
name is not what the databases index; PubMed silently rewrites what it is
given through MeSH, and an expansion that finds nothing looks exactly like a
query with no literature behind it; a topic may have a guideline that names
the concept in the field's own words. Each of those is a fact that exists in
a free public API, so the model is given the lookup instead of being asked to
guess well.

Two loops use these, with a different terminal tool each:

* the discovery loop's **research brief** explores, then submits query
  variants and the providers to search through ``submit_brief`` - once per
  topic, before the round-1 fan-out spends anything on an ungrounded guess;
* the discovery loop's **query reformulator** explores, then submits refined
  queries through ``submit_queries`` - once per later round, from the best
  hits found so far.

Exploration tools are shared between them, because "check what this query
actually retrieves before spending a round on it" is the same operation in
both. The terminal tool is what tells the loop the answer has arrived, and
:func:`~app.core.llm_client.generate_with_tools` forces it when the round
budget runs low, so a loop that explores too long still submits something.

This module is schema and dispatch only: every tool routes to the provider
that owns the HTTP call.
"""
from __future__ import annotations

import json
from collections.abc import Callable

from app.sources.europe_pmc import EuropePMCProvider
from app.sources.pubmed import PubMedProvider
from app.sources.rxnav import RxNavProvider

# Terminal tool names, exported so callers can extract the submission and ask
# the tool loop to force it.
SUBMIT_BRIEF_TOOL = "submit_brief"
SUBMIT_QUERIES_TOOL = "submit_queries"


# ══════════════════════════════════════════════════════════════
#  Exploration tools
# ══════════════════════════════════════════════════════════════

RESOLVE_DRUG_TOOL = {
    "type": "function",
    "function": {
        "name": "resolve_drug",
        "description": (
            "Resolve a drug name (brand or generic) to its active molecule "
            "(ingredient) via RxNorm. Use this before planning drug queries so "
            "searches target the generic/molecule name."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Drug name, brand or generic."},
            },
            "required": ["name"],
        },
    },
}

SEARCH_GUIDELINES_TOOL = {
    "type": "function",
    "function": {
        "name": "search_guidelines",
        "description": (
            "Find clinical practice guidelines on a topic via Europe PMC. Use "
            "this for treatment-standard / recommendation topics so the plan can "
            "reference real guideline titles."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Clinical topic to find guidelines for."},
            },
            "required": ["topic"],
        },
    },
}

PROBE_QUERY_TOOL = {
    "type": "function",
    "function": {
        "name": "probe_pubmed",
        "description": (
            "Test a query against PubMed before committing to it. Returns the "
            "number of matching papers, how PubMed translated the query into "
            "MeSH terms, any phrases it could not match, and a few sample "
            "titles. Use it to check that a query retrieves the right "
            "literature: a hit_count of 0 or a few, or a term appearing in "
            "unmatched_phrases, means the wording is wrong rather than the "
            "literature absent. Cheap - call it on any query you are unsure of."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The English query to test, as it would be sent to PubMed.",
                },
            },
            "required": ["query"],
        },
    },
}


def _resolve_drug(args: dict) -> str:
    return json.dumps(
        RxNavProvider().resolve_drug(args.get("name", "")), ensure_ascii=False,
    )


def _search_guidelines(args: dict) -> str:
    results = EuropePMCProvider().search_guidelines(args.get("topic", ""))
    payload = [
        {
            "title": r.title,
            "url": r.url,
            "year": (r.publication_date or "")[:4],
            "cited_by": r.citation_count,
        }
        for r in results
    ]
    return json.dumps(payload, ensure_ascii=False)


def _probe_pubmed(args: dict) -> str:
    return json.dumps(
        PubMedProvider.probe(args.get("query", "")), ensure_ascii=False,
    )


_EXPLORATION: dict[str, tuple[dict, Callable[[dict], str]]] = {
    "resolve_drug": (RESOLVE_DRUG_TOOL, _resolve_drug),
    "search_guidelines": (SEARCH_GUIDELINES_TOOL, _search_guidelines),
    "probe_pubmed": (PROBE_QUERY_TOOL, _probe_pubmed),
}


# ══════════════════════════════════════════════════════════════
#  Terminal tools
# ══════════════════════════════════════════════════════════════


def _submit_brief_tool(provider_ids: list[str]) -> dict:
    """Build the brief-submission schema, constrained to real provider ids.

    Passing the registry in as an enum means a hallucinated provider is
    rejected by the API before it ever reaches the discovery loop.
    """
    return {
        "type": "function",
        "function": {
            "name": SUBMIT_BRIEF_TOOL,
            "description": (
                "Submit the finished research brief: disambiguation, query "
                "variants, vocabulary and the providers this topic should search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "canonical_topic": {
                        "type": "string",
                        "description": "One precise, disambiguated restatement of the topic.",
                    },
                    "query_variants": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "4-6 DIVERSE English search queries, 4-8 words each.",
                    },
                    "query_variants_it": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "1-2 Italian queries for Italian sources.",
                    },
                    "must_include_concepts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "3-6 technical terms an on-topic paper will use.",
                    },
                    "negative_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Terms that indicate an OFF-topic homonym/wrong sense.",
                    },
                    "seminal_hints": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Named landmark trials, methods, drugs or acronyms, if any.",
                    },
                    "providers": {
                        "type": "array",
                        "items": {"type": "string", "enum": provider_ids},
                        "description": (
                            "Provider ids this topic should search. Always include pubmed. "
                            "Include openfda/dailymed/rxnav/ema whenever the topic names a "
                            "specific drug or concerns safety, interactions, dosage or "
                            "regulatory status; clinicaltrials for treatment-efficacy "
                            "questions; who_gho for epidemiology."
                        ),
                    },
                },
                "required": ["query_variants", "providers"],
            },
        },
    }


SUBMIT_QUERIES_SCHEMA = {
    "type": "function",
    "function": {
        "name": SUBMIT_QUERIES_TOOL,
        "description": (
            "Submit the refined search queries for the next discovery round. "
            "Call this once, at the end, with queries you have reason to "
            "believe will retrieve papers the previous rounds missed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "description": "2-4 new English queries, 4-8 words each.",
                    "items": {"type": "string"},
                },
                "rationale": {
                    "type": "string",
                    "description": "One sentence: what gap in the results so far these queries target.",
                },
            },
            "required": ["queries"],
        },
    },
}


def _accepted(args: dict, key: str) -> str:
    return json.dumps({"accepted": True, "count": len(args.get(key) or [])})


# ══════════════════════════════════════════════════════════════
#  Tool sets
# ══════════════════════════════════════════════════════════════


def brief_tools(medical: bool = True) -> tuple[list[dict], dict[str, Callable]]:
    """Return ``(schemas, handlers)`` for the discovery loop's research-brief grounding.

    ``resolve_drug`` and ``search_guidelines`` are only offered for medical
    domains - neither has anything to say about a non-medical topic, and an
    unusable tool is one more way for the model to waste a round.
    ``probe_pubmed`` is offered for both: PubMed indexes more than clinical
    medicine, and knowing whether a query retrieves anything is useful
    whatever the topic. The terminal ``submit_brief`` is always present.

    This runs once per topic, before the round-1 fan-out, so spending a tool
    round here does not multiply by the number of query variants the
    discovery loop issues.

    The provider registry is imported here rather than at module scope: the
    searcher imports this module for its schemas, so a top-level import would
    close the cycle.
    """
    from app.pipeline.searcher import PROVIDER_IDS

    names = ["resolve_drug", "search_guidelines", "probe_pubmed"] if medical else ["probe_pubmed"]

    schemas = [_EXPLORATION[n][0] for n in names]
    handlers: dict[str, Callable] = {n: _EXPLORATION[n][1] for n in names}

    schemas.append(_submit_brief_tool(list(PROVIDER_IDS)))
    handlers[SUBMIT_BRIEF_TOOL] = lambda args: _accepted(args, "query_variants")

    return schemas, handlers


def reformulation_tools() -> tuple[list[dict], dict[str, Callable]]:
    """Return ``(schemas, handlers)`` for the discovery loop's query reformulator.

    A reformulator's job is to guess which words the literature uses for
    something the previous round did not find. ``probe_pubmed`` turns that
    guess into a check: the model can try a phrasing, see the hit count and
    the MeSH translation, and submit the one that actually resolves. Drug
    resolution is offered alongside it because a refined query that names a
    brand instead of a molecule is the most common way for a promising
    reformulation to retrieve nothing.
    """
    names = ["probe_pubmed", "resolve_drug"]
    schemas = [_EXPLORATION[n][0] for n in names]
    handlers: dict[str, Callable] = {n: _EXPLORATION[n][1] for n in names}

    schemas.append(SUBMIT_QUERIES_SCHEMA)
    handlers[SUBMIT_QUERIES_TOOL] = lambda args: _accepted(args, "queries")

    return schemas, handlers
