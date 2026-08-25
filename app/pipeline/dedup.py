"""Source identity and deduplication.

The same paper reaches the pipeline repeatedly: PubMed and Europe PMC both
index it, the citation graph returns it as a neighbour of itself, and every
composite facet that touches it searches for it again. Each arrival carries a
different URL and a different subset of the metadata.

A duplicate consumes a result slot, is summarised a second time at full LLM cost,
and inflates the reciprocal-rank fusion score of whatever it duplicates.
So identity is resolved on the strongest available signal - DOI, then normalised title,
then URL - and the survivor inherits the best metadata seen across all its copies.
"""
from __future__ import annotations

import re

from app.sources.base import SourceResult, extract_doi

# Below this length a normalised title is too generic to be an identity
# ("Introduction", "Editorial"), so URL identity is used instead.
_MIN_TITLE_KEY_LENGTH = 15


def normalize_title(title: str) -> str:
    cleaned = re.sub(r"[^\w\s]", " ", (title or "").lower())
    return " ".join(cleaned.split())


def identity_key(url: str, title: str) -> str:
    doi = extract_doi(url)
    if doi:
        return f"doi:{doi}"
    normalised = normalize_title(title)
    if len(normalised) >= _MIN_TITLE_KEY_LENGTH:
        return f"title:{normalised}"
    return f"url:{url.rstrip('/').lower()}" if url else ""


def dedup_key(result: SourceResult) -> str:
    return identity_key(result.url, result.title) or f"id:{id(result)}"


def result_keys(result: SourceResult) -> list[str]:
    """Every identity *result* could match on.

    A result must be matched on DOI *and* title, not just its preferred key:
    one copy may expose a DOI while another has only a publisher link, and
    matching on the preferred key alone would leave them as separate papers.
    """
    keys: list[str] = []
    doi = extract_doi(result.url)
    if doi:
        keys.append(f"doi:{doi}")
    normalised = normalize_title(result.title)
    if len(normalised) >= _MIN_TITLE_KEY_LENGTH:
        keys.append(f"title:{normalised}")
    if not keys:
        keys.append(
            f"url:{result.url.rstrip('/').lower()}" if result.url else f"id:{id(result)}"
        )
    return keys


def absorb(kept: SourceResult, duplicate: SourceResult) -> None:
    """Fold *duplicate*'s metadata into *kept*, keeping the stronger of each.

    Providers are uneven: PubMed knows the citation count, Europe PMC has the
    abstract, bioRxiv has the date. Merging field by field means the survivor
    is better described than any single copy of it was.
    """
    kept.citation_count = max(kept.citation_count, duplicate.citation_count)
    if not kept.publication_date and duplicate.publication_date:
        kept.publication_date = duplicate.publication_date
    if len(duplicate.content or "") > len(kept.content or ""):
        kept.content = duplicate.content
    # Lower tier number = more authoritative provider.
    kept.reliability_tier = min(kept.reliability_tier, duplicate.reliability_tier)


def deduplicate(results: list[SourceResult]) -> list[SourceResult]:
    """Collapse results sharing a DOI or a normalised title, richest copy kept.

    Order-independent: whichever copy arrives first becomes the representative
    and absorbs the rest, so the output does not depend on provider timing.
    """
    key_to_group: dict[str, int] = {}
    groups: list[SourceResult] = []

    for result in results:
        keys = result_keys(result)
        group_index = next((key_to_group[k] for k in keys if k in key_to_group), None)
        if group_index is None:
            group_index = len(groups)
            groups.append(result)
        else:
            absorb(groups[group_index], result)
        for key in keys:
            key_to_group.setdefault(key, group_index)

    return groups
