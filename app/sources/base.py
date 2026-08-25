"""Shared contract for evidence-source providers."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field

from app.config import get_settings


def extract_doi(url: str) -> str | None:
    """Extract a bare DOI from a URL, or None if it carries none.

    The DOI is the only globally stable identifier a paper has: the same work
    appears under a dozen publisher and aggregator URLs, but one DOI.
    """
    match = re.search(r"10\.\d{4,9}/[^\s?#]+", url or "", re.IGNORECASE)
    if not match:
        return None
    return match.group().rstrip(").").lower()


def contact_email() -> str:
    """The address external APIs should use to reach whoever runs this."""
    return get_settings().contact_email


def user_agent() -> str:
    """Identify this client to external APIs, with a contact address if we have one.

    NCBI, Crossref and OpenAlex all ask callers to identify themselves and
    give identified traffic a more generous rate limit ("polite pool"); the
    address comes from ``CONTACT_EMAIL`` so a deployment can supply its own.
    Unset is left unset - a made-up address would satisfy the mailto: syntax
    but not what these services actually want, a way to reach a real
    operator.
    """
    email = get_settings().contact_email
    if not email:
        return "EBM-Lens/1.0"
    return f"EBM-Lens/1.0 (mailto:{email})"


def build_headers(
    *,
    accept: str | None = None,
    accept_language: str | None = None,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    headers = {"User-Agent": user_agent()}
    if accept:
        headers["Accept"] = accept
    if accept_language:
        headers["Accept-Language"] = accept_language
    if extra:
        headers.update(dict(extra))
    return headers


@dataclass
class SourceResult:
    """One piece of evidence returned by a source provider."""

    title: str
    url: str
    snippet: str
    content: str = ""
    source_type: str = ""           # pubmed, europe_pmc, wikipedia, ...
    reliability_tier: int = 3       # 1=gold (official DB/FDA), 2=strong (academic), 3=general
    publication_date: str = ""      # ISO date or year if available from provider
    citation_count: int = 0         # times-cited count if available (academic authority signal)
    content_hash: str = ""          # SHA256 of content, for dedup across providers
    language: str = ""              # "en", "it", detected or declared

    # Publication types as the provider itself classifies them, verbatim.
    # PubMed's come from NLM's own indexers ("Randomized Controlled Trial",
    # "Meta-Analysis"), which makes them a statement of fact about the paper
    # rather than a guess from its wording - the strongest evidence-grading
    # signal available, where a provider supplies one.
    publication_types: list[str] = field(default_factory=list)


class SourceProvider(ABC):
    """Interface every external source must implement."""

    source_type: str = "unknown"

    @abstractmethod
    def search(self, query: str, max_results: int = 3) -> list[SourceResult]:
        ...
