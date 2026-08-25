"""Whole-topic evidence cache.

The per-provider cache in :mod:`app.source_searcher` saves individual API
calls; this one saves the entire search. A repeat of the same topic skips
planning, every provider call and the retry tiers, and returns in
milliseconds instead of tens of seconds.

Lookup is exact-hash only, over a normalisation that lowercases, strips
punctuation and sorts words - so "aspirin and stroke prevention" and
"stroke prevention and aspirin" share an entry, but two genuinely different
phrasings of the same question do not. Catching those needs embedding
similarity over a vector store, which is exactly the infrastructure this
project does without. A miss costs time and tokens, never correctness.
"""
from __future__ import annotations

import dataclasses
import hashlib
import logging
import re

from app.core.cache import topic_evidence_cache
from app.sources.base import SourceResult

logger = logging.getLogger(__name__)


def _topic_hash(topic: str) -> str:
    # TODO: exact-hash matching, so two different phrasings of the same
    # question miss each other. Lifting this needs embedding similarity over
    # a vector store instead of a normalised hash.
    normalised = re.sub(r"[^\w\s]", "", topic.lower().strip())
    normalised = " ".join(sorted(normalised.split()))
    return hashlib.sha256(normalised.encode()).hexdigest()


def get_topic_evidence(topic: str) -> list[SourceResult] | None:
    cached = topic_evidence_cache.get(_topic_hash(topic))
    if cached is None:
        return None
    try:
        results = [SourceResult(**r) for r in cached]
    except TypeError as exc:
        # Entry written by an older build with a different SourceResult shape.
        logger.debug("Discarding incompatible cache entry: %s", exc)
        return None
    logger.info("Topic evidence cache HIT: %d sources for '%s'", len(results), topic[:60])
    return results


def save_topic_evidence(topic: str, evidence: list[SourceResult], ttl_days: int) -> None:
    """Cache aggregated evidence for *topic*. Never raises."""
    if not evidence:
        return
    try:
        topic_evidence_cache.set(
            _topic_hash(topic),
            [dataclasses.asdict(r) for r in evidence],
            ttl_days * 86400,
        )
    except Exception as exc:
        logger.debug("Topic evidence cache save failed (non-fatal): %s", exc)
