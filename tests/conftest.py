"""Shared fixtures and helpers for the source-provider test suite.

Two layers of tests live here:

* ``test_source_endpoints_live.py`` talks to the upstream APIs with plain
  ``httpx``. When one of these fails, **the database changed or is down** -
  it is not our bug.
* ``test_source_providers_live.py`` drives our ``SourceProvider`` classes.
  When one of these fails while the matching endpoint test passes, **our
  adapter is broken**.

Both layers are marked ``live`` because they need the network.

    pytest                       # everything, including live calls
    pytest -m "not live"         # offline unit tests only (fast, CI-safe)
    pytest -m live -x -q         # the periodic health check
    pytest -m live -rxX          # ...and show which known bugs are still open

Known defects are recorded as ``xfail`` with a ``BUG-nn`` reason. They keep
the suite green today and flip to XPASS the moment one is fixed or an
upstream API starts behaving again - so ``-rxX`` is the report you want.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# The app is importable from the repo root, and Settings requires a key even
# though no test here ever calls an LLM.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENROUTER_API_KEY", "test-key-not-used")
os.environ.setdefault("CONTACT_EMAIL", "tests@example.com")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "live: performs real network calls to a public API")
    config.addinivalue_line("markers", "slow: takes more than a couple of seconds")


# ── Realistic probe queries ───────────────────────────────────
#
# Deliberately picked so a provider cannot pass by accident:
# * a modern branded biologic (semaglutide) - the case that breaks
#   naive brand/generic name handling
# * a landmark-trial topic - the case that breaks recency-ordered search
# * an Italian query - the case that breaks English-only providers

DRUG = "semaglutide"
COMMON_DRUG = "metformin"
LITERATURE_TOPIC = "GLP-1 receptor agonists cardiovascular outcomes"
TRIAL_TOPIC = "semaglutide cardiovascular outcomes"
EPI_TOPIC = "life expectancy"
PREPRINT_TOPIC = "glioblastoma CAR-T"
ITALIAN_TOPIC = "diabete di tipo 2"


# ── Assertion helpers ─────────────────────────────────────────


def assert_wellformed(results: list, *, source_type: str, min_results: int = 1) -> None:
    """Every provider must return usable, correctly-labelled evidence."""
    assert len(results) >= min_results, (
        f"{source_type}: expected >= {min_results} results, got {len(results)}"
    )
    for r in results:
        assert r.title.strip(), f"{source_type}: result with empty title ({r.url})"
        assert r.url.startswith("http"), f"{source_type}: bad url {r.url!r}"
        assert r.source_type == source_type, (
            f"{source_type}: mislabelled as {r.source_type!r}"
        )
        assert 1 <= r.reliability_tier <= 3, (
            f"{source_type}: tier {r.reliability_tier} out of range"
        )


def assert_substantive(results: list, *, min_chars: int, source_type: str) -> None:
    """A result whose content is a stub teaches the summariser nothing."""
    thin = [r for r in results if len(r.content) < min_chars]
    assert not thin, (
        f"{source_type}: {len(thin)}/{len(results)} results carry < {min_chars} chars "
        f"of content, e.g. {thin[0].title[:60]!r} -> {thin[0].content[:120]!r}"
    )


def assert_mentions(results: list, term: str, *, source_type: str) -> None:
    """At least one result must actually be about the thing we asked for."""
    term = term.lower()
    hit = any(term in (r.title + " " + r.content).lower() for r in results)
    assert hit, (
        f"{source_type}: no result mentions {term!r}; got "
        f"{[r.title[:50] for r in results]}"
    )


# ── Throttling ────────────────────────────────────────────────
#
# EBI (Europe PMC, ChEMBL) and NCBI answer a burst with 429/406/503 rather
# than a slow 200. A periodic health check that red-lights on that is noise,
# so throttling is reported as a skip and only real breakage fails.

THROTTLE_STATUSES = frozenset({406, 429, 503})


def skip_if_throttled(response, who: str) -> None:
    """Turn an explicit throttling response into a skip."""
    if response.status_code in THROTTLE_STATUSES:
        pytest.skip(f"{who} throttled this client (HTTP {response.status_code})")


def search_or_skip(provider, query: str, max_results: int = 3, *, who: str = "") -> list:
    """Run ``provider.search``, skipping when the API is merely throttling us.

    Providers wrap ``search`` in tenacity, so a throttled call surfaces as a
    ``RetryError`` around the last ``HTTPStatusError`` - unwrap it before
    deciding whether this is our bug or their rate limiter.
    """
    import httpx
    from tenacity import RetryError

    who = who or getattr(provider, "source_type", type(provider).__name__)
    try:
        return provider.search(query, max_results=max_results)
    except RetryError as exc:
        cause = exc.last_attempt.exception() if exc.last_attempt.failed else None
    except httpx.HTTPStatusError as exc:
        cause = exc
    if isinstance(cause, httpx.HTTPStatusError) and (
        cause.response.status_code in THROTTLE_STATUSES
    ):
        pytest.skip(f"{who} throttled this client (HTTP {cause.response.status_code})")
    raise AssertionError(f"{who}.search({query!r}) failed: {cause!r}") from cause
