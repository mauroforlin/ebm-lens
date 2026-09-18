"""Unit tests for summary_verification.py - no network; claim_verification.
verify_claims is monkeypatched for the one path that would reach TypeSafe.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline import claim_verification
from app.pipeline.claim_verification import ClaimVerdict
from app.pipeline.summary_verification import SummaryChunk, split_into_chunks, verify_summary
from app.schemas import ArticleSummary


def _article() -> ArticleSummary:
    return ArticleSummary(url="https://example.invalid/doc", title="t")


# ── split_into_chunks ───────────────────────────────────────────


def test_split_single_citation_at_end():
    chunks = split_into_chunks("Metformin lowers HbA1c by 1.2% [0].")
    assert chunks == [SummaryChunk(text="Metformin lowers HbA1c by 1.2%", source_indices=[0])]


def test_split_multi_index_cluster():
    chunks = split_into_chunks("The drug is well tolerated [1][3].")
    assert chunks[0].source_indices == [1, 3]


def test_split_two_clauses_each_with_own_citation():
    text = "The drug reduced symptoms [0]. In older adults the effect disappeared [2]."
    chunks = split_into_chunks(text)
    assert [c.source_indices for c in chunks] == [[0], [2]]
    assert "reduced symptoms" in chunks[0].text
    assert "disappeared" in chunks[1].text


def test_split_trailing_uncited_text_becomes_its_own_chunk():
    text = "The drug reduced symptoms [0]. This warrants further investigation in larger cohorts."
    chunks = split_into_chunks(text)
    assert chunks[-1].source_indices == []
    assert "further investigation" in chunks[-1].text


def test_split_paragraph_with_no_citation_at_all():
    chunks = split_into_chunks("This paragraph cites nothing whatsoever here.")
    assert len(chunks) == 1
    assert chunks[0].source_indices == []


def test_split_uncited_tail_does_not_bleed_into_next_paragraph():
    text = (
        "First paragraph makes a point [0]. It also adds a trailing thought.\n\n"
        "Second paragraph makes another point [1]."
    )
    chunks = split_into_chunks(text)
    # The trailing thought must not have absorbed the second paragraph's citation.
    trailing = next(c for c in chunks if "trailing thought" in c.text)
    assert trailing.source_indices == []
    assert 1 not in trailing.source_indices


def test_split_tolerates_space_between_adjacent_clusters():
    chunks = split_into_chunks("A combined finding [1] [3].")
    assert chunks[0].source_indices == [1, 3]


# ── verify_summary ──────────────────────────────────────────────


def test_verify_summary_empty_input_returns_no_flags():
    assert verify_summary("", [_article()], settings=None) == []


def test_verify_summary_flags_substantive_uncited_statement(monkeypatch):
    def boom(*_a, **_kw):
        raise AssertionError("verify_claims should not be called with nothing cited")
    monkeypatch.setattr(claim_verification, "verify_claims", boom)

    flags = verify_summary(
        "This is a fairly long substantive claim about the drug's effect with no citation at all.",
        [_article()], settings=None, summary_language="en",
    )
    assert len(flags) == 1
    assert "no source citation" in flags[0]


def test_verify_summary_does_not_flag_short_uncited_connective_text(monkeypatch):
    def boom(*_a, **_kw):
        raise AssertionError("verify_claims should not be called")
    monkeypatch.setattr(claim_verification, "verify_claims", boom)

    assert verify_summary("However, this remains unclear.", [_article()], settings=None) == []


def test_verify_summary_no_flag_when_cited_chunk_is_supported(monkeypatch):
    def fake_verify_claims(claims, articles, settings, job_stats=None):
        return [ClaimVerdict(0, 0, "supports", 0.95, {})]
    monkeypatch.setattr(claim_verification, "verify_claims", fake_verify_claims)

    flags = verify_summary("The drug reduced symptoms [0].", [_article()], settings=None)
    assert flags == []


def test_verify_summary_flags_confidently_contradicted_chunk(monkeypatch):
    def fake_verify_claims(claims, articles, settings, job_stats=None):
        return [ClaimVerdict(0, 0, "contradicts", 0.95, {})]
    monkeypatch.setattr(claim_verification, "verify_claims", fake_verify_claims)

    flags = verify_summary(
        "The drug reduced symptoms [0].", [_article()], settings=None, summary_language="en",
    )
    assert len(flags) == 1
    assert "contradict" in flags[0]


def test_verify_summary_flags_confidently_unsupported_chunk(monkeypatch):
    def fake_verify_claims(claims, articles, settings, job_stats=None):
        return [ClaimVerdict(0, 0, "says_nothing", 0.9, {})]
    monkeypatch.setattr(claim_verification, "verify_claims", fake_verify_claims)

    flags = verify_summary(
        "The drug reduced symptoms [0].", [_article()], settings=None, summary_language="en",
    )
    assert len(flags) == 1
    assert "does not appear to address" in flags[0]


def test_verify_summary_does_not_flag_ambiguous_low_confidence_verdict(monkeypatch):
    def fake_verify_claims(claims, articles, settings, job_stats=None):
        return [ClaimVerdict(0, 0, "contradicts", 0.46, {})]
    monkeypatch.setattr(claim_verification, "verify_claims", fake_verify_claims)

    flags = verify_summary("The drug reduced symptoms [0].", [_article()], settings=None)
    assert flags == []


def _dist(**masses: float) -> ClaimVerdict:
    relation = max(masses, key=lambda k: masses[k])
    return ClaimVerdict(0, 0, relation, masses[relation], dict(masses))


def test_verify_summary_flags_on_contradiction_mass_the_argmax_hides(monkeypatch):
    # {says_nothing .35, contradicts .62}: argmax is contradicts here, but the
    # point is the bar is read off the mass, so a distribution whose winner
    # sits below FLAG_PROBABILITY does not flag while this one does.
    def fake_verify_claims(claims, articles, settings, job_stats=None):
        return [_dist(supports=0.03, contradicts=0.62, says_nothing=0.35)]
    monkeypatch.setattr(claim_verification, "verify_claims", fake_verify_claims)

    flags = verify_summary(
        "The drug reduced symptoms [0].", [_article()], settings=None, summary_language="en",
    )
    assert len(flags) == 1
    assert "contradict" in flags[0]


def test_verify_summary_does_not_flag_a_spread_distribution(monkeypatch):
    # No relation holds enough mass to clear FLAG_PROBABILITY - the passage is
    # genuinely undecided, which is not the same as being contradicted.
    def fake_verify_claims(claims, articles, settings, job_stats=None):
        return [_dist(supports=0.40, contradicts=0.35, says_nothing=0.25)]
    monkeypatch.setattr(claim_verification, "verify_claims", fake_verify_claims)

    flags = verify_summary("The drug reduced symptoms [0].", [_article()], settings=None)
    assert flags == []


def test_verify_summary_unsupported_flag_reads_says_nothing_mass(monkeypatch):
    def fake_verify_claims(claims, articles, settings, job_stats=None):
        return [_dist(supports=0.15, contradicts=0.05, says_nothing=0.80)]
    monkeypatch.setattr(claim_verification, "verify_claims", fake_verify_claims)

    flags = verify_summary(
        "The drug reduced symptoms [0].", [_article()], settings=None, summary_language="en",
    )
    assert len(flags) == 1
    assert "does not appear to address" in flags[0]


def test_verify_summary_error_verdict_is_not_flagged(monkeypatch):
    def fake_verify_claims(claims, articles, settings, job_stats=None):
        return [ClaimVerdict(0, 0, "error", None, {})]
    monkeypatch.setattr(claim_verification, "verify_claims", fake_verify_claims)

    flags = verify_summary("The drug reduced symptoms [0].", [_article()], settings=None)
    assert flags == []


def test_verify_summary_out_of_range_citation_treated_as_uncited(monkeypatch):
    def boom(*_a, **_kw):
        raise AssertionError("verify_claims should not be called - no valid indices")
    monkeypatch.setattr(claim_verification, "verify_claims", boom)

    flags = verify_summary(
        "This fairly long statement cites a source index that does not exist [7].",
        [_article()], settings=None, summary_language="en",
    )
    assert len(flags) == 1
    assert "no source citation" in flags[0]
