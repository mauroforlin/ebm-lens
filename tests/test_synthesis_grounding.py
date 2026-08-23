"""Unit tests for synthesis.py's citation-grounding invariants.

`eval/stance_eval.py`'s docstring explains why a distractor-based citation-
grounding eval against `synthesise` was designed and rejected (synthesise
writes its own claims, so "should claim k cite source j" has no SciFact gold
to check against - answering it needs the entailment machinery whose absence
the eval exists to measure). What *is* verifiable without a judge or a
network call is the structural machinery around citations: that an index has
to exist in the answer to be citable, that an unattributed claim is dropped
rather than kept with an empty citation list, that stated strength is
downgraded when no cited source is a direct match, and that the relevance
gate that decides which articles even reach synthesis is strict.

No network, no LLM calls except where `generate_json` is monkeypatched.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline import synthesis
from app.schemas import ArticleSummary


def _article(relevance_score: float, directness: str = "unclear", key_finding: str = "finding") -> ArticleSummary:
    return ArticleSummary(
        url="https://example.invalid/doc", title="t", relevance_score=relevance_score,
        directness=directness, key_finding=key_finding,
    )


# ── _valid_indices ──────────────────────────────────────────────


def test_valid_indices_drops_out_of_range():
    assert synthesis._valid_indices([0, 5, 2], allowed={0, 1, 2}) == [0, 2]


def test_valid_indices_drops_duplicates_keeping_first_occurrence():
    assert synthesis._valid_indices([1, 1, 0, 1], allowed={0, 1}) == [1, 0]


def test_valid_indices_rejects_bool_even_though_bool_is_an_int_subclass():
    # True == 1 and isinstance(True, int) is True in Python - the explicit
    # bool guard in _valid_indices exists precisely to reject this.
    assert synthesis._valid_indices([True, False, 1], allowed={0, 1}) == [1]


def test_valid_indices_non_list_input_is_empty():
    assert synthesis._valid_indices(None, allowed={0, 1}) == []
    assert synthesis._valid_indices("not a list", allowed={0, 1}) == []


# ── _read_claims ────────────────────────────────────────────────


def test_read_claims_drops_unattributed_claim():
    raw = [{"text": "some finding", "source_indices": [7], "strength": "strong"}]
    claims = synthesis._read_claims(raw, allowed={0, 1}, directness_by_index={0: "direct", 1: "direct"})
    assert claims == []


def test_read_claims_clamps_strength_to_weak_without_direct_source():
    raw = [{"text": "some finding", "source_indices": [0], "strength": "strong"}]
    claims = synthesis._read_claims(raw, allowed={0}, directness_by_index={0: "adjacent"})
    assert len(claims) == 1
    assert claims[0].strength == "weak"


def test_read_claims_keeps_stated_strength_when_a_cited_source_is_direct():
    raw = [{"text": "some finding", "source_indices": [0, 1], "strength": "strong"}]
    claims = synthesis._read_claims(raw, allowed={0, 1}, directness_by_index={0: "adjacent", 1: "direct"})
    assert claims[0].strength == "strong"


def test_read_claims_falls_back_to_moderate_for_unrecognised_strength():
    raw = [{"text": "some finding", "source_indices": [0], "strength": "extremely-strong"}]
    claims = synthesis._read_claims(raw, allowed={0}, directness_by_index={0: "direct"})
    assert claims[0].strength == "moderate"


def test_read_claims_drops_items_with_empty_text():
    raw = [{"text": "  ", "source_indices": [0]}, {"source_indices": [0]}]
    assert synthesis._read_claims(raw, allowed={0}, directness_by_index={0: "direct"}) == []


# ── _read_disagreements ─────────────────────────────────────────


def test_read_disagreements_drops_conflicts_with_fewer_than_two_valid_indices():
    raw = [{"issue": "dosage disagreement", "source_indices": [0], "positions": ["a", "b"]}]
    assert synthesis._read_disagreements(raw, allowed={0, 1}) == []


def test_read_disagreements_keeps_conflicts_with_two_or_more_valid_indices():
    raw = [{"issue": "dosage disagreement", "source_indices": [0, 1, 9], "positions": ["a", "b"]}]
    conflicts = synthesis._read_disagreements(raw, allowed={0, 1})
    assert len(conflicts) == 1
    assert conflicts[0].source_indices == [0, 1]  # index 9 is not in `allowed`, dropped


def test_read_disagreements_truncates_positions_to_four():
    raw = [{"issue": "x", "source_indices": [0, 1], "positions": ["a", "b", "c", "d", "e"]}]
    assert len(synthesis._read_disagreements(raw, allowed={0, 1})[0].positions) == 4


# ── synthesise's relevance gate ─────────────────────────────────


def test_synthesise_returns_low_evidence_message_when_nothing_clears_the_gate():
    articles = [_article(0.3), _article(0.0)]
    summary, claims, conflicts, gaps = synthesis.synthesise("topic", articles, settings=None)
    assert summary == synthesis.low_evidence_message("it")
    assert claims == conflicts == gaps == []


def test_synthesise_gate_is_strict_greater_than_not_greater_or_equal():
    # An article scoring exactly MIN_SYNTHESIS_RELEVANCE must NOT clear the
    # gate - if it did, this call would need a real LLM response instead of
    # taking the no-relevant-articles early return.
    articles = [_article(synthesis.MIN_SYNTHESIS_RELEVANCE)]
    summary, claims, conflicts, gaps = synthesis.synthesise("topic", articles, settings=None)
    assert summary == synthesis.low_evidence_message("it")
    assert claims == conflicts == gaps == []
