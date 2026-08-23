"""Unit tests for the graded-retrieval metrics in eval/retrieval_eval.py.

`_as_grades` has to accept either a flat list of ids (binary relevance) or a
`{doc_id: grade}` map and normalise both into the same shape, so any future
fixture with real relevance levels slots in without a metric rewrite. nDCG
must also behave at its edges - empty gradable pool, no relevant docs found -
since a mixed-source pool produces exactly those cases on real fixtures.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.retrieval_eval import _as_grades, _dcg, _gradable_ranking, _ndcg


def test_as_grades_flat_list_is_binary():
    assert _as_grades(["A", "B", "C"]) == {"A": 1, "B": 1, "C": 1}


def test_as_grades_map_passes_through_as_int():
    assert _as_grades({"A": 2, "B": "1"}) == {"A": 2, "B": 1}


def test_dcg_empty_is_zero():
    assert _dcg([], 10) == 0.0


def test_dcg_all_zero_grades_is_zero():
    assert _dcg([0, 0, 0], 10) == 0.0


def test_dcg_respects_k():
    # A relevant doc past k contributes nothing.
    assert _dcg([0, 0, 1], 2) == 0.0
    assert _dcg([1, 0, 0], 2) > 0.0


def test_ndcg_perfect_ranking_is_one():
    ranked = [2, 1, 1]
    ideal = [2, 1, 1]
    assert _ndcg(ranked, ideal, 10) == 1.0


def test_ndcg_empty_ideal_is_zero_not_error():
    # No relevant docs at all in gold - a degenerate topic, not a crash.
    assert _ndcg([], [], 10) == 0.0
    assert _ndcg([0, 0], [], 10) == 0.0


def test_ndcg_worse_ranking_scores_below_one():
    ranked = [1, 1, 2]  # the grade-2 doc buried last
    ideal = [2, 1, 1]
    assert 0.0 < _ndcg(ranked, ideal, 10) < 1.0


def test_ndcg_pool_ceiling_never_exceeds_full_ceiling():
    """IDCG computed over a subset of gold can only be <= IDCG over all of gold,
    so ndcg_at_10_pool (which uses only what was retrieved as its ideal) is
    always >= ndcg_at_10 (which uses the entire gold set as its ideal), for the
    same ranked_grades. Exercised here via the DCG/IDCG building blocks."""
    ranked_grades = [1, 0, 2]
    full_ideal = [2, 2, 1, 1, 1]       # entire gold set
    pool_ideal = [2, 1]                 # only what retrieval actually found

    ndcg_full = _ndcg(ranked_grades, full_ideal, 10)
    ndcg_pool = _ndcg(ranked_grades, pool_ideal, 10)
    assert ndcg_pool >= ndcg_full


class _FakeResult:
    def __init__(self, url: str):
        self.url = url


def test_gradable_ranking_drops_unjudgeable_urls_and_keeps_order():
    ranked = [
        _FakeResult("https://en.wikipedia.org/wiki/Something"),
        _FakeResult("https://pubmed.ncbi.nlm.nih.gov/12345"),
        _FakeResult("https://clinicaltrials.gov/study/NCT00000001"),
        _FakeResult("https://pubmed.ncbi.nlm.nih.gov/67890"),
    ]
    assert _gradable_ranking(ranked, "pmid") == ["12345", "67890"]


def test_gradable_ranking_empty_pool_is_empty_list():
    assert _gradable_ranking([], "pmid") == []
