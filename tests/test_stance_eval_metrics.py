"""Unit tests for eval/stance_eval.py's _summarise - no network, no LLM.

Builds synthetic rows in the exact shape _evaluate produces, so these are
sanity checks on the confusion-matrix and hallucination-rate arithmetic
against the graded 7-point + mixed + no_evidence scale, independent of
whatever a real model actually predicts.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.stance_eval import _summarise
from app.pipeline.synthesis import collapse_direction


def _pair(gold: str, predicted: str, *, relevance: float = 0.9, key_finding: str = "x") -> dict:
    return {
        "doc_id": "d", "gold": gold, "predicted": predicted,
        "predicted_collapsed": collapse_direction(predicted),
        "relevance_score": relevance, "key_finding": key_finding, "directness": "unclear",
    }


def _row(*pairs: dict) -> dict:
    return {"error": None, "pairs": list(pairs), "cost_usd": 0.0}


def test_perfect_predictions_score_full_accuracy_and_zero_hallucination():
    rows = [_row(
        _pair("SUPPORT", "strongly_supports"),
        _pair("CONTRADICT", "contradicts"),
        _pair("NOINFO", "no_evidence"),
    )]
    m = _summarise(rows)
    assert m["accuracy"] == 1.0
    assert m["hallucinated_finding_rate"] == 0.0
    assert m["strong_hallucination_rate"] == 0.0


def test_weak_assertion_on_noinfo_counts_as_hallucination_not_strong():
    rows = [_row(_pair("NOINFO", "weakly_supports"))]
    m = _summarise(rows)
    assert m["hallucinated_finding_rate"] == 1.0
    assert m["strong_hallucination_rate"] == 0.0
    assert m["n_hallucinated"] == 1
    assert m["n_strong_hallucinated"] == 0


def test_strong_assertion_on_noinfo_counts_as_both():
    rows = [_row(_pair("NOINFO", "strongly_contradicts"))]
    m = _summarise(rows)
    assert m["hallucinated_finding_rate"] == 1.0
    assert m["strong_hallucination_rate"] == 1.0


def test_noinfo_below_relevance_gate_is_never_a_hallucination():
    rows = [_row(_pair("NOINFO", "strongly_supports", relevance=0.1))]
    m = _summarise(rows)
    assert m["hallucinated_finding_rate"] == 0.0
    assert m["noinfo_below_gate_rate"] == 1.0


def test_no_evidence_on_noinfo_is_never_a_hallucination():
    rows = [_row(_pair("NOINFO", "no_evidence"))]
    m = _summarise(rows)
    assert m["hallucinated_finding_rate"] == 0.0


def test_empty_key_finding_is_never_a_hallucination_even_with_a_direction():
    # Defensive: judge_directions should never produce this combination
    # (no key_finding means the deterministic gate assigns no_evidence), but
    # the metric itself should not depend on that invariant holding upstream.
    rows = [_row(_pair("NOINFO", "strongly_supports", key_finding=""))]
    m = _summarise(rows)
    assert m["hallucinated_finding_rate"] == 0.0


def test_mixed_rate_by_gold_only_counts_the_raw_mixed_label():
    rows = [_row(
        _pair("SUPPORT", "mixed"),
        _pair("SUPPORT", "supports"),
        _pair("CONTRADICT", "contradicts"),
    )]
    m = _summarise(rows)
    assert m["mixed_rate_by_gold"]["SUPPORT"] == 0.5
    assert m["mixed_rate_by_gold"]["CONTRADICT"] == 0.0
    assert m["mixed_rate"] == 1 / 3


def test_confusion_matrix_never_keyerrors_on_any_graded_label():
    all_labels = (
        "strongly_contradicts", "contradicts", "weakly_contradicts",
        "no_evidence", "weakly_supports", "supports", "strongly_supports",
        "mixed",
    )
    rows = [_row(*(_pair("NOINFO", label) for label in all_labels))]
    m = _summarise(rows)  # must not raise
    assert sum(m["confusion"]["NOINFO"].values()) == len(all_labels)


def test_contradict_recall_matches_hand_computed_value():
    # 2 real CONTRADICT docs, only 1 correctly predicted -> recall 0.5.
    rows = [_row(
        _pair("CONTRADICT", "contradicts"),
        _pair("CONTRADICT", "no_evidence"),
    )]
    m = _summarise(rows)
    assert m["contradict_recall"] == 0.5


def test_errored_rows_are_excluded_but_counted():
    rows = [
        _row(_pair("SUPPORT", "supports")),
        {"error": "boom", "pairs": [], "cost_usd": 0.0},
    ]
    m = _summarise(rows)
    assert m["n_topics"] == 2
    assert m["n_errors"] == 1
    assert m["n_pairs"] == 1
