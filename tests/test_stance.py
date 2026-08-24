"""Unit tests for the stance-judgment vocabulary and judge_directions'
deterministic gates - no network; generate_json is monkeypatched for the
one path that reaches an LLM call.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline import synthesis
from app.pipeline.synthesis import collapse_direction, judge_directions, read_direction
from app.sources.base import SourceResult

_ALL_DIRECTIONS = (
    "strongly_contradicts", "contradicts", "weakly_contradicts",
    "no_evidence",
    "weakly_supports", "supports", "strongly_supports",
    "mixed",
)


# ── read_direction ────────────────────────────────────────────


def test_read_direction_accepts_every_known_value():
    for value in _ALL_DIRECTIONS:
        assert read_direction(value) == value


def test_read_direction_is_case_and_whitespace_insensitive():
    assert read_direction("  Supports \n") == "supports"


def test_read_direction_defaults_unknown_to_no_evidence():
    assert read_direction("definitely_supports") == "no_evidence"
    assert read_direction("neutral") == "no_evidence"  # the old 4-way label
    assert read_direction(None) == "no_evidence"
    assert read_direction(42) == "no_evidence"


# ── collapse_direction ───────────────────────────────────────


def test_collapse_direction_maps_every_supports_grade():
    for value in ("weakly_supports", "supports", "strongly_supports"):
        assert collapse_direction(value) == "SUPPORT"


def test_collapse_direction_maps_every_contradicts_grade():
    for value in ("weakly_contradicts", "contradicts", "strongly_contradicts"):
        assert collapse_direction(value) == "CONTRADICT"


def test_collapse_direction_maps_no_evidence_and_mixed_to_noinfo():
    assert collapse_direction("no_evidence") == "NOINFO"
    assert collapse_direction("mixed") == "NOINFO"


def test_collapse_direction_maps_unknown_to_noinfo():
    assert collapse_direction("garbage") == "NOINFO"


# ── judge_directions ──────────────────────────────────────────


def _source(content: str = "") -> SourceResult:
    return SourceResult(title="t", url="https://example.invalid/doc", snippet="", content=content)


def test_judge_directions_empty_key_finding_skips_the_call_entirely(monkeypatch):
    def boom(**_kwargs):
        raise AssertionError("generate_json should not be called")
    monkeypatch.setattr(synthesis, "generate_json", boom)

    appraisals = {0: {"key_finding": "", "evidence_quote": "", "relevance_score": 0.9}}
    out = judge_directions("topic", [_source()], appraisals, settings=None)
    assert out == {0: "no_evidence"}


def test_judge_directions_low_relevance_skips_the_call(monkeypatch):
    def boom(**_kwargs):
        raise AssertionError("generate_json should not be called")
    monkeypatch.setattr(synthesis, "generate_json", boom)

    appraisals = {0: {
        "key_finding": "drug reduces risk", "evidence_quote": "risk was reduced",
        "relevance_score": 0.5,  # at/below MIN_SYNTHESIS_RELEVANCE
    }}
    out = judge_directions(
        "topic", [_source(content="risk was reduced")], appraisals, settings=None,
    )
    assert out == {0: "no_evidence"}


def test_judge_directions_ungrounded_quote_skips_the_call(monkeypatch):
    def boom(**_kwargs):
        raise AssertionError("generate_json should not be called")
    monkeypatch.setattr(synthesis, "generate_json", boom)

    appraisals = {0: {
        "key_finding": "drug reduces risk",
        "evidence_quote": "a sentence never actually in the source",
        "relevance_score": 0.9,
    }}
    out = judge_directions(
        "topic", [_source(content="totally unrelated source text")], appraisals, settings=None,
    )
    assert out == {0: "no_evidence"}


def test_judge_directions_grounded_quote_reaches_the_llm_and_returns_its_direction(monkeypatch):
    captured = {}

    def fake_generate_json(*, prompt, **kwargs):
        captured["prompt"] = prompt
        return {"reasoning": "the quote reports a reduction", "direction": "strongly_supports"}
    monkeypatch.setattr(synthesis, "generate_json", fake_generate_json)

    appraisals = {0: {
        "key_finding": "drug reduces risk", "evidence_quote": "risk was reduced by 40%",
        "relevance_score": 0.9,
    }}
    out = judge_directions(
        "topic", [_source(content="Methods... risk was reduced by 40% in the trial.")],
        appraisals, settings=None,
    )
    assert out == {0: "strongly_supports"}
    assert "risk was reduced by 40%" in captured["prompt"]


def test_judge_directions_grounding_check_is_whitespace_and_case_insensitive(monkeypatch):
    monkeypatch.setattr(
        synthesis, "generate_json",
        lambda **_kwargs: {"reasoning": "ok", "direction": "supports"},
    )
    appraisals = {0: {
        "key_finding": "x", "evidence_quote": "Risk   Was\nReduced", "relevance_score": 0.9,
    }}
    out = judge_directions(
        "topic", [_source(content="...risk was reduced significantly...")],
        appraisals, settings=None,
    )
    assert out == {0: "supports"}


def test_judge_directions_rejects_a_label_outside_the_stance_vocabulary(monkeypatch):
    monkeypatch.setattr(
        synthesis, "generate_json",
        lambda **_kwargs: {"reasoning": "ok", "direction": "neutral"},  # the old 4-way label
    )
    appraisals = {0: {
        "key_finding": "x", "evidence_quote": "y", "relevance_score": 0.9,
    }}
    out = judge_directions("topic", [_source(content="y")], appraisals, settings=None)
    assert out == {0: "no_evidence"}


def test_judge_directions_llm_failure_falls_back_to_no_evidence(monkeypatch):
    def boom(**_kwargs):
        raise RuntimeError("upstream is down")
    monkeypatch.setattr(synthesis, "generate_json", boom)

    appraisals = {0: {
        "key_finding": "x", "evidence_quote": "y", "relevance_score": 0.9,
    }}
    out = judge_directions("topic", [_source(content="y")], appraisals, settings=None)
    assert out == {0: "no_evidence"}


def test_judge_directions_handles_a_mix_of_gated_and_judged_sources(monkeypatch):
    monkeypatch.setattr(
        synthesis, "generate_json",
        lambda **_kwargs: {"reasoning": "ok", "direction": "contradicts"},
    )
    appraisals = {
        0: {"key_finding": "", "evidence_quote": "", "relevance_score": 0.9},  # gated
        1: {"key_finding": "x", "evidence_quote": "the result", "relevance_score": 0.9},  # judged
    }
    results = [_source(), _source(content="the result was negative")]
    out = judge_directions("topic", results, appraisals, settings=None)
    assert out == {0: "no_evidence", 1: "contradicts"}


def test_judge_directions_empty_appraisals_returns_empty():
    assert judge_directions("topic", [], {}, settings=None) == {}
