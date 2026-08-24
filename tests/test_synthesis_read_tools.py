"""Unit tests for _summarise_batch's read_full_text / read_section tool
handlers - the fetch cache they share, and read_section's section matching.

No network, no LLM call: generate_with_tools is monkeypatched to invoke the
handlers directly with synthetic tool-call arguments, the same shape a real
model's tool calls would carry, then finalise via submit_appraisals exactly
as the real loop requires.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.llm_client import ToolInvocation
from app.pipeline import synthesis
from app.sources.base import SourceResult

_RAW_FULLTEXT = (
    "## Abstract\nBrief summary of the trial.\n"
    "## Methods\nHow the trial was run.\n"
    "## Results and Discussion\nThe hazard ratio was 0.65 (p=0.002).\n"
)


def _source(source_type: str = "europe_pmc") -> SourceResult:
    return SourceResult(
        title="A Trial", url="https://example.invalid/doc",
        snippet="short", source_type=source_type,
    )


def _run_batch(monkeypatch, calls: list[tuple[str, dict]], fetch_return: str | None = _RAW_FULLTEXT):
    """Run _summarise_batch with generate_with_tools faked to issue *calls*
    (a list of (tool_name, args) the "model" makes) before submitting."""
    monkeypatch.setattr(synthesis, "fetch_full_text", lambda result: fetch_return)

    captured: dict[str, object] = {}

    def fake_generate_with_tools(*, tool_handlers, **_kwargs):
        results = [(name, tool_handlers[name](args)) for name, args in calls]
        captured["results"] = results
        appraisals = [{"index": 0, "summary": "s", "relevance_score": 0.5}]
        tool_handlers["submit_appraisals"]({"appraisals": appraisals})
        return "", [ToolInvocation(name="submit_appraisals", arguments={"appraisals": appraisals})]

    monkeypatch.setattr(synthesis, "generate_with_tools", fake_generate_with_tools)

    results = [_source()]
    out = synthesis._summarise_batch(
        "topic", results, [0], settings=None, summary_language="en",
        job_stats=None, pico=None,
    )
    return out, captured["results"]


def test_read_section_returns_only_the_matched_section(monkeypatch):
    _out, results = _run_batch(monkeypatch, [("read_section", {"index": 0, "sections": ["results"]})])
    name, result = results[0]
    assert name == "read_section"
    assert "hazard ratio was 0.65" in result
    assert "How the trial was run" not in result


def test_read_section_matches_case_insensitively_and_by_substring(monkeypatch):
    # "results" should match the heading "Results and Discussion".
    _out, results = _run_batch(monkeypatch, [("read_section", {"index": 0, "sections": ["RESULTS"]})])
    assert "hazard ratio" in results[0][1]


def test_read_section_miss_lists_the_actual_headings(monkeypatch):
    _out, results = _run_batch(monkeypatch, [("read_section", {"index": 0, "sections": ["conclusion"]})])
    _name, result = results[0]
    assert "no section matching" in result
    assert "Results and Discussion" in result


def test_read_section_rejects_empty_sections_list(monkeypatch):
    _out, results = _run_batch(monkeypatch, [("read_section", {"index": 0, "sections": []})])
    assert "non-empty list" in results[0][1]


def test_read_full_text_and_read_section_share_one_fetch(monkeypatch):
    fetch_calls: list[SourceResult] = []

    def counting_fetch(result: SourceResult) -> str:
        fetch_calls.append(result)
        return _RAW_FULLTEXT

    monkeypatch.setattr(synthesis, "fetch_full_text", counting_fetch)
    monkeypatch.setattr(synthesis, "generate_with_tools", lambda **kwargs: _drive(kwargs))

    def _drive(kwargs):
        handlers = kwargs["tool_handlers"]
        handlers["read_full_text"]({"index": 0})
        handlers["read_section"]({"index": 0, "sections": ["methods"]})
        appraisals = [{"index": 0, "summary": "s", "relevance_score": 0.5}]
        handlers["submit_appraisals"]({"appraisals": appraisals})
        return "", [ToolInvocation(name="submit_appraisals", arguments={"appraisals": appraisals})]

    synthesis._summarise_batch(
        "topic", [_source()], [0], settings=None, summary_language="en",
        job_stats=None, pico=None,
    )
    assert len(fetch_calls) == 1


def test_read_section_reports_fetch_failure_not_budget_exhaustion(monkeypatch):
    out, results = _run_batch(
        monkeypatch, [("read_section", {"index": 0, "sections": ["results"]})], fetch_return="",
    )
    assert "not available" in results[0][1]
    assert "budget exhausted" not in results[0][1]


def test_read_full_text_still_returns_a_priority_filled_excerpt(monkeypatch):
    long_text = (
        "## Abstract\nshort\n"
        "## Introduction\n" + ("i" * 8000) + "\n"
        "## Results\nthe effect was significant\n"
    )
    _out, results = _run_batch(
        monkeypatch, [("read_full_text", {"index": 0})], fetch_return=long_text,
    )
    _name, result = results[0]
    assert "## Results" in result
