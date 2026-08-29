"""Unit tests for _summarise_batch's parsing of the raw submit_appraisals
payload into the `summaries` dict `judge_directions` and the orchestrator
consume - in particular, that a source's findings list is capped before
either of them ever sees it.

No network, no LLM call: generate_with_tools is monkeypatched to submit a
synthetic appraisals payload directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.llm_client import ToolInvocation
from app.pipeline import synthesis
from app.sources.base import SourceResult


def _source() -> SourceResult:
    return SourceResult(title="A Trial", url="https://example.invalid/doc", snippet="short")


def _submit(monkeypatch, appraisals: list[dict]) -> dict[int, dict]:
    def fake_generate_with_tools(*, tool_handlers, **_kwargs):
        tool_handlers["submit_appraisals"]({"appraisals": appraisals})
        return "", [ToolInvocation(name="submit_appraisals", arguments={"appraisals": appraisals})]

    monkeypatch.setattr(synthesis, "generate_with_tools", fake_generate_with_tools)
    return synthesis._summarise_batch(
        "topic", [_source()], [0], settings=None, summary_language="en",
        job_stats=None, pico=None,
    )


def test_findings_beyond_the_cap_are_dropped_before_judge_directions_sees_them(monkeypatch):
    # A model that ignores "up to MAX_FINDINGS_PER_SOURCE" must not get to
    # burn a stance call (in judge_directions, downstream of this function)
    # on the findings past the cap - they need to be gone by the time
    # summarise_sources returns, not just truncated later on read.
    findings = [{"text": f"finding {i}", "evidence_quote": f"quote {i}"} for i in range(5)]
    out = _submit(monkeypatch, [{"index": 0, "summary": "s", "relevance_score": 0.9, "findings": findings}])
    assert len(out[0]["findings"]) == synthesis.MAX_FINDINGS_PER_SOURCE
    assert [f["text"] for f in out[0]["findings"]] == ["finding 0", "finding 1", "finding 2"]


def test_findings_at_or_under_the_cap_are_left_untouched(monkeypatch):
    findings = [{"text": "only one", "evidence_quote": "q"}]
    out = _submit(monkeypatch, [{"index": 0, "summary": "s", "relevance_score": 0.9, "findings": findings}])
    assert out[0]["findings"] == findings


def test_missing_or_malformed_findings_field_is_left_as_is(monkeypatch):
    out = _submit(monkeypatch, [{"index": 0, "summary": "s", "relevance_score": 0.9}])
    assert "findings" not in out[0]

    out = _submit(monkeypatch, [{"index": 0, "summary": "s", "relevance_score": 0.9, "findings": "not a list"}])
    assert out[0]["findings"] == "not a list"  # untouched - downstream readers guard the type, not this parser
