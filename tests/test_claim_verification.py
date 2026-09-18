"""Unit tests for claim_verification.py's deterministic surface - no network;
ask_choice is monkeypatched for the one path that reaches TypeSafe.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.typesafe_client import ChoiceAnswer
from app.pipeline import claim_verification
from app.pipeline.claim_verification import (
    ClaimVerdict,
    _evidence_block,
    apply_verdicts,
    verify_claims,
)
from app.schemas import ArticleSummary, Claim, Finding


def _article(findings: list[Finding] | None = None, full_summary: str = "") -> ArticleSummary:
    return ArticleSummary(
        url="https://example.invalid/doc", title="t",
        findings=findings or [], full_summary=full_summary,
    )


# ── _evidence_block ─────────────────────────────────────────────


def test_evidence_block_falls_back_to_full_summary_when_no_findings():
    article = _article(full_summary="a scan summary")
    assert _evidence_block(article) == "a scan summary"


def test_evidence_block_renders_finding_text_and_quote():
    article = _article(findings=[Finding(text="drug reduces risk", evidence_quote="risk was reduced")])
    block = _evidence_block(article)
    assert "drug reduces risk" in block
    assert 'quote: "risk was reduced"' in block


def test_evidence_block_omits_quote_line_when_finding_has_none():
    article = _article(findings=[Finding(text="drug reduces risk", evidence_quote="")])
    assert _evidence_block(article) == "drug reduces risk"


def test_evidence_block_renders_every_finding_of_a_multi_finding_source():
    article = _article(findings=[
        Finding(text="efficacy improved", evidence_quote="efficacy up 30%"),
        Finding(text="safety worsened", evidence_quote="AEs increased"),
    ])
    block = _evidence_block(article)
    assert "efficacy improved" in block and "safety worsened" in block
    assert 'quote: "efficacy up 30%"' in block and 'quote: "AEs increased"' in block


# ── verify_claims ────────────────────────────────────────────────


def test_verify_claims_empty_claims_returns_empty():
    assert verify_claims([], [_article(full_summary="x")], settings=None) == []


def test_verify_claims_skips_out_of_range_indices_without_calling_typesafe(monkeypatch):
    def boom(**_kwargs):
        raise AssertionError("ask_choice should not be called")
    monkeypatch.setattr(claim_verification, "ask_choice", boom)

    claims = [Claim(text="some claim", source_indices=[5])]  # no article at index 5
    assert verify_claims(claims, [_article(full_summary="x")], settings=None) == []


def test_verify_claims_skips_the_call_when_source_has_no_evidence(monkeypatch):
    def boom(**_kwargs):
        raise AssertionError("ask_choice should not be called")
    monkeypatch.setattr(claim_verification, "ask_choice", boom)

    claims = [Claim(text="some claim", source_indices=[0])]
    articles = [_article(full_summary="")]  # no findings, empty full_summary
    out = verify_claims(claims, articles, settings=None)
    assert out == [ClaimVerdict(0, 0, "says_nothing", None, {})]


def test_verify_claims_reaches_typesafe_and_returns_its_verdict(monkeypatch):
    captured = {}

    def fake_ask_choice(*, state, **kwargs):
        captured["state"] = state
        return ChoiceAnswer(choice="supports", confidence=0.93, probabilities={"supports": 0.93})
    monkeypatch.setattr(claim_verification, "ask_choice", fake_ask_choice)

    claims = [Claim(text="the drug reduces risk", source_indices=[0])]
    articles = [_article(findings=[Finding(text="risk reduced", evidence_quote="risk was reduced by 40%")])]
    out = verify_claims(claims, articles, settings=None)

    assert out == [ClaimVerdict(0, 0, "supports", 0.93, {"supports": 0.93})]
    assert captured["state"]["claim"] == "the drug reduces risk"
    assert "risk was reduced by 40%" in captured["state"]["source_evidence"]


def test_verify_claims_typesafe_failure_yields_error_verdict(monkeypatch):
    def boom(**_kwargs):
        raise RuntimeError("upstream is down")
    monkeypatch.setattr(claim_verification, "ask_choice", boom)

    claims = [Claim(text="x", source_indices=[0])]
    articles = [_article(findings=[Finding(text="x", evidence_quote="y")])]
    out = verify_claims(claims, articles, settings=None)
    assert out == [ClaimVerdict(0, 0, "error", None, {})]


def test_verify_claims_one_verdict_per_cited_source_not_per_claim(monkeypatch):
    # A claim citing two sources gets independently checked against each -
    # collapsing to one verdict would hide a source that doesn't back it.
    def fake_ask_choice(*, state, **_kwargs):
        if "supports it" in state["source_evidence"]:
            return ChoiceAnswer(choice="supports", confidence=0.9, probabilities={})
        return ChoiceAnswer(choice="says_nothing", confidence=0.8, probabilities={})
    monkeypatch.setattr(claim_verification, "ask_choice", fake_ask_choice)

    claims = [Claim(text="x", source_indices=[0, 1])]
    articles = [
        _article(findings=[Finding(text="supports it", evidence_quote="q1")]),
        _article(findings=[Finding(text="unrelated", evidence_quote="q2")]),
    ]
    out = sorted(verify_claims(claims, articles, settings=None), key=lambda v: v.source_index)
    assert out == [
        ClaimVerdict(0, 0, "supports", 0.9, {}),
        ClaimVerdict(0, 1, "says_nothing", 0.8, {}),
    ]


# ── apply_verdicts ──────────────────────────────────────────────


def test_apply_verdicts_claim_with_no_verdicts_is_kept_unchanged():
    claim = Claim(text="x", source_indices=[0], strength="strong")
    assert apply_verdicts([claim], []) == ([claim], [])


def test_apply_verdicts_keeps_and_flags_claim_confidently_contradicted():
    # Never dropped, however confident: see CONTRADICT_FLAG_CONFIDENCE for the
    # base-rate reasoning. Demoted to weak, flagged, citations left intact so
    # the reader can check the source the flag is about.
    claim = Claim(text="x", source_indices=[0], strength="strong")
    verdicts = [ClaimVerdict(0, 0, "contradicts", 0.95, {})]
    out, flags = apply_verdicts([claim], verdicts)
    assert out == [Claim(text="x", source_indices=[0], strength="weak")]
    assert len(flags) == 1 and "x" in flags[0]


def test_apply_verdicts_contradicted_claim_keeps_the_contradicting_citation():
    claim = Claim(text="x", source_indices=[0, 1], strength="strong")
    verdicts = [
        ClaimVerdict(0, 0, "supports", 0.99, {}),
        ClaimVerdict(0, 1, "contradicts", 0.95, {}),
    ]
    out, flags = apply_verdicts([claim], verdicts)
    assert out == [Claim(text="x", source_indices=[0, 1], strength="weak")]
    assert len(flags) == 1


def test_apply_verdicts_keeps_claim_weakly_contradicted_without_flagging():
    claim = Claim(text="x", source_indices=[0], strength="strong")
    verdicts = [ClaimVerdict(0, 0, "contradicts", 0.46, {})]
    out, flags = apply_verdicts([claim], verdicts)
    assert out == [Claim(text="x", source_indices=[0], strength="weak")]
    assert flags == []


def test_apply_verdicts_drops_claim_all_sources_say_nothing_confidently():
    claim = Claim(text="x", source_indices=[0, 1], strength="moderate")
    verdicts = [
        ClaimVerdict(0, 0, "says_nothing", 0.99, {}),
        ClaimVerdict(0, 1, "says_nothing", 0.97, {}),
    ]
    out, flags = apply_verdicts([claim], verdicts)
    assert out == []
    # The removal is reported, not just logged - nothing leaves the response
    # silently.
    assert len(flags) == 1 and "x" in flags[0]


def test_apply_verdicts_says_nothing_below_the_raised_bar_is_kept():
    # 0.9 cleared the old 0.85 reject bar and would have deleted this claim;
    # UNSUPPORTED_REJECT_CONFIDENCE is 0.95, so it now survives as weak.
    claim = Claim(text="x", source_indices=[0], strength="moderate")
    verdicts = [ClaimVerdict(0, 0, "says_nothing", 0.9, {})]
    out, flags = apply_verdicts([claim], verdicts)
    assert out == [Claim(text="x", source_indices=[0], strength="weak")]
    assert flags == []


def test_apply_verdicts_narrows_to_the_source_that_actually_supports():
    claim = Claim(text="x", source_indices=[0, 1], strength="strong")
    verdicts = [
        ClaimVerdict(0, 0, "supports", 0.95, {}),
        ClaimVerdict(0, 1, "says_nothing", 0.6, {}),
    ]
    out, flags = apply_verdicts([claim], verdicts)
    assert out == [Claim(text="x", source_indices=[0], strength="strong")]
    assert flags == []


def test_apply_verdicts_fully_supported_claim_is_untouched():
    claim = Claim(text="x", source_indices=[0, 1], strength="strong")
    verdicts = [
        ClaimVerdict(0, 0, "supports", 0.95, {}),
        ClaimVerdict(0, 1, "supports", 0.99, {}),
    ]
    assert apply_verdicts([claim], verdicts) == ([claim], [])


def test_apply_verdicts_error_only_claim_is_kept_untouched():
    claim = Claim(text="x", source_indices=[0], strength="strong")
    verdicts = [ClaimVerdict(0, 0, "error", None, {})]
    assert apply_verdicts([claim], verdicts) == ([claim], [])


def test_apply_verdicts_keeps_unverified_source_alongside_a_supported_one():
    # source 1's TypeSafe call failed (error) - unverified, not refuted, so
    # it should survive the narrowing that drops a confirmed non-support.
    claim = Claim(text="x", source_indices=[0, 1], strength="strong")
    verdicts = [
        ClaimVerdict(0, 0, "supports", 0.95, {}),
        ClaimVerdict(0, 1, "error", None, {}),
    ]
    assert apply_verdicts([claim], verdicts) == ([claim], [])


def test_apply_verdicts_operates_per_claim_independently():
    claims = [
        Claim(text="good", source_indices=[0], strength="strong"),
        Claim(text="bad", source_indices=[1], strength="strong"),
    ]
    verdicts = [
        ClaimVerdict(0, 0, "supports", 0.95, {}),
        ClaimVerdict(1, 1, "says_nothing", 0.99, {}),
    ]
    out, flags = apply_verdicts(claims, verdicts)
    assert out == [claims[0]]
    assert len(flags) == 1


def test_apply_verdicts_flags_follow_the_summary_language():
    claim = Claim(text="x", source_indices=[0], strength="strong")
    verdicts = [ClaimVerdict(0, 0, "contradicts", 0.95, {})]
    _, it_flags = apply_verdicts([claim], verdicts, summary_language="it")
    _, en_flags = apply_verdicts([claim], verdicts, summary_language="en")
    assert it_flags != en_flags
    assert "contradict" in en_flags[0]
