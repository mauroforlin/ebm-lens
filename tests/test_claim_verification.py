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
    _mass,
    apply_verdicts,
    verify_claims,
)
from app.schemas import ArticleSummary, Claim, Finding


def _dist(claim_index: int, source_index: int, **masses: float) -> ClaimVerdict:
    """A verdict carrying a real distribution, argmax derived from it.

    Most tests below are about a rule reading one relation's mass, so naming
    the masses and letting the argmax fall out is closer to what TypeSafe
    actually returns than picking a label by hand and hoping it matches.
    """
    relation = max(masses, key=lambda k: masses[k])
    return ClaimVerdict(
        claim_index=claim_index, source_index=source_index, relation=relation,
        confidence=masses[relation], probabilities=dict(masses),
    )


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
    # Never dropped, however confident: see CONTRADICT_FLAG_PROBABILITY for the
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
    # UNSUPPORTED_REJECT_PROBABILITY is 0.95, so it now survives as weak.
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


# ── _mass ───────────────────────────────────────────────────────


def test_mass_reads_the_distribution_not_the_argmax():
    verdict = _dist(0, 0, supports=0.2, contradicts=0.3, says_nothing=0.5)
    assert verdict.relation == "says_nothing"
    assert _mass(verdict, "contradicts") == 0.3
    assert _mass(verdict, "supports") == 0.2


def test_mass_falls_back_to_confidence_without_a_distribution():
    # What every rule computed before there was a distribution to read, and
    # what a backend returning only a label and a confidence would give.
    verdict = ClaimVerdict(0, 0, "contradicts", 0.9, {})
    assert _mass(verdict, "contradicts") == 0.9
    assert _mass(verdict, "supports") == 0.0
    assert _mass(verdict, "says_nothing") == 0.0


def test_mass_falls_back_when_the_distribution_is_keyed_unexpectedly():
    # If probabilities ever came back keyed by something other than the
    # criteria names, a plain .get() would read 0.0 for every relation and
    # silently switch every rule off. The guard notices the dict does not
    # contain the verdict's own answer and reverts to the scalar reading.
    verdict = ClaimVerdict(0, 0, "contradicts", 0.9, {"0": 0.9, "1": 0.05, "2": 0.05})
    assert _mass(verdict, "contradicts") == 0.9


def test_mass_of_an_errored_verdict_is_zero_everywhere():
    verdict = ClaimVerdict(0, 0, "error", None, {})
    assert _mass(verdict, "contradicts") == 0.0
    assert _mass(verdict, "supports") == 0.0
    assert _mass(verdict, "says_nothing") == 0.0


def test_mass_of_the_no_evidence_shortcircuit_is_zero():
    # verify_claims fabricates this without calling TypeSafe: a says_nothing
    # with no confidence and no distribution. It must not count toward the
    # drop rule on the strength of a label alone.
    verdict = ClaimVerdict(0, 0, "says_nothing", None, {})
    assert _mass(verdict, "says_nothing") == 0.0


# ── apply_verdicts, reading the distribution ────────────────────


def test_bare_plurality_support_no_longer_counts_as_support():
    # The latent bug this fixes: `relation == "supports"` with no bar meant
    # {supports .34, contradicts .33, says_nothing .33} - a verdict closer to
    # "cannot tell" than to "the source backs this" - kept full strength.
    claim = Claim(text="x", source_indices=[0], strength="strong")
    verdicts = [_dist(0, 0, supports=0.34, contradicts=0.33, says_nothing=0.33)]
    out, flags = apply_verdicts([claim], verdicts)
    assert out == [Claim(text="x", source_indices=[0], strength="weak")]
    assert flags == []


def test_majority_support_still_counts_as_support():
    claim = Claim(text="x", source_indices=[0], strength="strong")
    verdicts = [_dist(0, 0, supports=0.6, contradicts=0.1, says_nothing=0.3)]
    assert apply_verdicts([claim], verdicts) == ([claim], [])


def test_split_direction_demotes_even_though_argmax_is_supports():
    # The model cannot tell whether the source backs or refutes the claim.
    # Under the old rule this was "supports" and sailed through untouched.
    claim = Claim(text="x", source_indices=[0], strength="strong")
    verdicts = [_dist(0, 0, supports=0.46, contradicts=0.45, says_nothing=0.09)]
    out, flags = apply_verdicts([claim], verdicts)
    assert out == [Claim(text="x", source_indices=[0], strength="weak")]
    assert flags == []


def test_unremarkable_argmax_with_hidden_contradiction_mass_is_left_alone():
    # {says_nothing .47, contradicts .45} used to be demoted by a "doubt
    # band". SciFact says that band's hit rate sits on the corpus base rate
    # at every bound tried, so it was removed: sub-threshold contradiction
    # mass is not evidence. The claim keeps its strength.
    claim = Claim(text="x", source_indices=[0, 1], strength="moderate")
    verdicts = [
        _dist(0, 0, supports=0.08, contradicts=0.45, says_nothing=0.47),
        _dist(0, 1, supports=0.90, contradicts=0.05, says_nothing=0.05),
    ]
    out, flags = apply_verdicts([claim], verdicts)
    assert out == [Claim(text="x", source_indices=[1], strength="moderate")]
    assert flags == []


def test_contradiction_below_the_flag_bar_leaves_a_supported_claim_alone():
    claim = Claim(text="x", source_indices=[0], strength="strong")
    verdicts = [_dist(0, 0, supports=0.7, contradicts=0.1, says_nothing=0.2)]
    assert apply_verdicts([claim], verdicts) == ([claim], [])


def test_a_well_supported_claim_keeps_its_strength_despite_a_quiet_dissenter():
    # One source backs it outright, another leans toward refuting it without
    # clearing the flag bar. The dissenting citation is narrowed away as
    # unsupported; the claim's strength survives, because the only rule that
    # would have taken it was the band the eval retired.
    claim = Claim(text="x", source_indices=[0, 1], strength="strong")
    verdicts = [
        _dist(0, 0, supports=0.95, contradicts=0.02, says_nothing=0.03),
        _dist(0, 1, supports=0.10, contradicts=0.50, says_nothing=0.40),
    ]
    out, flags = apply_verdicts([claim], verdicts)
    assert out == [Claim(text="x", source_indices=[0], strength="strong")]
    assert flags == []


def test_the_most_contradicting_source_decides_not_the_average():
    # Two quiet sources must not dilute one that refutes the claim outright.
    claim = Claim(text="x", source_indices=[0, 1, 2], strength="strong")
    verdicts = [
        _dist(0, 0, supports=0.05, contradicts=0.02, says_nothing=0.93),
        _dist(0, 1, supports=0.05, contradicts=0.03, says_nothing=0.92),
        _dist(0, 2, supports=0.03, contradicts=0.90, says_nothing=0.07),
    ]
    out, flags = apply_verdicts([claim], verdicts)
    assert out == [Claim(text="x", source_indices=[0, 1, 2], strength="weak")]
    assert len(flags) == 1


def test_narrowing_drops_a_source_whose_support_is_only_a_plurality():
    claim = Claim(text="x", source_indices=[0, 1], strength="strong")
    verdicts = [
        _dist(0, 0, supports=0.90, contradicts=0.02, says_nothing=0.08),
        _dist(0, 1, supports=0.40, contradicts=0.25, says_nothing=0.35),
    ]
    out, flags = apply_verdicts([claim], verdicts)
    assert out == [Claim(text="x", source_indices=[0], strength="strong")]
    assert flags == []


def test_drop_rule_needs_mass_on_says_nothing_from_every_source():
    claim = Claim(text="x", source_indices=[0, 1], strength="moderate")
    verdicts = [
        _dist(0, 0, supports=0.01, contradicts=0.01, says_nothing=0.98),
        _dist(0, 1, supports=0.30, contradicts=0.10, says_nothing=0.60),
    ]
    out, flags = apply_verdicts([claim], verdicts)
    assert out == [Claim(text="x", source_indices=[0, 1], strength="weak")]
    assert flags == []


def test_flag_rule_still_fires_on_an_overwhelming_contradiction():
    claim = Claim(text="x", source_indices=[0], strength="strong")
    verdicts = [_dist(0, 0, supports=0.02, contradicts=0.95, says_nothing=0.03)]
    out, flags = apply_verdicts([claim], verdicts)
    assert out == [Claim(text="x", source_indices=[0], strength="weak")]
    assert len(flags) == 1


def test_an_errored_source_never_contributes_mass_to_any_rule():
    # All three rules read only `real`; the error keeps its citation.
    claim = Claim(text="x", source_indices=[0, 1], strength="strong")
    verdicts = [
        _dist(0, 0, supports=0.9, contradicts=0.05, says_nothing=0.05),
        ClaimVerdict(0, 1, "error", None, {}),
    ]
    assert apply_verdicts([claim], verdicts) == ([claim], [])
