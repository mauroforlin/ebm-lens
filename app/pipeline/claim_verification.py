"""Claim-to-evidence verification via TypeSafe's Jev (Choice primitive).

`synthesise`'s citation check (`app/pipeline/synthesis.py::_valid_indices`)
only checks that a claim's cited index exists among the articles being
returned - not that the article actually backs the claim citing it, a gap
that function's own TODO names directly: "Verifying that would need a
separate entailment check between claim text and source content".
`eval/stance_eval.py`'s docstring explains why that check was never built as
a distractor-based eval against `synthesise` (synthesise writes its own
claims, so there is no gold label for "should claim k cite source j" -
answering it needs exactly the entailment machinery whose absence made the
eval unsound). This module is that machinery, built directly rather than
through an eval, one (claim, cited source) pair at a time. No pipeline stage
calls it - it is a standalone building block a caller runs directly against
`synthesise`'s output (see Usage below), not a step any request triggers on
its own.

For a pair, the "source" side is the same evidence `_source_lines` already
showed the synthesiser for that article - each `Finding`'s text and verbatim
`evidence_quote` (already checked against the source's own fetched content by
`is_quote_grounded`), or `full_summary` when the article had no finding that
cleared the bar. A Choice question then asks how that evidence relates to the
claim: `supports`, `contradicts`, or `says_nothing` - TypeSafe's own
citation_check cookbook's three-way relation (docs.typesafe.ai/cookbooks/
citation_check), not `judge_directions`' seven-grade scale, since this is a
different question (does source i back claim k as written) from what that
scale answers (which way does source i's finding point on the topic). The
`confidence` on each verdict is computed by TypeSafe from the model's own
probability distribution, not self-reported - see typesafe_client.py.

A claim citing several sources gets one verdict per cited source, not one
verdict for the claim as a whole: a claim can rest on one source that
supports it and another that says nothing, and collapsing that before it is
even inspected would hide exactly the case this module exists to catch.

`app/pipeline/orchestrator.py` runs `verify_claims` right after `synthesise`
returns, then folds the verdicts back into `key_findings` with
`apply_verdicts` before the response is built - gated on `settings.
typesafe_key` being set, since this stays optional (see typesafe_client.py).

Usage (run directly, e.g. against synthesise's own output):

    from app.pipeline.claim_verification import apply_verdicts, verify_claims
    verdicts = verify_claims(claims, articles, settings)
    claims, flags = apply_verdicts(claims, verdicts)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import Settings
from app.core.job_stats import JobStats
from app.core.llm_client import generate_json
from app.core.parallel import run_parallel
from app.core.typesafe_client import ask_choice
from app.schemas import ArticleSummary, Claim

logger = logging.getLogger(__name__)

# Mirrors _STANCE_SYSTEM's three underlying relations (synthesis.py) and
# TypeSafe's own citation_check cookbook - not judge_directions' seven-grade
# scale, since this checks a different thing (see module docstring).
_RELATION_CRITERIA: dict[str, str | None] = {
    "supports": "The source's evidence states the claim or directly implies that it is true.",
    "contradicts": "The source's evidence states the opposite of the claim or implies it is false.",
    "says_nothing": "The source's evidence does not address what the claim asserts, either way.",
}

_INSTRUCTIONS = "How does the source's evidence relate to the claim?"

# Pairs checked concurrently. Conservative pending a real TypeSafe rate
# limiter (see typesafe_client.py's module docstring) - this is sized for a
# person running a manual check over one topic's claims, not a production
# fan-out.
_VERIFY_MAX_WORKERS = 5


TYPESAFE_BACKEND = "typesafe"
LLM_BACKEND = "llm"


@dataclass
class ClaimVerdict:
    """One (claim, cited source) pair's verification result.

    *relation* is "error" when the judging call itself failed (network,
    auth, ...) - distinct from "says_nothing", which is a real answer the
    model gave, not a failure to get one.

    *backend* names which judge produced this, and is not decoration: the
    thresholds every rule in apply_verdicts compares against are per-backend
    (see _THRESHOLDS), and carrying the origin on the verdict is what makes
    it impossible to grade one backend's answers against the other's bars.
    Defaulted so verdicts built by hand still read as TypeSafe's.
    """

    claim_index: int
    source_index: int
    relation: str
    confidence: float | None
    probabilities: dict[str, float]
    backend: str = TYPESAFE_BACKEND


def _evidence_block(article: ArticleSummary) -> str:
    """Render one article's evidence exactly as `_source_lines` showed it to
    the synthesiser - its findings' text and verbatim quote, or its scan
    summary when no finding cleared the bar. Keeping this the same substrate
    means a verdict here answers "was the synthesiser's own input enough to
    write this claim", not some other, richer reading of the source.
    """
    if not article.findings:
        return article.full_summary
    lines: list[str] = []
    for finding in article.findings:
        lines.append(finding.text)
        if finding.evidence_quote:
            lines.append(f'quote: "{finding.evidence_quote}"')
    return "\n".join(lines)


_LLM_SYSTEM = (
    "You judge how a piece of scientific evidence relates to a claim. "
    'Answer with only a JSON object: {"relation": "supports" | "contradicts" '
    '| "says_nothing", "confidence": <float 0 to 1>}. "supports" means the '
    'evidence states the claim or directly implies it is true. "contradicts" '
    "means the evidence states the opposite of the claim or implies it is "
    'false. "says_nothing" means the evidence does not address the claim '
    "either way. confidence is your own honest estimate that your relation "
    "judgment is correct."
)


def _judge_typesafe(
    claim_text: str, evidence: str, settings: Settings, job_stats: JobStats | None,
) -> tuple[str, float, dict[str, float]]:
    answer = ask_choice(
        settings=settings,
        state={"claim": claim_text, "source_evidence": evidence},
        instructions=_INSTRUCTIONS,
        criteria=_RELATION_CRITERIA,
        purpose="claim_verification",
        job_stats=job_stats,
    )
    return answer.choice, answer.confidence, dict(answer.probabilities)


def _judge_llm(
    claim_text: str, evidence: str, settings: Settings, job_stats: JobStats | None,
) -> tuple[str, float, dict[str, float]]:
    """Ask the same question through the model the pipeline already uses.

    Word for word the prompt `eval/openrouter_stance_eval.py` graded, because
    _LLM_THRESHOLDS' numbers were measured through it - reworded prompt,
    invalidated thresholds.

    Returns no distribution: a prompted model reports one confidence and
    nothing about where the rest of its belief sits. `_mass` falls back to
    reading that scalar, which is exactly the shape this returns.
    """
    result = generate_json(
        settings=settings,
        prompt=f"Claim: {claim_text}\n\nEvidence: {evidence}",
        system_instruction=_LLM_SYSTEM,
        purpose="claim_verification_llm",
        job_stats=job_stats,
    )
    relation = str(result.get("relation", "")).strip().lower()
    if relation not in _RELATION_CRITERIA:
        raise ValueError(f"unexpected relation from the judge: {relation!r}")
    return relation, float(result.get("confidence", 0.0)), {}


def select_backend(settings: Settings) -> str:
    """Which judge this instance verifies with.

    TypeSafe when a key is configured, otherwise the OpenRouter model the
    pipeline already requires. Verification is therefore no longer optional:
    before this, an instance without a TypeSafe key ran no check at all and
    the two stages were dead code for anyone who cloned the repo. The two
    backends are not interchangeable at equal settings, which is why the
    thresholds travel with the verdict - see _THRESHOLDS.
    """
    return TYPESAFE_BACKEND if settings.typesafe_key else LLM_BACKEND


_JUDGES = {TYPESAFE_BACKEND: _judge_typesafe, LLM_BACKEND: _judge_llm}


def verify_claims(
    claims: list[Claim],
    articles: list[ArticleSummary],
    settings: Settings,
    job_stats: JobStats | None = None,
) -> list[ClaimVerdict]:
    """Verify every claim against each of its cited sources, one Choice call
    per (claim, source) pair, run concurrently.

    *articles* is indexed the same way `claim.source_indices` already is -
    the real position in the response, not a translated index - so this can
    run directly on `synthesise`'s own output with no remapping.
    """
    pairs = [
        (c_index, s_index)
        for c_index, claim in enumerate(claims)
        for s_index in claim.source_indices
        if 0 <= s_index < len(articles)
    ]
    if not pairs:
        return []

    backend = select_backend(settings)
    judge = _JUDGES[backend]

    def _check(pair: tuple[int, int]) -> ClaimVerdict:
        c_index, s_index = pair
        claim = claims[c_index]
        evidence = _evidence_block(articles[s_index])
        if not evidence.strip():
            # Nothing was ever shown to the synthesiser for this source - no
            # call needed, and no basis for a "supports" or "contradicts"
            # verdict either.
            return ClaimVerdict(c_index, s_index, "says_nothing", None, {}, backend)
        try:
            relation, confidence, probabilities = judge(
                claim.text, evidence, settings, job_stats,
            )
        except Exception as exc:
            logger.warning(
                "Claim verification failed for claim %d / source %d (%s): %s",
                c_index, s_index, backend, exc,
            )
            return ClaimVerdict(c_index, s_index, "error", None, {}, backend)
        return ClaimVerdict(c_index, s_index, relation, confidence, probabilities, backend)

    return run_parallel(_check, pairs, _VERIFY_MAX_WORKERS)


# Every threshold below reads a *probability mass* out of the verdict's own
# distribution, not `confidence`. `confidence` is one number describing how
# concentrated the distribution is; it cannot say where the remaining mass
# went, and every decision here turns on exactly that. "contradicts at 0.7"
# is {contradicts .55, supports .40, says_nothing .05} - the model cannot
# tell which way the source points - or {contradicts .55, says_nothing .42,
# supports .03}, which is far milder. Reading p(contradicts) directly tells
# the two apart; reading `relation` plus `confidence` cannot. See _mass for
# what happens when a verdict carries no distribution at all.
#
# Mass a "contradicts" needs before it demotes a claim and surfaces a flag -
# mirrors TypeSafe's own citation_check cookbook (AUTO_ACCEPT = 0.8), a
# little higher because a flag still costs a reader attention.
#
# This deliberately does not gate *dropping* a claim. Measured against
# SciFact, `contradicts` at >= 0.85 is 84.3% precise (226/268) - but that
# number is a function of SciFact's 21% CONTRADICT base rate, not a property
# of the model. Holding the measured sensitivity (85.3%) and false-positive
# rate (42/994 = 4.2%) fixed and moving only the base rate, precision falls
# to ~69% at 10% contradicted claims and ~35% at the ~2.6% rate the real
# pipeline actually showed. Silently deleting a correct claim is the worse
# failure for an evidence tool than showing a suspect one next to a warning,
# and at those precisions dropping loses in both directions. So a confident
# contradiction demotes and flags; it never deletes.
CONTRADICT_FLAG_PROBABILITY = 0.85

# There was a third bar here - a "doubt band" demoting a claim when enough
# mass sat on `contradicts` without clearing the flag bar, on the theory that
# {says_nothing .47, contradicts .45} is a claim in real trouble that the
# argmax hides. SciFact says it is not. Over 1,259 pairs the band's share of
# genuinely contradicted claims tracks the corpus base rate almost exactly
# (23.1% against 21.0%), and sweeping its lower bound from 0.15 to 0.70 never
# gets the lift above 1.19x. Sub-threshold contradiction mass carries close to
# no information about whether a claim is actually refuted, so the band only
# cost well-supported claims their stated strength. Removed rather than
# retuned: no bound made it work.

# Mass a "supports" needs before it counts as support - before it keeps its
# citation on the claim through the narrowing below, and before it saves the
# claim from being clamped to "weak". Previously there was no bar here at
# all: `relation == "supports"` was enough, so a bare plurality
# ({supports .34, contradicts .33, says_nothing .33}) - a verdict that is
# closer to "the model cannot tell" than to "the source backs this" - was
# treated exactly like {supports .99}. Requiring an outright majority is the
# weakest bar that still means the answer beat the alternatives combined.
#
# Honest size of the effect: against the old argmax-only rule this changes 5
# of 526 pairs on SciFact (85.7% -> 86.0% precision). Jev's distributions are
# concentrated enough that a bare plurality is rare. It is kept because it is
# the correct reading and costs nothing, not because it bought much - and
# because a backend with a flatter distribution would hit it far more often.
# The bar is a real dial, unlike the removed band: measured precision/recall
# runs 83.7%/91.1% at 0.34, 86.0%/88.2% here, 92.7%/75.2% at 0.90. Raising it
# trades recall for precision by taking citations away from claims, which is
# the direction this module has deliberately been moving away from.
SUPPORT_KEEP_PROBABILITY = 0.50

# The one destructive path, and a narrower claim than "a source disagrees":
# *nothing* a claim cites addresses it at all, which is what `_valid_indices`
# (synthesis.py) already drops one level up when an index points at no real
# article. Higher than the contradicts bar because the same base-rate
# argument above applies here too (measured 92.6% precision at 0.85, but
# again at SciFact's 38.6% NOINFO base rate). The exact precision at 0.95 is
# not measured - this buys specificity for a still-irreversible action, it
# does not certify it.
UNSUPPORTED_REJECT_PROBABILITY = 0.95


@dataclass(frozen=True)
class Thresholds:
    """One backend's bars. Every rule reads these through _over, never a bare
    module constant, because the two backends are not comparable at equal
    numbers - see _LLM_THRESHOLDS.
    """

    contradict_flag: float
    unsupported_reject: float
    support_keep: float
    summary_flag: float


_TYPESAFE_THRESHOLDS = Thresholds(
    contradict_flag=CONTRADICT_FLAG_PROBABILITY,
    unsupported_reject=UNSUPPORTED_REJECT_PROBABILITY,
    support_keep=SUPPORT_KEEP_PROBABILITY,
    summary_flag=0.6,
)

# Measured on the same 1,259 SciFact pairs, through the exact prompt in
# _judge_llm against the heavy model - not transplanted from TypeSafe's bars,
# which would have been the easy mistake and a bad one.
#
# The reason the numbers differ so much is not that a prompted model is worse
# at judging. It is that its self-reported confidence has three settings: it
# answers 0.8 on 6.6% of pairs, 0.9 on 74.4%, and 1.0 on 16.6%. A bar at 0.85
# therefore admits three quarters of everything and filters almost nothing,
# which is why at *equal numbers* this backend looks badly calibrated
# (contradicts 75.2% against TypeSafe's 84.3%). Move the bar to where its
# distribution actually separates and it is the more precise of the two:
#
#   contradicts   @0.95: 92.8% precision, 29.1% recall  (TypeSafe 85.8/77.7)
#   says_nothing  @0.95: 100%   precision, 18.1% recall  (TypeSafe 94.4/41.6)
#   supports      @0.90: 95.4% precision, 77.8% recall  (TypeSafe 93.9/70.3)
#
# So this backend is not a degraded fallback on precision; it is a quieter
# one. It catches roughly a third as many contradictions, and cries wolf less
# when it does. That is the right trade for the instance that has no TypeSafe
# key: fewer findings questioned, and the ones that are, worth reading.
#
# The two contradiction bars are deliberately not the same number, and the
# asymmetry is the whole design principle here: unsupported_reject deletes a
# claim, contradict_flag only annotates one. A deletion buys its 0.95 with
# recall it cannot afford to spend. A flag should be generous, the same
# reasoning that puts summary_flag below both.
#
# contradict_flag is 0.90, not 0.95, because at 0.95 this backend misses
# almost everything: 92.8% precision but 29.1% recall, against 75.2%/87.2% at
# 0.90. Roughly four times the flags to catch three times the real
# contradictions - worth it for a reversible annotation in a tool whose
# entire purpose is catching claims their sources do not support. A missed
# contradiction is the failure this feature exists to prevent; a false one
# costs a reader a second look. 0.90 is also the only bar below 0.95 this
# backend can express at all.
_LLM_THRESHOLDS = Thresholds(
    contradict_flag=0.90,
    unsupported_reject=0.95,
    support_keep=0.90,
    summary_flag=0.90,
)

_THRESHOLDS = {
    TYPESAFE_BACKEND: _TYPESAFE_THRESHOLDS,
    LLM_BACKEND: _LLM_THRESHOLDS,
}


def thresholds_for(backend: str) -> Thresholds:
    return _THRESHOLDS.get(backend, _TYPESAFE_THRESHOLDS)


def _over(verdict: ClaimVerdict, relation: str, bar: str) -> bool:
    """Does *verdict* put enough mass on *relation* to trip its own backend's
    *bar*? The verdict picks the thresholds, not the caller - a verdict can
    never be graded against a backend it did not come from.
    """
    return _mass(verdict, relation) >= getattr(thresholds_for(verdict.backend), bar)


def _mass(verdict: ClaimVerdict, relation: str) -> float:
    """How much probability *verdict* put on *relation*.

    Falls back to the scalar reading - `confidence` if *relation* is the one
    the verdict chose, 0.0 otherwise - whenever no usable distribution is
    attached, which is what the old `relation == X and confidence >= T`
    rules computed. That keeps three cases working: the no-evidence
    short-circuit in verify_claims (which fabricates a verdict without
    calling TypeSafe), a caller constructing verdicts by hand, and any
    future backend that returns a label and a confidence but no distribution
    at all - an LLM judge, say.

    The `verdict.relation in probabilities` guard is deliberate. If TypeSafe
    ever keys `probabilities` by something other than the criteria names,
    a plain `.get(relation, 0.0)` would silently return 0.0 for every
    relation and quietly switch the whole module off; this notices that the
    dict does not even contain the answer it came with, and reverts to the
    reading that cannot fail that way.
    """
    probabilities = verdict.probabilities
    if probabilities and verdict.relation in probabilities:
        return probabilities.get(relation, 0.0)
    return (verdict.confidence or 0.0) if verdict.relation == relation else 0.0


def _snippet(text: str, limit: int = 140) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _claim_contradicted_flag(text: str, language: str) -> str:
    snippet = _snippet(text)
    if language == "it":
        return f'Una delle fonti citate sembra contraddire questa affermazione: "{snippet}"'
    return f'One of the cited sources appears to contradict this claim: "{snippet}"'


def _claim_removed_flag(text: str, language: str) -> str:
    snippet = _snippet(text)
    if language == "it":
        return f'Affermazione rimossa: nessuna delle fonti citate la tratta: "{snippet}"'
    return f'Claim removed - none of its cited sources address it: "{snippet}"'


def apply_verdicts(
    claims: list[Claim],
    verdicts: list[ClaimVerdict],
    summary_language: str = "it",
) -> tuple[list[Claim], list[str]]:
    """Fold verify_claims' verdicts back into *claims*.

    Returns the surviving claims and human-readable flags in
    *summary_language*, the same shape `verify_summary` returns for the prose
    overview. Every verdict that changes a claim produces a flag: the point
    is that no verification result acts on what the reader sees without the
    reader being told, which a `logger.info` on the server does not achieve.

    A claim a cited source confidently contradicts is kept, clamped to
    "weak", and flagged - see CONTRADICT_FLAG_PROBABILITY for why this stops
    short of deleting it, and why its contradicting citation stays on the
    claim rather than being narrowed away (a flag pointing at a source the
    reader can no longer see is not checkable). A claim none of whose cited
    sources address at all is dropped, at the higher
    UNSUPPORTED_REJECT_PROBABILITY, and the removal is reported rather than
    just logged. A claim citing several sources where only some carry real
    support keeps only those - the same claim, minus the citation that
    didn't hold up. Anything left over is kept but not trusted at face
    value: its stated strength is clamped to "weak" when nothing actually
    supports it, the same mechanism _read_claims already applies for a claim
    resting only on non-direct sources.

    `relation == "error"` - the TypeSafe call itself failed - counts as
    neither a support nor a rejection anywhere in this function, so a
    TypeSafe outage degrades verification (nothing gets clamped, flagged or
    dropped on its account) rather than deleting otherwise-good claims.
    """
    by_claim: dict[int, list[ClaimVerdict]] = {}
    for verdict in verdicts:
        by_claim.setdefault(verdict.claim_index, []).append(verdict)

    kept: list[Claim] = []
    flags: list[str] = []
    for c_index, claim in enumerate(claims):
        claim_verdicts = by_claim.get(c_index)
        if not claim_verdicts:
            kept.append(claim)
            continue

        # Verdicts TypeSafe actually returned, as opposed to a failed call -
        # every decision below reads only this list, so a claim whose sources
        # all errored falls through untouched rather than being clamped on
        # the strength of zero real information.
        real = [v for v in claim_verdicts if v.relation != "error"]
        if not real:
            kept.append(claim)
            continue

        if all(_over(v, "says_nothing", "unsupported_reject") for v in real):
            logger.info("Dropping claim none of its cited sources actually address: %s", claim.text[:80])
            flags.append(_claim_removed_flag(claim.text, summary_language))
            continue

        # The single most contradicting source decides, not the average: one
        # source refuting a claim is the finding, however many others stay
        # quiet about it.
        if any(_over(v, "contradicts", "contradict_flag") for v in real):
            logger.info("Flagging claim contradicted by its own cited source: %s", claim.text[:80])
            flags.append(_claim_contradicted_flag(claim.text, summary_language))
            if claim.strength != "weak":
                claim = Claim(
                    text=claim.text, source_indices=claim.source_indices, strength="weak",
                )
            kept.append(claim)
            continue

        # A source with no real verdict (its call errored) is kept rather
        # than dropped from the citation list - unverified is not the same
        # as refuted, and an infra failure should not cost a claim a
        # citation that may well be good.
        supporting = {
            v.source_index for v in real if _over(v, "supports", "support_keep")
        }
        unverified = {v.source_index for v in claim_verdicts if v.relation == "error"}
        keep_indices = supporting | unverified

        indices = claim.source_indices
        if supporting and keep_indices != set(indices):
            indices = [i for i in indices if i in keep_indices]
        strength = claim.strength if supporting else "weak"

        if indices != claim.source_indices or strength != claim.strength:
            claim = Claim(text=claim.text, source_indices=indices, strength=strength)

        kept.append(claim)
    return kept, flags
