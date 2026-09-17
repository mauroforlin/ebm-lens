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
    claims = apply_verdicts(claims, verdicts)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import Settings
from app.core.job_stats import JobStats
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


@dataclass
class ClaimVerdict:
    """One (claim, cited source) pair's verification result.

    *relation* is "error" when the TypeSafe call itself failed (network,
    auth, ...) - distinct from "says_nothing", which is a real answer the
    model gave, not a failure to get one.
    """

    claim_index: int
    source_index: int
    relation: str
    confidence: float | None
    probabilities: dict[str, float]


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

    def _check(pair: tuple[int, int]) -> ClaimVerdict:
        c_index, s_index = pair
        claim = claims[c_index]
        evidence = _evidence_block(articles[s_index])
        if not evidence.strip():
            # Nothing was ever shown to the synthesiser for this source - no
            # call needed, and no basis for a "supports" or "contradicts"
            # verdict either.
            return ClaimVerdict(c_index, s_index, "says_nothing", None, {})
        try:
            answer = ask_choice(
                settings=settings,
                state={"claim": claim.text, "source_evidence": evidence},
                instructions=_INSTRUCTIONS,
                criteria=_RELATION_CRITERIA,
                purpose="claim_verification",
                job_stats=job_stats,
            )
        except Exception as exc:
            logger.warning(
                "Claim verification failed for claim %d / source %d: %s",
                c_index, s_index, exc,
            )
            return ClaimVerdict(c_index, s_index, "error", None, {})
        return ClaimVerdict(c_index, s_index, answer.choice, answer.confidence, answer.probabilities)

    return run_parallel(_check, pairs, _VERIFY_MAX_WORKERS)


# Confidence a verdict needs before it acts on its own, rather than just
# nudging strength - mirrors TypeSafe's own citation_check cookbook
# (AUTO_ACCEPT = 0.8), set a little higher because the action here is
# dropping a claim outright rather than flagging it for review. A verdict
# below this (the compound-claim case this was tuned against returned 0.46)
# is exactly the kind of genuinely ambiguous case apply_verdicts is meant to
# demote, not act on as if it were a clean answer.
REJECT_CONFIDENCE = 0.85


def apply_verdicts(claims: list[Claim], verdicts: list[ClaimVerdict]) -> list[Claim]:
    """Fold verify_claims' verdicts back into *claims*.

    A claim a cited source actively contradicts, confidently, is dropped -
    worse to show than one with no citation at all. A claim none of whose
    cited sources actually address (again, confidently) is dropped the same
    way _valid_indices already drops a claim citing nothing real, just
    caught one level deeper. A claim citing several sources where only some
    verify as "supports" keeps only those - the same claim, minus the
    citation that didn't hold up. Anything left over (an ambiguous verdict
    below REJECT_CONFIDENCE, or nothing verified as an outright "supports")
    is kept but not trusted at face value: its stated strength is clamped to
    "weak", the same mechanism _read_claims already applies for a claim
    resting only on non-direct sources.

    `relation == "error"` - the TypeSafe call itself failed - counts as
    neither a support nor a rejection anywhere in this function, so a
    TypeSafe outage degrades verification (nothing gets clamped or dropped
    on its account) rather than deleting otherwise-good claims.
    """
    by_claim: dict[int, list[ClaimVerdict]] = {}
    for verdict in verdicts:
        by_claim.setdefault(verdict.claim_index, []).append(verdict)

    kept: list[Claim] = []
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

        if any(
            v.relation == "contradicts" and (v.confidence or 0.0) >= REJECT_CONFIDENCE
            for v in real
        ):
            logger.info("Dropping claim contradicted by its own cited source: %s", claim.text[:80])
            continue

        if all(
            v.relation == "says_nothing" and (v.confidence or 0.0) >= REJECT_CONFIDENCE
            for v in real
        ):
            logger.info("Dropping claim none of its cited sources actually address: %s", claim.text[:80])
            continue

        # A source with no real verdict (its call errored) is kept rather
        # than dropped from the citation list - unverified is not the same
        # as refuted, and an infra failure should not cost a claim a
        # citation that may well be good.
        supporting = {v.source_index for v in real if v.relation == "supports"}
        unverified = {v.source_index for v in claim_verdicts if v.relation == "error"}
        keep_indices = supporting | unverified
        if supporting and keep_indices != set(claim.source_indices):
            claim = Claim(
                text=claim.text,
                source_indices=[i for i in claim.source_indices if i in keep_indices],
                strength=claim.strength,
            )
        elif not supporting and claim.strength != "weak":
            claim = Claim(text=claim.text, source_indices=claim.source_indices, strength="weak")

        kept.append(claim)
    return kept
