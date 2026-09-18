"""TypeSafe client - typed judgments via Jev (System One models).

Mirrors llm_client.py's shape (lazy singleton client, purpose-based job_stats
accounting) but calls a structurally different kind of model: Jev takes a
*state* plus one or more typed *questions* (Choice/Noul/Score) and returns
typed answers directly - no prose to generate, no JSON to parse or repair.
A Choice answer's `confidence` is computed by TypeSafe from the model's own
probability distribution over the given criteria, not self-reported by the
model in the same completion the way `judge_directions`' `_STANCE_SYSTEM`
prompt currently asks for a "reasoning" field - see docs.typesafe.ai/confidence.

app/pipeline/claim_verification.py is the one caller, itself called from
app/pipeline/orchestrator.py's stage 7 only when `settings.typesafe_key` is
set - `typesafe_key` is optional in Settings for exactly that reason: nothing
here is required to run the tool (see app/config.py).

No rate limiter, unlike llm_client.py's OpenRouter calls (see reserve_tokens
in ratelimiter.py) - TypeSafe's own published limits aren't known. The
pipeline caller now runs unattended, bounded only by claim_verification.py's
own `_VERIFY_MAX_WORKERS`; add a real limiter here, sized to TypeSafe's
actual limits, if that concurrency ever needs to grow.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.config import Settings

if TYPE_CHECKING:
    from app.core.job_stats import JobStats

# Pinned, not jev-latest. eval/typesafe_stance_eval.py grades this exact
# call and claim_verification.py's thresholds are argued from those numbers,
# so a TypeSafe-side model upgrade under `jev-latest` would silently
# invalidate the evidence for every threshold in that module while the code
# kept reporting the old figures. TypeSafe's own citation_check cookbook
# pins for the same reason. The failure mode of a pin - the version is
# eventually retired and calls start erroring - is the safe one here:
# verify_claims turns an exception into an "error" verdict, which
# apply_verdicts treats as neither support nor rejection, so verification
# degrades to a no-op instead of degrading quietly into different answers.
# Re-run the eval before moving this.
DEFAULT_MODEL = "jev-1.13.0"

_client = None
_client_lock = threading.Lock()


def get_typesafe_client(settings: Settings):
    """Lazy singleton - one TypeSafeClient for the process.

    Raises TypeSafeError (from the SDK) if *settings.typesafe_key* is unset -
    callers that want a softer failure should check it first, the way
    claim_verification.py does before running any batch.
    """
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                from typesafe_sdk import TypeSafeClient
                _client = TypeSafeClient(api_key=settings.typesafe_key)
    return _client


@dataclass
class ChoiceAnswer:
    """One Choice question's answer, trimmed to what callers need.

    Mirrors the SDK's own `ChoiceAnswer` shape (choice/probabilities/
    confidence) rather than returning it directly, so a caller's type hints
    don't reach into `typesafe_sdk` internals for a value this thin.
    """

    choice: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)


def ask_choice(
    *,
    settings: Settings,
    state: Any,
    instructions: str,
    criteria: dict[str, str | None],
    purpose: str = "",
    job_stats: JobStats | None = None,
    model: str = DEFAULT_MODEL,
) -> ChoiceAnswer:
    """Ask one Choice question against *state* and return its answer.

    One question per call, unlike TypeSafe's own "ask several questions
    together" guidance (see docs.typesafe.ai/primitives#ask-multiple-
    questions-together) - claim_verification.py's judgment is exactly one
    Choice per (claim, source) pair, with nothing else to batch into the same
    request. A second question type used the same way belongs here as its
    own ask_noul/ask_score, not folded into this one function's signature.
    """
    from typesafe_sdk import Choice

    client = get_typesafe_client(settings)
    t0 = time.monotonic()
    response = client.system_one(
        state=state,
        questions={"answer": Choice(instructions=instructions, criteria=criteria)},
        model=model,
    )
    if job_stats:
        usage = getattr(response, "usage", None)
        job_stats.record_llm_call(
            purpose or "typesafe_choice",
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            model=f"typesafe/{model}",
            # TypeSafe's usage doesn't carry a cost figure the way
            # OpenRouter's does (see llm_client.py's extract_cost) - job_stats'
            # total_cost_usd is therefore a floor, not the real total, while
            # any TypeSafe purpose has calls in by_purpose.
            cost_usd=0.0,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    answer = response.answers["answer"]
    return ChoiceAnswer(
        choice=answer.choice,
        confidence=answer.confidence,
        probabilities=dict(answer.probabilities),
    )


@dataclass
class ScoreAnswer:
    """One Score question's answer.

    `score` is the probability-weighted mean over the *criteria* list's
    indices (0-based) - e.g. a 5-point legend answered mostly at index 0 with
    a little mass on index 1 comes back as ~0.2, not a bucketed integer. Same
    computed (not self-reported) confidence as Choice - see ChoiceAnswer.
    """

    score: float
    legend: dict[int, str]
    confidence: float
    probabilities: dict[int, float] = field(default_factory=dict)


def ask_score(
    *,
    settings: Settings,
    state: Any,
    instructions: str,
    criteria: list[str],
    purpose: str = "",
    job_stats: JobStats | None = None,
    model: str = DEFAULT_MODEL,
) -> ScoreAnswer:
    """Ask one Score question against *state* - an ordinal judgment (*criteria*
    given low-to-high) rather than Choice's unordered categories.
    """
    from typesafe_sdk import Score

    client = get_typesafe_client(settings)
    t0 = time.monotonic()
    response = client.system_one(
        state=state,
        questions={"answer": Score(instructions=instructions, criteria=criteria)},
        model=model,
    )
    if job_stats:
        usage = getattr(response, "usage", None)
        job_stats.record_llm_call(
            purpose or "typesafe_score",
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            model=f"typesafe/{model}",
            cost_usd=0.0,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    answer = response.answers["answer"]
    return ScoreAnswer(
        score=answer.score,
        legend=dict(answer.legend),
        confidence=answer.confidence,
        probabilities=dict(answer.probabilities),
    )


@dataclass
class NoulAnswer:
    """One Noul question's answer - a bare 0-1 truth value, nothing else.

    Unlike Choice/Score, the SDK's own `NoulAnswer` has no `confidence` field
    at all (verified directly against the installed SDK's response types,
    not just its docs) - there is no separate calibrated-confidence number to
    read here the way there is for the other two primitives.
    """

    noul: float


def ask_noul(
    *,
    settings: Settings,
    state: Any,
    instructions: str,
    criteria: dict[str, str | None] | None = None,
    purpose: str = "",
    job_stats: JobStats | None = None,
    model: str = DEFAULT_MODEL,
) -> NoulAnswer:
    """Ask one Noul question against *state* - a single yes/no proposition,
    answered as a truth value rather than picked from criteria. *criteria*
    optionally describes the true/false outcomes (`{"true": ..., "false":
    ...}`); pass None to leave them undescribed.
    """
    from typesafe_sdk import Noul

    client = get_typesafe_client(settings)
    t0 = time.monotonic()
    response = client.system_one(
        state=state,
        questions={"answer": Noul(instructions=instructions, criteria=criteria)},
        model=model,
    )
    if job_stats:
        usage = getattr(response, "usage", None)
        job_stats.record_llm_call(
            purpose or "typesafe_noul",
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            model=f"typesafe/{model}",
            cost_usd=0.0,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    answer = response.answers["answer"]
    return NoulAnswer(noul=answer.noul)
