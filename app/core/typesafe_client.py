"""TypeSafe client - typed judgments via Jev (System One models).

Mirrors llm_client.py's shape (lazy singleton client, purpose-based job_stats
accounting) but calls a structurally different kind of model: Jev takes a
*state* plus one or more typed *questions* (Choice/Noul/Score) and returns
typed answers directly - no prose to generate, no JSON to parse or repair.
A Choice answer's `confidence` is computed by TypeSafe from the model's own
probability distribution over the given criteria, not self-reported by the
model in the same completion the way `judge_directions`' `_STANCE_SYSTEM`
prompt currently asks for a "reasoning" field - see docs.typesafe.ai/confidence.

No pipeline stage calls this - app/pipeline/claim_verification.py is the one
caller, itself a standalone building block with no caller of its own.
`typesafe_key` is optional in Settings for exactly that reason: nothing here
is required to run the tool (see app/config.py).

No rate limiter, unlike llm_client.py's OpenRouter calls (see reserve_tokens
in ratelimiter.py) - TypeSafe's own published limits aren't known, and the
one caller today runs a handful of calls by hand, not a production fan-out.
Add one here, sized to TypeSafe's real limits, before any pipeline stage
that runs unattended starts calling this.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.config import Settings

if TYPE_CHECKING:
    from app.core.job_stats import JobStats

# jev-latest always resolves to TypeSafe's current flagship model - fine for
# a single-caller integration where nothing depends on reproducing an exact
# verdict across a TypeSafe-side model upgrade; a caller that does needs that
# stability should pin an explicit version instead, as the citation_check
# cookbook does with jev-1.12.
DEFAULT_MODEL = "jev-latest"

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
