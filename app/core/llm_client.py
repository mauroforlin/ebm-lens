"""LLM client - centralised OpenRouter backend.

Every LLM call routes through OpenRouter (https://openrouter.ai) via one
shared OpenAI client. Model selection, rate limiting, retries,
cost tracking and JSON repair are all handled here.

Two entry points: generate_json() for one-shot structured output, and
generate_with_tools() for function-calling loops where the model grounds
its answer in real lookups before returning. Both accept an optional
JobStats; when provided, each call is recorded (tokens, real cost, latency)
and every retry is counted.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Settings
from app.core.ratelimiter import reserve as reserve_tokens

if TYPE_CHECKING:
    from app.core.job_stats import JobStats

logger = logging.getLogger(__name__)


# ── Shared OpenRouter client singleton ────────────────────────

_client = None
_client_lock = threading.Lock()


def get_openrouter_client(settings: Settings):
    """Lazy singleton - one OpenAI client pointed at OpenRouter.

    Shared by both chat completions and embeddings.
    """
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                import openai
                _client = openai.OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=settings.openrouter_api_key,
                    timeout=60.0,
                )
    return _client


# ── Cost extraction (shared) ─────────────────────────────────

def extract_cost(usage: Any) -> float:
    """Extract USD cost from an OpenRouter response usage object.

    OpenRouter returns usage.cost; the SDK may not surface it as an attribute,
    so fall back to model_extra.
    """
    if usage is None:
        return 0.0
    if hasattr(usage, "cost") and usage.cost is not None:
        return float(usage.cost)
    if hasattr(usage, "model_extra") and isinstance(usage.model_extra, dict):
        return float(usage.model_extra.get("cost", 0.0) or 0.0)
    return 0.0


# ── Model routing policy (centralised) ────────────────────────

# Maps a call ``purpose`` to the :class:`Settings` field that selects its
# model. Any purpose not listed here falls back to the default ``llm_model``.
_PURPOSE_MODEL_FIELD: dict[str, str] = {
    # Planning / query generation.
    "related_articles_brief": "llm_planner_model",
    "related_articles_reformulate": "llm_planner_model",
    # Quality-critical, long-form output.
    "related_articles_summarize": "llm_heavy_model",
    "related_articles_synthesis": "llm_heavy_model",
    "related_articles_rerank": "llm_heavy_model",
}


def resolve_model(settings: Settings, purpose: str, model_override: str | None) -> str:
    """Pick the model for a call.

    Precedence: explicit ``model_override`` → central purpose mapping →
    default ``llm_model``.
    """
    if model_override:
        return model_override
    field = _PURPOSE_MODEL_FIELD.get(purpose or "")
    return getattr(settings, field) if field else settings.llm_model


# ── LLM usage metadata ───────────────────────────────────────

@dataclass
class LLMUsage:
    """Token counts + cost from a single LLM call."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    cost_usd: float = 0.0


# ── Retry semantics ───────────────────────────────────────────

def _is_retryable(exc: BaseException) -> bool:
    import openai
    if isinstance(exc, openai.APIStatusError):
        # 429 here is OpenRouter's shared upstream pool asking to be asked
        # again shortly, not this account's own quota - its own error message
        # says so. A 5xx is the same kind of transient failure one layer down.
        return exc.status_code == 429 or 500 <= exc.status_code < 600
    return isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError))


def on_retry(retry_state: Any) -> None:
    """tenacity ``before_sleep`` hook - count the retry in the job's stats."""
    stats = retry_state.kwargs.get("job_stats")
    if stats is not None:
        stats.add_retry()


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=15),
    stop=stop_after_attempt(3),
    retry=retry_if_exception(_is_retryable),
    before_sleep=on_retry,
)
def _chat_completion(
    settings: Settings, *, model: str, messages: list[dict],
    temperature: float, tools: list[dict] | None = None,
    tool_choice: dict | str | None = None,
    job_stats: JobStats | None = None,
) -> tuple[Any, LLMUsage]:
    """Low-level chat completion via OpenRouter.

    Returns the raw ``ChatCompletionMessage`` (which carries either
    ``content`` or ``tool_calls``) plus token/cost usage. When *tools* is
    provided the request uses function calling; otherwise it requests a
    JSON object (the plain structured-output path).
    """
    client = get_openrouter_client(settings)
    est = sum(max(1, len(m.get("content") or "") // 4) for m in messages)

    kwargs: dict[str, Any] = {
        "model": model, "messages": messages, "temperature": temperature,
    }
    if tools:
        kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
    else:
        kwargs["response_format"] = {"type": "json_object"}

    with reserve_tokens("openrouter", settings.openrouter_rpm_limit,
                        settings.openrouter_tpm_limit, est):
        response = client.chat.completions.create(**kwargs)

    u = getattr(response, "usage", None)
    usage = (
        LLMUsage(
            input_tokens=getattr(u, "prompt_tokens", 0) or 0,
            output_tokens=getattr(u, "completion_tokens", 0) or 0,
            total_tokens=getattr(u, "total_tokens", 0) or 0,
            model=model,
            cost_usd=extract_cost(u),
        )
        if u else LLMUsage(model=model)
    )

    return response.choices[0].message, usage


# ── JSON repair ──────────────────────────────────────────────

def parse_json(raw: str) -> Any:
    """Parse JSON with repair fallback."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("LLM returned invalid JSON, attempting repair: %s…", raw[:200])
        start = min((raw.find(c) for c in "[{" if raw.find(c) != -1), default=0)
        end = max((raw.rfind(c) for c in "]}" if raw.rfind(c) != -1), default=len(raw) - 1)
        return json.loads(raw[start : end + 1])


# ── Public API ────────────────────────────────────────────────

def generate_json(
    *, settings: Settings, prompt: str, system_instruction: str = "",
    temperature: float = 0.1, purpose: str = "",
    job_stats: JobStats | None = None, model_override: str | None = None,
) -> Any:
    """Generate a JSON response via OpenRouter.

    The model is chosen by :func:`resolve_model` from *purpose*; pass
    *model_override* to bypass the central policy for a single call.
    """
    model = resolve_model(settings, purpose, model_override)
    messages: list[dict] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    t0 = time.monotonic()
    msg, usage = _chat_completion(
        settings, model=model, messages=messages,
        temperature=temperature, job_stats=job_stats,
    )

    if job_stats:
        job_stats.record_llm_call(
            purpose or "unknown",
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            model=usage.model, cost_usd=usage.cost_usd,
            latency_ms=(time.monotonic() - t0) * 1000,
        )
    return parse_json((msg.content or "").strip())


# ── Tool calling (function calling) ───────────────────────────

@dataclass
class ToolInvocation:
    """One tool call the model issued during a tool loop."""
    name: str
    arguments: dict


# A tool result is context the model pays for on every subsequent round of the
# loop, so a provider that answers with a hundred records would crowd out the
# conversation it was meant to inform. Handlers are expected to summarise;
# this is the backstop.
_MAX_TOOL_RESULT_CHARS = 6000

# Tool calls run concurrently within a round. The handlers are HTTP lookups
# against different services, so a model that asks for three at once should
# wait for the slowest, not for their sum.
_TOOL_WORKERS = 4


def _safe_loads_args(raw: str) -> dict:
    """Parse a tool's JSON arguments string defensively."""
    try:
        data = json.loads(raw or "{}")
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _assistant_message_dict(msg: Any) -> dict:
    """Re-serialise an assistant message (with optional tool_calls) for the API."""
    d: dict = {"role": "assistant", "content": getattr(msg, "content", None) or None}
    tcs = getattr(msg, "tool_calls", None) or []
    if tcs:
        d["tool_calls"] = [
            {
                "id": getattr(tc, "id", None),
                "type": "function",
                "function": {
                    "name": getattr(getattr(tc, "function", None), "name", ""),
                    "arguments": getattr(getattr(tc, "function", None), "arguments", "{}"),
                },
            }
            for tc in tcs
        ]
    return d


def _run_tool(
    name: str,
    args: dict,
    tool_handlers: dict[str, Any],
    memo: dict[tuple[str, str], str],
) -> str:
    """Dispatch one tool call to its handler and return a JSON string.

    A handler that raises answers with its error rather than aborting the
    loop: the model can read the failure and route around it, which is the
    whole point of giving it tools instead of a fixed pipeline.

    Repeated identical calls are served from *memo*. Models that get a
    disappointing result sometimes re-issue the same lookup verbatim, and
    paying for that twice buys nothing.
    """
    signature = (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
    if signature in memo:
        return memo[signature]

    handler = tool_handlers.get(name)
    if handler is None:
        return json.dumps({"error": f"unknown tool: {name}"})

    try:
        result = handler(args)
    except Exception as exc:
        logger.warning("Tool handler %s failed: %s", name, exc)
        return json.dumps({"error": str(exc)})

    if not isinstance(result, str):
        result = json.dumps(result, ensure_ascii=False)
    if len(result) > _MAX_TOOL_RESULT_CHARS:
        result = result[:_MAX_TOOL_RESULT_CHARS] + " …[truncated]"

    memo[signature] = result
    return result


def generate_with_tools(
    *,
    settings: Settings,
    prompt: str,
    tools: list[dict],
    tool_handlers: dict[str, Any],
    system_instruction: str = "",
    temperature: float = 0.1,
    purpose: str = "",
    job_stats: JobStats | None = None,
    model_override: str | None = None,
    max_tool_rounds: int = 4,
    final_tool: str | None = None,
) -> tuple[str, list[ToolInvocation]]:
    """Run an LLM conversation loop where the model may call tools.

    The model can call any tool in *tools*; each call is dispatched to the
    matching *tool_handlers* entry (a callable taking the arguments dict and
    returning a JSON string or a JSON-serialisable object). Tool results are
    fed back into the conversation until the model emits a final answer (or
    ``max_tool_rounds`` is reached). Returns ``(final_content, invocations)``
    where *invocations* records every tool call in order, so callers can
    extract structured tool arguments directly instead of re-parsing prose.

    Several tool calls issued in one round are executed concurrently: the
    handlers are independent lookups against different services, so the round
    costs the slowest of them rather than their sum.

    *final_tool* names the tool that terminates the loop by submitting the
    answer. When the round budget is about to run out it is forced through
    ``tool_choice``, which turns "the model explored for too long and returned
    nothing usable" into "the model submits what it has". Without it, a loop
    that hits its budget mid-exploration yields no result at all.
    """
    model = resolve_model(settings, purpose, model_override)
    messages: list[dict] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    invocations: list[ToolInvocation] = []
    memo: dict[tuple[str, str], str] = {}
    final_content = ""
    last_round = max_tool_rounds

    def _call(force: bool) -> Any:
        t0 = time.monotonic()
        msg, usage = _chat_completion(
            settings, model=model, messages=messages,
            temperature=temperature, tools=tools, job_stats=job_stats,
            tool_choice=(
                {"type": "function", "function": {"name": final_tool}}
                if force else None
            ),
        )
        if job_stats:
            job_stats.record_llm_call(
                purpose or "unknown",
                input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                model=usage.model, cost_usd=usage.cost_usd,
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        return msg

    for round_index in range(max_tool_rounds + 1):
        pending_final = final_tool is not None and not any(
            inv.name == final_tool for inv in invocations
        )
        force_submit = pending_final and round_index == last_round

        msg = _call(force_submit)

        tool_calls = list(getattr(msg, "tool_calls", None) or [])
        if not tool_calls:
            if pending_final and not force_submit:
                # The model answered in prose instead of calling the required terminal tool
                msg = _call(force=True)
                tool_calls = list(getattr(msg, "tool_calls", None) or [])
            if not tool_calls:
                final_content = (getattr(msg, "content", None) or "").strip()
                break

        messages.append(_assistant_message_dict(msg))

        pending: list[tuple[Any, str, dict]] = []
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", "") if fn else ""
            args = _safe_loads_args(getattr(fn, "arguments", "{}") if fn else "{}")
            invocations.append(ToolInvocation(name=name, arguments=args))
            if job_stats:
                job_stats.record_tool_call(name or "unknown")
            pending.append((tc, name, args))

        if len(pending) == 1:
            results = [_run_tool(pending[0][1], pending[0][2], tool_handlers, memo)]
        else:
            with ThreadPoolExecutor(max_workers=min(_TOOL_WORKERS, len(pending))) as pool:
                results = list(pool.map(
                    lambda item: _run_tool(item[1], item[2], tool_handlers, memo),
                    pending,
                ))

        for (tc, _, _), result in zip(pending, results, strict=True):
            messages.append({
                "role": "tool",
                "tool_call_id": getattr(tc, "id", None),
                "content": result,
            })

        # The terminal tool has been called: the answer is in its arguments,
        # and another round would only ask the model to narrate it.
        if final_tool is not None and any(inv.name == final_tool for inv in invocations):
            break

    return final_content, invocations
