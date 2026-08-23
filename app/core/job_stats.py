"""Thread-safe per-job statistics tracker.

Collects LLM token usage, provider call counts, and timing data for cost
monitoring and performance analysis. Every run returns its own accounting in
job_stats, so the cost of a query is a fact you can read rather than estimate.

All LLM/embedding costs are the real cost returned by OpenRouter

PS: The evidence providers are free public APIs and contribute nothing to the total.

Usage::

    stats = JobStats()
    stats.record_llm_call("summarize", input_tokens=1200, output_tokens=300,
                          model="google/gemini-2.5-flash-lite", cost_usd=0.00045)
    stats.record_provider_call("pubmed", cached=False)
    ...
    summary = stats.to_dict()
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class _LLMCallRecord:
    """A single LLM call record."""
    purpose: str          # e.g. "related_articles_domain", "related_articles_summarize", ...
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float = 0.0


class JobStats:
    """Thread-safe accumulator for per-job statistics.

    Designed to be created once per pipeline run, passed through all stages,
    and dumped at the end via to_dict().
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # LLM usage
        self._llm_calls: list[_LLMCallRecord] = []
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_llm_cost_usd: float = 0.0

        # Embedding usage
        self._embedding_calls: int = 0
        self._embedding_tokens: int = 0
        self._embedding_cost_usd: float = 0.0
        self._embedding_by_model: dict[str, dict[str, float]] = {}

        # Provider calls
        self._provider_calls: dict[str, int] = {}        # provider_id → count
        self._provider_cached: dict[str, int] = {}       # provider_id → cache-hit count
        self._provider_errors: dict[str, int] = {}       # provider_id → error count

        # Pipeline stages timing (name → elapsed ms)
        self._stage_timings: dict[str, float] = {}

        # Tool calls issued by the model inside a tool-calling loop
        self._tool_calls: dict[str, int] = {}      # tool name → count

        # Misc counters
        self._evidence_total: int = 0
        self._evidence_after_relevance: int = 0
        self._retries: int = 0

    # ── LLM tracking ─────────────────────────────────────────

    def record_llm_call(
        self,
        purpose: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = "",
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
    ) -> None:
        """Record a single LLM call.

        *cost_usd* should be the real cost returned by OpenRouter

        When unavailable (e.g. the provider doesn't return it), pass 0.0.
        """
        record = _LLMCallRecord(
            purpose=purpose,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )
        with self._lock:
            self._llm_calls.append(record)
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens
            self._total_llm_cost_usd += cost_usd

    # ── Embedding tracking ─────────────────────────────────────

    def record_embedding_call(
        self,
        model: str = "",
        tokens: int = 0,
        num_texts: int = 1,
        cost_usd: float = 0.0,
    ) -> None:
        """Record an embedding API call.

        *cost_usd* should be the real cost returned by OpenRouter.
        """
        with self._lock:
            self._embedding_calls += num_texts
            self._embedding_tokens += tokens
            self._embedding_cost_usd += cost_usd
            key = model or "unknown"
            entry = self._embedding_by_model.setdefault(
                key, {"calls": 0, "tokens": 0, "cost_usd": 0.0}
            )
            entry["calls"] += num_texts
            entry["tokens"] += tokens
            entry["cost_usd"] += cost_usd

    # ── Provider tracking (sources) ───────────────────────────

    def record_provider_call(
        self,
        provider_id: str,
        *,
        cached: bool = False,
        error: bool = False,
    ) -> None:
        with self._lock:
            self._provider_calls[provider_id] = self._provider_calls.get(provider_id, 0) + 1
            if cached:
                self._provider_cached[provider_id] = self._provider_cached.get(provider_id, 0) + 1
            if error:
                self._provider_errors[provider_id] = self._provider_errors.get(provider_id, 0) + 1

    # ── Stage timing ──────────────────────────────────────────

    def record_tool_call(self, name: str) -> None:
        """Record one tool invocation the model chose to make."""
        with self._lock:
            self._tool_calls[name] = self._tool_calls.get(name, 0) + 1

    def record_stage(self, name: str, elapsed_ms: float) -> None:
        with self._lock:
            self._stage_timings[name] = (
                self._stage_timings.get(name, 0.0) + elapsed_ms
            )

    # ── Misc counters ─────────────────────────────────────────

    def add_evidence(self, total: int, after_filter: int) -> None:
        with self._lock:
            self._evidence_total += total
            self._evidence_after_relevance += after_filter

    def add_retry(self) -> None:
        with self._lock:
            self._retries += 1

    # ── Export ─────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Produce a JSON-serialisable summary of all collected stats."""
        with self._lock:
            # Aggregate LLM calls by purpose
            calls_by_purpose: dict[str, dict[str, Any]] = {}
            for rec in self._llm_calls:
                entry = calls_by_purpose.setdefault(rec.purpose, {
                    "count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "total_latency_ms": 0.0,
                })
                entry["count"] += 1
                entry["input_tokens"] += rec.input_tokens
                entry["output_tokens"] += rec.output_tokens
                entry["cost_usd"] += rec.cost_usd
                entry["total_latency_ms"] += rec.latency_ms

            # Round costs
            for v in calls_by_purpose.values():
                v["cost_usd"] = round(v["cost_usd"], 8)
                v["total_latency_ms"] = round(v["total_latency_ms"], 1)

            # Grand total: LLM + embeddings
            grand_total_usd = self._total_llm_cost_usd + self._embedding_cost_usd

            return {
                "total_cost_usd": round(grand_total_usd, 6),
                "llm": {
                    "total_calls": len(self._llm_calls),
                    "total_input_tokens": self._total_input_tokens,
                    "total_output_tokens": self._total_output_tokens,
                    "total_tokens": self._total_input_tokens + self._total_output_tokens,
                    "total_cost_usd": round(self._total_llm_cost_usd, 8),
                    "by_purpose": calls_by_purpose,
                },
                "embeddings": {
                    "total_calls": self._embedding_calls,
                    "total_tokens": self._embedding_tokens,
                    "total_cost_usd": round(self._embedding_cost_usd, 8),
                    "by_model": {
                        k: {
                            "calls": v["calls"],
                            "tokens": v["tokens"],
                            "cost_usd": round(v["cost_usd"], 8),
                        }
                        for k, v in self._embedding_by_model.items()
                    },
                },
                "tools": {
                    "calls": dict(self._tool_calls),
                    "total_calls": sum(self._tool_calls.values()),
                },
                "providers": {
                    "calls": dict(self._provider_calls),
                    "cache_hits": dict(self._provider_cached),
                    "errors": dict(self._provider_errors),
                    "total_calls": sum(self._provider_calls.values()),
                    "total_cache_hits": sum(self._provider_cached.values()),
                },
                "pipeline": {
                    "evidence_total": self._evidence_total,
                    "evidence_after_relevance": self._evidence_after_relevance,
                    "retries": self._retries,
                },
                "stage_timings_ms": {
                    k: round(v, 1) for k, v in self._stage_timings.items()
                },
            }
