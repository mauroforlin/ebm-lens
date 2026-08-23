"""Embedding helpers with retry semantics (via OpenRouter)."""
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import Settings
from app.core.llm_client import extract_cost, get_openrouter_client, on_retry
from app.core.ratelimiter import reserve as reserve_tokens

if TYPE_CHECKING:
    from app.core.job_stats import JobStats

# Models accepting the `dimensions` parameter
_DIMENSIONS_CAPABLE = "text-embedding-3"


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=20),
    stop=stop_after_attempt(5),
    before_sleep=on_retry,
)
def _embed_batch(
    client, model: str, batch: list[str], settings: Settings,
    job_stats: JobStats | None = None, dimensions: int | None = None,
) -> list[list[float]]:
    est = sum(len(t) // 4 + 1 for t in batch)
    kwargs: dict = {"model": model, "input": batch}
    if dimensions:
        kwargs["dimensions"] = dimensions
    with reserve_tokens("openrouter", settings.openrouter_rpm_limit,
                        settings.openrouter_tpm_limit, est):
        resp = client.embeddings.create(**kwargs)
    if job_stats:
        u = getattr(resp, "usage", None)
        tokens = getattr(u, "total_tokens", est) if u else est
        job_stats.record_embedding_call(
            model=model, tokens=tokens, num_texts=len(batch),
            cost_usd=extract_cost(u),
        )
    return [r.embedding for r in resp.data]


def embed_texts(
    texts: Iterable[str], settings: Settings, batch_size: int = 50,
    job_stats: JobStats | None = None, model_override: str | None = None,
    dimensions: int | None = None,
) -> list[list[float]]:
    """Embed texts in batches returning vectors.

    *model_override* lets a caller pick a different embedding model than the
    configured default (e.g. the stronger pro model for related-articles
    discovery) without changing global configuration.

    *dimensions* truncates the output on models that support it.
    When the default model is used, it is forced to the configured
    embedding_dimension.
    """
    client = get_openrouter_client(settings)
    model = model_override or settings.embedding_model

    if dimensions is None and model_override is None:
        dimensions = settings.embedding_dimension
    base = model.split("/", 1)[-1] if "/" in model else model
    if not base.startswith(_DIMENSIONS_CAPABLE):
        dimensions = None

    results: list[list[float]] = []
    batch: list[str] = []
    for text in texts:
        batch.append(text)
        if len(batch) >= batch_size:
            results.extend(_embed_batch(
                client, model, batch, settings,
                job_stats=job_stats, dimensions=dimensions,
            ))
            batch = []
    if batch:
        results.extend(_embed_batch(
            client, model, batch, settings,
            job_stats=job_stats, dimensions=dimensions,
        ))
    return results
