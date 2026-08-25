"""Application configuration via Pydantic settings.

Every value is read from a ``.env`` file or the real environment (see
``.env.example``). Only ``OPENROUTER_API_KEY`` is mandatory: the tool is
meant to be cloned and run with one key and no infrastructure.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = Field("EBM Lens", alias="APP_NAME")

    # Optional API-key auth. If unset, the API is open (no auth required) -
    # fine for local/dev use, set it before exposing this publicly.
    api_key: str | None = Field(None, alias="API_KEY")

    # Origins allowed to call the API from a browser, comma-separated.
    # Defaults to none: the bundled UI is same-origin, so nothing needs
    # allowing for it to work. Set it only to serve a frontend from another
    # host. Held as a string rather than a list because pydantic-settings
    # JSON-decodes list-typed env vars, and a blank value - exactly what
    # .env.example ships - is not valid JSON and would stop the app booting.
    cors_allow_origins_raw: str = Field("", alias="CORS_ALLOW_ORIGINS")

    # ── LLM backend (OpenRouter - single provider for all models) ──
    openrouter_api_key: str = Field(..., alias="OPENROUTER_API_KEY")
    llm_model: str = Field("google/gemini-2.5-flash-lite", alias="LLM_MODEL")
    llm_heavy_model: str = Field("google/gemini-2.5-flash", alias="LLM_HEAVY_MODEL")
    llm_planner_model: str = Field("google/gemini-3-flash-preview", alias="LLM_PLANNER_MODEL")
    openrouter_rpm_limit: int = Field(200, alias="OPENROUTER_RPM_LIMIT")
    openrouter_tpm_limit: int = Field(200000, alias="OPENROUTER_TPM_LIMIT")

    # ── Embeddings (via OpenRouter) ──
    embedding_model: str = Field("openai/text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(1536, alias="EMBEDDING_DIM")
    # A stronger embedding model for the discovery loop's relevance gate and
    # the rerank step - higher quality seed/shortlist selection at negligible
    # extra cost on the small candidate pools involved.
    embedding_model_rerank: str = Field("openai/text-embedding-3-large", alias="EMBEDDING_MODEL_RERANK")

    # ── Outbound HTTP identity ──
    # Sent as the mailto: in the User-Agent of every external request. NCBI,
    # Crossref and OpenAlex ask for a real contact address and route
    # identified clients through a faster "polite pool"; anonymous traffic
    # gets throttled first when a service is under load. Set it to your own
    # address if you run this against the public APIs regularly.
    contact_email: str = Field("anonymous@example.com", alias="CONTACT_EMAIL")

    # ── NCBI (PubMed) ──
    # Optional. Without it PubMed calls are limited to 3 req/s (fine for
    # normal use); with a free key (https://www.ncbi.nlm.nih.gov/account/) it's 10 req/s.
    ncbi_api_key: str | None = Field(None, alias="NCBI_API_KEY")

    # ── openFDA ──
    # Optional. Without it openFDA calls are capped at 1,000 requests/day per
    # IP; with a free key (https://open.fda.gov/apis/authentication/) it's 120,000/day.
    openfda_api_key: str | None = Field(None, alias="OPENFDA_API_KEY")

    # ── Search engine tunables ──
    search_timeout_seconds: int = Field(45, alias="SEARCH_TIMEOUT_SECONDS")

    # ── In-process cache ──
    # Search results are cached in memory only (no database, by design - see
    # app/cache.py). The entry cap bounds memory on a long-lived process.
    source_cache_ttl_days: int = Field(30, alias="SOURCE_CACHE_TTL_DAYS")
    source_cache_max_entries: int = Field(2000, alias="SOURCE_CACHE_MAX_ENTRIES")
    topic_cache_max_entries: int = Field(500, alias="TOPIC_CACHE_MAX_ENTRIES")

    @property
    def cors_allow_origins(self) -> list[str]:
        """The configured CORS origins; empty when unset."""
        return [o.strip() for o in self.cors_allow_origins_raw.split(",") if o.strip()]

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
