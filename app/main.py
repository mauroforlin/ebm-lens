"""FastAPI entry point.

Wires the API router, the static frontend and startup configuration. All the
work happens in :mod:`app.pipeline`.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.endpoints import router as api_router
from app.config import get_settings
from app.core.cache import configure_caches

logging.basicConfig(level=logging.INFO)

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description=(
        "Evidence discovery for medical and clinical topics. Searches PubMed, "
        "Europe PMC, ClinicalTrials.gov, OpenFDA, DailyMed, RxNav, Open Targets, "
        "ChEMBL, WHO GHO, EMA, bioRxiv and Wikipedia, expands through the citation "
        "graph, then ranks, summarises and synthesises the results."
    ),
    version="0.1.0",
)

configure_caches(
    source_max_entries=settings.source_cache_max_entries,
    topic_max_entries=settings.topic_cache_max_entries,
)

# Cross-origin access is opt-in. The bundled UI is served from this same
# origin, so it needs no CORS at all; a wildcard would only widen who can
# spend this deployment's API budget.
if settings.cors_allow_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

app.include_router(api_router, prefix="/api")

_FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(_FRONTEND_DIR / "index.html"))
