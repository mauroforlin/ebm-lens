"""API dependencies: optional API-key auth."""
from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from app.config import Settings, get_settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Enforce ``X-API-Key`` only when ``API_KEY`` is configured.

    With no ``API_KEY`` set (the default), the API is open - fine for local
    use or a trusted network. Set ``API_KEY`` in ``.env`` before exposing
    this publicly, and clients must then send a matching ``X-API-Key`` header.
    """
    settings: Settings = get_settings()
    if not settings.api_key:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header",
        )
