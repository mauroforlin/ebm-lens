"""In-memory API rate limiter (fixed-window, per-process).

This service runs as a single process, so a per-process quota is enough -
no Redis, no shared state to coordinate across workers.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class _Quota:
    rpm: int
    tpm: int


class InMemoryRateLimiter:
    """Simple fixed-window (60s) rate limiter: N requests and M tokens per minute."""

    def __init__(self, rpm: int, tpm: int) -> None:
        self._quota = _Quota(rpm=rpm, tpm=tpm)
        self._lock = threading.Lock()
        self._window_start = time.monotonic()
        self._requests_used = 0
        self._tokens_used = 0

    def _reset_if_needed(self) -> None:
        now = time.monotonic()
        if now - self._window_start >= 60:
            self._window_start = now
            self._requests_used = 0
            self._tokens_used = 0

    def acquire(self, tokens: int) -> None:
        while True:
            with self._lock:
                self._reset_if_needed()
                can_req = self._requests_used < self._quota.rpm
                can_tok = self._tokens_used + tokens <= self._quota.tpm
                if can_req and can_tok:
                    self._requests_used += 1
                    self._tokens_used += tokens
                    return
                wait = max(0, 60 - (time.monotonic() - self._window_start))
            time.sleep(min(max(wait, 0.1), 5))


_limiters: dict[str, InMemoryRateLimiter] = {}
_limiters_lock = threading.Lock()


def get_limiter(key_prefix: str, rpm: int, tpm: int) -> InMemoryRateLimiter:
    """Return a singleton limiter for *key_prefix*, creating it on first use."""
    limiter = _limiters.get(key_prefix)
    if limiter is None:
        with _limiters_lock:
            limiter = _limiters.get(key_prefix)
            if limiter is None:
                limiter = InMemoryRateLimiter(rpm=rpm, tpm=tpm)
                _limiters[key_prefix] = limiter
    return limiter


@contextmanager
def reserve(key_prefix: str, rpm: int, tpm: int, tokens: int) -> Iterator[None]:
    """Block until *tokens* worth of quota is available under *key_prefix*."""
    get_limiter(key_prefix, rpm, tpm).acquire(tokens)
    yield
