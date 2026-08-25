"""In-process TTL cache - the whole persistence layer of this project.

There is no database and no Redis, on purpose: the tool is meant to be
cloned, given one API key, and run. Cached search results live in this
process's memory and are lost on restart, which costs a slower first query
after a restart and nothing else.

If the process stays up for days:

* Expiry is per entry (providers' results go stale at different rates).
* Capacity is bounded. TTL alone does not bound memory, an expired
  entry is only reclaimed if someone asks for that exact key again, and
  search keys are query-shaped, so most are never asked for twice. The
  cache therefore evicts least recently used entries once it is full, and
  opportunistically drops expired ones on insert.

If you need a more persistent solution, you can use Redis or SQLite.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")

_DEFAULT_MAX_ENTRIES = 2000


@dataclass
class _Entry(Generic[T]):
    value: T
    expires_at: float


class TTLCache(Generic[T]):
    """Thread-safe in-memory cache with per-entry TTL and an LRU capacity bound."""

    def __init__(self, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        self._lock = threading.Lock()
        self._store: OrderedDict[str, _Entry[T]] = OrderedDict()
        self._max_entries = max(1, max_entries)

    def get(self, key: str) -> T | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.expires_at < time.monotonic():
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return entry.value

    def set(self, key: str, value: T, ttl_seconds: float) -> None:
        with self._lock:
            self._store[key] = _Entry(value=value, expires_at=time.monotonic() + ttl_seconds)
            self._store.move_to_end(key)
            self._evict_locked()

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def set_max_entries(self, max_entries: int) -> None:
        """Resize the capacity bound, evicting immediately if now over it."""
        with self._lock:
            self._max_entries = max(1, max_entries)
            self._evict_locked()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def _evict_locked(self) -> None:
        """Drop expired entries, then LRU-evict until within capacity.

        Caller must hold the lock. Expired entries are swept first so that a
        full-but-stale cache does not evict live entries needlessly.
        """
        if len(self._store) <= self._max_entries:
            return
        now = time.monotonic()
        for key in [k for k, e in self._store.items() if e.expires_at < now]:
            del self._store[key]
        while len(self._store) > self._max_entries:
            self._store.popitem(last=False)  # least recently used


# TODO: source_cache and topic_evidence_cache live in this process's memory
# (module-level dict), so they cannot be shared across workers. Scaling this
# service horizontally needs a shared backend (Redis or similar) behind the
# same get/set interface.

source_cache: TTLCache[list[dict[str, Any]]] = TTLCache()
"""Per-(provider, query) raw search-result cache. Keyed by a hash of
(source_type, normalised query)."""

topic_evidence_cache: TTLCache[list[dict[str, Any]]] = TTLCache(max_entries=500)
"""Aggregated per-topic evidence cache, keyed by a normalised topic hash.

Exact-hash lookups only. Two phrasings of the same question miss each other
unless they normalise identically. This is a speed and cost loss on repeat queries
and the price of having no vector store to search."""
# TODO: exact-hash matching means two different phrasings of the same
# question miss each other. Lifting this needs embedding similarity over a
# vector store instead of a hash lookup.


def configure_caches(
    *,
    source_max_entries: int,
    topic_max_entries: int,
) -> None:
    source_cache.set_max_entries(source_max_entries)
    topic_evidence_cache.set_max_entries(topic_max_entries)
