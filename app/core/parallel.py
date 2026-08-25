"""Shared bounded thread-pool fan-out for independent per-item work (LLM
calls, HTTP lookups) where a caller wants whatever finishes rather than
paying for the slowest item's failure to cost every other item its result.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def run_parallel(
    fn: Callable[[T], R],
    items: Iterable[T],
    max_workers: int,
    *,
    timeout: float | None = None,
) -> list[R]:
    """Run ``fn(item)`` for every item in *items* concurrently.

    Returns whatever completes, in completion order rather than input order -
    callers that need to know which item produced which result should have
    *fn* carry that itself (e.g. return ``(item, fn_result)``). An item whose
    call raises is logged and dropped rather than failing the whole batch.

    Not a bare ``with ThreadPoolExecutor(...)``: its implicit
    ``shutdown(wait=True)`` on exit would block until every submitted item
    finished regardless of *timeout*, silently turning the deadline into a
    no-op. ``shutdown(wait=False, cancel_futures=True)`` returns as soon as
    the deadline (or the last result) arrives; anything still running
    finishes in the background with its result simply never collected.
    """
    items = list(items)
    if not items:
        return []

    results: list[R] = []
    pool = ThreadPoolExecutor(max_workers=min(max_workers, len(items)))
    try:
        futures = {pool.submit(fn, item): item for item in items}
        try:
            for future in as_completed(futures, timeout=timeout):
                item = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    logger.warning("run_parallel: item %r failed: %s", item, exc)
        except FutureTimeoutError:
            logger.warning(
                "run_parallel: timed out after %ss with %d/%d items still pending",
                timeout, len(items) - len(results), len(items),
            )
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return results
