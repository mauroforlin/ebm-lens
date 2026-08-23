"""Progress-event plumbing for streaming a run over Server-Sent Events.

`POST /api/related-articles/stream` (see `app/api/endpoints.py`) drives this:
the pipeline runs in a worker thread and calls `ProgressEmitter.emit` as each
stage starts; the request handler concurrently drains the same emitter's
`iter_sse` and forwards each frame to the client. The run ends with one
`result` frame carrying the same JSON `/api/related-articles` would have
returned (`status: "failed"` included, since the orchestrator never raises),
then the stream closes - still one request, one run, no job id, no polling.

`app/pipeline/orchestrator.py` and `app/pipeline/agentic.py` take an
`emitter: Emitter` argument, defaulting to the shared `NULL_EMITTER`, so the
plain `/api/related-articles` code path never has to branch on whether
anyone is listening.
"""
from __future__ import annotations

import json
import queue
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ProgressEvent:
    """One stage transition, timestamped when it was emitted."""

    stage: str
    message: str
    ts: float = field(default_factory=time.time)

    def to_sse(self) -> str:
        payload = json.dumps({"stage": self.stage, "message": self.message, "ts": self.ts})
        return f"event: progress\ndata: {payload}\n\n"


@dataclass(frozen=True)
class ResultEvent:
    """The run's final response, JSON-serialisable (a `RelatedArticlesResponse.model_dump()`)."""

    data: dict[str, Any]

    def to_sse(self) -> str:
        return f"event: result\ndata: {json.dumps(self.data)}\n\n"


@dataclass(frozen=True)
class ErrorEvent:
    """An uncaught failure outside the orchestrator's own try/except.

    The orchestrator never raises - it returns a ``status: "failed"``
    response instead - so in practice this only fires if something goes
    wrong in the streaming plumbing itself, not the pipeline.
    """

    message: str

    def to_sse(self) -> str:
        return f"event: error\ndata: {json.dumps({'error': self.message})}\n\n"


class Emitter(Protocol):
    """What pipeline stages call into. `ProgressEmitter` and `NullEmitter` both satisfy this."""

    def emit(self, stage: str, message: str) -> None: ...


class ProgressEmitter:
    """Thread-safe channel from a pipeline run to its SSE response.

    One instance per run - there is no job id or registry, matching the
    orchestrator's "one request, one run" shape (see its module docstring).
    The producer (a worker thread running the pipeline) calls `emit`,
    `emit_result` and finally `close`; the consumer (the async request
    handler) iterates `iter_sse` and streams each frame as it arrives.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[ProgressEvent | ResultEvent | ErrorEvent | None] = queue.Queue()

    def emit(self, stage: str, message: str) -> None:
        self._queue.put(ProgressEvent(stage=stage, message=message))

    def emit_result(self, data: dict[str, Any]) -> None:
        self._queue.put(ResultEvent(data=data))

    def emit_error(self, message: str) -> None:
        self._queue.put(ErrorEvent(message=message))

    def close(self) -> None:
        """Signal that no further events are coming; unblocks `iter_sse`."""
        self._queue.put(None)

    def iter_sse(self) -> Iterator[str]:
        """Yield SSE frames as they're emitted, until `close()`.

        Blocks on each `queue.get()`, so this belongs in a worker thread or
        behind a threadpool call - never awaited directly on the event loop.
        """
        while True:
            event = self._queue.get()
            if event is None:
                return
            yield event.to_sse()


class NullEmitter:
    """No-op stand-in so pipeline code can take an `Emitter` unconditionally."""

    def emit(self, stage: str, message: str) -> None:
        pass


NULL_EMITTER = NullEmitter()
