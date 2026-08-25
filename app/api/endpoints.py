"""API endpoints """
from __future__ import annotations

import threading
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app.api.deps import require_api_key
from app.core.events import ProgressEmitter
from app.pipeline.orchestrator import run_related_articles
from app.schemas import RelatedArticlesRequest, RelatedArticlesResponse

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post(
    "/related-articles",
    response_model=RelatedArticlesResponse,
    dependencies=[Depends(require_api_key)],
)
async def related_articles(payload: RelatedArticlesRequest) -> RelatedArticlesResponse:
    """Run the related-articles pipeline for *payload.topic* and return the result.

    This is synchronous from the client's point of view: the request blocks
    until the pipeline finishes (typically 60-180s). The pipeline itself is
    blocking (network calls, no async I/O), so it runs in FastAPI's
    threadpool to avoid blocking the event loop.
    """
    return await run_in_threadpool(
        run_related_articles,
        topic=payload.topic,
        domain_hint=payload.domain_hint,
        max_sources=payload.max_sources,
        summary_language=payload.summary_language,
    )


@router.post(
    "/related-articles/stream",
    dependencies=[Depends(require_api_key)],
)
async def related_articles_stream(payload: RelatedArticlesRequest) -> StreamingResponse:
    """Same pipeline as `/related-articles`, streamed as Server-Sent Events.

    Still one request, one run: no job id, no polling endpoint. The pipeline
    runs to completion in a plain background thread regardless of whether
    the client stays connected - there is nothing to cancel it into, and
    disconnecting only stops the stream, not the work. A `progress` frame is
    sent as each stage starts or finishes, then one `result` frame carries
    the same JSON `/related-articles` would have returned (`status:
    "failed"` included, since the orchestrator never raises), and the stream
    closes.
    """
    emitter = ProgressEmitter()

    def _run() -> None:
        try:
            result = run_related_articles(
                topic=payload.topic,
                domain_hint=payload.domain_hint,
                max_sources=payload.max_sources,
                summary_language=payload.summary_language,
                emitter=emitter,
            )
            emitter.emit_result(result.model_dump())
        except Exception as exc:  # pragma: no cover - the orchestrator already catches its own
            emitter.emit_error(str(exc))
        finally:
            emitter.close()

    # TODO: a client that disconnects here leaves this run to completion with
    # nobody reading the result - a real cost (LLM calls) spent for nothing,
    # and unbounded for any public deployment. Fixing it needs a cancellation
    # token threaded through and checked between pipeline stages; the
    # pipeline is synchronous today and has no such checkpoint.
    threading.Thread(target=_run, daemon=True).start()

    async def _frames() -> AsyncIterator[str]:
        # `iter_sse` blocks on a plain queue.Queue, so each `next()` is
        # bounced through the threadpool to avoid stalling the event loop
        # between frames.
        pending = emitter.iter_sse()
        while True:
            frame = await run_in_threadpool(next, pending, None)
            if frame is None:
                return
            yield frame

    return StreamingResponse(
        _frames(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
