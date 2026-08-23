"""Unit tests for the SSE progress-event channel in app/core/events.py.

Covers the plumbing in isolation from the pipeline that drives it: events
arrive in order, `close()` terminates the stream, a result/error frame is
shaped correctly, and `NullEmitter` is a safe no-op.
"""
from __future__ import annotations

import json
import threading

from app.core.events import NULL_EMITTER, NullEmitter, ProgressEmitter


def test_emit_then_close_yields_events_in_order():
    emitter = ProgressEmitter()
    emitter.emit("brief", "Writing the research brief")
    emitter.emit("search", "Round 1")
    emitter.close()

    frames = list(emitter.iter_sse())

    assert len(frames) == 2
    assert "Writing the research brief" in frames[0]
    assert "Round 1" in frames[1]


def test_sse_frame_is_well_formed():
    emitter = ProgressEmitter()
    emitter.emit("brief", "hello")
    emitter.close()

    frame = next(emitter.iter_sse())

    assert frame.startswith("event: progress\ndata: ")
    assert frame.endswith("\n\n")
    assert '"stage": "brief"' in frame
    assert '"message": "hello"' in frame


def test_iter_sse_blocks_until_producer_emits():
    emitter = ProgressEmitter()
    received = []

    def consume():
        received.extend(emitter.iter_sse())

    consumer = threading.Thread(target=consume)
    consumer.start()
    emitter.emit("brief", "first")
    emitter.close()
    consumer.join(timeout=2)

    assert not consumer.is_alive()
    assert len(received) == 1


def test_null_emitter_is_a_no_op():
    NullEmitter().emit("stage", "message")
    NULL_EMITTER.emit("stage", "message")


def test_emit_result_yields_a_result_frame():
    emitter = ProgressEmitter()
    emitter.emit_result({"status": "completed", "articles": []})
    emitter.close()

    frame = next(emitter.iter_sse())

    assert frame.startswith("event: result\ndata: ")
    payload = json.loads(frame.split("data: ", 1)[1])
    assert payload == {"status": "completed", "articles": []}


def test_emit_error_yields_an_error_frame():
    emitter = ProgressEmitter()
    emitter.emit_error("boom")
    emitter.close()

    frame = next(emitter.iter_sse())

    assert frame.startswith("event: error\ndata: ")
    assert json.loads(frame.split("data: ", 1)[1]) == {"error": "boom"}


def test_progress_then_result_then_close_is_the_full_run_shape():
    emitter = ProgressEmitter()
    emitter.emit("brief", "Writing the research brief")
    emitter.emit_result({"status": "completed"})
    emitter.close()

    frames = list(emitter.iter_sse())

    assert frames[0].startswith("event: progress\n")
    assert frames[1].startswith("event: result\n")
