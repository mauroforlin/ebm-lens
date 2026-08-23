"""End-to-end test of the SSE wiring in app/api/endpoints.py.

Exercises the actual thread + queue.Queue + async-generator bridge behind
`POST /api/related-articles/stream`, with `run_related_articles` replaced by
a fake that emits through the same `Emitter` protocol the real orchestrator
uses - this is what would break if the threadpool/queue bridging in the
endpoint were wrong, which a unit test of events.py alone can't catch.
"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

import app.api.endpoints as endpoints_module
from app.main import app
from app.schemas import RelatedArticlesResponse

client = TestClient(app)


def _parse_frames(raw_text: str) -> list[tuple[str, dict]]:
    frames = []
    for block in raw_text.split("\n\n"):
        if not block.strip():
            continue
        event = "message"
        data = None
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = json.loads(line[len("data:"):].strip())
        frames.append((event, data))
    return frames


def test_stream_endpoint_emits_progress_then_result(monkeypatch):
    def fake_run_related_articles(*, topic, domain_hint, max_sources, summary_language, emitter):
        emitter.emit("domain", "Analysing the topic…")
        emitter.emit("discovery", "Round 1: 5 candidates from 3 queries")
        return RelatedArticlesResponse(
            status="completed",
            topic=topic,
            domain_detected="medicine",
            duration_seconds=1.23,
        )

    monkeypatch.setattr(endpoints_module, "run_related_articles", fake_run_related_articles)

    with client.stream(
        "POST", "/api/related-articles/stream",
        json={"topic": "semaglutide cardiovascular outcomes"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    frames = _parse_frames(body)
    events = [event for event, _ in frames]

    assert events == ["progress", "progress", "result"]
    assert frames[0][1]["stage"] == "domain"
    assert frames[1][1]["message"] == "Round 1: 5 candidates from 3 queries"
    assert frames[2][1]["status"] == "completed"
    assert frames[2][1]["topic"] == "semaglutide cardiovascular outcomes"


def test_stream_endpoint_surfaces_unexpected_exceptions_as_error_frame(monkeypatch):
    def broken_run_related_articles(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(endpoints_module, "run_related_articles", broken_run_related_articles)

    with client.stream(
        "POST", "/api/related-articles/stream",
        json={"topic": "a topic that triggers the fake failure"},
    ) as response:
        body = "".join(response.iter_text())

    frames = _parse_frames(body)
    assert frames[-1][0] == "error"
    assert frames[-1][1]["error"] == "boom"
