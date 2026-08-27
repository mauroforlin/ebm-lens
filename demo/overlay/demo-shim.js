"use strict";

/* Loads before app.js and replaces window.fetch. app.js's only network call
   is `fetch(STREAM_API, {method: "POST", ...})` against
   /api/related-articles/stream (see frontend/app.js); everything else in
   app.js - progress log, citations, filters, run stats, history - is
   unmodified and has no idea this page has no backend.

   On a match, this resolves the posted topic to a recorded fixture and
   replays its `events` as the same SSE frames the real stream would have
   sent, then the fixture's full response as the `result` frame. app.js reads
   both through the same `response.body.getReader()` path either way. */

(function () {
  const STREAM_PATH = "/api/related-articles/stream";
  const FIXTURES_BASE = "fixtures/";

  // The real run took anywhere from ~60s to ~180s; compressing every replay
  // toward this budget keeps the demo watchable while preserving the
  // relative pacing between stages recorded in the fixture.
  const REPLAY_BUDGET_MS = 2500;
  const MIN_FRAME_GAP_MS = 50;

  const realFetch = window.fetch.bind(window);
  let indexPromise = null;

  function loadIndex() {
    if (!indexPromise) {
      indexPromise = realFetch(FIXTURES_BASE + "index.json").then((r) => r.json());
    }
    return indexPromise;
  }

  function normalize(text) {
    return (text || "").trim().toLowerCase();
  }

  async function resolveFixtureId(topic) {
    const index = await loadIndex();
    const target = normalize(topic);
    const hit = index.find((entry) => normalize(entry.topic) === target);
    return hit ? hit.id : null;
  }

  function sseFrame(event, data) {
    return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  }

  function buildReplayStream(fixture) {
    const events = fixture.events || [];
    const lastAt = events.length ? events[events.length - 1].at_ms : 0;
    const scale = lastAt > 0 ? Math.min(1, REPLAY_BUDGET_MS / lastAt) : 1;
    const encoder = new TextEncoder();

    return new ReadableStream({
      async start(controller) {
        let prevDelay = 0;
        for (const evt of events) {
          const targetDelay = Math.max(prevDelay + MIN_FRAME_GAP_MS, evt.at_ms * scale);
          const wait = targetDelay - prevDelay;
          if (wait > 0) await new Promise((resolve) => setTimeout(resolve, wait));
          prevDelay = targetDelay;
          controller.enqueue(encoder.encode(sseFrame("progress", {
            stage: evt.stage,
            message: evt.message,
            ts: Date.now() / 1000,
          })));
        }
        controller.enqueue(encoder.encode(sseFrame("result", fixture.response)));
        controller.close();
      },
    });
  }

  function notFoundResponse() {
    return new Response(
      JSON.stringify({ detail: "This is a read-only demo: pick one of the questions below to see a real run." }),
      { status: 404, headers: { "Content-Type": "application/json" } },
    );
  }

  window.fetch = async function (input, init) {
    const url = typeof input === "string" ? input : input && input.url;
    if (!url || !url.endsWith(STREAM_PATH)) {
      return realFetch(input, init);
    }

    let topic = "";
    try {
      topic = JSON.parse((init && init.body) || "{}").topic || "";
    } catch {
      // falls through to notFoundResponse below
    }

    const id = await resolveFixtureId(topic);
    if (!id) return notFoundResponse();

    const fixtureResp = await realFetch(FIXTURES_BASE + id + ".json");
    if (!fixtureResp.ok) return notFoundResponse();
    const fixture = await fixtureResp.json();

    return new Response(buildReplayStream(fixture), {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    });
  };
})();
