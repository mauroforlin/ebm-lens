"""Record real pipeline runs as static fixtures for the read-only demo.

Calls `run_related_articles` directly, the same entry point
`POST /api/related-articles` uses - see `app/pipeline/orchestrator.py`. Each
completed run is written to `demo/fixtures/<id>.json` as the progress events
the real stream would have emitted (offsets, not wall-clock timestamps) plus
the full response. `demo/overlay/demo-shim.js` replays that file instead of
calling a live backend.

Resumable like the eval/ scripts (see eval/_harness.py): a fixture already on
disk is skipped unless --fresh or --only names it. A run that doesn't finish
with status "completed" is never written, so a fixture on disk is always
usable.

Usage:
    python demo/scripts/record_demo.py                  # all 15, skipping done
    python demo/scripts/record_demo.py --only glp1-cv-risk
    python demo/scripts/record_demo.py --only glp1-cv-risk --fresh
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import get_settings
from app.pipeline.orchestrator import run_related_articles

_DEMO_DIR = Path(__file__).resolve().parents[1]
_QUERIES_PATH = _DEMO_DIR / "queries.json"
_FIXTURES_DIR = _DEMO_DIR / "fixtures"
_INDEX_PATH = _FIXTURES_DIR / "index.json"


class _Recorder:
    """Emitter that timestamps each stage as an offset from the first call.

    The offset, not the wall-clock time, is what the shim needs to replay the
    run's shape (see demo-shim.js) - a fixture recorded today and one recorded
    next month replay identically regardless of when they were made.
    """

    def __init__(self) -> None:
        self.events: list[dict] = []
        self._started: float | None = None

    def emit(self, stage: str, message: str) -> None:
        now = time.monotonic()
        if self._started is None:
            self._started = now
        self.events.append({"stage": stage, "message": message, "at_ms": round((now - self._started) * 1000)})


# Values that must never reach a committed fixture. Redacted wherever they
# appear in the serialised JSON, not just in known fields, because a failed
# provider call can surface an API key inside an httpx error string embedded
# anywhere in the response (see app/sources/pubmed.py, openfda.py).
_SECRET_ENV_VARS = ("NCBI_API_KEY", "OPENFDA_API_KEY", "OPENROUTER_API_KEY")


def _redact(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[redacted]")
    text = re.sub(r"(api[_-]?key=)[^&\s\"']+", r"\1[redacted]", text, flags=re.IGNORECASE)
    return text


def _record_one(query: dict) -> dict | None:
    print(f"  running: {query['topic']!r} (max_sources={query['max_sources']})", flush=True)
    recorder = _Recorder()
    resp = run_related_articles(
        topic=query["topic"],
        max_sources=query["max_sources"],
        summary_language="en",
        emitter=recorder,
    )
    if resp.status != "completed":
        print(f"  REJECTED ({resp.status}): {resp.error}", flush=True)
        return None

    data = resp.model_dump()
    data.pop("error", None)  # only ever meaningful on a failed run, which is never written

    fixture = {
        "id": query["id"],
        "topic": query["topic"],
        "max_sources": query["max_sources"],
        "summary_language": "en",
        "recorded_at": time.strftime("%Y-%m-%d"),
        "events": recorder.events,
        "response": data,
    }

    settings = get_settings()
    secrets = [getattr(settings, name.lower(), None) or "" for name in _SECRET_ENV_VARS]
    text = _redact(json.dumps(fixture, ensure_ascii=False), [s for s in secrets if s])
    return json.loads(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="Record a single query id, ignoring the skip-if-present rule.")
    parser.add_argument("--fresh", action="store_true", help="Re-record even if a fixture already exists.")
    args = parser.parse_args()

    queries = json.loads(_QUERIES_PATH.read_text(encoding="utf-8"))
    if args.only:
        queries = [q for q in queries if q["id"] == args.only]
        if not queries:
            sys.exit(f"no query with id {args.only!r} in {_QUERIES_PATH}")

    _FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    for i, query in enumerate(queries, start=1):
        out_path = _FIXTURES_DIR / f"{query['id']}.json"
        print(f"[{i}/{len(queries)}] {query['id']}")
        if out_path.exists() and not args.fresh and not args.only:
            print("  cached, skipping (use --only to force a single re-record)")
            continue

        started = time.monotonic()
        fixture = _record_one(query)
        elapsed = time.monotonic() - started
        if fixture is None:
            print(f"  ({elapsed:.1f}s) not written - rerun this id to retry")
            continue

        out_path.write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8")
        n = len(fixture["response"]["articles"])
        cost = fixture["response"]["job_stats"].get("total_cost_usd", 0)
        print(f"  ({elapsed:.1f}s) wrote {out_path.name}: {n} articles, ${cost:.4f}")

    _write_index()


def _write_index() -> None:
    queries = json.loads(_QUERIES_PATH.read_text(encoding="utf-8"))
    entries = []
    for q in queries:
        fixture_path = _FIXTURES_DIR / f"{q['id']}.json"
        if not fixture_path.exists():
            continue
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        resp = fixture["response"]
        entries.append({
            "id": q["id"],
            "topic": q["topic"],
            "max_sources": q["max_sources"],
            "showcase": q["showcase"],
            "domain_detected": resp.get("domain_detected", ""),
            "article_count": len(resp.get("articles", [])),
            "duration_seconds": resp.get("duration_seconds", 0),
        })
    _INDEX_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nindex: {_INDEX_PATH} ({len(entries)} of {len(queries)} queries have a fixture)")


if __name__ == "__main__":
    main()
