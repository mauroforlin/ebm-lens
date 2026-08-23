"""Shared scaffolding for the resumable per-topic eval scripts.

Every eval here follows the same shape: load a JSONL fixture, run one graded
call per row, append each result to a progress file as soon as it's done, and
rewrite a summary after every row - so a run can be Ctrl-C'd and resumed
without losing what it already paid for. This module owns that shape once;
`retrieval_eval.py`, `pool_relevance_eval.py`, `pico_eval.py` and
`stance_eval.py` each supply only what's specific to them: how to grade one
row (`evaluate`), how to fold rows into a summary (`summarise`), and how to
describe one row's result on the progress line (`describe`).

Callers keep their own `sys.path.insert(0, ...)` above their `app.*` imports -
this module is imported after that line, so it must not depend on it.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

_RESULTS_DIR = Path(__file__).resolve().parent / "results"
_HISTORY_PATH = _RESULTS_DIR / "history.jsonl"


def add_standard_args(parser: argparse.ArgumentParser, default_fixture: Path) -> None:
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fixture", type=Path, default=default_fixture)
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore any existing progress file for this fixture and start over.",
    )


def load_fixture(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def progress_path(fixture: Path, stem: str | None = None) -> Path:
    return _RESULTS_DIR / f"{stem or fixture.stem}_progress.jsonl"


def summary_path(fixture: Path, stem: str | None = None) -> Path:
    return _RESULTS_DIR / f"{stem or fixture.stem}_summary.json"


def load_progress(path: Path) -> dict[str, dict]:
    """Rows already on disk from a prior run, as ``{row_key: row}``."""
    done: dict[str, dict] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            done[row["id"]] = row
    return done


def append_progress(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _git_sha() -> str | None:
    """Short commit hash of the working tree, or None (e.g. no commits yet)."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5,
        )
    except OSError:
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def append_history(
    eval_id: str, fixture_name: str, summary: dict, n_done: int, n_total: int,
    timestamp: str | None = None, git_sha: str | None = ...,
) -> None:
    """Append one immutable row to results/history.jsonl.

    Unlike summary_path (overwritten every run), this file is append-only, so
    a run's numbers survive being superseded by a later, possibly incomplete,
    rerun of the same fixture - the gap this eval suite had until this row.
    """
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": timestamp or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "eval": eval_id,
        "fixture": fixture_name,
        "git_sha": _git_sha() if git_sha is ... else git_sha,
        "n_done": n_done,
        "n_total": n_total,
        "complete": n_done == n_total,
        "summary": summary,
    }
    with _HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_rows(
    rows: list[dict],
    done: dict[str, dict],
    progress_path: Path,
    summary_path: Path,
    evaluate: Callable[[dict], dict],
    summarise: Callable[[list[dict]], dict],
    describe: Callable[[dict, dict], str],
    fixture_name: str,
    row_key: Callable[[dict], str] = lambda row: row["id"],
) -> dict:
    """Run *evaluate* over every row not already in *done*, resumably.

    *evaluate* returns a metrics dict; *row_key* is merged in as ``"id"`` plus
    whatever else *evaluate* already put there, and appended to progress.
    Returns the final summary dict.
    """
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = list(done.values())

    n_done = sum(1 for r in rows if row_key(r) in done)
    print(f"running {len(rows)} rows from {fixture_name}"
          f" ({n_done} already done, resuming into {progress_path.name})")

    def _write_summary() -> dict:
        summary = summarise(results) if results else {}
        summary_path.write_text(
            json.dumps({"summary": summary, "rows": results}, indent=2),
            encoding="utf-8",
        )
        return summary

    try:
        for i, row in enumerate(rows, start=1):
            key = row_key(row)
            if key in done:
                print(f"[{i}/{len(rows)}] cached  {key}", flush=True)
                continue

            started = time.monotonic()
            try:
                metrics = evaluate(row)
                error = None
            except Exception as exc:  # noqa: BLE001 - one row's failure must not abort the run
                metrics, error = {}, str(exc)
            elapsed = time.monotonic() - started

            result = {**row, "id": key, "error": error, **metrics}
            results.append(result)
            done[key] = result
            append_progress(progress_path, result)

            status = f"ERROR: {error}" if error else describe(row, metrics)
            print(f"[{i}/{len(rows)}] ({elapsed:.1f}s) {status}", flush=True)
            _write_summary()
    except KeyboardInterrupt:
        print(f"\nInterrupted - {len(results)} results saved to "
              f"{progress_path.name}. Re-run the same command to resume.")

    summary = _write_summary()
    append_history(
        eval_id=summary_path.stem.removesuffix("_summary"),
        fixture_name=fixture_name,
        summary=summary,
        n_done=len(results),
        n_total=len(rows),
    )
    print("\n--- summary ---")
    _print_summary(summary)
    print(f"\nprogress: {progress_path}\nsummary:  {summary_path}\nhistory:  {_HISTORY_PATH}")
    return summary


def _print_summary(summary: dict, indent: str = "  ") -> None:
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"{indent}{key}:")
            _print_summary(value, indent + "  ")
        else:
            print(f"{indent}{key}: {value:.3f}" if isinstance(value, float) else f"{indent}{key}: {value}")
