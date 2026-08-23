"""Calibrate ``ranking.PRIOR_WEIGHTS`` against retrieval_eval's own metric,
instead of hand-picked constants.

`PRIOR_WEIGHTS` is a linear combination of per-candidate signals - exactly the
shape "coordinate ascent" was built for (Metzler & Croft, *Linear
Feature-based Models for Information Retrieval*, Information Retrieval 2007):
hold every weight fixed but one, grid-search that one against a rank metric,
move to the next, repeat until nothing improves. It needs no gradient, which
matters here because nDCG/precision/MRR are piecewise-constant in the
weights - a plain optimiser has nothing to climb.

Two phases, run separately because only one of them costs money:

    python eval/calibrate_ranking_weights.py --dump [--limit N]
        Runs the real pipeline (_discover, _build_signals) once per fixture
        topic - the expensive part, real API/LLM calls - and caches every
        candidate's raw signal dict plus its gold grade to
        results/weight_calibration_signals_progress.jsonl. Resumable, same
        shape as retrieval_eval.py's own progress file.

    python eval/calibrate_ranking_weights.py [--restarts N] [--folds N]
        Pure arithmetic over the cached signals - no network calls, runs in
        minutes. Reports the baseline weights' score, a cross-validated
        estimate of what a re-fit would generalise to, and a candidate
        weight vector fit on the whole fixture.

The optimisation target is ``ndcg_at_10_pool``, not ``ndcg_at_10`` or
``pool_recall``: those two are about what discovery found, which no ranking
weight can change. ``ndcg_at_10_pool`` isolates the one thing weights *can*
fix - whether the candidates discovery already retrieved get ordered well -
and that's the only thing this script has any business touching.

Caveat that matters more than any number this prints: the fixture is 40
BioASQ topics. That is enough to see whether a weight change points the right
direction, not enough to trust four decimal places or to treat the fitted
vector as final. The cross-validated score (fit on 4 folds, score on the
held-out one, repeated per fold) exists specifically to catch overfitting to
those 40 questions - a wide gap between the in-sample and cross-validated
numbers means the suggested weights are memorising the fixture, not finding a
real pattern, and should not be pasted into ranking.py as-is.

FINAL_WEIGHTS is out of scope: it includes `relevance`, the LLM's
post-summarisation judgement, which retrieval_eval.py doesn't compute (running
summarisation for 40 topics just for calibration would cost what the eval
itself is trying to avoid). Only PRIOR_WEIGHTS is calibrated here.

Usage:
    python eval/calibrate_ranking_weights.py --dump [--limit N] [--fixture PATH] [--fresh]
    python eval/calibrate_ranking_weights.py [--restarts N] [--folds N] [--seed N] [--limit N]
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eval._harness as harness
import eval.retrieval_eval as retrieval_eval
from app.config import get_settings
from app.core.job_stats import JobStats
from app.pipeline import orchestrator, ranking, topic_analysis
from app.pipeline.dedup import dedup_key, deduplicate

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "bioasq_retrieval.jsonl"
_CACHE_STEM = "weight_calibration_signals"

# Same keys ranking.build_signals emits; weighted_score already treats a
# missing key as 0.0, so nothing needs a special case for `relevance`
# (only present in FINAL_WEIGHTS, absent from every cached candidate here).
_SEARCH_KEYS = tuple(ranking.PRIOR_WEIGHTS)
# offtopic is a suppression term by design (see ranking.py) - constraining its
# sign keeps the search from finding a degenerate "reward the wrong sense"
# optimum that happens to fit 40 topics. Every other signal is non-negative
# by construction (citation_score, recency_score, ... all live in [0, 1]).
_BOUNDS: dict[str, tuple[float, float]] = {
    key: (-0.6, 0.0) if key == "offtopic" else (0.0, 0.5) for key in _SEARCH_KEYS
}
_GRID_STEP = 0.02
_COORD_PASSES = 4
_RESTARTS = 6
_FOLDS = 5


# ══════════════════════════════════════════════════════════════
#  Phase 1: dump per-candidate signals (costs API calls, resumable)
# ══════════════════════════════════════════════════════════════


def _dump_row(row: dict) -> dict:
    settings = get_settings()
    stats = JobStats()
    context, spec = topic_analysis.analyse_topic(row["question"], settings, job_stats=stats)
    discovery = orchestrator._discover(row["question"], spec, context, settings, stats)
    discovery.evidence = deduplicate(discovery.evidence)

    if not discovery.evidence:
        return {"candidates": [], "cost_usd": stats.to_dict()["total_cost_usd"]}

    signals = orchestrator._build_signals(row["question"], spec, discovery, settings, stats)
    pool = discovery.evidence  # _build_signals truncates this list in place

    id_type = row["id_type"]
    candidates = []
    for result in pool:
        key = dedup_key(result)
        built = ranking.build_signals(
            result,
            cosine=signals.cosine.get(key, 0.0),
            bm25=signals.bm25.get(key, 0.0),
            rrf=signals.rrf.get(key, 0.0),
            evidence=signals.evidence.get(key, 0.0),
            concept=signals.concept.get(key, 0.0),
            offtopic=signals.offtopic.get(key, 0.0),
            rerank=signals.rerank.get(key, 0.0),
        )
        doc_id = retrieval_eval.extract_id(result.url, id_type)
        candidates.append({"doc_id": doc_id, "signals": built})

    return {"candidates": candidates, "cost_usd": stats.to_dict()["total_cost_usd"]}


def _dump_summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["error"]]
    return {
        "n_topics": len(rows),
        "n_errors": len(rows) - len(ok),
        "mean_candidates": statistics.mean(len(r["candidates"]) for r in ok) if ok else 0.0,
        "total_cost_usd": round(sum(r.get("cost_usd", 0.0) for r in rows), 6),
    }


def _dump_describe(row: dict, metrics: dict) -> str:
    return f"n_candidates={len(metrics.get('candidates', []))}  {row['question'][:60]}"


def _run_dump(args: argparse.Namespace) -> None:
    progress_path = harness.progress_path(args.fixture, stem=_CACHE_STEM)
    summary_path = harness.summary_path(args.fixture, stem=_CACHE_STEM)
    if args.fresh and progress_path.exists():
        progress_path.unlink()

    done = harness.load_progress(progress_path)
    rows = retrieval_eval._load_fixture(args.fixture, args.limit)

    harness.run_rows(
        rows, done, progress_path, summary_path,
        evaluate=_dump_row, summarise=_dump_summarise, describe=_dump_describe,
        fixture_name=args.fixture.name,
    )


# ══════════════════════════════════════════════════════════════
#  Phase 2: search over cached signals (no network calls)
# ══════════════════════════════════════════════════════════════


def _row_metrics(row: dict, weights: dict[str, float]) -> dict[str, float]:
    """Rank-dependent metrics only - the ones a weight change can move.

    Mirrors retrieval_eval._evaluate's precision_at_10/ndcg_at_10_pool/mrr
    exactly, but over cached signals instead of a live pipeline run.
    ``pool_recall``/``recall_at_k``/``ndcg_at_10`` (against the full gold set)
    are deliberately absent: they're about what discovery found, not how the
    found candidates got ordered, so no weight vector can move them.
    """
    candidates = row["candidates"]
    gold = row["gold"]
    if not candidates:
        return {"precision_at_10": 0.0, "ndcg_at_10_pool": 0.0, "mrr": 0.0}

    ranked = sorted(
        candidates, key=lambda c: ranking.weighted_score(c["signals"], weights), reverse=True,
    )
    ranked_ids = [c["doc_id"] for c in ranked]
    gradable = [doc_id for doc_id in ranked_ids if doc_id]
    n_gradable = len(gradable)

    def grade(doc_id: str | None) -> int:
        return gold.get(doc_id, 0) if doc_id else 0

    gradable_top10 = gradable[:10]
    precision_at_10 = (
        sum(1 for doc_id in gradable_top10 if grade(doc_id) >= 1) / min(10, n_gradable)
        if n_gradable else 0.0
    )
    ranked_grades_10 = [grade(doc_id) for doc_id in gradable_top10]
    ndcg_at_10_pool = retrieval_eval._ndcg(ranked_grades_10, [grade(d) for d in gradable], 10)
    first_hit_rank = next(
        (rank for rank, doc_id in enumerate(ranked_ids, start=1) if grade(doc_id) >= 1), None,
    )
    mrr = 1.0 / first_hit_rank if first_hit_rank else 0.0

    return {"precision_at_10": precision_at_10, "ndcg_at_10_pool": ndcg_at_10_pool, "mrr": mrr}


def _objective(rows: list[dict], weights: dict[str, float]) -> float:
    if not rows:
        return 0.0
    return statistics.mean(_row_metrics(r, weights)["ndcg_at_10_pool"] for r in rows)


def _grid_values(lo: float, hi: float, step: float) -> list[float]:
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 4) for i in range(n + 1)]


def _coordinate_ascent(
    rows: list[dict], start: dict[str, float], rng: random.Random,
) -> tuple[dict[str, float], float]:
    weights = dict(start)
    best_score = _objective(rows, weights)
    for _ in range(_COORD_PASSES):
        improved = False
        keys = list(_SEARCH_KEYS)
        rng.shuffle(keys)
        for key in keys:
            lo, hi = _BOUNDS[key]
            current_val = weights[key]
            best_val = current_val
            for val in _grid_values(lo, hi, _GRID_STEP):
                trial = dict(weights)
                trial[key] = val
                score = _objective(rows, trial)
                if score > best_score:
                    best_score, best_val = score, val
                    improved = True
            weights[key] = best_val
        if not improved:
            break
    return weights, best_score


def _random_start(rng: random.Random) -> dict[str, float]:
    return {key: round(rng.uniform(*_BOUNDS[key]), 3) for key in _SEARCH_KEYS}


def _search(
    rows: list[dict], restarts: int, rng: random.Random,
) -> tuple[dict[str, float], float]:
    """Coordinate ascent from the current weights plus random restarts."""
    starts = [dict(ranking.PRIOR_WEIGHTS)] + [_random_start(rng) for _ in range(restarts - 1)]
    best_weights, best_score = dict(ranking.PRIOR_WEIGHTS), -1.0
    for start in starts:
        weights, score = _coordinate_ascent(rows, start, rng)
        if score > best_score:
            best_weights, best_score = weights, score
    return best_weights, best_score


def _cross_validate(
    rows: list[dict], folds: int, restarts: int, rng: random.Random,
) -> list[float]:
    """Held-out ndcg_at_10_pool per fold - the overfitting check.

    Fits weights on k-1 folds, scores them on the fold left out, repeated for
    every fold. This never trains and scores on the same topics, so it's the
    honest estimate of what a re-fit generalises to, unlike the in-sample
    number `_search` alone would report.
    """
    shuffled = list(rows)
    rng.shuffle(shuffled)
    bucket = [shuffled[i::folds] for i in range(folds)]

    scores = []
    for i in range(folds):
        test = bucket[i]
        train = [r for j, fold in enumerate(bucket) if j != i for r in fold]
        weights, _ = _search(train, restarts, rng)
        scores.append(_objective(test, weights))
    return scores


def _load_cache(fixture: Path, limit: int | None) -> list[dict]:
    cache_path = harness.progress_path(fixture, stem=_CACHE_STEM)
    if not cache_path.exists():
        raise SystemExit(
            f"No signal cache at {cache_path}.\n"
            f"Run `python eval/calibrate_ranking_weights.py --dump` first "
            f"(this is the phase that costs API calls)."
        )
    rows = [r for r in harness.load_progress(cache_path).values() if not r["error"] and r["candidates"]]
    return rows[:limit] if limit else rows


def _print_weights(weights: dict[str, float]) -> None:
    for key in _SEARCH_KEYS:
        print(f"    {key!r}: {weights[key]:.3f},")


def _run_search(args: argparse.Namespace) -> None:
    rows = _load_cache(args.fixture, args.limit)
    rng = random.Random(args.seed)

    baseline = _objective(rows, ranking.PRIOR_WEIGHTS)
    print(f"n_topics = {len(rows)} (cached candidates, no API calls in this phase)")
    print(f"baseline PRIOR_WEIGHTS: mean ndcg_at_10_pool = {baseline:.4f}\n")

    print(f"cross-validating ({args.folds} folds x {args.restarts} restarts) - "
          f"held-out estimate of what a re-fit generalises to...")
    held_out = _cross_validate(rows, args.folds, args.restarts, rng)
    cv_mean = statistics.mean(held_out)
    print(f"cross-validated mean ndcg_at_10_pool = {cv_mean:.4f}  "
          f"(per-fold: {[round(s, 3) for s in held_out]})\n")

    print("fitting final weights on the full fixture...")
    best_weights, in_sample = _search(rows, args.restarts, rng)
    print(f"in-sample mean ndcg_at_10_pool = {in_sample:.4f}  (baseline {baseline:.4f})")

    gap = in_sample - cv_mean
    print(f"\nin-sample vs cross-validated gap = {gap:.4f} "
          f"({'small - the direction of the change looks real' if gap < 0.05 else 'large - treat the fitted weights as a hypothesis, not a result'})")

    print("\nsuggested PRIOR_WEIGHTS (review before pasting into ranking.py):")
    _print_weights(best_weights)


def main() -> None:
    parser = argparse.ArgumentParser()
    harness.add_standard_args(parser, _FIXTURE)
    parser.add_argument("--dump", action="store_true",
                         help="Run the pipeline and cache per-candidate signals. Costs real API calls.")
    parser.add_argument("--restarts", type=int, default=_RESTARTS)
    parser.add_argument("--folds", type=int, default=_FOLDS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.dump:
        _run_dump(args)
    else:
        _run_search(args)


if __name__ == "__main__":
    main()
