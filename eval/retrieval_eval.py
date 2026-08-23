"""Retrieval + rerank eval: does EBM Lens find and rank the right documents?

Runs the real discovery and ranking stages (`topic_analysis.analyse_topic`,
`orchestrator._discover`, `orchestrator._build_signals`,
`selection.candidate_prior`) against a fixture's human-judged relevant-document
set - the same code path production uses, minus summarisation/synthesis,
which this eval does not score and would only add LLM cost to.

Fixture-agnostic: `--fixture` selects which judged set to grade against, and
the gold-id field present on each row (`gold_pmids`, see `_GOLD_FIELDS`) picks
the matching id-extraction pattern. Gold is normalised at load time to a
graded `{doc_id: relevance}` map (`_as_grades`) rather than a flat id list, so
the metrics below are graded-relevance-shaped even though BioASQ's own gold is
binary (every id gets relevance 1) - a future fixture with real relevance
levels needs no metric changes, only a new entry in `_GOLD_FIELDS`.

Grading is restricted to the *gradable subset* of the ranked pool: a pool item
whose URL yields no id of the fixture's `id_type` (a Wikipedia page against
BioASQ's PMIDs) cannot be judged relevant or non-relevant by the gold set at
all - it was simply never considered. Scoring it as non-relevant would assert
something the ground truth never said. `precision_at_10` / `ndcg_at_10` /
`mrr` are computed over this gradable subset (with `n_gradable`,
`gradable_share` reported so the subset size is visible); `recall_at_10` /
`recall_at_20` stay computed over the full ranked list, unchanged, since
that's what a user's shortlist actually contains.

Two nDCGs are reported for the same reason `pool_recall` and `recall_at_10`
diagnose different failures: `ndcg_at_10` grades against the *entire* gold set,
while `ndcg_at_10_pool` grades against only the grades the pool actually
retrieved - 1.0 there means "given what discovery found, ranking ordered it
perfectly". A gap between the two points at discovery; a low `ndcg_at_10_pool`
alone points at `ranking.py`/`selection.py`. `pool_recall_ceiling =
min(n_gradable, n_gold) / n_gold` is the maximum `pool_recall` could reach
given how much of the gold set the pool even had room to find, and
`mean_pool_recall_vs_ceiling` in the summary says how close the pipeline gets
to that ceiling - the more honest read of "how good is retrieval here" than
`pool_recall` alone whenever `n_gold` runs larger than the ranked pool.

Each topic's result is appended to a progress file as soon as it's done, and
the summary file is rewritten after every topic. Re-running the same
--fixture skips whatever is already in the progress file, so a run can be
Ctrl-C'd and resumed later without losing what it already paid for.
``--fresh`` discards existing progress and starts over.

Usage:
    python eval/retrieval_eval.py [--limit N] [--fixture PATH] [--fresh]
"""
from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eval._harness as harness
from app.config import get_settings
from app.core.job_stats import JobStats
from app.pipeline import orchestrator, selection, topic_analysis
from app.pipeline.dedup import deduplicate

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "bioasq_retrieval.jsonl"

# Which gold-id field a fixture row carries determines what kind of document
# id (and therefore which URL patterns) it's graded against.
_GOLD_FIELDS = {"gold_pmids": "pmid"}

_ID_PATTERNS: dict[str, list[re.Pattern]] = {
    "pmid": [
        re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)"),
        re.compile(r"ncbi\.nlm\.nih\.gov/pubmed/(\d+)"),
        re.compile(r"europepmc\.org/article/MED/(\d+)"),
    ],
}

_EMPTY_METRICS = {
    "n_discovered": 0, "n_ranked": 0, "n_gradable": 0, "gradable_share": 0.0, "n_gold": 0,
    "pool_recall": 0.0, "pool_recall_ceiling": 0.0,
    "recall_at_10": 0.0, "recall_at_20": 0.0,
    "precision_at_10": 0.0, "ndcg_at_10": 0.0, "ndcg_at_10_pool": 0.0,
    "mrr": 0.0, "cost_usd": 0.0,
}


def extract_id(url: str, id_type: str) -> str | None:
    for pattern in _ID_PATTERNS[id_type]:
        match = pattern.search(url or "")
        if match:
            return match.group(1)
    return None


def _as_grades(raw: list | dict) -> dict[str, int]:
    """Normalise a fixture's gold field to ``{doc_id: relevance_grade}``.

    BioASQ's `gold_pmids` is a flat list - every id equally relevant, grade 1.
    A dict is passed through as-is, so a fixture with real relevance levels
    can be added later without touching any metric below.
    """
    if isinstance(raw, dict):
        return {doc_id: int(grade) for doc_id, grade in raw.items()}
    return dict.fromkeys(raw, 1)


def _dcg(grades: list[int], k: int) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))


def _ndcg(ranked_grades: list[int], ideal_grades: list[int], k: int) -> float:
    idcg = _dcg(sorted(ideal_grades, reverse=True), k)
    return _dcg(ranked_grades, k) / idcg if idcg else 0.0


def _gradable_ranking(ranked: list, id_type: str) -> list[str]:
    """The ranked pool's ids, in ranked order, dropping items the gold set can't judge."""
    return [doc_id for doc_id in (extract_id(r.url, id_type) for r in ranked) if doc_id]


def _load_fixture(path: Path, limit: int | None) -> list[dict]:
    rows = harness.load_fixture(path, limit)
    for row in rows:
        gold_field = next((f for f in _GOLD_FIELDS if f in row), None)
        if gold_field is None:
            raise ValueError(f"fixture row {row.get('id')!r} has none of {sorted(_GOLD_FIELDS)}")
        row["gold"], row["id_type"] = _as_grades(row[gold_field]), _GOLD_FIELDS[gold_field]
    return rows


def _evaluate(question: str, context, spec, gold: dict[str, int], id_type: str) -> dict:
    settings = get_settings()
    stats = JobStats()
    n_gold = len(gold)

    discovery = orchestrator._discover(question, spec, context, settings, stats)
    discovery.evidence = deduplicate(discovery.evidence)
    pool = discovery.evidence
    n_discovered = len(pool)

    if not pool:
        empty = dict(_EMPTY_METRICS)
        empty["n_gold"] = n_gold
        empty["cost_usd"] = stats.to_dict()["total_cost_usd"]
        return empty

    discovered_ids = {extract_id(r.url, id_type) for r in pool} - {None}
    pool_recall = len(set(gold) & discovered_ids) / n_gold if n_gold else 0.0

    signals = orchestrator._build_signals(question, spec, discovery, settings, stats)
    n_ranked = len(pool)  # _build_signals truncates `pool` in place - measure after
    ranked = sorted(pool, key=lambda r: selection.candidate_prior(r, signals), reverse=True)
    ranked_ids = [extract_id(r.url, id_type) for r in ranked]
    gradable = [doc_id for doc_id in ranked_ids if doc_id]
    n_gradable = len(gradable)

    def grade(doc_id: str | None) -> int:
        return gold.get(doc_id, 0)

    hits_at_10 = {doc_id for doc_id in ranked_ids[:10] if doc_id in gold}
    hits_at_20 = {doc_id for doc_id in ranked_ids[:20] if doc_id in gold}

    # Rank of the first relevant hit as the user experiences it - counting
    # every ranked slot, not just the gradable ones, since an off-topic
    # result sitting above the first real hit still delays finding it.
    first_hit_rank = next(
        (rank for rank, doc_id in enumerate(ranked_ids, start=1) if grade(doc_id) >= 1), None,
    )

    gradable_top10 = gradable[:10]
    precision_at_10 = (
        sum(1 for doc_id in gradable_top10 if grade(doc_id) >= 1) / min(10, n_gradable)
        if n_gradable else 0.0
    )
    ranked_grades_10 = [grade(doc_id) for doc_id in gradable_top10]
    ndcg_at_10 = _ndcg(ranked_grades_10, list(gold.values()), 10)
    ndcg_at_10_pool = _ndcg(ranked_grades_10, [grade(doc_id) for doc_id in gradable], 10)

    return {
        "n_discovered": n_discovered,
        "n_ranked": n_ranked,
        "n_gradable": n_gradable,
        "gradable_share": n_gradable / n_ranked if n_ranked else 0.0,
        "n_gold": n_gold,
        "pool_recall": pool_recall,
        "pool_recall_ceiling": min(n_gradable, n_gold) / n_gold if n_gold else 0.0,
        "recall_at_10": len(hits_at_10) / n_gold if n_gold else 0.0,
        "recall_at_20": len(hits_at_20) / n_gold if n_gold else 0.0,
        "precision_at_10": precision_at_10,
        "ndcg_at_10": ndcg_at_10,
        "ndcg_at_10_pool": ndcg_at_10_pool,
        "mrr": 1.0 / first_hit_rank if first_hit_rank else 0.0,
        "cost_usd": stats.to_dict()["total_cost_usd"],
    }


def _evaluate_row(row: dict) -> dict:
    settings = get_settings()
    analysis_stats = JobStats()
    context, spec = topic_analysis.analyse_topic(row["question"], settings, job_stats=analysis_stats)
    metrics = _evaluate(row["question"], context, spec, row["gold"], row["id_type"])
    metrics["cost_usd"] = round(metrics["cost_usd"] + analysis_stats.to_dict()["total_cost_usd"], 8)
    return metrics


def _summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["error"]]

    def mean(key: str) -> float:
        return statistics.mean(r[key] for r in ok) if ok else 0.0

    ceiling_ratios = [r["pool_recall"] / r["pool_recall_ceiling"] for r in ok if r["pool_recall_ceiling"] > 0]

    return {
        "n_topics": len(rows),
        "n_errors": len(rows) - len(ok),
        "mean_pool_recall": mean("pool_recall"),
        "mean_pool_recall_ceiling": mean("pool_recall_ceiling"),
        "mean_pool_recall_vs_ceiling": statistics.mean(ceiling_ratios) if ceiling_ratios else 0.0,
        "mean_recall_at_10": mean("recall_at_10"),
        "mean_recall_at_20": mean("recall_at_20"),
        "mean_precision_at_10": mean("precision_at_10"),
        "mean_ndcg_at_10": mean("ndcg_at_10"),
        "mean_ndcg_at_10_pool": mean("ndcg_at_10_pool"),
        "mean_mrr": mean("mrr"),
        "mean_gradable_share": mean("gradable_share"),
        "zero_pool_recall_count": sum(1 for r in ok if r["pool_recall"] == 0.0),
        "mean_cost_usd": mean("cost_usd"),
        "total_cost_usd": sum(r["cost_usd"] for r in rows),
    }


def _describe(row: dict, metrics: dict) -> str:
    return (
        f"{row['type']:8s} pool_recall={metrics['pool_recall']:.2f} "
        f"ndcg@10={metrics['ndcg_at_10']:.2f} mrr={metrics['mrr']:.2f}  {row['question'][:60]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    harness.add_standard_args(parser, _FIXTURE)
    args = parser.parse_args()

    progress_path = harness.progress_path(args.fixture)
    summary_path = harness.summary_path(args.fixture)
    if args.fresh and progress_path.exists():
        progress_path.unlink()

    done = harness.load_progress(progress_path)
    rows = _load_fixture(args.fixture, args.limit)

    harness.run_rows(
        rows, done, progress_path, summary_path,
        evaluate=_evaluate_row, summarise=_summarise, describe=_describe,
        fixture_name=args.fixture.name,
    )


if __name__ == "__main__":
    main()
