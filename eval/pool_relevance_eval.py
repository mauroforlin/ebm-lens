"""Pool relevance eval: how much of the discovery pool is actually signal?

`retrieval_eval.py` scores exact-PMID recall against BioASQ's gold set, which
conflates two different failures: the pool missing the gold papers, and the
pool being full of off-topic noise. A topic can score zero pool_recall while
still being full of *genuinely relevant* evidence the gold set just didn't
happen to name (a review article covering the same fact as an older primary
study, say) - exact-PMID recall can't tell that apart from a pool that is
actually garbage.

This script judges the other axis: for each candidate the discovery pool
actually returned, is it topically relevant to the question at all? One LLM
call per topic classifies the whole pool at once (cheap, and the model sees
all candidates together for consistent judgments), and results are broken
down by source_type - so a provider that is mostly noise for a given kind of
question shows up directly, rather than being averaged away.

**No aggregate relevance number is reported.** Discovery is deliberately
recall-oriented - it casts a wide net across many source types and leaves
precision to ranking/selection downstream - so a low "% of the pool that is
relevant" is partly the design working as intended, not a regression by
itself; there's no target to chase it towards; a rising or falling aggregate
doesn't say whether the system got better or worse. Two things from the same
judgments stay actionable despite that: `relevance_by_source_type` catches a
provider contributing near-zero signal question after question (fix its
query construction or drop it), and `pool_size_relevance_correlation` checks
whether widening the pool dilutes signal faster than it adds it - relevant
because a diluted pool is more for ranking to sift through, raising the
chance the few correct documents get buried. What should be chased towards a
target is precision *after* ranking (see `retrieval_eval.py`'s
`precision_at_10` / `ndcg_at_10`), since that's the stage meant to be
precise.

Same resumable-progress pattern as retrieval_eval.py: each topic's judgment
is appended to disk as soon as it's done, so a run can be interrupted and
resumed without re-paying for work already done.

Usage:
    python eval/pool_relevance_eval.py [--limit N] [--fixture PATH] [--fresh]
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eval._harness as harness
from app.config import get_settings
from app.core.job_stats import JobStats
from app.core.llm_client import generate_json
from app.pipeline import orchestrator, topic_analysis
from app.pipeline.dedup import deduplicate

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "bioasq_retrieval.jsonl"

_JUDGE_SYSTEM = """\
You judge search results for topical relevance to a biomedical question.

A candidate is RELEVANT only if it specifically discusses the entities, drug,
gene, disease or claim the question is about - not merely if it shares a
keyword or a broad subject area (e.g. "gene expression", "cardiac") with it.
A candidate about a different gene, different drug, or a generic methods/
benchmark paper that happens to mention a shared term is NOT relevant.

For each numbered candidate, decide true (relevant) or false (not relevant).
Return JSON: {"judgments": [{"i": <int>, "relevant": <bool>}, ...]} covering
every candidate index exactly once.
"""

_MAX_SNIPPET_CHARS = 160


def _get_pool(question: str, settings, stats: JobStats) -> list:
    context, spec = topic_analysis.analyse_topic(question, settings, job_stats=stats)
    discovery = orchestrator._discover(question, spec, context, settings, stats)
    return deduplicate(discovery.evidence)


def _judge_pool(question: str, pool: list, settings, stats: JobStats) -> list[bool]:
    """One LLM call classifying every candidate in *pool* at once."""
    lines = [
        f"{i}. [{r.source_type}] {r.title} — {(r.content or r.snippet or '')[:_MAX_SNIPPET_CHARS]}"
        for i, r in enumerate(pool)
    ]
    prompt = f'QUESTION: "{question}"\n\nCANDIDATES:\n' + "\n".join(lines)

    result = generate_json(
        settings=settings, prompt=prompt, system_instruction=_JUDGE_SYSTEM,
        temperature=0.0, purpose="pool_relevance_judge", job_stats=stats,
    )
    judgments = result.get("judgments") if isinstance(result, dict) else None
    verdicts = [False] * len(pool)
    for j in judgments or []:
        if not isinstance(j, dict):
            continue
        i, relevant = j.get("i"), j.get("relevant")
        if isinstance(i, int) and 0 <= i < len(pool) and isinstance(relevant, bool):
            verdicts[i] = relevant
    return verdicts


def _evaluate(row: dict) -> dict:
    settings = get_settings()
    stats = JobStats()
    pool = _get_pool(row["question"], settings, stats)
    verdicts = _judge_pool(row["question"], pool, settings, stats) if pool else []

    by_source: dict[str, list[int]] = {}
    for r, relevant in zip(pool, verdicts, strict=True):
        counts = by_source.setdefault(r.source_type, [0, 0])
        counts[0] += 1
        counts[1] += int(relevant)

    return {
        "pool_size": len(pool),
        "relevant_count": sum(verdicts),
        "by_source_type": by_source,
        "cost_usd": round(stats.to_dict()["total_cost_usd"], 8),
    }


def _summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["error"] and r["pool_size"]]

    by_source: dict[str, list[int]] = {}
    for r in ok:
        for src, (n, n_rel) in r["by_source_type"].items():
            counts = by_source.setdefault(src, [0, 0])
            counts[0] += n
            counts[1] += n_rel

    sizes = [r["pool_size"] for r in ok]
    rates = [r["relevant_count"] / r["pool_size"] for r in ok]
    correlation = (
        statistics.correlation(sizes, rates)
        if len(ok) >= 2 and len(set(sizes)) > 1 and len(set(rates)) > 1
        else None
    )

    return {
        "n_topics": len(rows),
        "n_errors": len(rows) - len(ok),
        "pool_size_relevance_correlation": correlation,
        "mean_cost_usd": sum(r["cost_usd"] for r in ok) / len(ok) if ok else 0.0,
        "total_cost_usd": sum(r["cost_usd"] for r in rows),
        "relevance_by_source_type": {
            src: round(n_rel / n, 3) if n else 0.0
            for src, (n, n_rel) in sorted(by_source.items(), key=lambda kv: -kv[1][0])
        },
        "items_by_source_type": {src: n for src, (n, _) in by_source.items()},
    }


def _describe(row: dict, metrics: dict) -> str:
    if not metrics["pool_size"]:
        return "empty pool"
    return f"{metrics['relevant_count']}/{metrics['pool_size']} relevant  {row['question'][:60]}"


def main() -> None:
    parser = argparse.ArgumentParser()
    harness.add_standard_args(parser, _FIXTURE)
    args = parser.parse_args()

    progress_path = harness.progress_path(args.fixture, stem="pool_relevance")
    summary_path = harness.summary_path(args.fixture, stem="pool_relevance")
    if args.fresh and progress_path.exists():
        progress_path.unlink()

    done = harness.load_progress(progress_path)
    rows = harness.load_fixture(args.fixture, args.limit)

    harness.run_rows(
        rows, done, progress_path, summary_path,
        evaluate=_evaluate, summarise=_summarise, describe=_describe,
        fixture_name=args.fixture.name,
    )


if __name__ == "__main__":
    main()
