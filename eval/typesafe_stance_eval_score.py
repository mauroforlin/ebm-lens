"""Does TypeSafe's Score primitive agree with SciFact's expert labels - and
does its ordinal number carry information Choice's flat three-way pick
doesn't?

Sibling of `typesafe_stance_eval.py` (which grades the Choice primitive that
`claim_verification.py` actually uses in production): same SciFact rows
(`eval/_scifact_rows.py`), same evidence substrate, same gold labels, but a
Score question instead of a Choice question - `criteria` given low-to-high as
an ordered five-point legend rather than three unordered categories. No
pipeline stage uses Score today; this eval exists to answer, cheaply (no
OpenRouter calls, same as the Choice eval), whether it should.

Score's `score` field is not a bucketed integer - it is the probability-
weighted mean over the legend's indices (0-based), so a call answered mostly
"strongly contradicts" with a little mass on "somewhat contradicts" comes
back as e.g. 0.22, not 0 or 1 (see typesafe_client.py's ScoreAnswer). That
value is rounded to its nearest legend index and mapped to a gold label
below - a coarser read than the raw float, but the one comparable to
Choice's three-way accuracy. `confidence` is graded the same way as the
Choice eval's calibration table, since Score carries the same computed
confidence field Choice does.

Usage:
    python eval/typesafe_stance_eval_score.py [--limit N] [--fresh]
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
from app.core.typesafe_client import ask_score
from eval._scifact_rows import build_article, load_rows

_GOLD_LABELS = ("SUPPORT", "CONTRADICT", "NOINFO")

# Low-to-high, as Score requires. Index -> gold label for the rounded score.
_LEGEND = [
    "strongly contradicts the claim",
    "somewhat contradicts the claim",
    "does not address the claim either way",
    "somewhat supports the claim",
    "strongly supports the claim",
]
_INDEX_TO_GOLD = {0: "CONTRADICT", 1: "CONTRADICT", 2: "NOINFO", 3: "SUPPORT", 4: "SUPPORT"}

_INSTRUCTIONS = (
    "Rate how the source evidence relates to the claim, from strongly "
    "contradicting it to strongly supporting it."
)

_CONFIDENCE_BUCKETS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.001)]


def _evaluate(row: dict) -> dict:
    settings = get_settings()
    stats = JobStats()
    pairs = []
    for doc in row["docs"]:
        article = build_article(doc)
        try:
            answer = ask_score(
                settings=settings,
                state={"claim": row["claim"], "source_evidence": article.full_summary},
                instructions=_INSTRUCTIONS,
                criteria=_LEGEND,
                purpose="typesafe_stance_eval_score",
                job_stats=stats,
            )
        except Exception as exc:
            pairs.append({
                "doc_id": doc["doc_id"], "gold": doc["label"],
                "predicted": "error", "predicted_mapped": "ERROR",
                "score": None, "confidence": None, "error": str(exc),
            })
            continue
        rounded = round(answer.score)
        pairs.append({
            "doc_id": doc["doc_id"],
            "gold": doc["label"],
            "predicted": answer.legend.get(rounded, str(rounded)),
            "predicted_mapped": _INDEX_TO_GOLD.get(rounded, "ERROR"),
            "score": answer.score,
            "confidence": answer.confidence,
        })
    return {"pairs": pairs, "cost_usd": round(stats.to_dict()["total_cost_usd"], 8)}


def _precision_recall_f1(confusion: dict[str, dict[str, int]], label: str) -> tuple[float, float, float]:
    tp = confusion[label][label]
    predicted_as = sum(confusion[g][label] for g in _GOLD_LABELS)
    actual = sum(confusion[label].values())
    precision = tp / predicted_as if predicted_as else 0.0
    recall = tp / actual if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["error"]]
    all_pairs = [p for r in ok for p in r["pairs"]]
    errors = [p for p in all_pairs if p["predicted"] == "error"]
    pairs = [p for p in all_pairs if p["predicted"] != "error"]

    confusion: dict[str, dict[str, int]] = {g: dict.fromkeys(_GOLD_LABELS, 0) for g in _GOLD_LABELS}
    for p in pairs:
        confusion[p["gold"]][p["predicted_mapped"]] += 1

    per_class = {}
    for label in _GOLD_LABELS:
        precision, recall, f1 = _precision_recall_f1(confusion, label)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}

    accuracy = sum(confusion[g][g] for g in _GOLD_LABELS) / len(pairs) if pairs else 0.0

    calibration = {}
    for lo, hi in _CONFIDENCE_BUCKETS:
        bucket = [p for p in pairs if p["confidence"] is not None and lo <= p["confidence"] < hi]
        acc = statistics.mean(p["predicted_mapped"] == p["gold"] for p in bucket) if bucket else None
        calibration[f"{lo:.2f}-{hi:.2f}"] = {"n": len(bucket), "accuracy": acc}

    # Mean raw score per gold label - does the ordinal number itself move in
    # the right direction even when the rounded 3-way call is wrong?
    mean_score_by_gold = {}
    for label in _GOLD_LABELS:
        scored = [p["score"] for p in pairs if p["gold"] == label and p["score"] is not None]
        mean_score_by_gold[label] = statistics.mean(scored) if scored else None

    return {
        "n_rows": len(rows),
        "n_errors_rows": len(rows) - len(ok),
        "n_pairs": len(pairs),
        "n_typesafe_errors": len(errors),
        "accuracy": accuracy,
        "macro_f1": statistics.mean(c["f1"] for c in per_class.values()) if per_class else 0.0,
        "per_class": per_class,
        "confusion": confusion,
        "calibration_by_confidence": calibration,
        "mean_score_by_gold": mean_score_by_gold,
        "mean_cost_usd": statistics.mean(r["cost_usd"] for r in ok) if ok else 0.0,
        "total_cost_usd": sum(r.get("cost_usd", 0.0) for r in rows),
    }


def _describe(row: dict, metrics: dict) -> str:
    votes = ", ".join(f"{p['gold']}->score={p['score']}" for p in metrics.get("pairs", []))
    return f"[{votes}]  {row['claim'][:70]}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="cap the number of claims loaded")
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore any existing progress file and start over.",
    )
    args = parser.parse_args()

    fixture_stem = "typesafe_stance_score_full"
    progress_path = harness.progress_path(Path("scifact_full.jsonl"), stem=fixture_stem)
    summary_path = harness.summary_path(Path("scifact_full.jsonl"), stem=fixture_stem)
    if args.fresh and progress_path.exists():
        progress_path.unlink()

    done = harness.load_progress(progress_path)
    rows = load_rows(args.limit)

    harness.run_rows(
        rows, done, progress_path, summary_path,
        evaluate=_evaluate, summarise=_summarise, describe=_describe,
        fixture_name="scifact train+dev (full, Score primitive)",
    )


if __name__ == "__main__":
    main()
