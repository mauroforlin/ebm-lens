"""Does TypeSafe's Noul primitive agree with SciFact's expert labels - and
does its bare 0-1 value behave like a calibrated confidence even though the
SDK gives it no confidence field at all?

Sibling of `typesafe_stance_eval.py` (Choice) and `typesafe_stance_eval_score.py`
(Score). Noul answers a single yes/no proposition as a truth value, not a
pick from several criteria - there is no native three-way question to ask it
the way Choice and Score allow. SciFact's SUPPORT/CONTRADICT/NOINFO is
reconstructed here from two Noul calls per (claim, doc) pair - "does the
evidence support the claim" and "does the evidence contradict the claim" -
still two TypeSafe calls per pair, same zero marginal cost as the other two
evals. Whichever proposition comes back truer, past a coin-flip, wins the
predicted label; if neither clears 0.5, the pair is called NOINFO.

`typesafe_client.py`'s `NoulAnswer` is unverified against `docs.typesafe.ai`
for anything beyond the bare 0-1 value (confirmed by reading the installed
SDK's response types directly, see that module's docstring) - so there is no
`confidence` to grade a calibration table against, the way the Choice and
Score evals do. `decisive_value` below is this eval's own construction, not
a TypeSafe field: the truth value of whichever proposition decided the
label (for a NOINFO call, `1 - max(supports, contradicts)`, i.e. how sure
neither strong claim held). Bucketing accuracy by *that* is the actual
question this eval exists to answer: is Noul's raw number, unlabeled as a
confidence, still informative the way Choice's and Score's real confidence
fields are.

Usage:
    python eval/typesafe_stance_eval_noul.py [--limit N] [--fresh]
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
from app.core.typesafe_client import ask_noul
from eval._scifact_rows import build_article, load_rows

_GOLD_LABELS = ("SUPPORT", "CONTRADICT", "NOINFO")

_SUPPORTS_INSTRUCTIONS = "Does the source evidence support the claim being true?"
_SUPPORTS_CRITERIA = {
    "true": "The evidence states the claim or directly implies it is true.",
    "false": "The evidence does not establish that the claim is true.",
}
_CONTRADICTS_INSTRUCTIONS = "Does the source evidence contradict the claim?"
_CONTRADICTS_CRITERIA = {
    "true": "The evidence states the opposite of the claim or implies it is false.",
    "false": "The evidence does not establish that the claim is false.",
}

_VALUE_BUCKETS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.001)]


def _classify(supports: float, contradicts: float) -> tuple[str, float]:
    if supports >= 0.5 and supports > contradicts:
        return "SUPPORT", supports
    if contradicts >= 0.5 and contradicts > supports:
        return "CONTRADICT", contradicts
    return "NOINFO", 1 - max(supports, contradicts)


def _evaluate(row: dict) -> dict:
    settings = get_settings()
    stats = JobStats()
    pairs = []
    for doc in row["docs"]:
        article = build_article(doc)
        state = {"claim": row["claim"], "source_evidence": article.full_summary}
        try:
            supports = ask_noul(
                settings=settings, state=state,
                instructions=_SUPPORTS_INSTRUCTIONS, criteria=_SUPPORTS_CRITERIA,
                purpose="typesafe_stance_eval_noul", job_stats=stats,
            ).noul
            contradicts = ask_noul(
                settings=settings, state=state,
                instructions=_CONTRADICTS_INSTRUCTIONS, criteria=_CONTRADICTS_CRITERIA,
                purpose="typesafe_stance_eval_noul", job_stats=stats,
            ).noul
        except Exception as exc:
            pairs.append({
                "doc_id": doc["doc_id"], "gold": doc["label"],
                "predicted_mapped": "ERROR", "supports": None, "contradicts": None,
                "decisive_value": None, "error": str(exc),
            })
            continue
        predicted, decisive_value = _classify(supports, contradicts)
        pairs.append({
            "doc_id": doc["doc_id"],
            "gold": doc["label"],
            "predicted_mapped": predicted,
            "supports": supports,
            "contradicts": contradicts,
            "decisive_value": decisive_value,
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
    errors = [p for p in all_pairs if p["predicted_mapped"] == "ERROR"]
    pairs = [p for p in all_pairs if p["predicted_mapped"] != "ERROR"]

    confusion: dict[str, dict[str, int]] = {g: dict.fromkeys(_GOLD_LABELS, 0) for g in _GOLD_LABELS}
    for p in pairs:
        confusion[p["gold"]][p["predicted_mapped"]] += 1

    per_class = {}
    for label in _GOLD_LABELS:
        precision, recall, f1 = _precision_recall_f1(confusion, label)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}

    accuracy = sum(confusion[g][g] for g in _GOLD_LABELS) / len(pairs) if pairs else 0.0

    calibration = {}
    for lo, hi in _VALUE_BUCKETS:
        bucket = [p for p in pairs if lo <= p["decisive_value"] < hi]
        acc = statistics.mean(p["predicted_mapped"] == p["gold"] for p in bucket) if bucket else None
        calibration[f"{lo:.2f}-{hi:.2f}"] = {"n": len(bucket), "accuracy": acc}

    return {
        "n_rows": len(rows),
        "n_errors_rows": len(rows) - len(ok),
        "n_pairs": len(pairs),
        "n_typesafe_errors": len(errors),
        "accuracy": accuracy,
        "macro_f1": statistics.mean(c["f1"] for c in per_class.values()) if per_class else 0.0,
        "per_class": per_class,
        "confusion": confusion,
        "calibration_by_decisive_value": calibration,
        "mean_cost_usd": statistics.mean(r["cost_usd"] for r in ok) if ok else 0.0,
        "total_cost_usd": sum(r.get("cost_usd", 0.0) for r in rows),
    }


def _describe(row: dict, metrics: dict) -> str:
    votes = ", ".join(
        f"{p['gold']}->{p['predicted_mapped']}({p['decisive_value']:.2f})"
        if p["decisive_value"] is not None else f"{p['gold']}->ERROR"
        for p in metrics.get("pairs", [])
    )
    return f"[{votes}]  {row['claim'][:70]}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="cap the number of claims loaded")
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore any existing progress file and start over.",
    )
    args = parser.parse_args()

    fixture_stem = "typesafe_stance_noul_full"
    progress_path = harness.progress_path(Path("scifact_full.jsonl"), stem=fixture_stem)
    summary_path = harness.summary_path(Path("scifact_full.jsonl"), stem=fixture_stem)
    if args.fresh and progress_path.exists():
        progress_path.unlink()

    done = harness.load_progress(progress_path)
    rows = load_rows(args.limit)

    harness.run_rows(
        rows, done, progress_path, summary_path,
        evaluate=_evaluate, summarise=_summarise, describe=_describe,
        fixture_name="scifact train+dev (full, Noul primitive)",
    )


if __name__ == "__main__":
    main()
