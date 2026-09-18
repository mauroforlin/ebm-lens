"""Does a plain structured-output call to the model EBM Lens already uses for
a similar judgment do just as well as TypeSafe's Choice primitive, on the
same SciFact task?

`typesafe_stance_eval.py` grades `claim_verification.py`'s TypeSafe Choice
call against SciFact. This script asks the same three-way question (does the
evidence support/contradict/say nothing about the claim) about the same rows
(`eval/_scifact_rows.py`), but through `app/core/llm_client.py::generate_json`
against `google/gemini-2.5-flash` - the model `synthesis.py` already uses for
`related_articles_stance`, a comparably narrow stance judgment (see that
purpose's comment in llm_client.py: "getting a source's stance backwards is
the pipeline's worst possible error"). If EBM Lens's own existing model, asked
directly with no new vendor, does about as well as TypeSafe's dedicated
primitive, that is the real answer to "was TypeSafe actually necessary here",
not a comparison against an unrelated 2020 paper's fine-tuned baseline.

Unlike the TypeSafe evals, this one has a real, non-zero OpenRouter cost -
`job_stats` records the actual dollar cost per call here, not a $0 floor.

Confidence is self reported in the same JSON call (there is no separate
computed-from-logits number the way TypeSafe's Choice gives), which is
itself part of what is being compared: is a self reported number from a
prompted call as informative as TypeSafe's computed one.

Usage:
    python eval/openrouter_stance_eval.py [--limit N] [--fresh]
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
from app.pipeline.claim_verification import _LLM_SYSTEM, LLM_BACKEND, thresholds_for
from eval._scifact_rows import build_article, load_rows

_GOLD_LABELS = ("SUPPORT", "CONTRADICT", "NOINFO")
_RELATION_TO_GOLD = {"supports": "SUPPORT", "contradicts": "CONTRADICT", "says_nothing": "NOINFO"}

# The model synthesis.py already uses for related_articles_stance - the
# closest existing production analog to this judgment. Not the cheap default
# (llm_model), on purpose: this is the fair comparison, "the model already
# judged fit for a stance call like this", not "the cheapest model available".
_MODEL = "google/gemini-2.5-flash"

# The backend under test owns the prompt: _LLM_THRESHOLDS was measured
# through this exact wording, so a copy here that drifted would silently
# stop grading the thing production runs.
_SYSTEM = _LLM_SYSTEM

# This script grades the LLM backend, so it reads the LLM backend's bars -
# not TypeSafe's, which is the exact mix-up the per-backend split exists
# to prevent.
_BARS = thresholds_for(LLM_BACKEND)

_CONFIDENCE_BUCKETS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.001)]


def _evaluate(row: dict) -> dict:
    settings = get_settings()
    stats = JobStats()
    pairs = []
    for doc in row["docs"]:
        article = build_article(doc)
        prompt = f"Claim: {row['claim']}\n\nEvidence: {article.full_summary}"
        try:
            result = generate_json(
                settings=settings, prompt=prompt, system_instruction=_SYSTEM,
                purpose="openrouter_stance_eval", job_stats=stats,
                model_override=_MODEL,
            )
            relation = str(result.get("relation", "")).strip().lower()
            confidence = float(result.get("confidence"))
            if relation not in _RELATION_TO_GOLD:
                raise ValueError(f"unexpected relation: {relation!r}")
        except Exception as exc:
            pairs.append({
                "doc_id": doc["doc_id"], "gold": doc["label"],
                "predicted": "error", "predicted_mapped": "ERROR",
                "confidence": None, "error": str(exc),
            })
            continue
        pairs.append({
            "doc_id": doc["doc_id"],
            "gold": doc["label"],
            "predicted": relation,
            "predicted_mapped": _RELATION_TO_GOLD[relation],
            "confidence": confidence,
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

    # The decision-relevant number, mirroring apply_verdicts' own rule: how
    # often a confident call is right, each relation measured at the bar that
    # relation actually crosses in production. Read from
    # claim_verification.py rather than hardcoded, so this number cannot
    # silently stop describing the code it is supposed to be grading.
    #
    # Precision here is base-rate dependent, and SciFact's base rates are not
    # the pipeline's - see CONTRADICT_FLAG_PROBABILITY's own comment in
    # claim_verification.py for that arithmetic, and why it stopped
    # justifying a deletion.
    high_conf_precision = {}
    thresholds = [
        ("contradicts", "CONTRADICT", _BARS.contradict_flag),
        ("says_nothing", "NOINFO", _BARS.unsupported_reject),
        ("supports", "SUPPORT", _BARS.support_keep),
    ]
    for relation, mapped, threshold in thresholds:
        high = [
            p for p in pairs
            if p["predicted"] == relation and (p["confidence"] or 0) >= threshold
        ]
        correct = sum(1 for p in high if p["gold"] == mapped)
        high_conf_precision[relation] = {
            "n": len(high),
            "threshold": threshold,
            "precision": correct / len(high) if high else None,
        }

    return {
        "n_rows": len(rows),
        "n_errors_rows": len(rows) - len(ok),
        "n_pairs": len(pairs),
        "n_call_errors": len(errors),
        "accuracy": accuracy,
        "macro_f1": statistics.mean(c["f1"] for c in per_class.values()) if per_class else 0.0,
        "per_class": per_class,
        "confusion": confusion,
        "calibration_by_self_reported_confidence": calibration,
        "high_confidence_precision": high_conf_precision,
        "mean_cost_usd": statistics.mean(r["cost_usd"] for r in ok) if ok else 0.0,
        "total_cost_usd": sum(r.get("cost_usd", 0.0) for r in rows),
    }


def _describe(row: dict, metrics: dict) -> str:
    votes = ", ".join(f"{p['gold']}->{p['predicted']}" for p in metrics.get("pairs", []))
    return f"[{votes}]  {row['claim'][:70]}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="cap the number of claims loaded")
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore any existing progress file and start over.",
    )
    args = parser.parse_args()

    fixture_stem = "openrouter_stance_full"
    progress_path = harness.progress_path(Path("scifact_full.jsonl"), stem=fixture_stem)
    summary_path = harness.summary_path(Path("scifact_full.jsonl"), stem=fixture_stem)
    if args.fresh and progress_path.exists():
        progress_path.unlink()

    done = harness.load_progress(progress_path)
    rows = load_rows(args.limit)

    harness.run_rows(
        rows, done, progress_path, summary_path,
        evaluate=_evaluate, summarise=_summarise, describe=_describe,
        fixture_name="scifact train+dev (full, OpenRouter/Gemini comparison)",
    )


if __name__ == "__main__":
    main()
