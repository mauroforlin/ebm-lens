"""Does TypeSafe's Choice call agree with SciFact's expert labels, and is its
confidence actually calibrated - at the full scale SciFact offers, not the
90-claim sample `stance_eval.py` uses?

`stance_eval.py` grades the current production judge (`summarise_sources` +
`judge_directions`, both OpenRouter LLM calls) against SciFact. This script
grades a different thing: `claim_verification.verify_claims`'s TypeSafe
Choice call, in isolation, against the same kind of gold label - but it does
not run summarise_sources first to get there. SciFact's own annotators
already marked the exact sentences (`rationale_sentences`) that back a
SUPPORT or CONTRADICT label; those sentences are the same substrate
`_evidence_block` (claim_verification.py) builds from a finding's text and
verbatim quote in production. Handing TypeSafe those sentences directly, with
no LLM extraction step in between, isolates the one question this eval
exists to answer: given the right evidence, does TypeSafe's judgment agree
with a human expert's, and does its confidence number mean what it claims to
mean. A NOINFO document has no rationale sentences by definition - its full
abstract stands in, the same fallback `_evidence_block` takes when an
article has no findings.

This is also why the eval can run on SciFact's full train+dev split (1,109
claims, not the 90 curated into `fixtures/scifact_stance.jsonl` for
`stance_eval.py`) at negligible cost: no OpenRouter call happens anywhere in
this script, only the TypeSafe calls `verify_claims` itself makes - one per
cited document, same as production. Raw SciFact data is expected, gitignored,
at eval/data/scifact/data/ (see eval/scripts/curate_scifact.py's docstring
for the download command); nothing here writes a fixture back to
eval/fixtures/, since redistributing the full set - unlike the small curated
sample - is not this project's call to make (see eval/README.md).

Confidence calibration is the number stance_eval.py has no equivalent of,
because judge_directions never had a comparable confidence field to begin
with (see typesafe_client.py, claim_verification.py's module docstrings) -
this is the first time this codebase can check, against real outside labels,
whether a TypeSafe confidence bucket's accuracy actually tracks its number.
apply_verdicts' thresholds (CONTRADICT_FLAG_PROBABILITY 0.85,
UNSUPPORTED_REJECT_PROBABILITY 0.95) and summary_verification's
FLAG_PROBABILITY (0.6) started out chosen from a handful of hand-run
examples, not from anything like this; the per-bucket accuracy below, and
the per-relation precision at the bar each threshold actually sits at, are
what they should be argued from instead.

Usage:
    python eval/typesafe_stance_eval.py [--limit N] [--fresh]
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
from app.pipeline import claim_verification
from app.pipeline.claim_verification import (
    CONTRADICT_DOUBT_PROBABILITY,
    CONTRADICT_FLAG_PROBABILITY,
    SUPPORT_KEEP_PROBABILITY,
    UNSUPPORTED_REJECT_PROBABILITY,
)
from app.schemas import Claim
from eval._scifact_rows import build_article, load_rows

_GOLD_LABELS = ("SUPPORT", "CONTRADICT", "NOINFO")
_RELATION_TO_GOLD = {"supports": "SUPPORT", "contradicts": "CONTRADICT", "says_nothing": "NOINFO"}

_CONFIDENCE_BUCKETS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.001)]


def _evaluate(row: dict) -> dict:
    settings = get_settings()
    stats = JobStats()
    articles = [build_article(d) for d in row["docs"]]
    claim = Claim(text=row["claim"], source_indices=list(range(len(articles))))

    verdicts = claim_verification.verify_claims([claim], articles, settings, job_stats=stats)

    pairs = []
    for v in verdicts:
        doc = row["docs"][v.source_index]
        pairs.append({
            "doc_id": doc["doc_id"],
            "gold": doc["label"],
            "predicted": v.relation,
            "predicted_mapped": _RELATION_TO_GOLD.get(v.relation, "ERROR"),
            "confidence": v.confidence,
            # apply_verdicts decides on this, not on the argmax above - so a
            # run that does not record it cannot grade the rule the pipeline
            # actually applies. See _summarise's rule_precision.
            "probabilities": dict(v.probabilities),
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


def _mass(pair: dict, relation: str) -> float:
    """The eval-side mirror of claim_verification._mass, over a stored pair.

    Same fallback, and the same guard against a `probabilities` dict that is
    not keyed by relation name - so this grades what apply_verdicts would
    actually have done with the verdict, including on progress files written
    before probabilities were recorded at all.
    """
    probabilities = pair.get("probabilities") or {}
    if probabilities and pair["predicted"] in probabilities:
        return probabilities.get(relation, 0.0)
    return (pair["confidence"] or 0.0) if pair["predicted"] == relation else 0.0


def _precision_at(pairs: list[dict], relation: str, gold: str, threshold: float) -> dict:
    hits = [p for p in pairs if _mass(p, relation) >= threshold]
    correct = sum(1 for p in hits if p["gold"] == gold)
    return {
        "n": len(hits),
        "threshold": threshold,
        "precision": correct / len(hits) if hits else None,
    }


def _rule_precision(pairs: list[dict]) -> dict:
    """Grade the rules apply_verdicts actually applies, at their own bars.

    `per_class` above grades the argmax, which is not what the pipeline acts
    on: every bar in claim_verification is a probability mass on one named
    relation. These are the numbers its thresholds should be argued from.

    `contradicts_doubt_band` is the one with no counterpart in the old rules
    - pairs where enough mass sits on "contradicts" to stop calling a claim
    strong, while some other relation still wins the argmax. If the gold
    label there is CONTRADICT far more often than the base rate, the band is
    catching something the argmax throws away; if it is not, the band is
    only costing claims their strength for nothing.
    """
    doubt = [
        p for p in pairs
        if CONTRADICT_DOUBT_PROBABILITY <= _mass(p, "contradicts") < CONTRADICT_FLAG_PROBABILITY
    ]
    base_rate = (
        sum(1 for p in pairs if p["gold"] == "CONTRADICT") / len(pairs) if pairs else None
    )
    return {
        "contradicts_flag": _precision_at(
            pairs, "contradicts", "CONTRADICT", CONTRADICT_FLAG_PROBABILITY,
        ),
        "says_nothing_drop": _precision_at(
            pairs, "says_nothing", "NOINFO", UNSUPPORTED_REJECT_PROBABILITY,
        ),
        "supports_keep": _precision_at(
            pairs, "supports", "SUPPORT", SUPPORT_KEEP_PROBABILITY,
        ),
        "contradicts_doubt_band": {
            "n": len(doubt),
            "range": [CONTRADICT_DOUBT_PROBABILITY, CONTRADICT_FLAG_PROBABILITY],
            "share_truly_contradicted": (
                sum(1 for p in doubt if p["gold"] == "CONTRADICT") / len(doubt) if doubt else None
            ),
            "corpus_base_rate": base_rate,
        },
        # What the old rule would have kept as support: argmax alone, no bar.
        # The gap between this and supports_keep is what SUPPORT_KEEP_
        # PROBABILITY bought.
        "supports_argmax_only": {
            "n": sum(1 for p in pairs if p["predicted"] == "supports"),
            "precision": (
                sum(1 for p in pairs if p["predicted"] == "supports" and p["gold"] == "SUPPORT")
                / sum(1 for p in pairs if p["predicted"] == "supports")
                if any(p["predicted"] == "supports" for p in pairs) else None
            ),
        },
    }


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

    # Calibration: within each confidence bucket, does accuracy actually
    # track the number - the question this eval exists to answer that
    # stance_eval.py has no equivalent field to ask.
    calibration = {}
    for lo, hi in _CONFIDENCE_BUCKETS:
        bucket = [p for p in pairs if p["confidence"] is not None and lo <= p["confidence"] < hi]
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
        "calibration_by_confidence": calibration,
        "rule_precision": _rule_precision(pairs),
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

    fixture_stem = "typesafe_stance_full"
    progress_path = harness.progress_path(Path("scifact_full.jsonl"), stem=fixture_stem)
    summary_path = harness.summary_path(Path("scifact_full.jsonl"), stem=fixture_stem)
    if args.fresh and progress_path.exists():
        progress_path.unlink()

    done = harness.load_progress(progress_path)
    rows = load_rows(args.limit)

    harness.run_rows(
        rows, done, progress_path, summary_path,
        evaluate=_evaluate, summarise=_summarise, describe=_describe,
        fixture_name="scifact train+dev (full)",
    )


if __name__ == "__main__":
    main()
