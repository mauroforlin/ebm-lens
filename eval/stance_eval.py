"""Stance + grounding eval: does `summarise_sources` read evidence correctly?

Grades `synthesis.summarise_sources`'s per-source appraisal against SciFact's
expert-labelled claim/document pairs: `finding_direction` (supports /
contradicts / mixed / neutral) against SUPPORT / CONTRADICT / NOINFO, and
`relevance_score` against the same three-way split.

A citation-grounding eval built the other way - hand the model a claim plus
distractor abstracts, run the full `synthesise` pass, and check whether its
generated claims cite the right sources - was designed and rejected. Three
reasons, any one fatal: `synthesise` is instructed to synthesise every source
it's handed, and distractors only pass the relevance gate because *this eval*
sets `relevance_score`, so citing them is correct model behaviour that a
precision-of-citations metric would then punish; production filters
distractors out via `MIN_SYNTHESIS_RELEVANCE` before `synthesise` ever sees
them, so the scenario can't occur outside this eval; and `synthesise` writes
its *own* claims rather than scoring a given one, so "should claim k cite doc
j" has no SciFact gold to check against - answering it needs a semantic judge,
which is the entailment machinery whose absence this eval exists to measure.
The metric would measure itself. (A judge-free conflict-detection test can't
substitute either: no SciFact claim has both SUPPORT and CONTRADICT evidence
docs, so the corpus has no within-claim disagreement case to test against.)

Grounding is measured one stage upstream instead, where it's judge-free and
sound: `_source_lines` renders `key_finding or full_summary` as the entire
evidentiary substrate every downstream claim is built from, so if the
appraiser attributes a supporting/contradicting finding to a NOINFO document -
one the claim's own authors cited but that SciFact's annotators found no
evidence in - every claim built on it downstream would be ungrounded. That's
`hallucinated_finding_rate`, this eval's headline number: judge-free, and it
uses SciFact's NOINFO pairs as the hard negatives their designers intended.

What's actually verifiable about `synthesise` itself (bounds-checking on
cited indices, the strength clamp, the low-evidence fallback) is covered in
`tests/test_synthesis_grounding.py` instead - pure-function tests, no
network, no LLM cost.

Sources are built with `source_type="scifact"` and a `.invalid` URL (RFC 2606
reserved, never resolves), which guarantees `fetch_full_text`'s allowlist
check short-circuits with zero network I/O - this eval is hermetic apart from
the OpenRouter appraisal call itself. `summary_language="en"` overrides the
production default so `key_finding` stays readable for debugging (`finding_
direction` is a language-independent id either way). `pico=None`, so
`directness` reads "unclear" almost everywhere by design - it is not graded
here.

Usage:
    python eval/stance_eval.py [--limit N] [--fixture PATH] [--fresh]
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
from app.pipeline.synthesis import (
    MIN_SYNTHESIS_RELEVANCE,
    clamp_relevance,
    read_direction,
    summarise_sources,
)
from app.sources.base import SourceResult

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "scifact_stance.jsonl"

_GOLD_TO_DIRECTION = {"SUPPORT": "supports", "CONTRADICT": "contradicts", "NOINFO": "neutral"}
_DIRECTIONS = ("supports", "contradicts", "mixed", "neutral")


def _build_source(doc: dict) -> SourceResult:
    return SourceResult(
        title=doc["title"],
        url=f"https://scifact.invalid/doc/{doc['doc_id']}",
        snippet=doc["abstract"][0] if doc["abstract"] else "",
        content=" ".join(doc["abstract"]),
        source_type="scifact",
        reliability_tier=2,
        publication_types=["Journal Article"],
    )


def _evaluate(row: dict) -> dict:
    settings = get_settings()
    stats = JobStats()
    sources = [_build_source(d) for d in row["docs"]]

    raw = summarise_sources(row["claim"], sources, settings, summary_language="en", job_stats=stats, pico=None)

    pairs = []
    for i, doc in enumerate(row["docs"]):
        appraisal = raw.get(i, {})
        relevance = clamp_relevance(appraisal.get("relevance_score"))
        pairs.append({
            "doc_id": doc["doc_id"],
            "gold": doc["label"],
            "predicted": read_direction(appraisal.get("finding_direction")),
            "relevance_score": relevance,
            "key_finding": appraisal.get("key_finding", ""),
            "directness": appraisal.get("directness", ""),
        })

    return {"pairs": pairs, "cost_usd": round(stats.to_dict()["total_cost_usd"], 8)}


def _summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["error"]]
    pairs = [p for r in ok for p in r["pairs"]]

    confusion: dict[str, dict[str, int]] = {g: dict.fromkeys(_DIRECTIONS, 0) for g in _GOLD_TO_DIRECTION}
    for p in pairs:
        confusion[p["gold"]][p["predicted"]] += 1

    def n_gold(label: str) -> int:
        return sum(confusion[label].values())

    def precision_recall_f1(direction: str, gold_label: str) -> tuple[float, float, float]:
        tp = confusion[gold_label][direction]
        predicted_as = sum(confusion[g][direction] for g in confusion)
        actual = n_gold(gold_label)
        precision = tp / predicted_as if predicted_as else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return precision, recall, f1

    per_class = {}
    for gold_label, direction in _GOLD_TO_DIRECTION.items():
        precision, recall, f1 = precision_recall_f1(direction, gold_label)
        per_class[gold_label] = {"precision": precision, "recall": recall, "f1": f1}

    accuracy = (
        sum(confusion[g][_GOLD_TO_DIRECTION[g]] for g in _GOLD_TO_DIRECTION) / len(pairs) if pairs else 0.0
    )
    mixed_by_gold = {g: (confusion[g]["mixed"] / n_gold(g) if n_gold(g) else 0.0) for g in confusion}

    noinfo_pairs = [p for p in pairs if p["gold"] == "NOINFO"]
    hallucinated = [
        p for p in noinfo_pairs
        if p["key_finding"] and p["predicted"] != "neutral" and p["relevance_score"] > MIN_SYNTHESIS_RELEVANCE
    ]
    below_gate = [p for p in noinfo_pairs if p["relevance_score"] <= MIN_SYNTHESIS_RELEVANCE]

    relevance_by_label = {
        g: statistics.mean(p["relevance_score"] for p in pairs if p["gold"] == g) if n_gold(g) else 0.0
        for g in confusion
    }

    return {
        "n_topics": len(rows),
        "n_errors": len(rows) - len(ok),
        "n_pairs": len(pairs),
        "accuracy": accuracy,
        "macro_f1": statistics.mean(c["f1"] for c in per_class.values()) if per_class else 0.0,
        "per_class": per_class,
        "confusion": confusion,
        "contradict_recall": per_class.get("CONTRADICT", {}).get("recall", 0.0),
        "mixed_rate": statistics.mean(p["predicted"] == "mixed" for p in pairs) if pairs else 0.0,
        "mixed_rate_by_gold": mixed_by_gold,
        "hallucinated_finding_rate": len(hallucinated) / len(noinfo_pairs) if noinfo_pairs else 0.0,
        "n_hallucinated": len(hallucinated),
        "n_noinfo_pairs": len(noinfo_pairs),
        "noinfo_below_gate_rate": len(below_gate) / len(noinfo_pairs) if noinfo_pairs else 0.0,
        "mean_relevance_by_gold_label": relevance_by_label,
        "mean_cost_usd": statistics.mean(r["cost_usd"] for r in ok) if ok else 0.0,
        "total_cost_usd": sum(r["cost_usd"] for r in rows),
    }


def _describe(row: dict, metrics: dict) -> str:
    votes = ", ".join(f"{p['gold']}->{p['predicted']}" for p in metrics["pairs"])
    return f"{row['gold_label']:10s} [{votes}]  {row['claim'][:60]}"


def main() -> None:
    parser = argparse.ArgumentParser()
    harness.add_standard_args(parser, _FIXTURE)
    args = parser.parse_args()

    progress_path = harness.progress_path(args.fixture)
    summary_path = harness.summary_path(args.fixture)
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
