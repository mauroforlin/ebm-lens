"""Stance + grounding eval: does `summarise_sources` + `judge_directions`
read evidence correctly?

Grades the two-stage appraisal - `synthesis.summarise_sources` for evidence,
`synthesis.judge_directions` for the verdict - against SciFact's expert-
labelled claim/document pairs: `finding_direction`, collapsed from its graded
scale (`strongly_contradicts` .. `strongly_supports`, plus `mixed` and
`no_evidence`) via `synthesis.collapse_direction`, against SciFact's own
SUPPORT / CONTRADICT / NOINFO; and `relevance_score` against the same split.

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
sound: `_source_lines` renders each source's `findings` (or `full_summary`
when it has none) as the entire evidentiary substrate every downstream claim
is built from, so if `judge_directions` attributes a supporting/contradicting
finding to a NOINFO document - one the claim's own authors cited but that
SciFact's annotators found no evidence in - every claim built on it
downstream would be ungrounded. That's `hallucinated_finding_rate`, this
eval's headline number: judge-free, and it uses SciFact's NOINFO pairs as the
hard negatives their designers intended. `judge_directions` also carries its
own judge-free grounding gate (an `evidence_quote` that doesn't verify
against the source's own content is treated as no evidence, never reaching
the stance call at all) - this eval exercises that gate exactly as production
does, since it calls the real function rather than reimplementing its logic.

`summarise_sources` can return more than one finding per document (see
`synthesis.MAX_FINDINGS_PER_SOURCE`); `judge_directions` grades each one
independently, but SciFact's gold label is per document, not per finding. A
multi-finding document is collapsed to a single prediction by
`_aggregate_direction`: the first finding whose direction isn't `no_evidence`
wins, since a document that supports OR contradicts the claim anywhere in it
is not a NOINFO document. This almost never actually triggers here - each
row scopes the claim tightly enough that a document yields 0 or 1 relevant
findings in practice - but the policy is applied uniformly rather than
assumed away.

What's actually verifiable about `synthesise` itself (bounds-checking on
cited indices, the strength clamp, the low-evidence fallback) is covered in
`tests/test_synthesis_grounding.py` instead - pure-function tests, no
network, no LLM cost.

Sources are built with `source_type="scifact"` and a `.invalid` URL (RFC 2606
reserved, never resolves), which guarantees `fetch_full_text`'s allowlist
check short-circuits with zero network I/O - this eval is hermetic apart from
the OpenRouter appraisal and stance calls themselves. `summary_language="en"`
overrides the production default so `key_finding` stays readable for
debugging (`finding_direction` is a language-independent id either way).
`pico=None`, so `directness` reads "unclear" almost everywhere by design - it
is not graded here.

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
    collapse_direction,
    judge_directions,
    summarise_sources,
)
from app.sources.base import SourceResult

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "scifact_stance.jsonl"

# SciFact's own three-way vocabulary. Both axes of the confusion matrix use
# it: gold labels natively, predictions via collapse_direction - the same
# collapse production would need if it ever had to compare itself to a
# 3-way gold standard, imported rather than re-derived here.
_GOLD_LABELS = ("SUPPORT", "CONTRADICT", "NOINFO")


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


def _aggregate_direction(directions: list[str]) -> str:
    """Collapse one document's per-finding directions to a single prediction.

    SciFact grades a document as a whole against one claim; where a document
    yielded more than one finding, the first decisive one (not
    `no_evidence`) wins - a document that supports or contradicts the claim
    anywhere in it is not a NOINFO document. Ties keep extraction order.
    """
    for direction in directions:
        if direction != "no_evidence":
            return direction
    return "no_evidence"


def _evaluate(row: dict) -> dict:
    settings = get_settings()
    stats = JobStats()
    sources = [_build_source(d) for d in row["docs"]]

    raw = summarise_sources(row["claim"], sources, settings, summary_language="en", job_stats=stats, pico=None)
    directions = judge_directions(row["claim"], sources, raw, settings, job_stats=stats)

    pairs = []
    for i, doc in enumerate(row["docs"]):
        appraisal = raw.get(i, {})
        relevance = clamp_relevance(appraisal.get("relevance_score"))
        predicted = _aggregate_direction(directions.get(i, []))
        findings = appraisal.get("findings")
        findings = findings if isinstance(findings, list) else []
        key_finding = "; ".join(
            f["text"] for f in findings if isinstance(f, dict) and f.get("text")
        )
        pairs.append({
            "doc_id": doc["doc_id"],
            "gold": doc["label"],
            "predicted": predicted,                             # graded scale, for distribution
            "predicted_collapsed": collapse_direction(predicted),  # SciFact's 3-way, for grading
            "relevance_score": relevance,
            "key_finding": key_finding,                         # joined, for debugging only
            "n_findings": len(findings),
            "directness": appraisal.get("directness", ""),
        })

    return {"pairs": pairs, "cost_usd": round(stats.to_dict()["total_cost_usd"], 8)}


_WEAK_LABELS = ("weakly_supports", "weakly_contradicts")


def _summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["error"]]
    pairs = [p for r in ok for p in r["pairs"]]

    # Both axes share SciFact's own 3-way vocabulary - gold natively,
    # predictions via collapse_direction - so this can never KeyError on a
    # label collapse_direction doesn't produce, unlike indexing a fixed
    # confusion matrix by the raw graded scale would.
    confusion: dict[str, dict[str, int]] = {g: dict.fromkeys(_GOLD_LABELS, 0) for g in _GOLD_LABELS}
    for p in pairs:
        confusion[p["gold"]][p["predicted_collapsed"]] += 1

    def n_gold(label: str) -> int:
        return sum(confusion[label].values())

    def precision_recall_f1(label: str) -> tuple[float, float, float]:
        tp = confusion[label][label]
        predicted_as = sum(confusion[g][label] for g in _GOLD_LABELS)
        actual = n_gold(label)
        precision = tp / predicted_as if predicted_as else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return precision, recall, f1

    per_class = {}
    for label in _GOLD_LABELS:
        precision, recall, f1 = precision_recall_f1(label)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}

    accuracy = sum(confusion[g][g] for g in _GOLD_LABELS) / len(pairs) if pairs else 0.0
    mixed_by_gold = {
        g: statistics.mean(p["predicted"] == "mixed" for p in pairs if p["gold"] == g) if n_gold(g) else 0.0
        for g in _GOLD_LABELS
    }

    noinfo_pairs = [p for p in pairs if p["gold"] == "NOINFO"]
    # The graded scale's "no_evidence" is the collapse target of hallucinated_
    # finding_rate's != check below, same role "neutral" played before the
    # split into judge_directions - a NOINFO document is not supposed to
    # clear the gate to any asserted direction at all, weak or strong.
    hallucinated = [
        p for p in noinfo_pairs
        if p["key_finding"] and p["predicted"] != "no_evidence" and p["relevance_score"] > MIN_SYNTHESIS_RELEVANCE
    ]
    # Strict subset of the above: excludes weakly_supports/weakly_contradicts,
    # so this isolates hallucinations the graded scale can't excuse as "a
    # real but small effect" - the gradient the new scale is meant to buy.
    strong_hallucinated = [p for p in hallucinated if p["predicted"] not in _WEAK_LABELS]
    below_gate = [p for p in noinfo_pairs if p["relevance_score"] <= MIN_SYNTHESIS_RELEVANCE]

    relevance_by_label = {
        g: statistics.mean(p["relevance_score"] for p in pairs if p["gold"] == g) if n_gold(g) else 0.0
        for g in _GOLD_LABELS
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
        "strong_hallucination_rate": len(strong_hallucinated) / len(noinfo_pairs) if noinfo_pairs else 0.0,
        "n_strong_hallucinated": len(strong_hallucinated),
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
