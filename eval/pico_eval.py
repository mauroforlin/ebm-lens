"""PICO extraction eval: does `analyse_topic` name the right P/I/C/O concepts?

Grades `topic_analysis.analyse_topic`'s `spec.pico` against EBM-NLP's
crowd-annotated spans. Gold gives many spans per element (a 280-token
abstract can carry a dozen population spans); `analyse_topic` emits one short
phrase. Requiring span-level alignment would grade a task the analyser isn't
performing, so gold spans for an element are pooled into one token set and
compared against the analyser's single phrase as sets - see `_norm_tokens`.
This is a deliberately lenient, one-directional metric (it can only make the
score look better than a strict span-F1 would), so it's reported as overlap,
never presented as span-F1.

**The headline number is `hit_rate`, not F1.** An outcome gold set can exceed
40 distinct tokens against a four-word prediction, so recall (and therefore
F1) is structurally near-zero regardless of extraction quality - it's
reported alongside `mean_{element}_gold_tokens` so that ceiling is visible,
but `hit_rate` ("did the prediction land on the right concept at all") and
`precision` ("was what it named actually annotated as this element") are what
the summary leads with.

**Known soundness caveat - not a bug, a genuine tension.** `analyse_topic`'s
prompt explicitly instructs standard medical terminology over the source's
wording ("myocardial infarction", not "heart attack"); EBM-NLP's gold spans
are lifted verbatim from the abstract. This metric can penalise exactly the
behaviour the prompt asks for. `--judge` (off by default) runs one LLM call
per document asking whether a predicted phrase names the same concept as any
gold span, reporting `{element}_judge_hit_rate` as a bound on how much that
terminology gap is costing - the deterministic token-overlap number stays the
one actually committed to, since it's free and reproducible.

**Task-shape caveat.** EBM-NLP annotates the PICO of an RCT *abstract*;
production feeds `analyse_topic` a user's *question*. Adjacent tasks. An
abstract states its PICO more explicitly than a question typically does, but
wraps it in ~250 words of methods prose a question doesn't contain, and the
analyser's prompt is written for questions, not abstracts. Read this as a
floor on element-identification competence, not an end-to-end score.

`pico is None` - `topic_analysis._build_pico` discards the whole PICO unless
`intervention` and (`outcome` or `population`) are non-empty - is what
production actually gets on those topics: no PICO reaches the query builder
or the ranker at all. Those rows are scored 0 on every headline element
rather than excluded, since excluding them would flatter the system;
`pico_none_rate` and a parallel `*_specified` block (scored over only the
rows where a PICO was returned) separate "didn't extract" from "extracted
wrong" - the same two-axis logic `retrieval_eval.py` uses for pool-recall vs
recall@10.

Usage:
    python eval/pico_eval.py [--limit N] [--fixture PATH] [--fresh] [--judge]
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eval._harness as harness
from app.config import get_settings
from app.core.job_stats import JobStats
from app.core.llm_client import generate_json
from app.pipeline import topic_analysis

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ebm_nlp_pico.jsonl"

_ELEMENTS = ("population", "intervention", "comparison", "outcome")

# English function words only - deliberately hand-written rather than pulled
# from nltk, so this eval doesn't acquire a dependency or a downloadable
# corpus just to strip "the"/"and" before a token-overlap comparison.
_STOPWORDS = frozenset(
    "a an and are as at be been by for from had has have in into is it its "  # noqa: SIM905
    "of on or that the their this to was were with which who whom".split()
)

_JUDGE_SYSTEM = """\
You judge whether a predicted PICO element phrase names the same clinical
concept as any of the gold-annotated spans for that element, even if the
wording differs (e.g. "myocardial infarction" matches "heart attack").

Return JSON: {"match": true/false}
"""


def _norm_tokens(text: str) -> set[str]:
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    tokens = set()
    for token in text.split():
        if len(token) <= 1 or token in _STOPWORDS:
            continue
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        tokens.add(token)
    return tokens


def _element_score(predicted: str, gold_spans: list[str]) -> dict:
    gold_tokens: set[str] = set()
    for span in gold_spans:
        gold_tokens |= _norm_tokens(span)
    pred_tokens = _norm_tokens(predicted)

    overlap = len(pred_tokens & gold_tokens)
    empty = not pred_tokens
    precision = overlap / len(pred_tokens) if pred_tokens else None
    recall = overlap / len(gold_tokens) if gold_tokens else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else 0.0
    )
    return {
        "hit": 1 if overlap >= 1 else 0,
        "precision": precision,
        "recall": recall if gold_tokens else None,
        "f1": f1,
        "n_gold_tokens": len(gold_tokens),
        "empty": empty,
    }


def _judge_element(claim_text: str, predicted: str, gold_spans: list[str], settings) -> bool:
    if not predicted.strip() or not gold_spans:
        return False
    prompt = f'PREDICTED: "{predicted}"\n\nGOLD SPANS:\n' + "\n".join(f"- {s}" for s in gold_spans)
    result = generate_json(
        settings=settings, prompt=prompt, system_instruction=_JUDGE_SYSTEM,
        temperature=0.0, purpose="pico_element_judge",
    )
    return bool(isinstance(result, dict) and result.get("match") is True)


def _make_evaluate(use_judge: bool):
    def evaluate(row: dict) -> dict:
        settings = get_settings()
        stats = JobStats()
        _, spec = topic_analysis.analyse_topic(row["abstract"], settings, job_stats=stats)
        pico = spec.pico

        metrics: dict = {"pico_none": pico is None, "cost_usd": round(stats.to_dict()["total_cost_usd"], 8)}
        predicted = {elem: (getattr(pico, elem, "") if pico else "") for elem in _ELEMENTS}
        metrics["predicted"] = predicted

        for elem in _ELEMENTS:
            gold_spans = row["gold"][elem]
            if elem == "comparison" and not row["has_comparison_gold"]:
                metrics[elem] = {"graded": False}
                continue
            score = _element_score(predicted[elem], gold_spans)
            score["graded"] = True
            if use_judge:
                score["judge_hit"] = int(_judge_element(row["abstract"], predicted[elem], gold_spans, settings))
            metrics[elem] = score

        return metrics

    return evaluate


def _summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["error"]]
    specified = [r for r in ok if not r["pico_none"]]

    def block(subset: list[dict]) -> dict:
        out: dict = {"n": len(subset)}
        for elem in _ELEMENTS:
            graded = [r[elem] for r in subset if r[elem].get("graded")]
            non_empty = [s for s in graded if not s["empty"]]
            out[f"{elem}_hit_rate"] = statistics.mean(s["hit"] for s in graded) if graded else 0.0
            out[f"{elem}_precision"] = (
                statistics.mean(s["precision"] for s in non_empty) if non_empty else 0.0
            )
            out[f"{elem}_recall"] = statistics.mean(s["recall"] for s in graded if s["recall"] is not None) if graded else 0.0
            out[f"{elem}_empty_rate"] = statistics.mean(s["empty"] for s in graded) if graded else 0.0
            out[f"{elem}_mean_gold_tokens"] = statistics.mean(s["n_gold_tokens"] for s in graded) if graded else 0.0
            out[f"n_{elem}_graded"] = len(graded)
            if graded and "judge_hit" in graded[0]:
                out[f"{elem}_judge_hit_rate"] = statistics.mean(s["judge_hit"] for s in graded)
        return out

    return {
        "n_topics": len(rows),
        "n_errors": len(rows) - len(ok),
        "pico_none_rate": statistics.mean(r["pico_none"] for r in ok) if ok else 0.0,
        "all_rows": block(ok),
        "pico_specified_only": block(specified),
        "mean_cost_usd": statistics.mean(r["cost_usd"] for r in ok) if ok else 0.0,
        "total_cost_usd": sum(r["cost_usd"] for r in rows),
    }


def _describe(row: dict, metrics: dict) -> str:
    if metrics["pico_none"]:
        return f"pico=None  {row['abstract'][:60]}"
    hits = "".join("Y" if metrics[e].get("hit") else ("-" if metrics[e].get("graded") else ".") for e in _ELEMENTS)
    return f"PICO[{hits}]  {row['abstract'][:60]}"


def main() -> None:
    parser = argparse.ArgumentParser()
    harness.add_standard_args(parser, _FIXTURE)
    parser.add_argument(
        "--judge", action="store_true",
        help="Also run an LLM judge per element, bounding the cost of the terminology-mismatch caveat.",
    )
    args = parser.parse_args()

    progress_path = harness.progress_path(args.fixture)
    summary_path = harness.summary_path(args.fixture)
    if args.fresh and progress_path.exists():
        progress_path.unlink()

    done = harness.load_progress(progress_path)
    rows = harness.load_fixture(args.fixture, args.limit)

    harness.run_rows(
        rows, done, progress_path, summary_path,
        evaluate=_make_evaluate(args.judge), summarise=_summarise, describe=_describe,
        fixture_name=args.fixture.name,
    )


if __name__ == "__main__":
    main()
