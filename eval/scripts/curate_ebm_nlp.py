"""Curate a small, stable PICO-extraction eval fixture from EBM-NLP.

EBM-NLP annotates ~5000 RCT abstracts with crowd-sourced Population,
Intervention and Outcome spans - exactly the ground truth
`topic_analysis.analyse_topic`'s PICO extraction needs, and the one part of
the pipeline no other fixture here touches at all.

Two annotation phases, two different corpus partitions - this looks like a
bug the first time you meet it, so it's spelled out here. `starting_spans/`
(phase 1: workers highlight P/I/O spans) covers a 191-document gold test set,
annotated by medical professionals - the actual target test set, preferred
over the crowd-annotated version. `hierarchical_labels/interventions/`
(phase 2: workers sub-classify already-highlighted intervention spans, one of
which is label `7` = "Control" - the only source of a PICO Comparison
element) was collected against a *different* test/train split: only 6 of the
191 starting_spans gold PMIDs land in its own test/gold folder, and another
140 are recoverable from its train folder instead - nothing here is being
trained, so mixing test/train for this second, unrelated task is fine, but
it means Comparison is gradable on the ~60 documents that (a) fall in this
merged 146-document intersection and (b) actually contain a Control span, not
on all 191. `has_comparison_gold` on each fixture row is what keeps the other
~115 documents' missing annotation from reading as a Comparison miss.

Requires the raw corpus at `eval/data/ebm-nlp/ebm_nlp_1_00/` (documents/ +
annotations/), from https://github.com/bepnye/EBM-NLP - not fetched by this
script; download and extract it there first.

Usage: python eval/scripts/curate_ebm_nlp.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

_RAW = Path(__file__).resolve().parent.parent / "data" / "ebm-nlp" / "ebm_nlp_1_00"
_OUT = Path(__file__).resolve().parent.parent / "fixtures" / "ebm_nlp_pico.jsonl"

_DOCS = _RAW / "documents"
_STARTING_SPANS = _RAW / "annotations" / "aggregated" / "starting_spans"
_HIER_INTERVENTIONS = _RAW / "annotations" / "aggregated" / "hierarchical_labels" / "interventions"

_PER_GROUP = 30  # with comparison gold / without, 60 rows total
_SEED = 42


def _spans(tokens: list[str], labels: list[str], mark: str) -> list[str]:
    """Contiguous runs of *mark* in *labels*, joined back into span text."""
    out: list[str] = []
    current: list[str] = []
    for token, label in zip(tokens, labels, strict=True):
        if label == mark:
            current.append(token)
        elif current:
            out.append(" ".join(current))
            current = []
    if current:
        out.append(" ".join(current))
    return out


def _load_labels(ann_path: Path) -> list[str]:
    return [x for x in ann_path.read_text(encoding="utf-8").strip().split(",") if x]


def _pio_spans(pmid: str, tokens: list[str]) -> dict[str, list[str]] | None:
    gold: dict[str, list[str]] = {}
    for elem, key in [("participants", "population"), ("interventions", "intervention"), ("outcomes", "outcome")]:
        ann = _STARTING_SPANS / elem / "test" / "gold" / f"{pmid}_AGGREGATED.ann"
        labels = _load_labels(ann)
        if len(labels) != len(tokens):
            raise ValueError(f"{pmid}/{elem}: {len(tokens)} tokens vs {len(labels)} labels - misaligned")
        spans = _spans(tokens, labels, mark="1")
        if not spans:
            return None
        gold[key] = spans
    return gold


def _comparison_spans(pmid: str, tokens: list[str]) -> tuple[list[str], str | None]:
    """Control (label 7) spans for *pmid*, preferring the phase-2 test/gold
    folder and falling back to its train folder - see the module docstring
    for why hierarchical_labels uses a different partition than starting_spans.
    """
    for subdir, source in [("test/gold", "hier_test_gold"), ("train", "hier_train")]:
        ann = _HIER_INTERVENTIONS / subdir / f"{pmid}_AGGREGATED.ann"
        if not ann.exists():
            continue
        labels = _load_labels(ann)
        if len(labels) != len(tokens):
            continue
        return _spans(tokens, labels, mark="7"), source
    return [], None


def main() -> None:
    pmids = sorted(p.name.split("_")[0] for p in (_STARTING_SPANS / "participants" / "test" / "gold").glob("*.ann"))

    with_comparison: list[dict] = []
    without_comparison: list[dict] = []

    for pmid in pmids:
        tokens = (_DOCS / f"{pmid}.tokens").read_text(encoding="utf-8").split()
        gold = _pio_spans(pmid, tokens)
        if gold is None:
            continue  # missing P, I or O gold entirely - not a usable row

        comparison, source = _comparison_spans(pmid, tokens)
        row = {
            "id": pmid, "pmid": pmid, "type": "ebm_nlp",
            "abstract": (_DOCS / f"{pmid}.text").read_text(encoding="utf-8").strip(),
            "gold": {**gold, "comparison": comparison},
            "has_comparison_gold": bool(comparison),
            "comparison_source": source,
        }
        (with_comparison if comparison else without_comparison).append(row)

    rng = random.Random(_SEED)
    rng.shuffle(with_comparison)
    rng.shuffle(without_comparison)
    sample = with_comparison[:_PER_GROUP] + without_comparison[:_PER_GROUP]
    rng.shuffle(sample)

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with _OUT.open("w", encoding="utf-8") as f:
        for row in sample:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(sample)} documents to {_OUT}")
    print(f"  with comparison gold: {sum(1 for r in sample if r['has_comparison_gold'])}")
    print(f"  without: {sum(1 for r in sample if not r['has_comparison_gold'])}")


if __name__ == "__main__":
    main()
