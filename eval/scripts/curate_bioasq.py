"""Curate a small, stable retrieval-eval fixture from the raw BioASQ training set.

BioASQ's `documents` field gives, per question, the PubMed articles its human
assessors judged relevant - exactly the retrieval ground truth
`app/pipeline/searcher.py` and `app/pipeline/ranking.py` need to be graded
against. The raw file is too large and too repetitive across years to run an
eval over directly, so this script draws one stratified sample and freezes it:
re-running EBM Lens's ranking should be judged against the same questions
every time, not a fresh draw.

Only "summary" and "yesno" questions are drawn. BioASQ also has "factoid" and
"list" types, which ask for one extractable fact ("which virus does cidofovir
treat?") rather than a synthesised answer. For an established fact, dozens of
equally-valid papers support it and BioASQ's gold set names only a handful -
checked against the actual gold PMIDs for one such question, all four were
generic review articles that happened to mention the fact, not a landmark or
otherwise necessary source. Exact-PMID recall on those types mostly measures
which arbitrary subset of interchangeable papers a search happened to return,
not whether it found real evidence. "summary"/"yesno" ask a bounded question
a synthesised answer must actually address - the shape `orchestrator.py`
targets - so their gold set is a meaningful retrieval target.

Usage: python eval/scripts/curate_bioasq.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

_RAW = Path(__file__).resolve().parent.parent / "data" / "bioasq" / "training14b.json"
_OUT = Path(__file__).resolve().parent.parent / "fixtures" / "bioasq_retrieval.jsonl"

_INCLUDED_TYPES = {"summary", "yesno"}
_PER_TYPE = 20
_MIN_DOCS = 3
_MAX_DOCS = 12
_SEED = 42


def _pmid_from_url(url: str) -> str | None:
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    return tail if tail.isdigit() else None


def main() -> None:
    raw = json.loads(_RAW.read_text(encoding="utf-8"))
    questions = raw["questions"]

    seen_bodies: set[str] = set()
    by_type: dict[str, list[dict]] = {}
    for q in questions:
        body = (q.get("body") or "").strip()
        qtype = q.get("type", "")
        docs = q.get("documents", [])
        if qtype not in _INCLUDED_TYPES:
            continue
        if not body or body in seen_bodies:
            continue
        if not (_MIN_DOCS <= len(docs) <= _MAX_DOCS):
            continue
        pmids = [p for p in (_pmid_from_url(d) for d in docs) if p]
        if len(pmids) < _MIN_DOCS:
            continue
        seen_bodies.add(body)
        by_type.setdefault(qtype, []).append({
            "id": q["id"],
            "question": body,
            "type": qtype,
            "gold_pmids": pmids,
        })

    rng = random.Random(_SEED)
    sample: list[dict] = []
    for _qtype, items in sorted(by_type.items()):
        rng.shuffle(items)
        sample.extend(items[:_PER_TYPE])

    # Interleaved, not grouped by type: a `--limit N` slice of the fixture
    # should draw from every included type, not silently become a
    # single-type sample because the file happened to be sorted.
    rng.shuffle(sample)

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with _OUT.open("w", encoding="utf-8") as f:
        for row in sample:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(sample)} questions to {_OUT}")
    for qtype in sorted(by_type):
        n = sum(1 for q in sample if q["type"] == qtype)
        print(f"  {qtype}: {n}")


if __name__ == "__main__":
    main()
