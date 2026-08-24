"""Curate a small, stable stance-eval fixture from SciFact.

SciFact pairs expert-written claims with PubMed-style abstracts their authors
cited, each doc labelled SUPPORT, CONTRADICT, or - critically - cited but
carrying no evidence for the claim at all (NOINFO). That last case is a hard
negative no synthetic distractor set could match: it's a document the claim's
own authors thought worth citing, on the record as containing no evidence for
it. `stance_eval.py` uses exactly this to grade `judge_directions`'s
`finding_direction` (collapsed to this same three-way vocabulary via
`synthesis.collapse_direction`) and `summarise_sources`'s relevance gate.

No claim in the training set mixes SUPPORT and CONTRADICT evidence across its
cited docs (verified: 332 pure-SUPPORT / 173 pure-CONTRADICT / 304 NOINFO / 0
mixed), so a single `gold_label` per claim is well defined - it's the shared
label of the claim's evidence docs, or NOINFO when the claim has cited docs
but no evidence entries at all.

Drawn from `claims_train.jsonl` rather than dev/test: nothing here is being
trained, so the split is irrelevant, and train has the most claims to sample
a clean 30/30/30 stratification from.

Requires the raw corpus at `eval/data/scifact/data/` (claims_train.jsonl +
corpus.jsonl), from https://github.com/allenai/scifact - not fetched by this
script; download and extract it there first.

--exclude-fixture draws a sample disjoint from claims already used in another
curated fixture (by claim id) - how `scifact_stance_holdout.jsonl` is built:
a set a prompt was iterated against carries no evidence about generalisation,
since the same data used to pick a change is not evidence the change works
beyond it; a disjoint sample, looked at once, is.

Usage: python eval/scripts/curate_scifact.py [--seed N] [--per-label N]
    [--out PATH] [--exclude-fixture PATH ...]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

_RAW = Path(__file__).resolve().parent.parent / "data" / "scifact" / "data"
_DEFAULT_OUT = Path(__file__).resolve().parent.parent / "fixtures" / "scifact_stance.jsonl"

_DEFAULT_PER_LABEL = 30
_DEFAULT_SEED = 42


def _used_claim_ids(fixture_paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in fixture_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.add(json.loads(line)["id"])
    return ids


def _gold_label(claim: dict) -> str | None:
    labels = {ev["label"] for doc_evidence in claim["evidence"].values() for ev in doc_evidence}
    if not labels:
        return "NOINFO" if claim["cited_doc_ids"] else None
    return labels.pop() if len(labels) == 1 else None  # drop the (empirically absent) mixed case


def _rationale_sentences(claim: dict, doc_id: str) -> list[int]:
    return sorted({s for ev in claim["evidence"].get(doc_id, []) for s in ev["sentences"]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--per-label", type=int, default=_DEFAULT_PER_LABEL)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--exclude-fixture", type=Path, action="append", default=[])
    args = parser.parse_args()

    exclude_ids = _used_claim_ids(args.exclude_fixture)

    claims = [json.loads(line) for line in (_RAW / "claims_train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    corpus = {}
    for line in (_RAW / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            doc = json.loads(line)
            corpus[doc["doc_id"]] = doc

    by_label: dict[str, list[dict]] = {"SUPPORT": [], "CONTRADICT": [], "NOINFO": []}
    for claim in claims:
        if f"scifact-{claim['id']}" in exclude_ids:
            continue
        label = _gold_label(claim)
        if label:
            by_label[label].append(claim)

    rng = random.Random(args.seed)
    sample: list[dict] = []
    for label, claims_for_label in by_label.items():
        rng.shuffle(claims_for_label)
        for claim in claims_for_label[:args.per_label]:
            docs = []
            for doc_id in claim["cited_doc_ids"]:
                doc = corpus.get(doc_id)
                if doc is None:
                    continue
                doc_evidence = claim["evidence"].get(str(doc_id), [])
                doc_labels = {ev["label"] for ev in doc_evidence}
                doc_label = doc_labels.pop() if len(doc_labels) == 1 else "NOINFO"
                docs.append({
                    "doc_id": doc_id,
                    "label": doc_label,
                    "title": doc["title"],
                    "abstract": doc["abstract"],
                    "rationale_sentences": _rationale_sentences(claim, str(doc_id)),
                })
            if not docs:
                continue
            sample.append({
                "id": f"scifact-{claim['id']}", "type": "scifact",
                "claim": claim["claim"], "gold_label": label, "docs": docs,
            })

    rng.shuffle(sample)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in sample:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(sample)} claims to {args.out}")
    for label in by_label:
        print(f"  {label}: {sum(1 for r in sample if r['gold_label'] == label)}")
    n_docs = [len(r["docs"]) for r in sample]
    print(f"  docs per claim: min={min(n_docs)} max={max(n_docs)} mean={sum(n_docs) / len(n_docs):.2f}")


if __name__ == "__main__":
    main()
