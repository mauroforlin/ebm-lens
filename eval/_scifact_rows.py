"""Shared SciFact row-loading for the TypeSafe primitive evals.

`typesafe_stance_eval.py` (Choice), `typesafe_stance_eval_score.py` (Score) and
`typesafe_stance_eval_noul.py` (Noul) all need the same (claim, cited docs)
rows built from the raw SciFact release - one place for that, rather than
three copies of the same per-doc labeling logic drifting apart. See
`typesafe_stance_eval.py`'s module docstring for why this reads the raw
release directly instead of `curate_scifact.py`'s curated fixture.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.schemas import ArticleSummary

RAW = Path(__file__).resolve().parent / "data" / "scifact" / "data"
SPLITS = ("train", "dev")


def load_rows(limit: int | None) -> list[dict]:
    """Every SUPPORT/CONTRADICT/NOINFO (claim, cited doc) pair in SciFact's
    train+dev split.
    """
    corpus: dict[int, dict] = {}
    for line in (RAW / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            doc = json.loads(line)
            corpus[doc["doc_id"]] = doc

    rows: list[dict] = []
    for split in SPLITS:
        path = RAW / f"claims_{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            claim = json.loads(line)
            docs = []
            for doc_id in claim.get("cited_doc_ids", []):
                doc = corpus.get(doc_id)
                if doc is None:
                    continue
                doc_evidence = claim["evidence"].get(str(doc_id), [])
                doc_labels = {ev["label"] for ev in doc_evidence}
                doc_label = doc_labels.pop() if len(doc_labels) == 1 else "NOINFO"
                sentences = sorted({s for ev in doc_evidence for s in ev["sentences"]})
                docs.append({
                    "doc_id": doc_id, "label": doc_label, "title": doc["title"],
                    "abstract": doc["abstract"], "rationale_sentences": sentences,
                })
            if docs:
                rows.append({
                    "id": f"scifact-{split}-{claim['id']}", "type": "scifact",
                    "claim": claim["claim"], "docs": docs,
                })
            if limit and len(rows) >= limit:
                return rows
    return rows


def build_article(doc: dict) -> ArticleSummary:
    sentences = doc["abstract"]
    rationale = doc["rationale_sentences"]
    text = " ".join(sentences[i] for i in rationale) if rationale else " ".join(sentences)
    return ArticleSummary(
        url=f"https://scifact.invalid/doc/{doc['doc_id']}",
        title=doc["title"], source_type="scifact", full_summary=text,
    )
