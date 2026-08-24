# Eval

Per-stage evaluation against public gold-standard datasets, instead of one
end-to-end judged set. EBM Lens is a multi-stage pipeline (retrieval → rerank
→ selection → synthesis) and an end-to-end score cannot say which stage
regressed; a per-stage eval, graded against ground truth nobody at this
project wrote.

## Datasets

Raw downloads live in `eval/data/` and are **not** committed — see
"Provenance and licensing" below. `eval/fixtures/` holds the small, frozen
subsets actually used by the eval scripts, and those *are* committed.

| Dataset | Tests | Source |
|---|---|---|
| **BioASQ** (Task B training set) | Retrieval + rerank (`searcher.py`, `ranking.py`, `selection.py`) — questions paired with PubMed articles a human assessor judged relevant. | [bioasq.org](http://www.bioasq.org/) / [participants-area.bioasq.org](https://participants-area.bioasq.org/) (registration required). Tsatsaronis et al., *An overview of the BIOASQ large-scale biomedical semantic indexing and question answering competition*, BMC Bioinformatics 2015 — [DOI](https://doi.org/10.1186/s12859-015-0564-6). |
| **EBM-NLP** | PICO extraction (`topic_analysis.py`) — RCT abstracts with crowd-annotated Population/Intervention/Comparison/Outcome spans. | [github.com/bepnye/EBM-NLP](https://github.com/bepnye/EBM-NLP). Nye et al., *A Corpus with Multi-Level Annotations of Patients, Interventions and Outcomes to Support Language Processing for Medical Literature*, ACL 2018 — [arXiv:1806.04185](https://arxiv.org/abs/1806.04185). |
| **SciFact** | Stance + grounding (`synthesis.py`'s `summarise_sources` and `judge_directions`) — expert-written claims paired with PubMed-style abstracts labelled SUPPORT/CONTRADICT, plus documents the claim's own authors cited but that carry no evidence for it (NOINFO). | [github.com/allenai/scifact](https://github.com/allenai/scifact). Wadden et al., *Fact or Fiction: Verifying Scientific Claims*, EMNLP 2020 — [arXiv:2004.14974](https://arxiv.org/abs/2004.14974). |

## Provenance and licensing

`eval/data/` is gitignored on purpose: BioASQ's terms require a registered
account to obtain the data, and none of these datasets are mine to
redistribute. Anyone reproducing this eval downloads their own copy, see
each dataset's row above for where; only the curated fixture below is
committed.

`eval/fixtures/` is a curated, frozen *sample* of the raw data (question text
+ gold labels only, no corpus text reproduced beyond what SciFact's own
`corpus.jsonl` and EBM-NLP's own abstracts already ship under their dataset
licenses), small enough that redistributing it is normal academic practice
for a benchmark subset. Each fixture file names the dataset and questions/ids
it was drawn from, so it traces back to the row above.

## Layout

```
eval/
  data/                      gitignored raw downloads (see table above)
  fixtures/                  curated, frozen subsets (committed)
    bioasq_retrieval.jsonl
    ebm_nlp_pico.jsonl
    scifact_stance.jsonl
  scripts/
    curate_bioasq.py         draws eval/fixtures/bioasq_retrieval.jsonl from eval/data/bioasq/
    curate_ebm_nlp.py        draws eval/fixtures/ebm_nlp_pico.jsonl from eval/data/ebm-nlp/
    curate_scifact.py        draws eval/fixtures/scifact_stance.jsonl from eval/data/scifact/
  _harness.py                shared resumable-progress/summary/CLI scaffolding for the four eval scripts below
  retrieval_eval.py          retrieval + rerank, against a fixture (default bioasq_retrieval.jsonl)
  pool_relevance_eval.py     is the discovery pool topically relevant, source-type by source-type
  pico_eval.py               PICO extraction, against ebm_nlp_pico.jsonl
  stance_eval.py             stance + grounding, against scifact_stance.jsonl
  calibrate_ranking_weights.py  fits ranking.PRIOR_WEIGHTS against retrieval_eval's own ndcg_at_10_pool
                                 metric instead of hand-picked constants (coordinate ascent, cross-validated)
  results/                   eval run output (gitignored — regenerated, not a fixture)
```

## Running

```bash
pip install -r requirements.txt        # OPENROUTER_API_KEY must be set (.env)

# get each raw dataset into eval/data/ (see table above for where), then curate
python eval/scripts/curate_bioasq.py            # -> eval/fixtures/bioasq_retrieval.jsonl
python eval/scripts/curate_ebm_nlp.py           # -> eval/fixtures/ebm_nlp_pico.jsonl
python eval/scripts/curate_scifact.py           # -> eval/fixtures/scifact_stance.jsonl

# run an eval — --fixture picks which judged set to grade against
python eval/retrieval_eval.py [--limit N]
python eval/pool_relevance_eval.py [--limit N]
python eval/pico_eval.py [--limit N]
python eval/stance_eval.py [--limit N]

# calibrate PRIOR_WEIGHTS: dump costs API calls, search is pure arithmetic
python eval/calibrate_ranking_weights.py --dump [--limit N]
python eval/calibrate_ranking_weights.py [--restarts N] [--folds N]
```

Each run hits real biomedical APIs and a couple of cheap LLM calls per topic
(domain detection, decomposition), so cost stays low.

Grading only counts a ranked item if its URL yields an id the fixture's gold
set can judge (a PubMed id against BioASQ's PMIDs) — an item the qrels never
considered isn't scored as irrelevant, since the ground truth never said that.
`n_gradable` / `gradable_share` report how much of the ranked pool that
covers, and `precision_at_10` / `ndcg_at_10` / `mrr` are computed over it;
`recall_at_10` / `recall_at_20` stay over the full ranked list, since that's
what a user's shortlist actually contains.

### EBM-NLP — what the numbers do and don't mean

`pico_eval.py` grades `analyse_topic`'s PICO extraction against EBM-NLP's
crowd-annotated spans, pooling each element's many gold spans into one token
set and comparing it to the analyser's single phrase (`_norm_tokens`) - gold
gives a dozen population spans per abstract, the analyser gives one short
phrase, so span-level alignment would grade a task it isn't performing. This
leniency is one-directional (it can only flatter), so the summary leads with
`hit_rate` ("did it land on the right concept") and `precision`, not F1 -
recall is structurally near-zero (an outcome gold set can exceed 40 tokens
against a four-word prediction) and would dominate any F1 with that artefact.

Two caveats worth reading before trusting a number here:

- **The prompt and the metric disagree on wording.** `analyse_topic` is told
  to use standard terminology over the source's own wording ("myocardial
  infarction", not "heart attack"); EBM-NLP's gold spans are lifted verbatim
  from the abstract. The metric can penalise exactly the behaviour the prompt
  asks for. `--judge` bounds this with an LLM judge that accepts a
  differently-worded match; the deterministic overlap number stays the
  committed one.
- **Task shape.** EBM-NLP annotates the PICO of an RCT *abstract*; production
  feeds `analyse_topic` a user's *question*. Read this as a floor on
  element-identification competence, not an end-to-end score. Comparison is
  gradable on only ~60 of 191 documents (the second annotation phase used a
  different corpus split - see `curate_ebm_nlp.py`'s docstring), so its `n`
  is reported alongside it rather than silently averaged over all 191.

### SciFact — grounding measured at the appraisal stage, not `synthesise`

A citation-grounding eval built against `synthesise` directly - mix a claim's
true cited abstracts with distractors, run synthesis, check whether the
generated claims cite the right ones - was designed and rejected. `synthesise`
is instructed to cite every source it's handed, and distractors only pass the
relevance gate because the eval itself sets `relevance_score`; production
filters them out before `synthesise` ever sees them; and `synthesise` writes
its *own* claims, so "should claim k cite source j" has no gold to check
against without a semantic judge - the exact entailment machinery whose
absence the eval was meant to measure. See `stance_eval.py`'s docstring for
the full reasoning.

Grounding is measured one stage upstream instead, where it's judge-free:
`hallucinated_finding_rate` is the share of NOINFO pairs - a document the
claim's own authors cited, that SciFact's annotators found no evidence for -
where `summarise_sources` nonetheless returns a non-empty `key_finding` and a
`relevance_score` above the synthesis gate, *and* the separate
`judge_directions` call - graded `strongly_contradicts` through
`strongly_supports`, plus `mixed` and a deterministic `no_evidence` - returns
anything other than `no_evidence`. `strong_hallucination_rate` narrows that to
non-`weakly_*` assertions, isolating the cases the graded scale can't excuse
as "a real but small effect". What *is* directly verifiable about
`synthesise` (bounds-checked citations, the strength clamp, the low-evidence
fallback) is covered in `tests/test_synthesis_grounding.py` instead -
pure-function tests, no network.

### Weight calibration — a hypothesis generator, not an auto-tuner

`calibrate_ranking_weights.py` fits `PRIOR_WEIGHTS` against `ndcg_at_10_pool`
on the same 40-topic BioASQ fixture `retrieval_eval.py` uses. 40 topics is
enough to tell a fitted weight vector's direction is plausible, not enough to
trust its precision - the script cross-validates (fit on 4 folds, score on
the held-out one) specifically to surface that: a wide gap between the
in-sample and cross-validated score means the fit is memorising the fixture,
not finding a real pattern, and the suggested weights should not go into
`ranking.py` unreviewed. It also only touches `PRIOR_WEIGHTS` - `FINAL_WEIGHTS`
needs the LLM's post-summarisation `relevance` score, which running
summarisation on 40 topics just to calibrate would cost more than the eval
it's calibrating against.

