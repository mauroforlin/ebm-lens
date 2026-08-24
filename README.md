# EBM Lens

Evidence discovery for biomedical questions. Give it a topic and it searches
twelve public biomedical databases, expands the result set through the
citation graph, ranks what it found by relevance and by study design, and
returns appraised sources plus an overview whose every claim cites the
sources under it.

![EBM Lens UI](assets/demo.gif)

```
"GLP-1 agonists and cardiovascular risk reduction"
   ↓
PubMed, Europe PMC, ClinicalTrials.gov, OpenFDA, DailyMed, RxNav,
Open Targets, ChEMBL, WHO GHO, EMA, bioRxiv/medRxiv, Wikipedia
   ↓  + Semantic Scholar / OpenAlex citation graph
   ↓
ranked sources, each appraised (design, population, direction)
   ↓
an overview, as claims that cite [0][3], plus the conflicts and the gaps
```

Every evidence source is a free public API. The only account you need is for
the LLM.

EBM Lens started as a feature inside [Sbobby](https://www.sbobby.com), a
lecture-transcription product I built. That feature, "Approfondimenti",
searches a similar number of sources for the topics covered in a lecture and
has its own basic query planning and reranking. This repository is a full
rewrite of it: study-design ranking, citation verification and the
multi-round search loop are all new.

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/mauroforlin/ebm-lens.git
cd ebm-lens
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements.txt

cp .env.example .env             # then set OPENROUTER_API_KEY
uvicorn app.main:app --reload
```

Open <http://localhost:8000> for the bundled UI, or call the API directly:

```bash
curl -X POST http://localhost:8000/api/related-articles \
  -H "Content-Type: application/json" \
  -d '{
        "topic": "GLP-1 agonists and cardiovascular risk reduction",
        "max_sources": 10,
        "summary_language": "en"
      }'
```

`/api/related-articles/stream` takes the same body and returns Server-Sent
Events instead of a single JSON response, if you want progress as it happens.

There is no database, no Redis, no job queue and no container to build. A
search runs synchronously inside the request.

### Configuration

Only `OPENROUTER_API_KEY` is required. Everything else has a working
default, see `.env.example`. Worth knowing about:

| Variable | Why you might set it |
|---|---|
| `CONTACT_EMAIL` | Goes in the `User-Agent`. NCBI, Crossref and OpenAlex give identified clients a more generous rate limit. |
| `NCBI_API_KEY` | Free, raises PubMed from 3 to 10 req/s. |
| `OPENFDA_API_KEY` | Free, raises openFDA from 1,000 to 120,000 requests/day. |
| `API_KEY` | If set, clients must send a matching `X-API-Key`. Set it before exposing the service beyond localhost. |
| `CORS_ALLOW_ORIGINS` | Only needed for a frontend on another origin. The bundled UI is same-origin. |

## How it works

### Twelve databases, one planned query each

Sending the same string to every provider wastes most of the calls: PubMed
wants MeSH-flavoured English, DailyMed wants a bare drug name and returns
nothing for a disease, and so on for the rest. An LLM plans the search
instead, choosing providers and wording each query for the one it is going
to.

The planner checks its own queries before committing to them. `probe_pubmed`
runs a candidate query and returns the hit count, PubMed's MeSH translation
of it, and any phrases PubMed could not match. That last part matters:
PubMed does not search the words it is given, it expands them, and a term it
does not recognise expands to nothing while looking exactly like a topic with
no literature behind it. The planner sees the difference and rewords.

If planning fails for any reason, a deterministic router picks providers from
the topic type instead. The result is worse-targeted, but the pipeline
returns an answer either way.

### PICO before retrieval

A clinical question has parts, and the databases
are indexed by them. Before anything is searched, the topic is put into
Population, Intervention, Comparison, Outcome form, the structure a
clinician is taught to break a question into. Only the elements the topic
actually states get filled in; inventing a comparator would send the search
after literature nobody asked about. The elements steer the plan, sharpen the
queries and weight the ranking.

### Two retrieval signals, fused

Embeddings read meaning but compress a
whole paper into one vector, so a rare decisive term (an INN, a trial
acronym, a gene symbol) counts for about as much as any other word in the
abstract. BM25 has the opposite blind spot: it cannot read meaning, but it
weights a term by how rare it is. Both run, and their rankings are fused
across providers alongside citation authority, recency, provider tier and an
LLM reranker that reads title and abstract. A candidate has to be wrong along
several of these dimensions at once to end up ranked low.

### Study design as its own ranking signal

A case report and a Cochrane
meta-analysis on the same subject are equally on-topic and not equally worth
believing, so design gets scored separately from relevance. Three sources can
say what a paper's design is, checked in order of trust: the provider's own
classification where one exists, the publication types NLM's indexers
assigned by hand, and only when neither is available, the wording of the
title and abstract. An unrecognised design scores neutral rather than low, so
a wrong guess never demotes real evidence. Every source is shown with the
design it was judged to have, which is what makes the ranking checkable.

### Composite questions searched per axis

"CAR-T therapy, BBB disruption and
ctDNA monitoring in glioblastoma" sent as one query returns whatever its
most-published axis happens to be, and does so silently. A topic detected as
composite gets a query per axis instead, plus one for their intersection, all
folded into the same discovery run, so the axis with the thinnest literature
is not starved by the others.

### Multi-round discovery

A single query pass fails frontier topics in two
specific ways: the seminal papers often use vocabulary the user's phrasing
does not contain, and a highly-cited paper about a different sense of the
same words can outrank everything actually on-topic. So discovery runs as a
loop. An LLM first writes a research brief: the disambiguated topic, the
vocabulary an on-topic paper is likely to use, the terms that mark the wrong
sense, and several deliberately diverse query variants. Those fan out in
parallel, and only the papers that clear a relevance gate seed citation-graph
expansion, since seeding on the most-cited hit expands around the wrong
paper. An LLM then reads the strongest results, extracts the terms the
literature actually uses, and issues refined queries, each probed against
PubMed first so a round is spent on a query with real literature behind it
rather than a plausible phrasing that expands to nothing. The loop repeats
while a round still surfaces new on-topic work and stops as soon as one
doesn't. Final selection applies a relevance floor and can return fewer
sources than requested: six on-topic papers beat ten padded with four that
merely share words.

### Synthesis you can check

The overview comes back as claims rather than a
paragraph of prose. Each claim names the articles it rests on by index, and
every index is checked against the articles actually being returned; a claim
citing nothing real is dropped before it reaches the response. Citation
hallucination is a well-documented failure mode of this kind of system, and a
better prompt does not fix it on its own. The citations are checked in code.

### Where the sources disagree, that disagreement is the finding

Roughly half of post-2010 biomedical papers conflict with something else in
their own literature, and an overview that averages two opposing findings
reads exactly like settled science. Conflicts come back with the sources on
each side named, alongside a separate note on what these particular sources
leave unsettled.

### Progress and cost, during and after the run

A run takes roughly 60 to 180 seconds: real multi-round search plus several LLM
calls. The streaming endpoint
(`/api/related-articles/stream`, Server-Sent Events) emits a frame at each
stage, so the bundled UI can show what stage is running instead of a bare
spinner. Every response, streamed or not, also carries a `job_stats`
breakdown: real USD cost from the provider, token counts, per-stage timing,
and per-source call and cache counts.

## What the model is allowed to do

Most of this pipeline is fixed: the stages, the providers, the ranking. Two
steps are not, because the right move in those two depends on facts nobody
has looked up yet, like which words a given database indexes or what a brand
name is actually called. Those two run as tool loops instead of fixed code.

| Tool | Answers |
|---|---|
| `probe_pubmed(query)` | Hit count, PubMed's MeSH translation of the query, the phrases it could not match, sample titles. |
| `resolve_drug(name)` | The active molecule behind a brand name, via RxNorm. |
| `search_guidelines(topic)` | Real clinical practice guideline titles, via Europe PMC. |
| `submit_brief(...)` / `submit_queries(queries)` | Terminal tools: the research brief and the query batch, as typed arguments rather than prose to re-parse. |

The loop itself lives in `app/core/llm_client.py` and does three things worth
naming. Tool calls issued in the same turn run concurrently, so a model
probing three phrasings at once waits for the slowest one rather than their
sum. Identical repeated calls are served from a memo, since a model that
dislikes a result will sometimes re-issue it verbatim. And when the round
budget is about to run out, the terminal tool is forced through
`tool_choice`, turning "explored too long, returned nothing" into "submits
what it has."

Every invocation is counted in `job_stats.tools`, so which lookups a model
actually reaches for, and how often, is visible in the response.

## Layout

```
app/
  main.py          FastAPI app, static frontend, startup config
  config.py        settings (pydantic-settings, reads .env)
  schemas.py       request/response models and the internal TopicSpec
  api/             endpoints.py (sync + streaming routes), deps.py (API-key auth)

  core/            infrastructure that knows nothing about medicine
    llm_client.py    OpenRouter wrapper: model routing, retries, JSON repair,
                     tool-calling loop, cost capture
    embeddings.py    batched embeddings
    events.py        SSE progress events for the streaming endpoint
    cache.py         in-process TTL cache, LRU-bounded
    ratelimiter.py   in-process token/request limiter
    job_stats.py     per-run cost, token and timing accounting

  sources/         everything that talks to the outside world
    base.py          SourceProvider / SourceResult contract
    blocklist.py     domain quality gate applied to every result
    <12 providers>   one module each
    citation_expander.py   Semantic Scholar + OpenAlex neighbours
    content_extractor.py   full-text fetch, allowlisted hosts only

  pipeline/        the discovery logic
    orchestrator.py  entry point; the run, stage by stage
    topic_analysis.py  domain detection and composite decomposition
    searcher.py      the tiered multi-provider search
    agentic.py       the multi-round discovery loop
    dedup.py         source identity (DOI, then title, then URL)
    ranking.py       pure scoring functions and weight profiles
    relevance.py     builds per-candidate signal maps
    lexical.py       BM25 over the candidate pool, no index, no dependency
    evidence_grade.py  study design detection and the evidence hierarchy
    evidence_cache.py  whole-topic result cache, keyed on the normalised query
    selection.py     final ranking and selection policy
    synthesis.py     per-source appraisal and the grounded overview
    planner_tools.py the tools the model may call, and their dispatch

frontend/  index.html, app.js, style.css - plain files, no build step
```

## Design notes and limits

- **No persistence.** Caches are in-memory and reset on restart, which costs
  a slower first query afterwards and nothing else. `app/core/cache.py`
  exposes only `get`/`set`, so swapping in a Redis or SQLite backing is a
  single-module change.
- **Both caches match on exact hashes.** `evidence_cache.py`'s topic lookup
  and the per-provider cache in `cache.py` both normalise (lowercase, strip
  punctuation, sort words) before hashing, so word-order variants of the same
  question share an entry, but two genuinely different phrasings do not.
  Catching those needs embedding similarity over a vector store, the
  infrastructure this project deliberately does without. A miss just costs
  a slower response.
- **Single process.** Fine for local use or one container. Scaling
  horizontally means giving `cache.py` and `ratelimiter.py` a shared backend,
  since both are per-process today.
- **Summaries are a reading aid.** They exist to help decide which sources
  are worth opening. Every result also links back to its source, with
  provider, study design and reliability tier shown alongside it.
- **The design tier maps to GRADE's first input, nothing more.** GRADE grades
  a body of evidence for a specific question, weighing risk of bias,
  imprecision and indirectness, none of which this pipeline can see. What the
  ranker produces is the design tier, GRADE's first and largest input,
  reported by name so the rest of the appraisal can be applied by hand.
> [!IMPORTANT]
> **Citation checking stops at existence (Working on it)**
> A claim citing an article that is not in the response gets dropped before the response is built. Whether a cited article actually supports the sentence citing it is left to the reader; the response links straight to the source for that check.

## Evaluation

`eval/` grades each pipeline stage against a public, externally-labelled
dataset, instead of one end-to-end judged score: retrieval and reranking
against BioASQ's PubMed relevance judgments, per-source stance and citation
grounding against SciFact's expert-labelled claims, and PICO extraction
against EBM-NLP's crowd-annotated abstracts.

```bash
python eval/retrieval_eval.py
python eval/pool_relevance_eval.py
python eval/pico_eval.py
python eval/stance_eval.py
```

Each script writes a resumable run to `eval/results/` and appends a row to
`eval/results/history.jsonl`, so a change to the pipeline can be compared
against the run before it. See `eval/README.md` for dataset provenance,
licensing and the caveats specific to each benchmark.

