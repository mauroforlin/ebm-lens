"""PubMed E-utilities provider - smart query construction, resilient API
usage.

Supports optional NCBI API key (env ``NCBI_API_KEY``) for 10 req/s instead
of the anonymous 3 req/s.  Uses adaptive rate-limiting, circuit-breaker
logic, a persistent ``httpx.Client`` for connection reuse, and multi-stage
query strategies (MeSH-tagged → Boolean → simplified keywords).

Search results carry the abstract only - it is what the ranking signals
(BM25, embeddings) are meant to score, and what they score for every other
provider too. Full text, when a paper has an open-access copy in PubMed
Central, is fetched on demand by the synthesis stage instead (see
``fetch_full_text_by_url`` below and ``synthesis.py``'s ``read_full_text``
tool) - only for sources that actually reach the shortlist, after ranking
has already run.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from xml.etree import ElementTree

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings
from app.sources.base import SourceProvider, SourceResult, user_agent

logger = logging.getLogger(__name__)

# ── NCBI E-utilities endpoints ────────────────────────────────
_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_ESEARCH = f"{_EUTILS_BASE}/esearch.fcgi"
_EFETCH = f"{_EUTILS_BASE}/efetch.fcgi"
_ELINK = f"{_EUTILS_BASE}/elink.fcgi"
_PMC_FETCH = f"{_EUTILS_BASE}/efetch.fcgi"

# ── API key & rate-limiting ───────────────────────────────────
# Read lazily via get_settings() rather than os.environ - pydantic-settings
# parses .env into Settings' own fields, it never injects them into
# os.environ, so a key set only in .env would otherwise go unnoticed.
def _ncbi_api_key() -> str | None:
    return get_settings().ncbi_api_key


def _min_interval() -> float:
    # NCBI's key-holder cap is 10 req/s (0.10s gap); 0.11s left no slack and
    # the agentic loop's concurrent query variants were still tripping 429s
    # against it, so this keeps a real margin below the ceiling rather than
    # riding it exactly.
    # Without a key: 3 req/s → 0.35 s gap.
    return 0.15 if _ncbi_api_key() else 0.35


_last_request_time = 0.0
_rate_lock = threading.Lock()


# ── Circuit breaker state ─────────────────────────────────────
_consecutive_failures = 0
_circuit_open_until = 0.0
_CIRCUIT_THRESHOLD = 5        # open after 5 consecutive failures
_CIRCUIT_COOLDOWN = 60.0      # stay open for 60 s
_circuit_lock = threading.Lock()


def _rate_limit() -> None:
    """Adaptive rate limiter - respects NCBI limits and backs off on errors."""
    global _last_request_time
    with _rate_lock:
        now = time.monotonic()
        gap = _min_interval()
        with _circuit_lock:
            if _consecutive_failures > 0:
                gap = min(gap * (1 + _consecutive_failures * 0.5), 3.0)
        elapsed = now - _last_request_time
        if elapsed < gap:
            time.sleep(gap - elapsed)
        _last_request_time = time.monotonic()


def _check_circuit() -> bool:
    """Return True if the circuit is open (we should NOT make requests)."""
    with _circuit_lock:
        tripped = _consecutive_failures >= _CIRCUIT_THRESHOLD
        if tripped and time.monotonic() < _circuit_open_until:
            return True
    # Not tripped, or the cooldown has elapsed - half-open: allow one probe.
    return False


def _record_success() -> None:
    global _consecutive_failures, _circuit_open_until
    with _circuit_lock:
        _consecutive_failures = 0
        _circuit_open_until = 0.0


def _record_failure() -> None:
    global _consecutive_failures, _circuit_open_until
    with _circuit_lock:
        _consecutive_failures += 1
        if _consecutive_failures >= _CIRCUIT_THRESHOLD:
            _circuit_open_until = time.monotonic() + _CIRCUIT_COOLDOWN
            logger.warning(
                "PubMed circuit breaker OPEN - %d consecutive failures, "
                "cooling down for %.0fs",
                _consecutive_failures, _CIRCUIT_COOLDOWN,
            )


def _is_retryable_status(exc: BaseException) -> bool:
    """Whether *exc* is a transient E-utilities failure worth a local retry.

    A 429 or a 5xx is NCBI asking to be asked again later; anything else (a
    malformed query, for instance) fails identically on retry, so only these
    two are worth the extra round trip. Scoped to `_esearch` itself rather
    than left to `search()`'s own outer retry, so one rate-limited call among
    several in a multi-stage search absorbs its own backoff instead of
    discarding the stages that already succeeded and starting the whole
    search over.
    """
    return isinstance(exc, httpx.HTTPStatusError) and (
        exc.response.status_code == 429 or exc.response.status_code >= 500
    )


# ── Shared HTTP client (connection pooling) ───────────────────
_http_client: httpx.Client | None = None
_client_lock = threading.Lock()
_MAX_CONTENT_LEN = 8000  # chars for search-result content (abstract + metadata prefix)

# Safety ceiling on a PMC full-text fetch, well above what synthesis.py's own
# trimming ever keeps - it exists to bound one pathological article's memory
# and network cost, not to shape what a model sees. Covers the large majority
# of open-access RCT/systematic-review full text; the rare article past it
# still gets the abstract plus whatever of the body arrives first.
_MAX_RAW_FULLTEXT_CHARS = 60000


def _get_client() -> httpx.Client:
    """Lazily create a module-level httpx.Client for connection reuse."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        with _client_lock:
            if _http_client is None or _http_client.is_closed:
                _http_client = httpx.Client(
                    timeout=httpx.Timeout(20.0, connect=10.0),
                    limits=httpx.Limits(
                        max_connections=6,
                        max_keepalive_connections=4,
                    ),
                    follow_redirects=True,
                    headers={"User-Agent": user_agent()},
                )
    return _http_client


def _api_params(extra: dict | None = None) -> dict:
    """Base params that include the API key when available."""
    params: dict = {}
    key = _ncbi_api_key()
    if key:
        params["api_key"] = key
    if extra:
        params.update(extra)
    return params


# ── Stopwords ─────────────────────────────────────────────────

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "to",
    "for", "with", "on", "at", "from", "by", "about", "as", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "out", "off", "over", "under", "again", "further", "then", "once",
    "that", "this", "these", "those", "what", "which", "who", "whom",
    "when", "where", "why", "how", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "no", "not", "only",
    "same", "so", "than", "too", "very", "and", "but", "or", "if",
    "it", "its", "they", "their", "them", "he", "she", "his", "her",
})

# ── Publication-type filters (PubMed [pt] field) ─────────────

_PT_FILTERS: dict[str, str] = {
    "systematic_review": "(systematic review[pt] OR meta-analysis[pt])",
    "cochrane_review": '(systematic review[pt] OR meta-analysis[pt]) AND "cochrane database syst rev"[ta]',
    "rct": "randomized controlled trial[pt]",
    "guideline": "(practice guideline[pt] OR guideline[pt])",
    "clinical_trial": "clinical trial[pt]",
    "review": "review[pt]",
}


# ══════════════════════════════════════════════════════════════
#  Query construction helpers
# ══════════════════════════════════════════════════════════════


def _strip_stopwords(text: str, max_tokens: int = 8) -> str:
    """Extract keywords from text, removing stopwords."""
    clean = re.sub(r"[\"'()[\]{}<>:;,./\\]", " ", text)
    tokens = clean.split()
    keywords = [t for t in tokens if t.lower() not in _STOPWORDS and len(t) > 2]
    return " ".join(keywords[:max_tokens])


def _has_field_tags(query: str) -> bool:
    """Check if the query already contains PubMed field tags like [MeSH Terms]."""
    return bool(re.search(r"\[(?:MeSH|tiab|pt|au|Title|Abstract)", query, re.I))


def build_pubmed_query(
    query: str,
    *,
    mesh_terms: list[str] | None = None,
    drug_names: list[str] | None = None,
    diseases: list[str] | None = None,
    evidence_type: str | None = None,
) -> str:
    """Build a well-structured PubMed Boolean query.

    Constructs queries using PubMed field tags:
    - ``"term"[MeSH Terms]`` for controlled vocabulary
    - ``term[tiab]`` for title/abstract free-text
    - ``type[pt]`` for publication type filters

    Parameters
    ----------
    query : str
        Base query (may already contain field tags).
    mesh_terms : list[str], optional
        Pre-resolved MeSH descriptors to use as primary search terms.
    drug_names : list[str], optional
        INN drug names to include with [tiab] tags.
    diseases : list[str], optional
        Disease/condition names to include.
    evidence_type : str, optional
        Desired evidence level ('systematic_review', 'rct', 'guideline', etc.).
    """
    # A query with field tags is left as-is; the publication-type filter
    # below still gets appended on top of it.
    if _has_field_tags(query):
        base = query
    elif mesh_terms:
        # Build MeSH-anchored query - use [Majr] (Major MeSH Heading) for
        # the first term to find articles where it's the primary focus,
        # and [MeSH Terms] for supplementary terms for recall.
        mesh_parts = []
        for i, t in enumerate(mesh_terms[:3]):
            tag = "Majr" if i == 0 else "MeSH Terms"
            mesh_parts.append(f'"{t}"[{tag}]')
        mesh_block = " AND ".join(mesh_parts)

        # Add free-text keywords for recall
        keywords = _strip_stopwords(query, max_tokens=4)
        if keywords:
            kw_parts = [f"{w}[tiab]" for w in keywords.split()[:3]]
            kw_block = " AND ".join(kw_parts)
            base = f"({mesh_block}) AND ({kw_block})"
        else:
            base = mesh_block
    else:
        # No MeSH - build from drug names + diseases or plain keywords
        parts: list[str] = []
        if drug_names:
            drug_terms = [f'"{d}"[tiab]' for d in drug_names[:2]]
            parts.append("(" + " OR ".join(drug_terms) + ")")
        if diseases:
            disease_terms = [f'"{d}"[tiab]' for d in diseases[:2]]
            parts.append("(" + " OR ".join(disease_terms) + ")")
        if parts:
            base = " AND ".join(parts)
            # Also add remaining keywords for context
            extra_kw = _strip_stopwords(query, max_tokens=3)
            if extra_kw and len(parts) < 2:
                base = f"({base}) AND ({extra_kw}[tiab])"
        else:
            # Pure keyword query - tag with [tiab] for precision
            keywords = _strip_stopwords(query, max_tokens=6)
            if not keywords:
                return query
            kw_list = keywords.split()
            if len(kw_list) <= 3:
                base = " AND ".join(f"{w}[tiab]" for w in kw_list)
            else:
                # Group first 3 as AND, rest as optional
                core = " AND ".join(f"{w}[tiab]" for w in kw_list[:3])
                rest = " ".join(kw_list[3:])
                base = f"({core}) AND ({rest})"

    if evidence_type and evidence_type in _PT_FILTERS:
        base = f"({base}) AND {_PT_FILTERS[evidence_type]}"

    return base


def _simplify_query(query: str) -> str:
    """Fallback: strip to plain keywords for broader PubMed matching."""
    clean = re.sub(r"\[.*?\]", " ", query)  # Remove field tags
    clean = re.sub(r"[\"'()[\]{}<>]", " ", clean)
    tokens = clean.split()
    keywords = [t for t in tokens if t.lower() not in _STOPWORDS and len(t) > 2]
    return " ".join(keywords[:6])


def _broaden_query(query: str) -> str:
    """Even broader: keep only 3-4 core medical terms."""
    clean = re.sub(r"\[.*?\]", " ", query)
    clean = re.sub(r"[\"'()[\]{}<>]", " ", clean)
    tokens = clean.split()
    keywords = [t for t in tokens if t.lower() not in _STOPWORDS and len(t) > 3]
    return " ".join(keywords[:4])


# ══════════════════════════════════════════════════════════════
#  PMC full-text retrieval
# ══════════════════════════════════════════════════════════════


# ELink returns several dbto=="pmc" linksetdbs per PMID: "pubmed_pmc" and
# "pubmed_pmc_local" point at the paper's own copy when it's open-access in
# PMC, while "pubmed_pmc_refs" points at unrelated PMC articles that cite the
# paper. Only the former identify the paper's own full text.
_PMC_OWN_COPY_LINKNAMES = ("pubmed_pmc", "pubmed_pmc_local")


def _pmids_to_pmcids(pmids: list[str]) -> dict[str, str]:
    """Use ELink to find PMC IDs for a list of PMIDs (batch, single call)."""
    if not pmids:
        return {}
    try:
        _rate_limit()
        resp = _get_client().get(
            _ELINK,
            params=_api_params({
                "dbfrom": "pubmed",
                "db": "pmc",
                "id": ",".join(pmids),
                "retmode": "json",
            }),
        )
        resp.raise_for_status()
        data = resp.json()
        mapping: dict[str, str] = {}
        for linkset in data.get("linksets", []):
            pmid = str(linkset.get("ids", [""])[0])
            by_linkname: dict[str, str] = {}
            for linksetdb in linkset.get("linksetdbs", []):
                linkname = linksetdb.get("linkname")
                if linksetdb.get("dbto") == "pmc" and linkname in _PMC_OWN_COPY_LINKNAMES:
                    pmc_ids = linksetdb.get("links", [])
                    if pmc_ids:
                        by_linkname[linkname] = str(pmc_ids[0])
            pmc_id = by_linkname.get("pubmed_pmc") or by_linkname.get("pubmed_pmc_local")
            if pmc_id:
                mapping[pmid] = pmc_id
        _record_success()
        return mapping
    except Exception as exc:
        logger.debug("ELink PMID→PMC failed: %s", exc)
        _record_failure()
        return {}


_PMID_URL_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)")


def fetch_full_text_by_url(url: str) -> str:
    """Full text for a ``pubmed.ncbi.nlm.nih.gov/{pmid}/`` URL, via PMC.

    The abstract page itself cannot be scraped (see content_extractor.py's
    allowlist comment - it serves a cookie-challenge to a plain GET), so this
    is the only route to a PubMed source's full text: PMID -> the paper's own
    PMC copy, if it has one.
    """
    match = _PMID_URL_RE.search(url or "")
    if not match:
        return ""
    pmid = match.group(1)
    pmc_map = _pmids_to_pmcids([pmid])
    pmc_id = pmc_map.get(pmid)
    return _fetch_pmc_fulltext(pmc_id) if pmc_id else ""


def _fetch_pmc_fulltext(pmc_id: str) -> str:
    """Fetch full-text body from PMC for an open-access article."""
    try:
        _rate_limit()
        resp = _get_client().get(
            _PMC_FETCH,
            params=_api_params({
                "db": "pmc",
                "id": pmc_id,
                "rettype": "xml",
                "retmode": "xml",
            }),
        )
        if resp.status_code != 200:
            return ""
        root = ElementTree.fromstring(resp.content)
        # Extract body text from the JATS XML
        body = root.find(".//body")
        if body is None:
            return ""
        parts: list[str] = []
        # The abstract lives in <front>, not <body> - JATS never repeats it
        # there, so without this the text a reader gets starts with the
        # Introduction's generic background instead of the paper's own dense
        # summary of what it found.
        abstract_el = root.find(".//abstract")
        if abstract_el is not None:
            abstract_text = re.sub(r"\s+", " ", "".join(abstract_el.itertext())).strip()
            if abstract_text:
                parts.append(f"## Abstract\n{abstract_text}\n")
        for sec in body.iter("sec"):
            title_el = sec.find("title")
            # itertext() picks up a title wrapped in inline markup (e.g.
            # "<title><italic>In vivo</italic> results</title>") that
            # title_el.text alone would miss entirely, leaving the section's
            # paragraphs unmarked and silently folded into whichever section
            # precedes it.
            if title_el is not None:
                title = re.sub(r"\s+", " ", "".join(title_el.itertext())).strip()
                if title:
                    parts.append(f"\n## {title}\n")
            for p in sec.findall("p"):
                text = "".join(p.itertext()).strip()
                if text:
                    parts.append(text)
        fulltext = "\n".join(parts)
        if not fulltext:
            return ""
        _record_success()
        # Returned with its `## Section` markers intact and only a generous
        # safety ceiling applied, not the tight budget a model's context
        # window needs - synthesis.py reads sections out of this by name
        # (read_section) or asks for a priority-filled excerpt of it
        # (read_full_text); trimming here would throw away the material
        # either of those needs before it ever reaches that decision.
        return fulltext[:_MAX_RAW_FULLTEXT_CHARS]
    except Exception as exc:
        logger.debug("PMC full-text fetch failed for %s: %s", pmc_id, exc)
        return ""


# ══════════════════════════════════════════════════════════════
#  Rich metadata extraction from PubMed XML
# ══════════════════════════════════════════════════════════════


def _extract_article_metadata(article: ElementTree.Element) -> dict:
    """Extract comprehensive metadata from a PubmedArticle XML element."""
    # Mixed content (sub/sup/italic tags inside the title) needs itertext(),
    # not .text, to pick up everything.
    title_el = article.find(".//ArticleTitle")
    title = "".join(title_el.itertext()).strip() if title_el is not None else ""

    # A structured abstract carries a Label per AbstractText element
    # (Background, Methods, Results, ...); an unstructured one has none.
    abs_parts: list[str] = []
    for abs_el in article.findall(".//AbstractText"):
        label = abs_el.get("Label", "")
        text = "".join(abs_el.itertext()).strip()
        if label:
            abs_parts.append(f"**{label}**: {text}")
        elif text:
            abs_parts.append(text)
    abstract = "\n".join(abs_parts)

    pmid_el = article.find(".//PMID")
    pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else ""

    doi = ""
    for aid in article.findall(".//ArticleId"):
        if aid.get("IdType") == "doi" and aid.text:
            doi = aid.text.strip()
            break
    # A record's DOI isn't always in ArticleId; ELocationID is the other
    # place PubMed puts it.
    if not doi:
        for eloc in article.findall(".//ELocationID"):
            if eloc.get("EIdType") == "doi" and eloc.text:
                doi = eloc.text.strip()
                break

    authors: list[str] = []
    for author in article.findall(".//Author"):
        last = author.findtext("LastName", "")
        initials = author.findtext("Initials", "")
        if last:
            authors.append(f"{last} {initials}".strip())
    author_str = ", ".join(authors[:6])
    if len(authors) > 6:
        author_str += " et al."

    journal = article.findtext(".//Journal/Title", "") or article.findtext(
        ".//Journal/ISOAbbreviation", ""
    )

    pub_date = ""
    date_el = article.find(".//PubDate")
    if date_el is not None:
        year = date_el.findtext("Year", "")
        month = date_el.findtext("Month", "")
        day = date_el.findtext("Day", "")
        pub_date = "-".join(p for p in [year, month, day] if p)

    pub_types: list[str] = []
    for pt in article.findall(".//PublicationType"):
        if pt.text:
            pub_types.append(pt.text.strip())

    mesh_headings: list[str] = []
    for mh in article.findall(".//MeshHeading/DescriptorName"):
        if mh.text:
            mesh_headings.append(mh.text.strip())

    return {
        "title": title,
        "abstract": abstract,
        "pmid": pmid,
        "doi": doi,
        "authors": author_str,
        "journal": journal,
        "pub_date": pub_date,
        "pub_types": pub_types,
        "mesh_headings": mesh_headings,
    }


# ══════════════════════════════════════════════════════════════
#  Main provider
# ══════════════════════════════════════════════════════════════


class PubMedProvider(SourceProvider):
    """Fetch biomedical evidence from PubMed with smart, resilient queries.

    Features
    --------
    * **Multi-stage query strategy**: structured Boolean → simplified → broadened
    * **MeSH-tagged queries** when mesh_terms/drug_names/diseases are provided
    * **Publication type filters** (systematic reviews, RCTs, guidelines)
    * **Rich metadata**: authors, journal, DOI, publication types, MeSH headings
    * **Circuit breaker**: stops hitting NCBI after 5 consecutive failures
    * **Adaptive rate limiting** with NCBI API key support (10 req/s)
    * **Connection pooling** via persistent httpx.Client
    """

    source_type = "pubmed"

    def __init__(
        self,
        *,
        mesh_terms: list[str] | None = None,
        drug_names: list[str] | None = None,
        diseases: list[str] | None = None,
        evidence_type: str | None = None,
        prefer_reviews: bool = False,
        recent_only: bool = False,
    ):
        self._mesh_terms = mesh_terms or []
        self._drug_names = drug_names or []
        self._diseases = diseases or []
        self._evidence_type = evidence_type
        self._prefer_reviews = prefer_reviews
        self._recent_only = recent_only

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=15),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
    )
    def search(self, query: str, max_results: int = 5) -> list[SourceResult]:
        if _check_circuit():
            logger.warning("PubMed circuit breaker is OPEN - skipping search")
            return []

        if not query or len(query.strip()) < 3:
            return []

        # Date range for time-sensitive claims (epidemiology, statistics)
        date_range: dict[str, str] | None = None
        if self._recent_only:
            import datetime as _dt
            _today = _dt.date.today()
            date_range = {
                "datetype": "pdat",
                "mindate": str(_today.year - 5),
                "maxdate": str(_today.year),
            }

        # ── Stage -1: Cochrane Reviews first (gold standard) ──
        # Cochrane systematic reviews are the highest-quality evidence
        # source. Search specifically in the Cochrane Database of
        # Systematic Reviews journal before broader SR search.
        sr_pmids: list[str] = []
        all_pmids: list[str] = []
        if self._prefer_reviews or self._evidence_type in ("systematic_review", "review"):
            cochrane_q = build_pubmed_query(
                query,
                mesh_terms=self._mesh_terms,
                drug_names=self._drug_names,
                diseases=self._diseases,
                evidence_type="cochrane_review",
            )
            logger.debug("PubMed query (Cochrane reviews): %s", cochrane_q[:200])
            cochrane_pmids = self._esearch(cochrane_q, max_results, extra_params=date_range)
            all_pmids.extend(cochrane_pmids)

        # ── Stage 0: systematic reviews & meta-analyses ──
        # Top of the evidence hierarchy: a systematic review aggregates the
        # primary studies, so retrieving one is worth more than retrieving
        # several of the studies it already summarises.
        wants_reviews = (
            self._prefer_reviews
            or self._evidence_type in ("systematic_review", "review")
        )
        if wants_reviews and len(all_pmids) < max_results:
                sr_q = build_pubmed_query(
                    query,
                    mesh_terms=self._mesh_terms,
                    drug_names=self._drug_names,
                    diseases=self._diseases,
                    evidence_type="systematic_review",
                )
                logger.debug("PubMed query (systematic reviews): %s", sr_q[:200])
                sr_pmids = self._esearch(sr_q, max_results, extra_params=date_range)
                for pid in sr_pmids:
                    if pid not in all_pmids:
                        all_pmids.append(pid)

        # ── Stage 0b: Reviews (broader than systematic reviews) ──
        if self._prefer_reviews and len(all_pmids) < max_results:
            review_q = build_pubmed_query(
                query,
                mesh_terms=self._mesh_terms,
                drug_names=self._drug_names,
                diseases=self._diseases,
                evidence_type="review",
            )
            logger.debug("PubMed query (reviews): %s", review_q[:200])
            review_pmids = self._esearch(review_q, max_results, extra_params=date_range)
            for pid in review_pmids:
                if pid not in all_pmids:
                    all_pmids.append(pid)

        # ── Stage 1: Structured Boolean query ──
        if not all_pmids:
            structured_q = build_pubmed_query(
                query,
                mesh_terms=self._mesh_terms,
                drug_names=self._drug_names,
                diseases=self._diseases,
                evidence_type=self._evidence_type,
            )
            logger.debug("PubMed query (structured): %s", structured_q[:200])
            all_pmids = self._esearch(structured_q, max_results * 2, extra_params=date_range)

        # ── Stage 2: Simplified keywords (no field tags) ──
        # Extends rather than replaces: a structured query that matched only
        # a couple of PMIDs was almost certainly over-constrained by its AND
        # clauses, and the papers a looser query would add are as real as the
        # ones already found - only a query that matched *nothing at all*
        # tells you the wording, not the constraint count, was the problem.
        if len(all_pmids) < max(3, max_results // 2):
            simplified = _simplify_query(query)
            if simplified and simplified != query:
                logger.debug("PubMed retry (simplified): %s", simplified[:120])
                for pid in self._esearch(simplified, max_results * 2, extra_params=date_range):
                    if pid not in all_pmids:
                        all_pmids.append(pid)

        # ── Stage 3: Broadened (very few core terms) ──
        if len(all_pmids) < max(3, max_results // 2):
            broadened = _broaden_query(query)
            if broadened and broadened != query:
                logger.debug("PubMed retry (broadened): %s", broadened[:120])
                for pid in self._esearch(broadened, max_results, extra_params=date_range):
                    if pid not in all_pmids:
                        all_pmids.append(pid)

        if not all_pmids:
            return []

        articles = self._efetch(all_pmids[:max_results])
        if not articles:
            return []

        # ── Prioritize systematic reviews and meta-analyses in results ──
        articles = self._sort_by_evidence_quality(articles)

        return articles

    @staticmethod
    def _sort_by_evidence_quality(results: list[SourceResult]) -> list[SourceResult]:
        """Sort results so systematic reviews and meta-analyses come first."""
        def _quality_score(r: SourceResult) -> int:
            text = (r.content + " " + r.snippet).lower()
            if "systematic review" in text or "meta-analysis" in text or "meta-analys" in text:
                return 0
            if "randomized controlled" in text or "rct" in text:
                return 1
            if "guideline" in text or "practice recommendation" in text:
                return 2
            if "cohort" in text or "prospective" in text:
                return 3
            if "review" in text:
                return 4
            return 5
        return sorted(results, key=_quality_score)

    # ── E-utilities calls ─────────────────────────────────────

    @staticmethod
    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(2),
        retry=retry_if_exception(_is_retryable_status),
    )
    def _esearch(
        query: str,
        max_results: int,
        extra_params: dict[str, str] | None = None,
    ) -> list[str]:
        _rate_limit()
        try:
            params = _api_params({
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "sort": "relevance",
                "usehistory": "n",
            })
            if extra_params:
                params.update(extra_params)
            resp = _get_client().get(_ESEARCH, params=params)
            resp.raise_for_status()
            data = resp.json()
            result = data.get("esearchresult", {})

            # Check for errors/warnings in the response
            error_list = result.get("errorlist", {})
            if error_list:
                phrases_not_found = error_list.get("phrasesnotfound", [])
                if phrases_not_found:
                    logger.debug(
                        "PubMed: phrases not found: %s",
                        phrases_not_found,
                    )

            pmids = result.get("idlist", [])
            if pmids:
                _record_success()
                count = result.get("count", len(pmids))
                logger.debug(
                    "PubMed esearch: %s total hits, returning %d PMIDs",
                    count, len(pmids),
                )
            return pmids
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                logger.warning("PubMed rate-limited (429) - backing off")
            elif exc.response.status_code >= 500:
                logger.warning("PubMed server error (%d)", exc.response.status_code)
            _record_failure()
            raise
        except Exception as exc:
            logger.debug("PubMed esearch failed: %s", exc)
            _record_failure()
            raise

    @staticmethod
    def probe(query: str, max_titles: int = 5) -> dict:
        """Test a query against PubMed without running a full search.

        Returns the total hit count, PubMed's own translation of the query,
        and a few matching titles.

        The translation is the interesting part. PubMed does not match the
        words it is given: it expands them through MeSH and its synonym
        tables, and the expansion decides what comes back. ``heart attack``
        becomes ``"myocardial infarction"[MeSH Terms] OR …``; a misspelled
        drug name or an acronym PubMed does not know expands to nothing and
        the query quietly retrieves the wrong literature. Exposing the
        translation lets a caller see that before committing a search to it.

        Never raises: a failed probe answers with an error field, so a model
        using this as a tool can read the failure and try something else.
        """
        query = (query or "").strip()
        if not query:
            return {"query": query, "error": "empty query"}

        _rate_limit()
        try:
            resp = _get_client().get(_ESEARCH, params=_api_params({
                "db": "pubmed",
                "term": query,
                "retmax": max_titles,
                "retmode": "json",
                "sort": "relevance",
            }))
            resp.raise_for_status()
            result = resp.json().get("esearchresult", {})
        except Exception as exc:
            logger.debug("PubMed probe failed for '%s': %s", query[:60], exc)
            return {"query": query, "error": str(exc)[:200]}

        pmids = result.get("idlist", []) or []
        titles: list[str] = []
        if pmids:
            try:
                titles = [r.title for r in PubMedProvider._efetch(pmids[:max_titles])]
            except Exception as exc:
                logger.debug("PubMed probe efetch failed: %s", exc)

        return {
            "query": query,
            "hit_count": int(result.get("count", 0) or 0),
            "query_translation": (result.get("querytranslation") or "")[:600],
            "unmatched_phrases": (result.get("errorlist", {}) or {}).get("phrasesnotfound", [])[:5],
            "sample_titles": titles,
        }

    @staticmethod
    def _efetch(pmids: list[str]) -> list[SourceResult]:
        _rate_limit()
        try:
            resp = _get_client().get(
                _EFETCH,
                params=_api_params({
                    "db": "pubmed",
                    "id": ",".join(pmids),
                    "rettype": "abstract",
                    "retmode": "xml",
                }),
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.debug("PubMed efetch failed: %s", exc)
            _record_failure()
            return []

        _record_success()
        root = ElementTree.fromstring(resp.content)
        results: list[SourceResult] = []

        for article_el in root.iter("PubmedArticle"):
            meta = _extract_article_metadata(article_el)
            if not meta["abstract"] and not meta["title"]:
                continue

            url = f"https://pubmed.ncbi.nlm.nih.gov/{meta['pmid']}/" if meta["pmid"] else ""

            snippet_parts: list[str] = []
            if meta["authors"]:
                snippet_parts.append(meta["authors"])
            if meta["journal"]:
                snippet_parts.append(meta["journal"])
            if meta["pub_date"]:
                snippet_parts.append(meta["pub_date"])
            if meta["pub_types"]:
                snippet_parts.append("; ".join(meta["pub_types"][:3]))
            snippet = " | ".join(snippet_parts)

            # Content = abstract. Full text, when useful, is fetched on demand
            # by the synthesis stage (fetch_full_text_by_url below) - not here.
            content = meta["abstract"]

            # Prepend key metadata to content for the verifier
            content_prefix_parts: list[str] = []
            if meta["pub_types"]:
                content_prefix_parts.append(
                    f"Study type: {', '.join(meta['pub_types'][:3])}"
                )
            if meta["mesh_headings"]:
                content_prefix_parts.append(
                    f"MeSH: {', '.join(meta['mesh_headings'][:6])}"
                )
            if content_prefix_parts:
                prefix = " | ".join(content_prefix_parts)
                content = f"[{prefix}]\n\n{content}"

            results.append(
                SourceResult(
                    publication_types=list(meta["pub_types"]),
                    title=meta["title"],
                    url=url,
                    snippet=snippet[:300],
                    content=content[:_MAX_CONTENT_LEN],
                    source_type="pubmed",
                    reliability_tier=2,
                    publication_date=meta["pub_date"],
                    language="en",
                )
            )

        return results
