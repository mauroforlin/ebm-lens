"use strict";

const STREAM_API = "/api/related-articles/stream";

const el = (id) => document.getElementById(id);
const topicInput   = el("topic");
const searchBtn    = el("searchBtn");
const progress     = el("progress");
const progressStage= el("progressStage");
const elapsedEl    = el("elapsed");
const progressLog  = el("progressLog");
const errorEl      = el("error");
const resultsEl    = el("results");
const historyToggle= el("historyToggle");
const historyCount = el("historyCount");
const historyCountDrawer = el("historyCountDrawer");
const historyPanel = el("historyPanel");
const historyBackdrop = el("historyBackdrop");
const historyCloseBtn = el("historyCloseBtn");
const historyFilter = el("historyFilter");
const historyList  = el("historyList");
const clearHistoryBtn = el("clearHistoryBtn");
const sourcesPreset  = el("sourcesPreset");
const legendToggle = el("legendToggle");
const legendPopover = el("legendPopover");

let maxSources = 10;
let timerId = null;

function setMaxSources(value) {
  maxSources = value;
  [...sourcesPreset.querySelectorAll("button")].forEach((b) =>
    b.setAttribute("aria-pressed", String(Number(b.dataset.value) === value)));
}

sourcesPreset.addEventListener("click", (event) => {
  const btn = event.target.closest("button");
  if (btn) setMaxSources(Number(btn.dataset.value));
});

/* ── Search history (IndexedDB) ─────────────────────────────────
   Kept entirely client-side: the backend is deliberately stateless
   (see README), so "revisitable" means storing the full response
   here, not just the query — a saved entry reopens instantly, with
   nothing re-sent to the API.                                      */

const HISTORY_DB = "ebm-lens-history";
const HISTORY_STORE = "searches";
const HISTORY_LIMIT = 50;
const hasIndexedDB = typeof indexedDB !== "undefined";

function openHistoryDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(HISTORY_DB, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(HISTORY_STORE)) {
        db.createObjectStore(HISTORY_STORE, { keyPath: "id" }).createIndex("ts", "ts");
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function txDone(tx) {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function getAllHistory(db) {
  const tx = db.transaction(HISTORY_STORE, "readonly");
  const records = await new Promise((resolve, reject) => {
    const req = tx.objectStore(HISTORY_STORE).getAll();
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return records.sort((a, b) => b.ts - a.ts);
}

async function saveToHistory(record) {
  const db = await openHistoryDB();
  const tx = db.transaction(HISTORY_STORE, "readwrite");
  tx.objectStore(HISTORY_STORE).put(record);
  await txDone(tx);

  const all = await getAllHistory(db);
  if (all.length > HISTORY_LIMIT) {
    const trimTx = db.transaction(HISTORY_STORE, "readwrite");
    all.slice(HISTORY_LIMIT).forEach((r) => trimTx.objectStore(HISTORY_STORE).delete(r.id));
    await txDone(trimTx);
  }
  db.close();
}

async function loadHistory() {
  const db = await openHistoryDB();
  const all = await getAllHistory(db);
  db.close();
  return all;
}

async function getHistoryEntry(id) {
  const db = await openHistoryDB();
  const tx = db.transaction(HISTORY_STORE, "readonly");
  const record = await new Promise((resolve, reject) => {
    const req = tx.objectStore(HISTORY_STORE).get(id);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  db.close();
  return record;
}

async function deleteHistoryEntry(id) {
  const db = await openHistoryDB();
  const tx = db.transaction(HISTORY_STORE, "readwrite");
  tx.objectStore(HISTORY_STORE).delete(id);
  await txDone(tx);
  db.close();
}

async function clearAllHistory() {
  const db = await openHistoryDB();
  const tx = db.transaction(HISTORY_STORE, "readwrite");
  tx.objectStore(HISTORY_STORE).clear();
  await txDone(tx);
  db.close();
}

function formatRelativeTime(ts) {
  const sec = Math.floor((Date.now() - ts) / 1000);
  if (sec < 60) return "just now";
  const min = Math.floor(sec / 60);
  if (min < 60) return min + "m ago";
  const hr = Math.floor(min / 60);
  if (hr < 24) return hr + "h ago";
  const day = Math.floor(hr / 24);
  if (day < 7) return day + "d ago";
  return new Date(ts).toLocaleDateString();
}

/* Buckets read as "when", not exact dates: the drawer is a register you
   skim, and a relative shelf finds a search faster than a sorted list. */
function historyBucket(ts) {
  const startOfDay = (ms) => { const d = new Date(ms); d.setHours(0, 0, 0, 0); return d.getTime(); };
  const diffDays = Math.round((startOfDay(Date.now()) - startOfDay(ts)) / 86400000);
  if (diffDays <= 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (diffDays < 7) return "This week";
  return "Earlier";
}

function groupHistory(records) {
  const order = ["Today", "Yesterday", "This week", "Earlier"];
  const groups = new Map();
  for (const r of records) {
    const bucket = historyBucket(r.ts);
    if (!groups.has(bucket)) groups.set(bucket, []);
    groups.get(bucket).push(r);
  }
  return order.filter((b) => groups.has(b)).map((b) => [b, groups.get(b)]);
}

function historyItemHtml(r) {
  return `
    <div class="history-item">
      <button type="button" class="history-item-main" data-id="${esc(r.id)}">
        <span class="history-topic">${esc(r.topic)}</span>
        <span class="history-meta">
          ${r.domain ? `<span class="history-domain">${esc(r.domain)}</span>` : ""}
          <span>${r.articleCount} source${r.articleCount === 1 ? "" : "s"}</span>
          <span class="history-time">${formatRelativeTime(r.ts)}</span>
        </span>
      </button>
      <button type="button" class="history-delete" data-id="${esc(r.id)}" title="Remove from history" aria-label="Remove from history">×</button>
    </div>`;
}

let allHistoryRecords = [];

function renderHistoryList() {
  const query = historyFilter.value.trim().toLowerCase();
  const filtered = query
    ? allHistoryRecords.filter((r) => r.topic.toLowerCase().includes(query))
    : allHistoryRecords;

  const total = allHistoryRecords.length;
  historyCount.textContent = total ? String(total) : "";
  historyCountDrawer.textContent = total ? String(total) : "";
  clearHistoryBtn.disabled = !total;

  if (!total) {
    historyList.innerHTML = `<p class="history-empty">No searches yet: they'll appear here after you run one.</p>`;
    return;
  }
  if (!filtered.length) {
    historyList.innerHTML = `<p class="history-empty">No saved search matches "${esc(historyFilter.value.trim())}".</p>`;
    return;
  }
  historyList.innerHTML = groupHistory(filtered).map(([label, items]) => `
    <div class="history-group">
      <h3 class="history-group-label">${label}</h3>
      <div class="history-item-list">${items.map(historyItemHtml).join("")}</div>
    </div>`).join("");
}

async function refreshHistoryPanel() {
  if (!hasIndexedDB) return;
  allHistoryRecords = await loadHistory().catch(() => []);
  renderHistoryList();
}

function applyHistoryEntry(record) {
  topicInput.value = record.topic;
  el("language").value = record.language || "it";
  setMaxSources(record.maxSources || 10);
  errorEl.hidden = true;
  stopProgress();
  render(record.data);
  closeHistoryPanel();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function openHistoryPanel() {
  historyBackdrop.hidden = false;
  historyPanel.hidden = false;
  requestAnimationFrame(() => {
    historyBackdrop.classList.add("in");
    historyPanel.classList.add("in");
  });
  historyPanel.setAttribute("aria-hidden", "false");
  historyToggle.setAttribute("aria-expanded", "true");
  document.body.style.overflow = "hidden";
  refreshHistoryPanel();
  historyFilter.focus();
}

function closeHistoryPanel() {
  if (historyPanel.hidden) return;
  historyBackdrop.classList.remove("in");
  historyPanel.classList.remove("in");
  historyPanel.setAttribute("aria-hidden", "true");
  historyToggle.setAttribute("aria-expanded", "false");
  document.body.style.overflow = "";
  historyFilter.value = "";
  setTimeout(() => { historyBackdrop.hidden = true; historyPanel.hidden = true; }, 200);
  historyToggle.focus();
}

if (hasIndexedDB) {
  historyToggle.addEventListener("click", () => {
    if (historyPanel.hidden) openHistoryPanel(); else closeHistoryPanel();
  });
  historyCloseBtn.addEventListener("click", closeHistoryPanel);
  historyBackdrop.addEventListener("click", closeHistoryPanel);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !historyPanel.hidden) closeHistoryPanel();
  });

  historyFilter.addEventListener("input", renderHistoryList);

  clearHistoryBtn.addEventListener("click", async () => {
    if (!confirm("Clear all saved search history? This cannot be undone.")) return;
    await clearAllHistory();
    refreshHistoryPanel();
  });

  historyList.addEventListener("click", async (event) => {
    const delBtn = event.target.closest(".history-delete");
    if (delBtn) {
      await deleteHistoryEntry(delBtn.dataset.id);
      refreshHistoryPanel();
      return;
    }
    const mainBtn = event.target.closest(".history-item-main");
    if (mainBtn) {
      const record = await getHistoryEntry(mainBtn.dataset.id);
      if (record) applyHistoryEntry(record);
    }
  });

  refreshHistoryPanel();
} else {
  historyToggle.hidden = true;
}

if (legendToggle && legendPopover) {
  legendToggle.addEventListener("click", (event) => {
    event.stopPropagation();
    const isOpen = legendPopover.classList.toggle("is-open");
    legendToggle.setAttribute("aria-expanded", String(isOpen));
  });

  document.addEventListener("click", (event) => {
    if (!legendPopover.contains(event.target) && !legendToggle.contains(event.target)) {
      legendPopover.classList.remove("is-open");
      legendToggle.setAttribute("aria-expanded", "false");
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && legendPopover.classList.contains("is-open")) {
      legendPopover.classList.remove("is-open");
      legendToggle.setAttribute("aria-expanded", "false");
      legendToggle.focus();
    }
  });
}

/* ── Progress feedback ──────────────────────────────────────────
   Real backend events over SSE (see app/core/events.py and the
   /related-articles/stream endpoint): each line below is a stage the
   orchestrator actually just ran, with its real counts, not a guess
   from elapsed time.                                                */

let progressStarted = 0;

function startProgress() {
  progressStarted = performance.now();
  progress.hidden = false;
  progressStage.innerHTML = '<span class="live-dot"></span>Connecting…';
  elapsedEl.textContent = "0.0s";
  progressLog.innerHTML = "";

  timerId = setInterval(() => {
    elapsedEl.textContent = ((performance.now() - progressStarted) / 1000).toFixed(1) + "s";
  }, 100);
}

function stopProgress() {
  clearInterval(timerId);
  progress.hidden = true;
}

function pushProgressEvent(message) {
  const seconds = ((performance.now() - progressStarted) / 1000).toFixed(1);
  progressStage.innerHTML = `<span class="live-dot"></span>${esc(message)}`;
  const li = document.createElement("li");
  li.innerHTML = `<span class="progress-log-time">${seconds}s</span><span>${esc(message)}</span>`;
  progressLog.appendChild(li);
  progressLog.scrollTop = progressLog.scrollHeight;
}

/* ── SSE parsing ────────────────────────────────────────────────
   Plain fetch + a manual reader, not EventSource: EventSource can only
   GET, and the topic (free text, can be long) belongs in a POST body. */

async function streamEvents(response, onFrame) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const rawFrame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const frame = parseSseFrame(rawFrame);
      if (frame) onFrame(frame);
    }
  }
}

function parseSseFrame(raw) {
  let eventType = "message";
  const dataLines = [];
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) eventType = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!dataLines.length) return null;
  try {
    return { event: eventType, data: JSON.parse(dataLines.join("\n")) };
  } catch {
    return null;
  }
}

/* ── Search ─────────────────────────────────────────────────── */

async function runSearch() {
  const topic = topicInput.value.trim();
  if (topic.length < 3) {
    showError("Enter a topic of at least 3 characters.");
    return;
  }

  searchBtn.disabled = true;
  errorEl.hidden = true;
  resultsEl.innerHTML = "";
  startProgress();

  try {
    const response = await fetch(STREAM_API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        topic,
        max_sources: maxSources,
        summary_language: el("language").value
      })
    });

    if (!response.ok) {
      const data = await response.json().catch(() => null);
      showError(describeHttpError(response, data));
      return;
    }

    let finalData = null;
    let streamError = null;
    await streamEvents(response, (frame) => {
      if (frame.event === "progress") pushProgressEvent(frame.data.message);
      else if (frame.event === "result") finalData = frame.data;
      else if (frame.event === "error") streamError = frame.data.error;
    });

    if (streamError) {
      showError(streamError);
      return;
    }
    if (!finalData || finalData.status === "failed") {
      showError((finalData && finalData.error) || "The pipeline failed without a message.");
      return;
    }
    render(finalData);
    if (hasIndexedDB) {
      saveToHistory({
        id: (crypto.randomUUID ? crypto.randomUUID() : Date.now() + "-" + Math.random().toString(36).slice(2)),
        ts: Date.now(),
        topic,
        language: el("language").value,
        maxSources: maxSources,
        articleCount: finalData.articles.length,
        domain: finalData.domain_detected || "",
        data: finalData
      }).then(refreshHistoryPanel).catch(() => {});
    }
  } catch (err) {
    showError("Could not reach the API: " + err.message);
  } finally {
    stopProgress();
    searchBtn.disabled = false;
  }
}

function describeHttpError(response, data) {
  if (response.status === 401) return "Unauthorised: this instance requires an API key.";
  if (data && data.detail) {
    return typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
  }
  return "Request failed with status " + response.status + ".";
}

function showError(message) {
  errorEl.textContent = message;
  errorEl.hidden = false;
}

/* ── Rendering ──────────────────────────────────────────────── */

const TIER = {
  1: { cls: "t1", label: "institutional" },
  2: { cls: "t2", label: "curated" },
  3: { cls: "t3", label: "general" }
};

function esc(value) {
  const node = document.createElement("div");
  node.textContent = value == null ? "" : String(value);
  return node.innerHTML;
}

const num = (value) => (value || 0).toLocaleString("en-US");

/* Mirrors the prefixes app/pipeline/synthesis.py._no_direct_evidence_prefix
   prepends server-side when every cited source is adjacent at best - split
   out so it renders as its own flag instead of the summary's first sentence. */
const NO_DIRECT_PREFIXES = [
  "No source directly addresses the question as asked: what follows is "
    + "extrapolated from sources on related but different populations or "
    + "conditions. ",
  "Nessuna fonte affronta direttamente la domanda posta: quanto segue e' "
    + "estrapolato da fonti su popolazioni o condizioni correlate ma diverse. ",
];

function splitNoDirectFlag(summary) {
  for (const prefix of NO_DIRECT_PREFIXES) {
    if (summary.startsWith(prefix)) return { flagged: true, rest: summary.slice(prefix.length) };
  }
  return { flagged: false, rest: summary };
}

function render(data) {
  const parts = [];

  if (data.global_summary) {
    const { flagged, rest } = splitNoDirectFlag(data.global_summary);
    parts.push(`
      <section class="overview">
        <h2>Overview</h2>
        ${flagged ? `<p class="no-direct-flag">No source directly addresses this exact question
          — the summary below is extrapolated from related evidence.</p>` : ""}
        <div class="overview-text">${renderOverviewMarkdown(esc(rest))}</div>
        ${renderFindings(data.key_findings)}
        ${renderProfile(data.evidence_profile)}
        ${renderConflicts(data.disagreements)}
        ${renderGaps(data.evidence_gaps)}
      </section>`);
  }

  parts.push(`
    <div class="result-count">
      <span>${data.articles.length} source${data.articles.length === 1 ? "" : "s"} selected
        from ${num(data.total_sources_consulted)} consulted</span>
      <span>${esc(data.domain_detected)} · ${data.duration_seconds}s</span>
    </div>`);

  if (!data.articles.length) {
    parts.push(`<p class="empty">No source cleared the relevance bar for this topic.
      Try rephrasing it, or naming the clinical term directly.</p>`);
  } else {
    parts.push(renderFilters(data.articles));
    parts.push(data.articles.map(renderArticle).join(""));
  }

  parts.push(renderStats(data));
  resultsEl.innerHTML = parts.join("");
}

function stripInlineCitations(text) {
  if (!text) return "";
  return text.replace(/\s*\[\d+(?:\s*[,–-]\s*\d+)*\]/g, "").trim();
}

/* Turn the [n] markers the synthesis writes into links to the source itself.
   Runs on already-escaped text, so the only markup it introduces is its own. */
function linkCitations(escaped) {
  return escaped.replace(/\[(\d+)\]/g, (match, index) =>
    `<a class="cite" href="#src-${index}">[${index}]</a>`);
}

/* The overview is allowed a small, closed Markdown subset - bold, italic,
   blank-line paragraphs - see the synthesis prompt's formatting rule. Runs
   on already-escaped text, same as linkCitations, so it only ever introduces
   the two tags below; nothing else in the input can produce markup. */
function renderOverviewMarkdown(escaped) {
  const paragraphs = escaped
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean);
  return paragraphs
    .map((block) => {
      let html = block
        .replace(/\*\*([^*\n]+?)\*\*/g, "<strong>$1</strong>")
        .replace(/(^|[^*])\*([^*\n]+?)\*(?!\*)/g, "$1<em>$2</em>");
      return `<p>${linkCitations(html)}</p>`;
    })
    .join("");
}

function citeLinks(indices) {
  if (!indices || !indices.length) return "";
  return indices.map((i) => `<a class="cite" href="#src-${i}">[${i}]</a>`).join("");
}

/* Each finding carries the sources it rests on: claims the synthesis could not
   attribute are dropped server-side, so everything here is followable. */
function renderFindings(findings) {
  if (!findings || !findings.length) return "";
  const items = findings.map((finding) => {
    const strength = ["strong", "moderate", "weak"].includes(finding.strength)
      ? finding.strength : "moderate";
    return `<li class="finding">
      <span class="strength ${strength}">${strength}</span>
      <span>${esc(stripInlineCitations(finding.text))}${citeLinks(finding.source_indices)}</span>
    </li>`;
  }).join("");
  return `<ul class="findings">${items}</ul>`;
}

/* Where the sources disagree, the disagreement is the finding. */
function renderConflicts(conflicts) {
  if (!conflicts || !conflicts.length) return "";
  const items = conflicts.map((conflict) => {
    const positions = (conflict.positions || []).length
      ? `<span class="positions">: ${conflict.positions.map(esc).join(" vs ")}</span>`
      : "";
    return `<div class="conflict">
      <b>${esc(stripInlineCitations(conflict.issue))}</b>${positions}${citeLinks(conflict.source_indices)}
    </div>`;
  }).join("");
  return `<div class="conflicts"><h3>Sources disagree</h3>${items}</div>`;
}

function renderGaps(gaps) {
  if (!gaps || !gaps.length) return "";
  return `<div class="gaps">
    <h3>Not settled by these sources</h3>
    <ul>${gaps.map((gap) => `<li>${esc(gap)}</li>`).join("")}</ul>
  </div>`;
}

/* What kind of evidence the answer is made of - ten case reports and ten
   randomised trials look alike in a list and are not the same answer. */
function renderProfile(profile) {
  if (!profile || !profile.designs || !Object.keys(profile.designs).length) return "";
  const chips = Object.entries(profile.designs)
    .map(([design, count]) => `<span>${esc(design)} ×${count}</span>`).join("");
  const high = Math.round((profile.high_tier_share || 0) * 100);
  const preprints = Math.round((profile.preprint_share || 0) * 100);
  const notes = [`<span class="high">${high}% high-tier</span>`];
  if (preprints) notes.push(`<span>${preprints}% preprint</span>`);
  return `<div class="profile">${chips}${notes.join("")}</div>`;
}

/* Present only when a search actually mixes tiers - no point filtering a
   list that's all one kind. */
function renderFilters(articles) {
  const present = new Set(articles.map((a) => a.reliability_tier || 3));
  if (present.size < 2) return "";
  const chips = [1, 2, 3].filter((t) => present.has(t)).map((t) => {
    const tier = TIER[t];
    return `<button type="button" class="tag ${tier.cls} filter-chip" data-tier="${t}" aria-pressed="true">${tier.label}</button>`;
  }).join("");
  return `<div class="filters" role="group" aria-label="Filter by source tier">${chips}</div>`;
}

resultsEl.addEventListener("click", (event) => {
  const chip = event.target.closest(".filter-chip");
  if (!chip) return;
  chip.setAttribute("aria-pressed", String(chip.getAttribute("aria-pressed") !== "true"));
  const active = new Set([...resultsEl.querySelectorAll(".filter-chip[aria-pressed=\"true\"]")]
    .map((c) => c.dataset.tier));
  resultsEl.querySelectorAll("article[data-tier]").forEach((article) => {
    article.hidden = active.size > 0 && !active.has(article.dataset.tier);
  });
});

function renderArticle(article, index) {
  const tier = TIER[article.reliability_tier] || TIER[3];
  const relevance = Math.round((article.relevance_score || 0) * 100);
  const relClass = relevance >= 85 ? "r-vhigh"
                 : relevance >= 70 ? "r-high"
                 : relevance >= 55 ? "r-mid"
                 : relevance >= 40 ? "r-low"
                 : "r-vlow";
  const body = article.full_summary || article.snippet || "";

  const meta = [`<span class="tag source">${esc(article.source_type)}</span>`,
                `<span class="tag ${tier.cls}">${tier.label}</span>`];
  if (article.study_design) {
    // Shade the badge by position on the evidence hierarchy, so the strength
    // of a source is readable without knowing the ladder by heart.
    const level = article.evidence_level || 0;
    const rank = level >= 0.8 ? " d-high" : level >= 0.5 ? " d-mid" : "";
    meta.push(`<span class="tag design${rank}"
      title="Study design, appraised from the source itself">${esc(article.study_design)}</span>`);
  }
  if (article.publication_date) meta.push(`<span class="tag">${esc(article.publication_date)}</span>`);
  if (article.citation_count)  meta.push(`<span class="tag">${num(article.citation_count)} citations</span>`);
  if (article.directness === "adjacent") {
    meta.push(`<span class="tag extrapolated"
      title="Studies a related population or intervention, not the one asked about">extrapolated</span>`);
  }

  const population = article.population
    ? `<div class="article-facets"><span>studied: ${esc(article.population)}</span></div>`
    : "";

  return `
    <article id="src-${index}" data-tier="${article.reliability_tier || 3}">
      <h3><a href="${esc(article.url)}" target="_blank" rel="noopener noreferrer">${esc(article.title || article.url)}</a></h3>
      <div class="meta">
        ${meta.join("")}
        <span class="relevance" title="Relevance to your topic, judged after reading the source">
          <span class="relevance-track"><span class="relevance-fill ${relClass}" style="width:${relevance}%"></span></span>
          <span class="relevance-num">${relevance}%</span>
        </span>
      </div>
      ${body ? `<p>${esc(body)}</p>` : ""}
      ${population}
    </article>`;
}

/* ── Run statistics ─────────────────────────────────────────── */

/* Which lookups the model chose to make. The mix is the readable measure of
   whether the tools on offer are earning their place in the catalogue. */
function renderToolTable(calls) {
  if (!calls || !Object.keys(calls).length) return "";
  const rows = Object.entries(calls)
    .sort((a, b) => b[1] - a[1])
    .map(([name, count]) => `<tr><td>${esc(name)}</td><td>${num(count)}</td></tr>`)
    .join("");
  return `<table>
    <thead><tr><th>tool</th><th>calls</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

function renderStats(data) {
  const stats = data.job_stats;
  if (!stats || !Object.keys(stats).length) return "";

  const llm = stats.llm || {};
  const embeddings = stats.embeddings || {};
  const providers = stats.providers || {};
  const tools = stats.tools || {};

  const kpis = [
    ["$" + (stats.total_cost_usd || 0).toFixed(5), "total cost"],
    [num((llm.total_tokens || 0) + (embeddings.total_tokens || 0)), "tokens"],
    [num(llm.total_calls), "LLM calls"],
    [num(providers.total_calls), "provider calls"],
    [num(tools.total_calls), "tool calls"],
    [num(providers.total_cache_hits), "cache hits"],
    [data.duration_seconds + "s", "wall clock"]
  ].map(([value, label]) => `<div class="kpi"><div class="v">${value}</div><div class="l">${label}</div></div>`).join("");

  return `
    <details class="stats">
      <summary>Run statistics</summary>
      <div class="stats-body">
        <div class="kpis">${kpis}</div>
        ${renderPurposeTable(llm.by_purpose)}
        ${renderToolTable(tools.calls)}
        ${renderProviderTable(providers)}
        ${renderStageTable(stats.stage_timings_ms)}
      </div>
    </details>`;
}

function renderPurposeTable(byPurpose) {
  if (!byPurpose || !Object.keys(byPurpose).length) return "";
  const rows = Object.entries(byPurpose)
    .sort((a, b) => b[1].cost_usd - a[1].cost_usd)
    .map(([purpose, entry]) => `
      <tr>
        <td>${esc(purpose.replace(/^related_articles_/, ""))}</td>
        <td class="num">${entry.count}</td>
        <td class="num">${num(entry.input_tokens + entry.output_tokens)}</td>
        <td class="num">$${entry.cost_usd.toFixed(5)}</td>
      </tr>`).join("");
  return `
    <div class="stats-section">
      <h4>LLM calls by purpose</h4>
      <table>
        <tr><th>Stage</th><th class="num">Calls</th><th class="num">Tokens</th><th class="num">Cost</th></tr>
        ${rows}
      </table>
    </div>`;
}

function renderProviderTable(providers) {
  const calls = providers.calls || {};
  if (!Object.keys(calls).length) return "";
  const cached = providers.cache_hits || {};
  const errors = providers.errors || {};
  const rows = Object.entries(calls)
    .sort((a, b) => b[1] - a[1])
    .map(([provider, count]) => `
      <tr>
        <td>${esc(provider)}</td>
        <td class="num">${count}</td>
        <td class="num">${cached[provider] || 0}</td>
        <td class="num">${errors[provider] || 0}</td>
      </tr>`).join("");
  return `
    <div class="stats-section">
      <h4>Sources queried</h4>
      <table>
        <tr><th>Provider</th><th class="num">Calls</th><th class="num">Cached</th><th class="num">Errors</th></tr>
        ${rows}
      </table>
    </div>`;
}

function renderStageTable(timings) {
  if (!timings || !Object.keys(timings).length) return "";
  const rows = Object.entries(timings)
    .sort((a, b) => b[1] - a[1])
    .map(([stage, ms]) => `
      <tr>
        <td>${esc(stage.replace(/_/g, " "))}</td>
        <td class="num">${(ms / 1000).toFixed(2)}s</td>
      </tr>`).join("");
  return `
    <div class="stats-section">
      <h4>Time per stage</h4>
      <table>
        <tr><th>Stage</th><th class="num">Duration</th></tr>
        ${rows}
      </table>
    </div>`;
}

/* ── Wiring ─────────────────────────────────────────────────── */

searchBtn.addEventListener("click", runSearch);
topicInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) runSearch();
});

topicInput.focus();
