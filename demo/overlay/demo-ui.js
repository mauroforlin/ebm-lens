"use strict";

/* Loads after app.js. Everything here talks to app.js only through the DOM
   it already exposes (#topic, #searchBtn, #sourcesPreset, #language,
   #discoveryNote) plus the one global app.js leaves behind as a plain
   top-level function declaration in a classic script - window.setMaxSources -
   used instead of clicking the (disabled) preset buttons directly. Nothing
   here reaches into app.js's closures or module state, because there isn't
   any to reach into. */

(function () {
  const LANG_KEY = "ebmlens-demo-lang";
  const REPO_URL = "https://github.com/mauroforlin/ebm-lens";

  const STRINGS = {
    en: {
      bannerLabel: "Read-only demo",
      bannerBody: "Explore 15 pre-recorded searches. The loading animation is sped up, "
        + "but the sources, summaries, costs, and timings are exactly what the app "
        + "generated in real time.",
      bannerCta: "Clone the repo",
      bannerCtaSuffix: "to run your own searches.",
      pickerTitle: "Pick a question",
      pickerHint: "sources",
      discoveryNote: "This is a replay of a real search. The loading time is sped up, "
        + "but all the sources, summaries, and numbers below are exactly what the app "
        + "generated.",
      topicPlaceholder: "Pick a question below",
      langDisabledTitle: "Demo results are only available in English.",
      sourcesDisabledTitle: "The number of sources is fixed for these examples.",
    },
    it: {
      bannerLabel: "Demo in sola lettura",
      bannerBody: "Esplora 15 ricerche pre-registrate. L'animazione di caricamento è accelerata, "
        + "ma le fonti, i riassunti, i costi e i tempi sono esattamente quelli "
        + "generati dall'app in tempo reale.",
      bannerCta: "Clona il repo",
      bannerCtaSuffix: "per fare le tue ricerche.",
      pickerTitle: "Scegli una domanda",
      pickerHint: "fonti",
      discoveryNote: "Questo è il replay di una ricerca reale. L'attesa è accelerata, "
        + "ma tutte le fonti, i riassunti e i numeri qui sotto sono esattamente quelli "
        + "generati dall'app.",
      topicPlaceholder: "Scegli una domanda qui sotto",
      langDisabledTitle: "I risultati della demo sono disponibili solo in inglese.",
      sourcesDisabledTitle: "Il numero di fonti è fisso per questi esempi.",
    },
  };

  function currentLang() {
    const stored = localStorage.getItem(LANG_KEY);
    return stored === "it" ? "it" : "en";
  }

  function esc(value) {
    const node = document.createElement("div");
    node.textContent = value == null ? "" : String(value);
    return node.innerHTML;
  }

  async function loadQueries() {
    const res = await fetch("fixtures/index.json");
    return res.json();
  }

  function lockRealSearchControls() {
    const topic = document.getElementById("topic");
    const language = document.getElementById("language");
    const presetButtons = document.querySelectorAll("#sourcesPreset button");

    topic.readOnly = true;
    topic.value = "";

    language.value = "en";
    language.disabled = true;

    presetButtons.forEach((btn) => { btn.disabled = true; });
  }

  function renderBanner(lang) {
    const t = STRINGS[lang];
    const banner = document.createElement("div");
    banner.className = "demo-banner";
    banner.innerHTML = `
      <div class="demo-banner-text">
        <strong>${esc(t.bannerLabel)}.</strong> ${esc(t.bannerBody)}
        <a href="${esc(REPO_URL)}" target="_blank" rel="noopener noreferrer">${esc(t.bannerCta)}</a>
        ${esc(t.bannerCtaSuffix)}
      </div>
      <div class="demo-lang-toggle" role="group" aria-label="Banner language">
        <button type="button" data-lang="en" aria-pressed="${lang === "en"}">EN</button>
        <button type="button" data-lang="it" aria-pressed="${lang === "it"}">IT</button>
      </div>`;
    banner.querySelectorAll("[data-lang]").forEach((btn) => {
      btn.addEventListener("click", () => {
        localStorage.setItem(LANG_KEY, btn.dataset.lang);
        refresh();
      });
    });
    return banner;
  }

  function cardHtml(query, lang, active) {
    const showcase = lang === "it" ? query.showcases_it : query.showcases_en;
    const t = STRINGS[lang];
    return `
      <button type="button" class="demo-picker-card" data-id="${esc(query.id)}"
        aria-pressed="${active}">
        <span class="demo-picker-topic">${esc(query.topic)}</span>
        <span class="demo-picker-showcase">${esc(showcase)}</span>
        <span class="demo-picker-meta">${query.max_sources} ${esc(t.pickerHint)} &middot; ${query.article_count} ${esc(lang === "it" ? "selezionate" : "selected")}</span>
      </button>`;
  }

  function renderPicker(lang, queries, activeId, onPick) {
    const t = STRINGS[lang];
    const section = document.createElement("section");
    section.className = "demo-picker";
    section.innerHTML = `
      <h2 class="demo-picker-title">${esc(t.pickerTitle)}</h2>
      <div class="demo-picker-grid">${queries.map((q) => cardHtml(q, lang, q.id === activeId)).join("")}</div>`;
    section.querySelectorAll(".demo-picker-card").forEach((card) => {
      card.addEventListener("click", () => onPick(card.dataset.id));
    });
    return section;
  }

  let queriesCache = null;
  let activeId = null;

  function pickQuery(id) {
    const query = queriesCache.find((q) => q.id === id);
    if (!query) return;
    activeId = id;

    document.getElementById("topic").value = query.topic;
    window.setMaxSources(query.max_sources);
    refresh();

    document.getElementById("searchBtn").click();
    document.getElementById("results").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function refresh() {
    const lang = currentLang();
    document.documentElement.lang = lang;
    const t = STRINGS[lang];

    document.getElementById("topic").placeholder = t.topicPlaceholder;
    document.getElementById("language").title = t.langDisabledTitle;
    document.querySelectorAll("#sourcesPreset button").forEach((btn) => { btn.title = t.sourcesDisabledTitle; });

    const note = document.getElementById("discoveryNote");
    if (note) note.textContent = t.discoveryNote;

    const oldBanner = document.querySelector(".demo-banner");
    const banner = renderBanner(lang);
    if (oldBanner) oldBanner.replaceWith(banner);
    else document.querySelector("header").insertAdjacentElement("afterend", banner);

    const oldPicker = document.querySelector(".demo-picker");
    const picker = renderPicker(lang, queriesCache, activeId, pickQuery);
    if (oldPicker) oldPicker.replaceWith(picker);
    else document.querySelector(".panel").insertAdjacentElement("afterend", picker);
  }

  async function init() {
    lockRealSearchControls();
    queriesCache = await loadQueries();
    refresh();
  }

  init();
})();
