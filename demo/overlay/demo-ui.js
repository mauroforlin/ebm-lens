"use strict";

/* Loads after app.js. Everything here talks to app.js only through the DOM
   it already exposes (#topic, #searchBtn, #language, .panel, .hint,
   #discoveryNote, #results) plus the one global app.js leaves behind as a
   plain top-level function declaration in a classic script -
   window.setMaxSources - used instead of clicking the (hidden) preset
   buttons directly. Nothing here reaches into app.js's closures or module
   state, because there isn't any to reach into. */

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
      steps: [
        "Pick a question below",
        "Watch it search 12 medical databases",
        "Read an answer where every claim links back to a source",
      ],
      pickerTitle: "Pick a question",
      pickerHint: "sources",
      showMore: (n) => `Show ${n} more question${n === 1 ? "" : "s"}`,
      discoveryNote: "This is a replay of a real search. The loading time is sped up, "
        + "but all the sources, summaries, and numbers below are exactly what the app "
        + "generated.",
    },
    it: {
      bannerLabel: "Demo in sola lettura",
      bannerBody: "Esplora 15 ricerche pre-registrate. L'animazione di caricamento è accelerata, "
        + "ma le fonti, i riassunti, i costi e i tempi sono esattamente quelli "
        + "generati dall'app in tempo reale.",
      bannerCta: "Clona il repo",
      bannerCtaSuffix: "per fare le tue ricerche.",
      steps: [
        "Scegli una domanda qui sotto",
        "Guarda la ricerca su 12 database medici",
        "Leggi una risposta dove ogni affermazione rimanda a una fonte",
      ],
      pickerTitle: "Scegli una domanda",
      pickerHint: "fonti",
      showMore: (n) => `Mostra altr${n === 1 ? "a" : "e"} ${n} domand${n === 1 ? "a" : "e"}`,
      discoveryNote: "Questo è il replay di una ricerca reale. L'attesa è accelerata, "
        + "ma tutte le fonti, i riassunti e i numeri qui sotto sono esattamente quelli "
        + "generati dall'app.",
    },
  };

  // First view shows this many cards - enough to feel like real choice
  // without forcing a mobile visitor to scroll past all 15 before the
  // page shows them anything else.
  const INITIAL_CARDS = 4;

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

  /* The real search bar (topic box, language/source controls, Search button)
     has nothing for a demo visitor to do - every value it could show is
     already decided by the fixture. Hiding it outright, rather than greying
     it out in place, is what actually removes the visual weight; app.js
     still drives it by setting #topic.value and calling #searchBtn.click(),
     neither of which needs the element to be visible. */
  function hideRealSearchBar() {
    document.querySelector(".panel").hidden = true;
    document.getElementById("language").value = "en";
    // "Ctrl+Enter to search" refers to the textarea just hidden above.
    const hint = document.querySelector(".hint");
    if (hint) hint.hidden = true;
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

  /* Three short steps, always visible (no disclosure): what the picker
     below actually does, stated once, before the visitor has to guess it
     from fifteen cards of medical topics. */
  function renderSteps(lang) {
    const t = STRINGS[lang];
    const steps = document.createElement("ol");
    steps.className = "demo-steps";
    steps.innerHTML = t.steps
      .map((text, i) => `<li><span class="demo-step-num">${i + 1}</span>${esc(text)}</li>`)
      .join("");
    return steps;
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

  /* A <details> disclosure, not a plain section: fifteen full cards are fine
     as a first choice but too heavy to leave standing once one is picked, so
     picking one collapses it to a single summary line (title + the active
     topic) - reopen it the same way the Run Statistics panel below opens,
     to change the question.

     The grid itself only ever shows INITIAL_CARDS up front, even while open:
     fifteen full cards is still a wall of scroll on a phone before anything
     else on the page is reachable. A "show more" button, not a second
     details, since the picker is already the disclosure. */
  function renderPicker(lang, queries, activeId, onPick, openByDefault, showAllCards, onShowAll) {
    const t = STRINGS[lang];
    const active = queries.find((q) => q.id === activeId);
    const visible = showAllCards ? queries : queries.slice(0, INITIAL_CARDS);
    const remaining = queries.length - visible.length;
    const details = document.createElement("details");
    details.className = "demo-picker";
    details.open = openByDefault;
    details.innerHTML = `
      <summary>
        <span class="demo-picker-summary-label">${esc(t.pickerTitle)}</span>
        ${active ? `<span class="demo-picker-summary-active">${esc(active.topic)}</span>` : ""}
      </summary>
      <div class="demo-picker-grid">${visible.map((q) => cardHtml(q, lang, q.id === activeId)).join("")}</div>
      ${remaining > 0
        ? `<button type="button" class="demo-picker-more">${esc(t.showMore(remaining))}</button>`
        : ""}`;
    details.querySelectorAll(".demo-picker-card").forEach((card) => {
      card.addEventListener("click", () => onPick(card.dataset.id));
    });
    const moreBtn = details.querySelector(".demo-picker-more");
    if (moreBtn) moreBtn.addEventListener("click", onShowAll);
    return details;
  }

  let queriesCache = null;
  let activeId = null;
  let showAllCards = false;

  function pickQuery(id) {
    const query = queriesCache.find((q) => q.id === id);
    if (!query) return;
    activeId = id;

    document.getElementById("topic").value = query.topic;
    window.setMaxSources(query.max_sources);
    refresh();

    const picker = document.querySelector(".demo-picker");
    if (picker) picker.open = false;

    document.getElementById("searchBtn").click();
    document.getElementById("results").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function refresh() {
    const lang = currentLang();
    document.documentElement.lang = lang;
    const t = STRINGS[lang];

    const note = document.getElementById("discoveryNote");
    if (note) note.textContent = t.discoveryNote;

    const oldBanner = document.querySelector(".demo-banner");
    const banner = renderBanner(lang);
    if (oldBanner) oldBanner.replaceWith(banner);
    else document.querySelector("header").insertAdjacentElement("afterend", banner);

    const oldSteps = document.querySelector(".demo-steps");
    const steps = renderSteps(lang);
    if (oldSteps) oldSteps.replaceWith(steps);
    else banner.insertAdjacentElement("afterend", steps);

    // A language toggle re-renders the picker too (card text is bilingual),
    // so its open/closed state has to survive the rebuild - only pickQuery
    // forces it shut, by setting .open on the element this returns.
    const oldPicker = document.querySelector(".demo-picker");
    const openByDefault = oldPicker ? oldPicker.open : true;
    const picker = renderPicker(lang, queriesCache, activeId, pickQuery, openByDefault, showAllCards, () => {
      showAllCards = true;
      refresh();
    });
    if (oldPicker) oldPicker.replaceWith(picker);
    else steps.insertAdjacentElement("afterend", picker);
  }

  // Fisher-Yates. Run once at load, not per render: reshuffling on every
  // pick or language toggle would make the "N more" pile shift under the
  // visitor's finger.
  function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
  }

  async function init() {
    hideRealSearchBar();
    queriesCache = shuffle(await loadQueries());
    refresh();
  }

  init();
})();
