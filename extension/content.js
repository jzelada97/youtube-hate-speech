"use strict";

(() => {
  if (window.__hatedetLoaded) return;
  window.__hatedetLoaded = true;

  const SELECTOR = "ytd-comment-thread-renderer #content-text, ytd-comment-view-model #content-text";
  const BATCH = 40;
  const DEBOUNCE_MS = 500;
  const RETRY_MS = 15000;
  const LABELS = { permitir: "OK", revisar: "Revisar", ocultar: "Posible odio" };
  const DEFAULTS = { enabled: true, mode: "label", showAll: false };

  const state = { settings: { ...DEFAULTS }, timer: null, inflight: false, stats: { analyzed: 0, flagged: 0 }, error: null };

  function pending() {
    return Array.from(document.querySelectorAll(SELECTOR)).filter((el) => !el.dataset.hatedet && el.textContent.trim());
  }

  function badgeFor(el) {
    const next = el.nextElementSibling;
    return next && next.classList.contains("hatedet-badge") ? next : null;
  }

  function apply(el) {
    const band = el.dataset.hatedet;
    const score = Number(el.dataset.hatedetScore || 0);
    const old = badgeFor(el);
    if (old) old.remove();
    el.classList.remove("hatedet-blur", "hatedet-collapsed");
    if (!band || band === "pending") return;
    const flagged = band !== "permitir";
    if (!flagged && !state.settings.showAll) return;

    const badge = document.createElement("span");
    badge.className = `hatedet-badge hatedet-${band}`;
    badge.textContent = `${LABELS[band]} · ${Math.round(score * 100)} %`;
    badge.title = "Estimación automática de un modelo de aprendizaje automático. Puede equivocarse.";
    el.after(badge);

    const mode = state.settings.mode;
    if (flagged && (mode === "blur" || (mode === "hide" && band !== "ocultar"))) {
      el.classList.add("hatedet-blur");
    } else if (band === "ocultar" && mode === "hide") {
      el.classList.add("hatedet-collapsed");
    }
    if (el.classList.contains("hatedet-blur") || el.classList.contains("hatedet-collapsed")) {
      badge.classList.add("hatedet-clickable");
      badge.addEventListener("click", () => {
        el.classList.remove("hatedet-blur", "hatedet-collapsed");
        badge.classList.remove("hatedet-clickable");
      });
    }
  }

  function saveStats() {
    try {
      chrome.storage.local.set({ hatedetStats: { ...state.stats, error: state.error, at: Date.now() } });
    } catch (_) { /* contexto invalidado: ignorar */ }
  }

  async function run() {
    if (!state.settings.enabled || state.inflight) return;
    const els = pending().slice(0, BATCH);
    if (!els.length) return;
    els.forEach((el) => { el.dataset.hatedet = "pending"; });
    state.inflight = true;
    let res;
    try {
      res = await chrome.runtime.sendMessage({ type: "classify", texts: els.map((el) => el.textContent.trim()) });
    } catch (e) {
      res = { error: String(e) };
    }
    state.inflight = false;
    if (!res || res.error || !Array.isArray(res.predictions)) {
      els.forEach((el) => { delete el.dataset.hatedet; });
      state.error = (res && res.error) || "Sin respuesta";
      saveStats();
      setTimeout(schedule, RETRY_MS);
      return;
    }
    state.error = null;
    els.forEach((el, i) => {
      const p = res.predictions[i] || { band: "permitir", score: 0 };
      el.dataset.hatedet = p.band;
      el.dataset.hatedetScore = String(p.score);
      state.stats.analyzed += 1;
      if (p.band !== "permitir") state.stats.flagged += 1;
      apply(el);
    });
    saveStats();
    if (pending().length) schedule();
  }

  function schedule() {
    clearTimeout(state.timer);
    state.timer = setTimeout(run, DEBOUNCE_MS);
  }

  function reapplyAll() {
    document.querySelectorAll(SELECTOR).forEach((el) => { if (el.dataset.hatedet) apply(el); });
  }

  async function init() {
    try {
      state.settings = { ...DEFAULTS, ...(await chrome.storage.sync.get(DEFAULTS)) };
      chrome.storage.onChanged.addListener((changes) => {
        for (const [k, v] of Object.entries(changes)) if (k in DEFAULTS) state.settings[k] = v.newValue;
        reapplyAll();
        schedule();
      });
    } catch (_) { /* sin storage (p. ej. pruebas): usar valores por defecto */ }
    new MutationObserver(schedule).observe(document.body, { childList: true, subtree: true });
    schedule();
  }

  window.HateDet = { SELECTOR, pending, apply, state, run };
  init();
})();
