"use strict";

const DEFAULTS = { apiUrl: "http://localhost:8000", apiKey: "", enabled: true, mode: "label", showAll: false };
const MAX_CACHE = 3000;
const MAX_TEXTS = 200;
const MAX_CHARS = 5000;
const TIMEOUT_MS = 15000;
const cache = new Map();

function isAllowedApiUrl(raw) {
  try {
    const u = new URL(raw);
    const local = u.hostname === "localhost" || u.hostname === "127.0.0.1";
    return (u.protocol === "https:" || (u.protocol === "http:" && local)) && !u.username && !u.password;
  } catch {
    return false;
  }
}

async function settings() {
  return chrome.storage.sync.get(DEFAULTS);
}

async function callApi(path, options = {}) {
  const s = await settings();
  if (!isAllowedApiUrl(s.apiUrl)) throw new Error("La URL de la API debe ser HTTPS (o http://localhost)");
  const headers = { "Content-Type": "application/json" };
  if (s.apiKey) headers["X-API-Key"] = s.apiKey;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(s.apiUrl.replace(/\/+$/, "") + path, { ...options, headers, signal: ctrl.signal });
    if (!res.ok) throw new Error(`La API respondió ${res.status}`);
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

async function classify(texts) {
  if (!Array.isArray(texts) || !texts.length || texts.length > MAX_TEXTS) throw new Error("Petición no válida");
  const clean = texts.map((t) => (typeof t === "string" ? t.slice(0, MAX_CHARS) : ""));
  const results = clean.map((t) => cache.get(t));
  const missing = clean.map((t, i) => (results[i] === undefined && t.trim() ? i : -1)).filter((i) => i >= 0);
  if (missing.length) {
    const data = await callApi("/predict/batch", { method: "POST", body: JSON.stringify({ texts: missing.map((i) => clean[i]) }) });
    data.predictions.forEach((p, k) => {
      const value = { score: p.score, band: p.band };
      if (cache.size >= MAX_CACHE) cache.delete(cache.keys().next().value);
      cache.set(clean[missing[k]], value);
      results[missing[k]] = value;
    });
  }
  return { predictions: results.map((r) => r || { score: 0, band: "permitir" }) };
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (sender.id !== chrome.runtime.id) return false;
  const task = msg && msg.type === "classify" ? classify(msg.texts)
    : msg && msg.type === "health" ? callApi("/health")
    : Promise.reject(new Error("Mensaje desconocido"));
  task.then(sendResponse).catch((e) => sendResponse({ error: String((e && e.message) || e) }));
  return true;
});
