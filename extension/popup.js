"use strict";

const DEFAULTS = { apiUrl: "http://localhost:8000", apiKey: "", enabled: true, mode: "label", showAll: false };
const $ = (id) => document.getElementById(id);

function apiUrlProblem(raw) {
  let u;
  try { u = new URL(raw); } catch { return "La URL no es válida"; }
  const local = u.hostname === "localhost" || u.hostname === "127.0.0.1";
  if (u.username || u.password) return "La URL no puede incluir usuario ni contraseña";
  if (u.protocol !== "https:" && !(u.protocol === "http:" && local)) return "Usa HTTPS (HTTP solo se admite en localhost)";
  return null;
}

function show(msg, kind) {
  const el = $("status");
  el.textContent = msg;
  el.className = kind || "";
}

async function load() {
  const s = await chrome.storage.sync.get(DEFAULTS);
  $("apiUrl").value = s.apiUrl;
  $("apiKey").value = s.apiKey;
  $("mode").value = s.mode;
  $("enabled").checked = s.enabled;
  $("showAll").checked = s.showAll;
  const { hatedetStats } = await chrome.storage.local.get("hatedetStats");
  if (hatedetStats) {
    $("stats").textContent = `Última página: ${hatedetStats.analyzed} comentarios analizados, ${hatedetStats.flagged} marcados.` +
      (hatedetStats.error ? ` Error: ${hatedetStats.error}` : "");
  }
}

async function save() {
  const apiUrl = $("apiUrl").value.trim().replace(/\/+$/, "");
  const problem = apiUrlProblem(apiUrl);
  if (problem) { show(problem, "err"); return false; }
  const origin = new URL(apiUrl).origin + "/*";
  const local = ["localhost", "127.0.0.1"].includes(new URL(apiUrl).hostname);
  if (!local) {
    const granted = await chrome.permissions.request({ origins: [origin] });
    if (!granted) { show("Sin permiso para conectar con ese servidor", "err"); return false; }
  }
  await chrome.storage.sync.set({
    apiUrl, apiKey: $("apiKey").value, mode: $("mode").value,
    enabled: $("enabled").checked, showAll: $("showAll").checked,
  });
  show("Guardado", "ok");
  return true;
}

async function test() {
  if (!(await save())) return;
  show("Probando…");
  const res = await chrome.runtime.sendMessage({ type: "health" });
  if (res && !res.error) show(`Conectado · modelo ${res.model_version}`, "ok");
  else show(`No se pudo conectar: ${(res && res.error) || "sin respuesta"}`, "err");
}

$("save").addEventListener("click", save);
$("test").addEventListener("click", test);
load();
