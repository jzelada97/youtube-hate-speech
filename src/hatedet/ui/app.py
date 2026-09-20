"""Demo en Streamlit del detector de odio. Ejecutar: streamlit run src/hatedet/ui/app.py

Es un cliente de la API (no carga el modelo): asi la demo prueba exactamente lo que consumiran
otros clientes (p. ej. la futura extension de navegador). El texto se muestra con `st.text`
(nunca como HTML/Markdown) para evitar inyeccion de contenido desde los comentarios.
"""

import os

import httpx
import streamlit as st

DEFAULT_API_URL = os.getenv("HATEDET_API_URL", "http://localhost:8000")
BAND_STYLE = {
    "permitir": ("🟢", "Permitir", "No parece odio."),
    "revisar": ("🟠", "Revisar", "Sospechoso: enviar a un moderador humano."),
    "ocultar": ("🔴", "Ocultar (recomendado)", "Muy probable odio. Es una recomendación, no una acción."),
}

st.set_page_config(page_title="Detector de odio", page_icon="🛡️", layout="centered")
st.title("🛡️ Detector de mensajes de odio")
st.caption("Comentarios de YouTube · clasificador TF-IDF + regresión logística (Nivel Esencial)")

with st.sidebar:
    st.header("Conexión")
    api_url = st.text_input("URL de la API", DEFAULT_API_URL)
    api_key = st.text_input("Clave de API (opcional)", type="password")
    headers = {"X-API-Key": api_key} if api_key else {}


def call(path: str, payload: dict | None = None, method: str | None = None, quiet: bool = False):
    method = method or ("post" if payload is not None else "get")
    try:
        r = getattr(httpx, method)(
            f"{api_url.rstrip('/')}{path}", **({"json": payload} if payload is not None else {}),
            headers=headers, timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        if not quiet:
            st.error(f"La API respondió {e.response.status_code}: {e.response.text[:200]}")
    except httpx.HTTPError:
        if not quiet:
            st.error(f"No se pudo conectar con la API en {api_url}. ¿Está en marcha? "
                     "`uvicorn hatedet.api.main:app --port 8000`")
    return None


def show(pred: dict):
    icon, name, hint = BAND_STYLE[pred["band"]]
    st.subheader(f"{icon} {name}")
    st.progress(min(max(pred["score"], 0.0), 1.0), text=f"Probabilidad de odio: {pred['score']:.1%}")
    st.caption(f"{hint} · modelo `{pred['model_version']}`")


health = call("/health")
if health:
    thr = health["thresholds"]
    with st.sidebar:
        st.success(f"API conectada · modelo `{health['model_version']}`")
        st.caption(f"Umbrales: revisar ≥ {thr['t_low']:.2f} · ocultar ≥ {thr['t_high']:.2f}")

tab_one, tab_many, tab_video, tab_live, tab_review = st.tabs(
    ["Un comentario", "Varios comentarios", "Vídeo de YouTube", "En directo", "Revisión humana"]
)

with tab_one:
    text = st.text_area("Escribe o pega un comentario", height=120, max_chars=5000)
    if st.button("Analizar", type="primary", disabled=not text.strip()):
        if pred := call("/predict", {"text": text}):
            show(pred)

with tab_many:
    lines = st.text_area("Un comentario por línea (máx. 200)", height=200, key="many")
    if st.button("Analizar todos", disabled=not lines.strip()):
        items = [l for l in lines.splitlines() if l.strip()][:200]
        if out := call("/predict/batch", {"texts": items}):
            for text_i, pred in zip(items, out["predictions"]):
                icon, name, _ = BAND_STYLE[pred["band"]]
                st.text(f"{icon} {pred['score']:.0%}  {text_i[:160]}")

with tab_video:
    st.caption("Descarga los comentarios más recientes del vídeo y muestra cuántos caen en cada banda.")
    video_url = st.text_input("URL del vídeo", placeholder="https://www.youtube.com/watch?v=...")
    max_c = st.slider("Comentarios a analizar", 20, 500, 100, step=20)
    if st.button("Analizar vídeo", type="primary", disabled=not video_url.strip()):
        with st.spinner("Descargando y analizando comentarios…"):
            report = call("/analyze/video", {"url": video_url, "max_comments": max_c})
        if report:
            c1, c2, c3 = st.columns(3)
            c1.metric("Comentarios analizados", report["n_comments"])
            c2.metric("Marcados (revisar u ocultar)", f"{report['share_flagged']:.0%}")
            c3.metric("Ocultar (recomendado)", report["counts"]["ocultar"])
            st.bar_chart({"comentarios": report["counts"]}, horizontal=True)
            st.subheader("Más sospechosos")
            for f in report["flagged"]:
                st.text(f"{BAND_STYLE[f['band']][0]} {f['score']:.0%}  {f['text'][:200]}")
            if not report["flagged"]:
                st.success("Ningún comentario supera el umbral de revisión.")

with tab_live:
    st.caption("Seguimiento en vivo por sondeo (YouTube no ofrece avisos en tiempo real): la API consulta "
               "los comentarios nuevos cada pocos segundos. Máximo 30 minutos por sesión.")
    ss = st.session_state
    live_url = st.text_input("URL del vídeo a vigilar", key="live_url", placeholder="https://www.youtube.com/watch?v=...")
    poll = st.select_slider("Cada cuántos segundos consultar", options=[10, 15, 30, 60, 120], value=30)
    start_col, stop_col = st.columns(2)
    if start_col.button("Empezar seguimiento", type="primary", disabled=not live_url.strip()):
        started = call("/monitor/start", {"url": live_url, "poll_seconds": poll})
        if started:
            ss.update(live_sid=started["session_id"], live_cursor=0, live_events=[], live_error=None)
    if stop_col.button("Detener", disabled="live_sid" not in ss):
        call(f"/monitor/{ss.pop('live_sid')}", method="delete", quiet=True)

    @st.fragment(run_every=5)
    def live_panel():
        sid = st.session_state.get("live_sid")
        if not sid:
            st.info("Sin seguimiento activo.")
            return
        data = call(f"/monitor/{sid}/events?after={st.session_state.live_cursor}", quiet=True)
        if data is None:
            st.warning("La sesión terminó o caducó.")
            st.session_state.pop("live_sid", None)
            return
        st.session_state.live_events = (st.session_state.live_events + data["events"])[-200:]
        st.session_state.live_cursor = data["next"]
        events = st.session_state.live_events
        flagged = [e for e in events if e["band"] != "permitir"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Comentarios vistos", len(events))
        m2.metric("Marcados", len(flagged))
        m3.metric("Caduca en", f"{data['expires_in'] // 60} min")
        if data["error"]:
            st.caption(f"Último error al consultar YouTube: {data['error']}")
        for e in reversed(events[-30:]):
            st.text(f"{BAND_STYLE[e['band']][0]} {e['score']:.0%}  {e['text'][:160]}")

    live_panel()

with tab_review:
    st.caption("Los comentarios marcados esperan aquí a que una persona decida. El veredicto y las etiquetas "
               "alimentan el reentrenamiento (`python scripts/retrain.py`).")
    if not (health or {}).get("review_enabled"):
        st.info("La cola de revisión está desactivada. Arranca la API con `HATEDET_REVIEW_DB=data/review.db` "
                "para activarla (guarda el texto de los comentarios: ver la nota de RGPD del README).")
    else:
        stats = call("/review/stats", quiet=True) or {}
        s1, s2, s3 = st.columns(3)
        s1.metric("Pendientes", stats.get("pendientes", 0))
        s2.metric("Ya revisados", stats.get("revisados", 0))
        acuerdo = stats.get("acuerdo_con_el_modelo")
        s3.metric("Acuerdo con el modelo", "—" if acuerdo is None else f"{acuerdo:.0%}",
                  help="Cuántas veces el veredicto humano coincidió con el modelo. Si baja, toca reentrenar.")

        queue_text = st.text_area("Encolar comentarios para revisar (uno por línea)", height=100, key="rq")
        if st.button("Encolar", disabled=not queue_text.strip()):
            items = [{"text": t} for t in queue_text.splitlines() if t.strip()][:200]
            if out := call("/review/queue", {"items": items}):
                st.success(f"{out['encolados']} encolados de {len(items)} enviados "
                           "(solo se encola lo que cae en revisar u ocultar).")

        pending = (call("/review/pending?limit=10", quiet=True) or {}).get("items", [])
        etiquetas = (call("/review/pending?limit=1", quiet=True) or {}).get("etiquetas_disponibles", [])
        if not pending:
            st.success("No hay nada pendiente de revisar.")
        for item in pending:
            with st.container(border=True):
                st.text(f"{BAND_STYLE[item['band']][0]} {item['score']:.0%}  ·  {item['model_version']}")
                st.text(item["text"][:1000])
                tags = st.multiselect("Etiquetas", etiquetas, key=f"tags{item['id']}",
                                      help="Las mismas 12 del dataset del cliente, para que los datos sean compatibles.")
                yes, no = st.columns(2)
                for label, verdict, col in [("Es odio", True, yes), ("No es odio", False, no)]:
                    if col.button(label, key=f"{label}{item['id']}"):
                        if call(f"/review/{item['id']}", {"is_hatespeech": verdict, "tags": tags}):
                            st.rerun()

with st.expander("¿Qué tan fiable es? (léelo antes de fiarte)"):
    st.markdown(
        "- Entrenado con **~1000 comentarios** en inglés sobre un mismo suceso (Ferguson); **casi todo el odio de "
        "entrenamiento es racismo**. No generaliza a otros tipos de odio ni a otros temas.\n"
        "- Estimación honesta (validación cruzada): **F1-macro ≈ 0.70**; detecta ~70 % del odio a costa de "
        "~60 % de falsas alarmas entre lo marcado.\n"
        "- Por eso **«revisar» es la banda principal**: el sistema recomienda, un humano decide."
    )
