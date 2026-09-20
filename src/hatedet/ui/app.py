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


def call(path: str, payload: dict | None = None):
    try:
        r = (httpx.post if payload is not None else httpx.get)(
            f"{api_url.rstrip('/')}{path}", **({"json": payload} if payload is not None else {}),
            headers=headers, timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"La API respondió {e.response.status_code}: {e.response.text[:200]}")
    except httpx.HTTPError:
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

tab_one, tab_many = st.tabs(["Un comentario", "Varios comentarios"])

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

with st.expander("¿Qué tan fiable es? (léelo antes de fiarte)"):
    st.markdown(
        "- Entrenado con **~1000 comentarios** en inglés sobre un mismo suceso (Ferguson); **casi todo el odio de "
        "entrenamiento es racismo**. No generaliza a otros tipos de odio ni a otros temas.\n"
        "- Estimación honesta (validación cruzada): **F1-macro ≈ 0.70**; detecta ~70 % del odio a costa de "
        "~60 % de falsas alarmas entre lo marcado.\n"
        "- Por eso **«revisar» es la banda principal**: el sistema recomienda, un humano decide."
    )
