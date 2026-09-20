"""API de inferencia del detector de odio.

Ejecutar: uvicorn hatedet.api.main:app --host 0.0.0.0 --port 8000

Decisiones (ver docs/DECISIONES.md, seccion 12):
- Sirve el Pipeline serializado completo (texto crudo -> probabilidad): sin skew train/serve.
- NO registra el texto de los comentarios (datos personales, RGPD).
- Clave de API opcional (cabecera X-API-Key) y CORS acotado por defecto a localhost y extensiones.
- El sistema RECOMIENDA una banda (permitir/revisar/ocultar); no ejecuta acciones.
"""

import json
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator

from hatedet.data.schema import LABEL_COLUMNS
from hatedet.db.store import ReviewStore
from hatedet.models.bands import band_for
from hatedet.youtube.analysis import VideoReport, analyze
from hatedet.youtube.monitor import MonitorRegistry, RegistryFull
from hatedet.youtube.sources import CommentSource, SourceError, default_source
from hatedet.youtube.urls import parse_video_id

MAX_TEXT_CHARS = 5000
MAX_BATCH = 200
MAX_VIDEO_COMMENTS = 500
DEFAULT_MODEL_PATH = "models/baseline_v3.joblib"
PREFERRED_MODEL_PATH = "models/ensemble_v1.joblib"
DEFAULT_CORS_REGEX = r"^(chrome-extension://.*|https?://(localhost|127\.0\.0\.1)(:\d+)?)$"


class Comment(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("el texto no puede estar vacio")
        return v


class Prediction(BaseModel):
    label: bool = Field(description="True si score >= 0.5 (decision binaria por defecto)")
    score: float = Field(description="Probabilidad estimada de odio, 0-1")
    band: str = Field(description="permitir | revisar | ocultar")
    model_version: str


class BatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_BATCH)

    @field_validator("texts")
    @classmethod
    def valid_texts(cls, v: list[str]) -> list[str]:
        for t in v:
            if not t.strip() or len(t) > MAX_TEXT_CHARS:
                raise ValueError(f"cada texto debe tener entre 1 y {MAX_TEXT_CHARS} caracteres")
        return v


class BatchResponse(BaseModel):
    predictions: list[Prediction]


class MonitorStart(BaseModel):
    url: str = Field(min_length=11, max_length=300)
    poll_seconds: int = Field(default=30, ge=10, le=300)


class VideoRequest(BaseModel):
    url: str = Field(min_length=11, max_length=300, description="URL (o id) de un video de YouTube")
    max_comments: int = Field(default=100, ge=1, le=MAX_VIDEO_COMMENTS)


class QueueItem(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    comment_id: str | None = Field(default=None, max_length=100)
    video_id: str | None = Field(default=None, max_length=32)


class QueueRequest(BaseModel):
    items: list[QueueItem] = Field(min_length=1, max_length=MAX_BATCH)
    only_flagged: bool = Field(default=True, description="Encolar solo lo que cae en revisar/ocultar")


class Verdict(BaseModel):
    """Veredicto humano. `tags` acepta los nombres de etiqueta del dataset (IsRacist, IsThreat...)."""

    is_hatespeech: bool
    tags: list[str] = Field(default_factory=list, max_length=len(LABEL_COLUMNS))
    reviewer: str | None = Field(default=None, max_length=80)

    @field_validator("tags")
    @classmethod
    def known_tags(cls, v: list[str]) -> list[str]:
        unknown = [t for t in v if t not in LABEL_COLUMNS]
        if unknown:
            raise ValueError(f"etiquetas desconocidas: {unknown}")
        return v


def default_model_path() -> str:
    """Modelo por defecto: el ensemble solo si existe Y cumple el requisito de gap (< 5 pp); si no, el baseline."""
    preferred = Path(PREFERRED_MODEL_PATH)
    meta = preferred.with_suffix("").with_suffix(".metadata.json")
    try:
        if preferred.exists() and json.loads(meta.read_text()).get("meets_gap_requirement"):
            return str(preferred)
    except (OSError, ValueError):
        pass
    return DEFAULT_MODEL_PATH


def _load_artifacts(model_path: str):
    path = Path(model_path)
    pipeline = joblib.load(path)
    meta_path = path.with_suffix("").with_suffix(".metadata.json")
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return pipeline, metadata


def default_review_store() -> ReviewStore | None:
    """Cola de revision: apagada salvo que se defina HATEDET_REVIEW_DB (guarda texto, ver db/store.py)."""
    path = os.getenv("HATEDET_REVIEW_DB")
    return ReviewStore(path) if path else None


def create_app(
    pipeline=None, metadata: dict | None = None, comment_source: CommentSource | None = None,
    monitor_autostart: bool = True, review_store: ReviewStore | None = None,
) -> FastAPI:
    """Fabrica la app. `pipeline`/`metadata`/`comment_source`/`review_store` se inyectan en los tests."""
    store = review_store if review_store is not None else default_review_store()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if pipeline is not None:
            app.state.pipeline, app.state.metadata = pipeline, metadata or {}
        else:
            app.state.pipeline, app.state.metadata = _load_artifacts(
                os.getenv("HATEDET_MODEL_PATH") or default_model_path()
            )
        thr = app.state.metadata.get("thresholds", {})
        app.state.t_low = float(os.getenv("HATEDET_T_LOW", thr.get("t_low", 0.5)))
        app.state.t_high = float(os.getenv("HATEDET_T_HIGH", thr.get("t_high", 0.9)))
        app.state.version = app.state.metadata.get("model_version", "unknown")
        app.state.pipeline.predict_proba(["warm up"])  # evita ~2 s de arranque en frio en la 1a peticion
        app.state.monitors = MonitorRegistry(
            app.state.pipeline, app.state.t_low, app.state.t_high,
            max_sessions=int(os.getenv("HATEDET_MAX_MONITORS", "5")), autostart=monitor_autostart,
        )
        yield
        app.state.monitors.stop_all()

    app = FastAPI(title="Detector de odio en comentarios de YouTube", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=os.getenv("HATEDET_CORS_REGEX", DEFAULT_CORS_REGEX),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key"],
    )
    key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def check_key(provided: str | None = Security(key_header)) -> None:
        expected = os.getenv("HATEDET_API_KEY")
        if expected and not (provided and secrets.compare_digest(provided, expected)):
            raise HTTPException(status_code=401, detail="API key invalida o ausente")

    def score_texts(request: Request, texts: list[str]) -> list[Prediction]:
        st = request.app.state
        scores = st.pipeline.predict_proba(texts)[:, 1]
        return [
            Prediction(
                label=bool(s >= 0.5), score=round(float(s), 4),
                band=band_for(float(s), st.t_low, st.t_high), model_version=st.version,
            )
            for s in scores
        ]

    def require_store() -> ReviewStore:
        if store is None:
            raise HTTPException(status_code=503, detail="Cola de revision desactivada (define HATEDET_REVIEW_DB)")
        return store

    @app.get("/health")
    def health(request: Request):
        st = request.app.state
        return {"status": "ok", "model_version": st.version, "review_enabled": store is not None,
                "thresholds": {"t_low": st.t_low, "t_high": st.t_high}}

    @app.post("/predict", response_model=Prediction, dependencies=[Depends(check_key)])
    def predict(comment: Comment, request: Request):
        return score_texts(request, [comment.text])[0]

    @app.post("/predict/batch", response_model=BatchResponse, dependencies=[Depends(check_key)])
    def predict_batch(body: BatchRequest, request: Request):
        return BatchResponse(predictions=score_texts(request, body.texts))

    @app.post("/analyze/video", response_model=VideoReport, dependencies=[Depends(check_key)])
    def analyze_video(body: VideoRequest, request: Request):
        try:
            video_id = parse_video_id(body.url)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        try:
            comments = list((comment_source or default_source()).iter_comments(video_id, body.max_comments))
        except SourceError as e:
            raise HTTPException(status_code=502, detail=f"No se pudieron obtener los comentarios: {e}")
        st = request.app.state
        return analyze(comments, st.pipeline, st.t_low, st.t_high, video_id=video_id, model_version=st.version)

    @app.post("/monitor/start", dependencies=[Depends(check_key)])
    def monitor_start(body: MonitorStart, request: Request):
        try:
            video_id = parse_video_id(body.url)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        try:
            sid = request.app.state.monitors.start(video_id, comment_source or default_source(), body.poll_seconds)
        except RegistryFull as e:
            raise HTTPException(status_code=429, detail=str(e))
        return {"session_id": sid, "video_id": video_id, "poll_seconds": body.poll_seconds}

    @app.get("/monitor/{session_id}/events", dependencies=[Depends(check_key)])
    def monitor_events(session_id: str, request: Request, after: int = 0):
        monitor = request.app.state.monitors.get(session_id)
        if monitor is None:
            raise HTTPException(status_code=404, detail="Sesion desconocida o caducada")
        events, cursor = monitor.events_after(max(0, after))
        return {"events": events, "next": cursor, "active": monitor.active,
                "expires_in": monitor.expires_in, "error": monitor.last_error}

    @app.delete("/monitor/{session_id}", dependencies=[Depends(check_key)])
    def monitor_stop(session_id: str, request: Request):
        if not request.app.state.monitors.stop(session_id):
            raise HTTPException(status_code=404, detail="Sesion desconocida o caducada")
        return {"stopped": True}

    @app.post("/review/queue", dependencies=[Depends(check_key)])
    def review_queue(body: QueueRequest, request: Request):
        """Puntua los comentarios y encola los que necesitan mirada humana."""
        db = require_store()
        st = request.app.state
        preds = score_texts(request, [i.text for i in body.items])
        selected = [
            {"text": item.text, "comment_id": item.comment_id, "video_id": item.video_id,
             "score": pred.score, "band": pred.band}
            for item, pred in zip(body.items, preds)
            if not body.only_flagged or pred.band != "permitir"
        ]
        return {"encolados": db.enqueue(selected, st.version), "candidatos": len(selected)}

    @app.get("/review/pending", dependencies=[Depends(check_key)])
    def review_pending(limit: int = 20):
        db = require_store()
        return {"items": [vars(i) for i in db.pending(max(1, min(limit, MAX_BATCH)))],
                "etiquetas_disponibles": LABEL_COLUMNS}

    @app.post("/review/{review_id}", dependencies=[Depends(check_key)])
    def review_resolve(review_id: int, body: Verdict):
        db = require_store()
        if not db.resolve(review_id, body.is_hatespeech, body.tags, body.reviewer):
            raise HTTPException(status_code=404, detail="No existe ese elemento en la cola")
        return {"resuelto": review_id}

    @app.get("/review/stats", dependencies=[Depends(check_key)])
    def review_stats():
        return require_store().stats()

    return app


app = create_app()
