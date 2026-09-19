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

from hatedet.models.bands import band_for

MAX_TEXT_CHARS = 5000
MAX_BATCH = 200
DEFAULT_MODEL_PATH = "models/baseline_v3.joblib"
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


def _load_artifacts(model_path: str):
    path = Path(model_path)
    pipeline = joblib.load(path)
    meta_path = path.with_suffix("").with_suffix(".metadata.json")
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return pipeline, metadata


def create_app(pipeline=None, metadata: dict | None = None) -> FastAPI:
    """Fabrica la app. `pipeline`/`metadata` permiten inyectar un modelo (tests) sin tocar disco."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if pipeline is not None:
            app.state.pipeline, app.state.metadata = pipeline, metadata or {}
        else:
            app.state.pipeline, app.state.metadata = _load_artifacts(
                os.getenv("HATEDET_MODEL_PATH", DEFAULT_MODEL_PATH)
            )
        thr = app.state.metadata.get("thresholds", {})
        app.state.t_low = float(os.getenv("HATEDET_T_LOW", thr.get("t_low", 0.5)))
        app.state.t_high = float(os.getenv("HATEDET_T_HIGH", thr.get("t_high", 0.9)))
        app.state.version = app.state.metadata.get("model_version", "unknown")
        app.state.pipeline.predict_proba(["warm up"])  # evita ~2 s de arranque en frio en la 1a peticion
        yield

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

    @app.get("/health")
    def health(request: Request):
        st = request.app.state
        return {"status": "ok", "model_version": st.version,
                "thresholds": {"t_low": st.t_low, "t_high": st.t_high}}

    @app.post("/predict", response_model=Prediction, dependencies=[Depends(check_key)])
    def predict(comment: Comment, request: Request):
        return score_texts(request, [comment.text])[0]

    @app.post("/predict/batch", response_model=BatchResponse, dependencies=[Depends(check_key)])
    def predict_batch(body: BatchRequest, request: Request):
        return BatchResponse(predictions=score_texts(request, body.texts))

    return app


app = create_app()
