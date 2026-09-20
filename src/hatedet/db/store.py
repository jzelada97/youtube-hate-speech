"""Cola de revision humana: persistencia de los comentarios marcados y del veredicto del moderador.

Por que existe: el modelo clasifica en bandas (permitir / revisar / ocultar) pero la banda "revisar"
no servia de nada si no habia donde revisarlos. Aqui se guardan los comentarios marcados, un moderador
emite veredicto y etiquetas, y esos veredictos alimentan el reentrenamiento (scripts/retrain.py).

RGPD: ESTA es la unica pieza del sistema que guarda el texto de comentarios reales, porque revisarlos
sin verlos es imposible. Por eso:
- Esta apagada por defecto: solo se activa si se define HATEDET_REVIEW_DB.
- No se guarda el nombre del autor en ningun caso.
- `purge_older_than` permite fijar una politica de retencion.
- El fichero .db esta fuera de git (.gitignore).

Se usa sqlite3 de la libreria estandar: cero dependencias nuevas, un solo fichero, y suficiente para
el volumen de una cola de moderacion (miles de filas, un escritor).
"""

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

PENDING, RESOLVED = "pendiente", "resuelto"

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    comment_id    TEXT    NOT NULL UNIQUE,
    video_id      TEXT,
    text          TEXT    NOT NULL,
    score         REAL    NOT NULL,
    band          TEXT    NOT NULL,
    model_version TEXT    NOT NULL,
    created_at    TEXT    NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'pendiente',
    verdict       INTEGER,
    tags          TEXT,
    reviewer      TEXT,
    reviewed_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_reviews_status ON reviews(status, score DESC);
"""


def local_id(text: str) -> str:
    """Id estable para un texto sin id de YouTube (p. ej. escrito a mano en la demo)."""
    return "local:" + hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ReviewItem:
    """Un comentario marcado, tal y como se devuelve al moderador."""

    id: int
    comment_id: str
    video_id: str | None
    text: str
    score: float
    band: str
    model_version: str
    created_at: str
    status: str
    verdict: bool | None
    tags: list[str]
    reviewer: str | None
    reviewed_at: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ReviewItem":
        return cls(
            id=row["id"], comment_id=row["comment_id"], video_id=row["video_id"], text=row["text"],
            score=row["score"], band=row["band"], model_version=row["model_version"],
            created_at=row["created_at"], status=row["status"],
            verdict=None if row["verdict"] is None else bool(row["verdict"]),
            tags=json.loads(row["tags"]) if row["tags"] else [],
            reviewer=row["reviewer"], reviewed_at=row["reviewed_at"],
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ReviewStore:
    """Cola de revision sobre SQLite. Crea el esquema al instanciarse."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.parent != Path(""):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")  # varios lectores mientras se escribe
        return con

    def enqueue(self, items: Iterable[dict], model_version: str) -> int:
        """Encola comentarios marcados. Devuelve cuantos se anadieron (los repetidos se ignoran)."""
        rows = [
            (
                item.get("comment_id") or local_id(item["text"]), item.get("video_id"), item["text"],
                float(item["score"]), item["band"], model_version, _now(),
            )
            for item in items
        ]
        if not rows:
            return 0
        with self._connect() as con:
            before = con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
            con.executemany(
                "INSERT OR IGNORE INTO reviews "
                "(comment_id, video_id, text, score, band, model_version, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            return con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] - before

    def pending(self, limit: int = 20) -> list[ReviewItem]:
        """Pendientes, los mas sospechosos primero (es donde el moderador aporta mas)."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM reviews WHERE status = ? ORDER BY score DESC LIMIT ?", (PENDING, limit)
            ).fetchall()
        return [ReviewItem.from_row(r) for r in rows]

    def get(self, review_id: int) -> ReviewItem | None:
        with self._connect() as con:
            row = con.execute("SELECT * FROM reviews WHERE id = ?", (review_id,)).fetchone()
        return ReviewItem.from_row(row) if row else None

    def resolve(self, review_id: int, verdict: bool, tags: Sequence[str] = (), reviewer: str | None = None) -> bool:
        """Registra el veredicto humano. Devuelve False si el id no existe."""
        with self._connect() as con:
            cur = con.execute(
                "UPDATE reviews SET status = ?, verdict = ?, tags = ?, reviewer = ?, reviewed_at = ? WHERE id = ?",
                (RESOLVED, int(verdict), json.dumps(list(tags)), reviewer, _now(), review_id),
            )
            return cur.rowcount > 0

    def stats(self) -> dict:
        """Resumen para la interfaz y para decidir si toca reentrenar."""
        with self._connect() as con:
            total = con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
            pend = con.execute("SELECT COUNT(*) FROM reviews WHERE status = ?", (PENDING,)).fetchone()[0]
            hate = con.execute("SELECT COUNT(*) FROM reviews WHERE verdict = 1").fetchone()[0]
            not_hate = con.execute("SELECT COUNT(*) FROM reviews WHERE verdict = 0").fetchone()[0]
            agree = con.execute(
                "SELECT COUNT(*) FROM reviews WHERE status = ? AND verdict = (score >= 0.5)", (RESOLVED,)
            ).fetchone()[0]
        resolved = hate + not_hate
        return {
            "total": total, "pendientes": pend, "revisados": resolved,
            "es_odio": hate, "no_es_odio": not_hate,
            "acuerdo_con_el_modelo": round(agree / resolved, 4) if resolved else None,
        }

    def training_rows(self) -> list[dict]:
        """Veredictos humanos en el formato del dataset, listos para reentrenar (ver scripts/retrain.py)."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT comment_id, video_id, text, verdict, tags FROM reviews WHERE status = ? AND verdict IS NOT NULL",
                (RESOLVED,),
            ).fetchall()
        return [
            {"CommentId": r["comment_id"], "VideoId": r["video_id"] or "revision",
             "Text": r["text"], "IsHatespeech": bool(r["verdict"]),
             "tags": json.loads(r["tags"]) if r["tags"] else []}
            for r in rows
        ]

    def purge_older_than(self, days: int) -> int:
        """Politica de retencion (RGPD): borra los revisados mas antiguos que `days`. Devuelve cuantos."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
        with self._connect() as con:
            cur = con.execute("DELETE FROM reviews WHERE status = ? AND reviewed_at < ?", (RESOLVED, cutoff))
            return cur.rowcount
