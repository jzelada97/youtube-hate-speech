"""Analisis agregado de los comentarios de un video con el clasificador."""

from collections import Counter
from typing import Iterable

from pydantic import BaseModel

from hatedet.models.bands import BAND_ALLOW, BAND_HIDE, BAND_REVIEW, band_for
from hatedet.youtube.sources import YouTubeComment

MODEL_MAX_CHARS = 5000
CHUNK = 200


class FlaggedComment(BaseModel):
    id: str
    text: str
    score: float
    band: str


class VideoReport(BaseModel):
    video_id: str
    n_comments: int
    model_version: str
    counts: dict[str, int]
    share_flagged: float
    flagged: list[FlaggedComment]


def score_comments(pipeline, comments: list[YouTubeComment]) -> list[float]:
    texts = [c.text[:MODEL_MAX_CHARS] for c in comments]
    scores: list[float] = []
    for i in range(0, len(texts), CHUNK):
        scores.extend(float(p) for p in pipeline.predict_proba(texts[i : i + CHUNK])[:, 1])
    return scores


def analyze(
    comments: Iterable[YouTubeComment], pipeline, t_low: float, t_high: float,
    video_id: str, model_version: str, top_n: int = 10, snippet_chars: int = 240,
) -> VideoReport:
    comments = list(comments)
    scores = score_comments(pipeline, comments) if comments else []
    bands = [band_for(s, t_low, t_high) for s in scores]
    counts = Counter(bands)
    ranked = sorted(zip(comments, scores, bands), key=lambda x: -x[1])
    flagged = [
        FlaggedComment(id=c.id, text=c.text[:snippet_chars], score=round(s, 4), band=b)
        for c, s, b in ranked if b != BAND_ALLOW
    ][:top_n]
    n = len(comments)
    return VideoReport(
        video_id=video_id, n_comments=n, model_version=model_version,
        counts={b: counts.get(b, 0) for b in (BAND_ALLOW, BAND_REVIEW, BAND_HIDE)},
        share_flagged=round((counts.get(BAND_REVIEW, 0) + counts.get(BAND_HIDE, 0)) / n, 4) if n else 0.0,
        flagged=flagged,
    )
