"""Fuentes de comentarios de YouTube intercambiables.

- `ScraperSource`: sin clave de API (libreria `youtube-comment-downloader`). Util para la demo; depende del HTML de
  YouTube, que puede cambiar, y su uso masivo puede contravenir las condiciones de servicio.
- `ApiSource`: YouTube Data API v3 (requiere `YOUTUBE_API_KEY`, con cuota diaria).
Por privacidad (RGPD) NO se conserva el nombre del autor: solo id, texto y fecha.
"""

import os
from dataclasses import dataclass
from itertools import islice
from typing import Iterator, Protocol

import httpx

from hatedet.youtube.urls import watch_url

API_URL = "https://www.googleapis.com/youtube/v3/commentThreads"


class SourceError(Exception):
    """No se pudieron obtener comentarios (video inexistente, comentarios desactivados, cuota, red...)."""


@dataclass(frozen=True)
class YouTubeComment:
    id: str
    text: str
    published: float | None = None


class CommentSource(Protocol):
    def iter_comments(self, video_id: str, limit: int) -> Iterator[YouTubeComment]: ...


class ScraperSource:
    def __init__(self, downloader=None):
        self._downloader = downloader

    def _get_downloader(self):
        if self._downloader is None:
            from youtube_comment_downloader import YoutubeCommentDownloader

            self._downloader = YoutubeCommentDownloader()
        return self._downloader

    def iter_comments(self, video_id: str, limit: int) -> Iterator[YouTubeComment]:
        try:
            raw = self._get_downloader().get_comments_from_url(watch_url(video_id), sort_by=1)
            for c in islice(raw, limit):
                text = (c.get("text") or "").strip()
                if text:
                    yield YouTubeComment(id=c.get("cid", ""), text=text, published=c.get("time_parsed"))
        except SourceError:
            raise
        except Exception as e:
            raise SourceError(f"{type(e).__name__}: {e}") from e


class ApiSource:
    def __init__(self, api_key: str, client: httpx.Client | None = None):
        self._key = api_key
        self._client = client or httpx.Client(timeout=15)

    def iter_comments(self, video_id: str, limit: int) -> Iterator[YouTubeComment]:
        token, produced = None, 0
        while produced < limit:
            params = {"part": "snippet", "videoId": video_id, "maxResults": min(100, limit - produced),
                      "order": "time", "textFormat": "plainText", "key": self._key}
            if token:
                params["pageToken"] = token
            try:
                r = self._client.get(API_URL, params=params)
            except httpx.HTTPError as e:
                raise SourceError(f"error de red: {e}") from e
            if r.status_code != 200:
                reason = (r.json().get("error", {}).get("errors", [{}])[0].get("reason", "")
                          if r.headers.get("content-type", "").startswith("application/json") else "")
                raise SourceError(f"la API respondio {r.status_code} {reason}".strip())
            data = r.json()
            for item in data.get("items", []):
                top = item["snippet"]["topLevelComment"]
                text = (top["snippet"].get("textDisplay") or "").strip()
                if text:
                    yield YouTubeComment(id=top["id"], text=text, published=None)
                    produced += 1
                    if produced >= limit:
                        return
            token = data.get("nextPageToken")
            if not token:
                return


def default_source() -> CommentSource:
    """API oficial si hay `YOUTUBE_API_KEY`; si no, scraping."""
    key = os.getenv("YOUTUBE_API_KEY")
    return ApiSource(key) if key else ScraperSource()
