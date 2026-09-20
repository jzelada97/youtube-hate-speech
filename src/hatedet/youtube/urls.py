"""Extraccion segura del id de un video a partir de una URL.

Solo se extrae el id (11 caracteres); la URL que se consulta despues la construimos nosotros. Asi nunca se
hace una peticion a una URL arbitraria del usuario (evita SSRF).
"""

import re
from urllib.parse import parse_qs, urlparse

_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_PATH_ID = re.compile(r"^/(?:shorts|embed|live|v)/([^/?#]+)")
_HOSTS = {"youtube.com", "music.youtube.com", "youtube-nocookie.com"}


def parse_video_id(url_or_id: str) -> str:
    """Devuelve el id de video de una URL de YouTube (watch, youtu.be, shorts, embed, live) o de un id suelto."""
    text = (url_or_id or "").strip()
    if _ID.match(text):
        return text
    parsed = urlparse(text if "://" in text else "https://" + text)
    host = (parsed.hostname or "").lower()
    for prefix in ("www.", "m."):
        host = host.removeprefix(prefix)

    video_id = ""
    if host == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/")[0]
    elif host in _HOSTS:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            match = _PATH_ID.match(parsed.path)
            video_id = match.group(1) if match else ""
    else:
        raise ValueError("La URL no es de YouTube")

    if not _ID.match(video_id):
        raise ValueError("No se encontro un id de video valido en la URL")
    return video_id


def watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"
