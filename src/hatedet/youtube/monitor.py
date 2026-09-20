"""Seguimiento (cuasi) en tiempo real de los comentarios de un video.

YouTube no ofrece webhooks de comentarios, asi que se hace SONDEO: cada `poll_seconds` se piden los comentarios
mas recientes, se descartan los ya vistos (por id), se clasifican los nuevos y se guardan como eventos numerados
(`seq`) que los clientes recogen con `events_after(cursor)`.

Limites de recursos (el sondeo consume red y CPU): maximo de sesiones simultaneas, duracion maxima por sesion y
un tope de eventos retenidos en memoria por sesion.
"""

import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass

from hatedet.models.bands import band_for
from hatedet.youtube.analysis import score_comments
from hatedet.youtube.sources import CommentSource, SourceError

FETCH_LIMIT = 50
SNIPPET_CHARS = 240


class RegistryFull(Exception):
    pass


@dataclass(frozen=True)
class MonitorEvent:
    seq: int
    id: str
    text: str
    score: float
    band: str
    initial: bool


class VideoMonitor:
    def __init__(self, video_id, source: CommentSource, pipeline, t_low, t_high,
                 poll_seconds=30, backlog=20, max_events=500, ttl_seconds=1800, clock=time.monotonic):
        self.video_id, self.poll_seconds, self.backlog = video_id, poll_seconds, backlog
        self._source, self._pipeline, self._t_low, self._t_high = source, pipeline, t_low, t_high
        self._events: deque[MonitorEvent] = deque(maxlen=max_events)
        self._seen: set[str] = set()
        self._seq = 0
        self._first = True
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._clock, self._deadline = clock, clock() + ttl_seconds
        self.last_error: str | None = None
        self._thread: threading.Thread | None = None

    @property
    def active(self) -> bool:
        return not self._stop.is_set() and self._clock() < self._deadline

    @property
    def expires_in(self) -> int:
        return max(0, int(self._deadline - self._clock()))

    def poll_once(self) -> int:
        """Una ronda de sondeo. Devuelve cuantos comentarios nuevos se registraron."""
        try:
            fresh = [c for c in self._source.iter_comments(self.video_id, FETCH_LIMIT)]
            self.last_error = None
        except SourceError as e:
            self.last_error = str(e)
            return 0
        new = [c for c in fresh if c.id not in self._seen]
        for c in new:
            self._seen.add(c.id)
        if self._first:
            new = new[: self.backlog]
        if not new:
            self._first = False
            return 0
        scores = score_comments(self._pipeline, new)
        initial = self._first
        self._first = False
        with self._lock:
            for c, s in reversed(list(zip(new, scores))):
                self._seq += 1
                self._events.append(MonitorEvent(self._seq, c.id, c.text[:SNIPPET_CHARS], round(s, 4),
                                                 band_for(s, self._t_low, self._t_high), initial))
        return len(new)

    def events_after(self, cursor: int) -> tuple[list[dict], int]:
        with self._lock:
            out = [asdict(e) for e in self._events if e.seq > cursor]
            return out, (self._events[-1].seq if self._events else cursor)

    def _loop(self):
        while self.active:
            self.poll_once()
            self._stop.wait(self.poll_seconds)

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"monitor-{self.video_id}")
        self._thread.start()

    def stop(self):
        self._stop.set()


class MonitorRegistry:
    def __init__(self, pipeline, t_low, t_high, max_sessions=5, ttl_seconds=1800, autostart=True):
        self._pipeline, self._t_low, self._t_high = pipeline, t_low, t_high
        self._max, self._ttl, self._autostart = max_sessions, ttl_seconds, autostart
        self._sessions: dict[str, VideoMonitor] = {}
        self._lock = threading.Lock()

    def _purge(self):
        for sid in [k for k, m in self._sessions.items() if not m.active]:
            self._sessions.pop(sid).stop()

    def start(self, video_id, source, poll_seconds=30) -> str:
        with self._lock:
            self._purge()
            if len(self._sessions) >= self._max:
                raise RegistryFull(f"maximo de {self._max} sesiones simultaneas")
            monitor = VideoMonitor(video_id, source, self._pipeline, self._t_low, self._t_high,
                                   poll_seconds=poll_seconds, ttl_seconds=self._ttl)
            sid = uuid.uuid4().hex
            self._sessions[sid] = monitor
        if self._autostart:
            monitor.start()
        return sid

    def get(self, sid: str) -> VideoMonitor | None:
        with self._lock:
            self._purge()
            return self._sessions.get(sid)

    def stop(self, sid: str) -> bool:
        with self._lock:
            monitor = self._sessions.pop(sid, None)
        if monitor:
            monitor.stop()
        return monitor is not None

    def stop_all(self):
        with self._lock:
            sessions, self._sessions = list(self._sessions.values()), {}
        for m in sessions:
            m.stop()
