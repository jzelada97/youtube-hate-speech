import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hatedet.api.main import create_app
from hatedet.youtube.monitor import MonitorRegistry, RegistryFull, VideoMonitor
from hatedet.youtube.sources import SourceError, YouTubeComment

VID = "dQw4w9WgXcQ"
META = {"model_version": "fake", "thresholds": {"t_low": 0.5, "t_high": 0.9}}


class FakePipeline:
    def predict_proba(self, texts):
        p = np.array([0.95 if "hate" in t else 0.05 for t in texts])
        return np.column_stack([1 - p, p])


class ScriptedSource:
    """Cada llamada devuelve la siguiente lista de comentarios (mas recientes primero)."""

    def __init__(self, rounds):
        self.rounds, self.calls = rounds, 0

    def iter_comments(self, video_id, limit):
        batch = self.rounds[min(self.calls, len(self.rounds) - 1)]
        self.calls += 1
        if isinstance(batch, Exception):
            raise batch
        yield from batch[:limit]


def c(i, text="ok"):
    return YouTubeComment(str(i), f"{text} {i}")


def make(rounds, **kw):
    return VideoMonitor(VID, ScriptedSource(rounds), FakePipeline(), 0.5, 0.9, **kw)


def test_first_poll_emits_backlog_and_marks_it_initial():
    m = make([[c(3), c(2), c(1)]])
    assert m.poll_once() == 3
    events, cursor = m.events_after(0)
    assert [e["id"] for e in events] == ["1", "2", "3"] and all(e["initial"] for e in events)
    assert cursor == 3


def test_backlog_limit_applies_only_to_first_poll():
    m = make([[c(i) for i in range(30, 0, -1)]], backlog=5)
    assert m.poll_once() == 5


def test_later_polls_emit_only_unseen_comments_in_chronological_order():
    m = make([[c(2), c(1)], [c(4, "hate"), c(3), c(2), c(1)]])
    m.poll_once()
    assert m.poll_once() == 2
    events, cursor = m.events_after(2)
    assert [e["id"] for e in events] == ["3", "4"] and not any(e["initial"] for e in events)
    assert events[1]["band"] == "ocultar" and events[0]["band"] == "permitir"
    assert m.events_after(cursor)[0] == []


def test_source_error_is_recorded_and_does_not_break_the_monitor():
    m = make([SourceError("cuota"), [c(1)]])
    assert m.poll_once() == 0 and m.last_error == "cuota"
    assert m.poll_once() == 1 and m.last_error is None


def test_events_are_capped_in_memory():
    m = make([[c(i) for i in range(10, 0, -1)]], backlog=10, max_events=3)
    m.poll_once()
    assert len(m.events_after(0)[0]) == 3


def test_monitor_expires_after_ttl_and_can_be_stopped():
    t = {"now": 0.0}
    m = make([[c(1)]], ttl_seconds=100, clock=lambda: t["now"])
    assert m.active and m.expires_in == 100
    t["now"] = 101
    assert not m.active and m.expires_in == 0
    m2 = make([[c(1)]])
    m2.stop()
    assert not m2.active


def test_background_thread_polls_and_stops():
    m = make([[c(1)]], poll_seconds=0.05)
    m.start()
    time.sleep(0.3)
    m.stop()
    assert m.events_after(0)[0]


def test_registry_enforces_session_limit_and_releases_slots():
    reg = MonitorRegistry(FakePipeline(), 0.5, 0.9, max_sessions=2, autostart=False)
    a = reg.start(VID, ScriptedSource([[c(1)]]))
    reg.start(VID, ScriptedSource([[c(1)]]))
    with pytest.raises(RegistryFull):
        reg.start(VID, ScriptedSource([[c(1)]]))
    assert reg.stop(a) and not reg.stop(a)
    reg.start(VID, ScriptedSource([[c(1)]]))


def test_registry_purges_expired_sessions():
    reg = MonitorRegistry(FakePipeline(), 0.5, 0.9, ttl_seconds=-1, autostart=False)
    sid = reg.start(VID, ScriptedSource([[c(1)]]))
    assert reg.get(sid) is None


@pytest.fixture()
def client():
    app = create_app(FakePipeline(), META, ScriptedSource([[c(2, "hate"), c(1)]]), monitor_autostart=False)
    with TestClient(app) as cl:
        yield cl, app


def test_api_monitor_full_flow(client):
    cl, app = client
    sid = cl.post("/monitor/start", json={"url": VID, "poll_seconds": 15}).json()["session_id"]
    app.state.monitors.get(sid).poll_once()
    r = cl.get(f"/monitor/{sid}/events?after=0").json()
    assert [e["id"] for e in r["events"]] == ["1", "2"] and r["active"] and r["next"] == 2
    assert cl.get(f"/monitor/{sid}/events?after=2").json()["events"] == []
    assert cl.delete(f"/monitor/{sid}").json() == {"stopped": True}
    assert cl.get(f"/monitor/{sid}/events").status_code == 404
    assert cl.delete(f"/monitor/{sid}").status_code == 404


@pytest.mark.parametrize("payload", [
    {"url": "https://example.com/x"}, {"url": VID, "poll_seconds": 1}, {"url": VID, "poll_seconds": 9999}, {},
])
def test_api_monitor_validates_input(client, payload):
    assert client[0].post("/monitor/start", json=payload).status_code == 422


def test_api_monitor_returns_429_when_full(client, monkeypatch):
    cl, app = client
    app.state.monitors._max = 1
    assert cl.post("/monitor/start", json={"url": VID}).status_code == 200
    assert cl.post("/monitor/start", json={"url": VID}).status_code == 429


def test_api_monitor_requires_key_when_configured(monkeypatch):
    monkeypatch.setenv("HATEDET_API_KEY", "k")
    app = create_app(FakePipeline(), META, ScriptedSource([[c(1)]]), monitor_autostart=False)
    with TestClient(app) as cl:
        assert cl.post("/monitor/start", json={"url": VID}).status_code == 401
        assert cl.get("/monitor/x/events").status_code == 401
        assert cl.post("/monitor/start", json={"url": VID}, headers={"X-API-Key": "k"}).status_code == 200
