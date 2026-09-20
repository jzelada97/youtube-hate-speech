import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from hatedet.api.main import create_app
from hatedet.youtube.analysis import analyze
from hatedet.youtube.sources import ApiSource, ScraperSource, SourceError, YouTubeComment, default_source
from hatedet.youtube.urls import parse_video_id, watch_url

VID = "dQw4w9WgXcQ"


@pytest.mark.parametrize("url", [
    f"https://www.youtube.com/watch?v={VID}",
    f"https://youtube.com/watch?v={VID}&t=42s",
    f"http://m.youtube.com/watch?v={VID}",
    f"https://youtu.be/{VID}?si=abc",
    f"https://www.youtube.com/shorts/{VID}",
    f"https://www.youtube.com/embed/{VID}",
    f"https://www.youtube.com/live/{VID}?feature=share",
    f"www.youtube.com/watch?v={VID}",
    VID,
])
def test_parse_video_id_accepts_common_forms(url):
    assert parse_video_id(url) == VID


@pytest.mark.parametrize("url", [
    "", "https://example.com/watch?v=" + VID, "https://www.youtube.com/watch?v=corto",
    "https://www.youtube.com/", "https://evil.com/?u=https://youtube.com/watch?v=" + VID, "no es una url",
])
def test_parse_video_id_rejects_invalid_or_foreign_urls(url):
    with pytest.raises(ValueError):
        parse_video_id(url)


def test_watch_url_is_built_by_us():
    assert watch_url(VID) == f"https://www.youtube.com/watch?v={VID}"


class FakeDownloader:
    def __init__(self, items=None, boom=False):
        self.items, self.boom = items or [], boom

    def get_comments_from_url(self, url, sort_by=1):
        assert url == watch_url(VID)
        if self.boom:
            raise RuntimeError("html cambiado")
        yield from self.items


def test_scraper_source_maps_fields_skips_blank_and_drops_author():
    items = [{"cid": "a", "text": "hola", "author": "Persona Real", "time_parsed": 1.0},
             {"cid": "b", "text": "   "}, {"cid": "c", "text": "adios"}]
    out = list(ScraperSource(FakeDownloader(items)).iter_comments(VID, 10))
    assert [c.id for c in out] == ["a", "c"]
    assert not hasattr(out[0], "author")


def test_scraper_source_respects_limit_and_wraps_errors():
    items = [{"cid": str(i), "text": f"t{i}"} for i in range(20)]
    assert len(list(ScraperSource(FakeDownloader(items)).iter_comments(VID, 5))) == 5
    with pytest.raises(SourceError):
        list(ScraperSource(FakeDownloader(boom=True)).iter_comments(VID, 5))


def _api_client(pages):
    calls = {"n": 0}

    def handler(request: httpx.Request):
        assert request.url.params["videoId"] == VID and request.url.params["key"] == "K"
        page = pages[calls["n"]]
        calls["n"] += 1
        return httpx.Response(page[0], json=page[1])

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def _thread(i, text):
    return {"snippet": {"topLevelComment": {"id": f"c{i}", "snippet": {"textDisplay": text}}}}


def test_api_source_paginates_and_stops_at_limit():
    pages = [(200, {"items": [_thread(1, "uno"), _thread(2, "dos")], "nextPageToken": "T"}),
             (200, {"items": [_thread(3, "tres"), _thread(4, "cuatro")]})]
    client, calls = _api_client(pages)
    assert [c.text for c in ApiSource("K", client).iter_comments(VID, 3)] == ["uno", "dos", "tres"]
    assert calls["n"] == 2


def test_api_source_reports_api_errors():
    body = {"error": {"errors": [{"reason": "commentsDisabled"}]}}
    client, _ = _api_client([(403, body)])
    client.headers  # noqa: B018
    with pytest.raises(SourceError, match="403"):
        list(ApiSource("K", client).iter_comments(VID, 5))


def test_default_source_uses_api_only_when_key_is_set(monkeypatch):
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    assert isinstance(default_source(), ScraperSource)
    monkeypatch.setenv("YOUTUBE_API_KEY", "x")
    assert isinstance(default_source(), ApiSource)


class FakePipeline:
    def predict_proba(self, texts):
        p = np.array([0.95 if "hate" in t else 0.6 if "maybe" in t else 0.05 for t in texts])
        return np.column_stack([1 - p, p])


def _comments():
    return [YouTubeComment("1", "nice"), YouTubeComment("2", "so much hate"),
            YouTubeComment("3", "maybe not ok"), YouTubeComment("4", "lovely")]


def test_analyze_counts_bands_and_ranks_flagged_by_score():
    rep = analyze(_comments(), FakePipeline(), 0.5, 0.9, video_id=VID, model_version="v")
    assert rep.n_comments == 4 and rep.counts == {"permitir": 2, "revisar": 1, "ocultar": 1}
    assert rep.share_flagged == 0.5
    assert [f.id for f in rep.flagged] == ["2", "3"]


def test_analyze_handles_no_comments_and_truncates_snippets():
    empty = analyze([], FakePipeline(), 0.5, 0.9, video_id=VID, model_version="v")
    assert empty.n_comments == 0 and empty.share_flagged == 0.0 and empty.flagged == []
    long = analyze([YouTubeComment("x", "hate " * 200)], FakePipeline(), 0.5, 0.9, VID, "v", snippet_chars=50)
    assert len(long.flagged[0].text) == 50


class FakeSource:
    def __init__(self, fail=False):
        self.fail = fail

    def iter_comments(self, video_id, limit):
        if self.fail:
            raise SourceError("comentarios desactivados")
        yield from _comments()[:limit]


META = {"model_version": "fake", "thresholds": {"t_low": 0.5, "t_high": 0.9}}


def test_endpoint_returns_report():
    with TestClient(create_app(FakePipeline(), META, FakeSource())) as c:
        r = c.post("/analyze/video", json={"url": f"https://youtu.be/{VID}", "max_comments": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["video_id"] == VID and body["n_comments"] == 3 and "author" not in str(body)


@pytest.mark.parametrize("payload,code", [
    ({"url": "https://example.com/x", "max_comments": 5}, 422),
    ({"url": f"https://youtu.be/{VID}", "max_comments": 0}, 422),
    ({"url": f"https://youtu.be/{VID}", "max_comments": 501}, 422),
    ({}, 422),
])
def test_endpoint_validates_input(payload, code):
    with TestClient(create_app(FakePipeline(), META, FakeSource())) as c:
        assert c.post("/analyze/video", json=payload).status_code == code


def test_endpoint_maps_source_failure_to_502():
    with TestClient(create_app(FakePipeline(), META, FakeSource(fail=True))) as c:
        r = c.post("/analyze/video", json={"url": VID})
    assert r.status_code == 502 and "desactivados" in r.json()["detail"]


def test_endpoint_respects_api_key(monkeypatch):
    monkeypatch.setenv("HATEDET_API_KEY", "secret")
    with TestClient(create_app(FakePipeline(), META, FakeSource())) as c:
        assert c.post("/analyze/video", json={"url": VID}).status_code == 401
        assert c.post("/analyze/video", json={"url": VID}, headers={"X-API-Key": "secret"}).status_code == 200
