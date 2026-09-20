import numpy as np
import pytest
from fastapi.testclient import TestClient

from hatedet.api.main import MAX_BATCH, MAX_TEXT_CHARS, create_app


class FakePipeline:
    """Puntua alto si aparece 'hate', medio si 'maybe', bajo en otro caso."""

    def predict_proba(self, texts):
        p = [0.95 if "hate" in t else 0.6 if "maybe" in t else 0.05 for t in texts]
        return np.column_stack([1 - np.array(p), np.array(p)])


META = {"model_version": "fake_v1", "thresholds": {"t_low": 0.5, "t_high": 0.9}}


@pytest.fixture()
def client():
    with TestClient(create_app(FakePipeline(), META)) as c:
        yield c


def test_health_reports_version_and_thresholds(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_version"] == "fake_v1"
    assert body["thresholds"] == {"t_low": 0.5, "t_high": 0.9}


def test_predict_returns_contract_fields(client):
    r = client.post("/predict", json={"text": "I hate this"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"label", "score", "band", "model_version"}
    assert body["band"] == "ocultar" and body["label"] is True


def test_bands_allow_review_hide(client):
    bands = [client.post("/predict", json={"text": t}).json()["band"] for t in ("nice", "maybe", "hate")]
    assert bands == ["permitir", "revisar", "ocultar"]


def test_batch_preserves_order(client):
    r = client.post("/predict/batch", json={"texts": ["nice", "hate", "maybe"]})
    assert [p["band"] for p in r.json()["predictions"]] == ["permitir", "ocultar", "revisar"]


@pytest.mark.parametrize("payload", [
    {"text": ""}, {"text": "   "}, {"text": "x" * (MAX_TEXT_CHARS + 1)}, {}, {"text": 123},
])
def test_predict_rejects_invalid_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("payload", [
    {"texts": []}, {"texts": ["ok", " "]}, {"texts": ["x"] * (MAX_BATCH + 1)},
])
def test_batch_rejects_invalid_input(client, payload):
    assert client.post("/predict/batch", json=payload).status_code == 422


def test_api_key_enforced_when_configured(monkeypatch):
    monkeypatch.setenv("HATEDET_API_KEY", "secret")
    with TestClient(create_app(FakePipeline(), META)) as c:
        assert c.post("/predict", json={"text": "hi"}).status_code == 401
        assert c.post("/predict", json={"text": "hi"}, headers={"X-API-Key": "bad"}).status_code == 401
        assert c.post("/predict", json={"text": "hi"}, headers={"X-API-Key": "secret"}).status_code == 200
        assert c.get("/health").status_code == 200


def test_threshold_override_from_env(monkeypatch):
    monkeypatch.setenv("HATEDET_T_LOW", "0.01")
    with TestClient(create_app(FakePipeline(), META)) as c:
        assert c.post("/predict", json={"text": "nice"}).json()["band"] == "revisar"


def test_cors_allows_extension_and_localhost_but_not_other_origins(client):
    ok = client.options("/predict", headers={"Origin": "chrome-extension://abc", "Access-Control-Request-Method": "POST"})
    bad = client.options("/predict", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "chrome-extension://abc"
    assert "access-control-allow-origin" not in bad.headers


@pytest.mark.requires_data
def test_end_to_end_with_real_pipeline_from_raw_text():
    from hatedet.data.loader import load_raw_comments
    from hatedet.models.baseline import build_baseline_pipeline

    df = load_raw_comments()
    pipe = build_baseline_pipeline().fit(df.Text, df.IsHatespeech.astype(int))
    with TestClient(create_app(pipe, {"model_version": "real_test", "thresholds": {"t_low": 0.49, "t_high": 0.88}})) as c:
        r = c.post("/predict", json={"text": "What a lovely video, thanks for sharing!"})
        assert r.status_code == 200 and 0.0 <= r.json()["score"] <= 1.0
