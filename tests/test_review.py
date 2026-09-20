import numpy as np
import pytest
from fastapi.testclient import TestClient

from hatedet.api.main import create_app
from hatedet.db.store import PENDING, RESOLVED, ReviewStore, local_id


@pytest.fixture(autouse=True)
def _no_ambient_queue(monkeypatch):
    """La cola se inyecta en cada test; si el entorno tuviera HATEDET_REVIEW_DB usaria la base real."""
    monkeypatch.delenv("HATEDET_REVIEW_DB", raising=False)


@pytest.fixture
def store(tmp_path):
    return ReviewStore(tmp_path / "review.db")


def _items(n=3):
    return [{"text": f"comentario {i}", "score": 0.9 - i * 0.1, "band": "revisar", "comment_id": f"c{i}"}
            for i in range(n)]


def test_a_bad_path_fails_loudly_instead_of_pretending_to_store(tmp_path):
    blocker = tmp_path / "soy_un_fichero"
    blocker.write_text("no soy un directorio")
    with pytest.raises(RuntimeError, match="HATEDET_REVIEW_DB"):
        ReviewStore(blocker / "review.db")


def test_enqueue_returns_how_many_were_added_and_ignores_repeats(store):
    assert store.enqueue(_items(3), "v1") == 3
    assert store.enqueue(_items(3), "v1") == 0
    assert store.enqueue([{"text": "otro", "score": 0.7, "band": "revisar", "comment_id": "nuevo"}], "v1") == 1
    assert store.stats()["total"] == 4


def test_enqueue_count_matches_reality_for_one_item_and_for_mixed_batches(store):
    """El conteo se llevo un fallo en Docker: cuenta las filas insertadas de verdad, no las enviadas."""
    assert store.enqueue(_items(1), "v1") == 1
    assert store.stats()["total"] == 1
    mixed = _items(3)  # c0 ya esta; c1 y c2 son nuevos
    assert store.enqueue(mixed, "v1") == 2
    assert store.stats()["total"] == 3


def test_text_without_comment_id_gets_a_stable_id(store):
    item = {"text": "  escrito a mano  ", "score": 0.8, "band": "revisar"}
    store.enqueue([item], "v1")
    store.enqueue([{"text": "escrito a mano", "score": 0.8, "band": "revisar"}], "v1")
    assert store.stats()["total"] == 1
    assert store.pending()[0].comment_id == local_id("escrito a mano")


def test_pending_is_ordered_by_score_so_the_moderator_sees_the_worst_first(store):
    store.enqueue(_items(3), "v1")
    scores = [i.score for i in store.pending()]
    assert scores == sorted(scores, reverse=True)
    assert all(i.status == PENDING for i in store.pending())


def test_resolve_records_the_verdict_and_removes_it_from_pending(store):
    store.enqueue(_items(2), "v1")
    first = store.pending()[0]
    assert store.resolve(first.id, verdict=True, tags=["IsRacist"], reviewer="jose")
    again = store.get(first.id)
    assert again.status == RESOLVED and again.verdict is True and again.tags == ["IsRacist"]
    assert again.reviewer == "jose" and again.reviewed_at
    assert first.id not in [i.id for i in store.pending()]


def test_resolve_unknown_id_returns_false(store):
    assert store.resolve(9999, verdict=True) is False


def test_stats_counts_verdicts_and_agreement_with_the_model(store):
    store.enqueue([{"text": "a", "score": 0.9, "band": "ocultar", "comment_id": "a"},
                   {"text": "b", "score": 0.2, "band": "permitir", "comment_id": "b"}], "v1")
    by_id = {i.comment_id: i.id for i in store.pending()}
    store.resolve(by_id["a"], verdict=True)   # el modelo decia odio (0.9) y acierta
    store.resolve(by_id["b"], verdict=True)   # el modelo decia no-odio (0.2) y falla
    stats = store.stats()
    assert stats == {"total": 2, "pendientes": 0, "revisados": 2, "es_odio": 2, "no_es_odio": 0,
                     "acuerdo_con_el_modelo": 0.5}


def test_stats_agreement_is_none_when_nothing_has_been_reviewed(store):
    store.enqueue(_items(1), "v1")
    assert store.stats()["acuerdo_con_el_modelo"] is None


def test_training_rows_only_include_resolved_items_in_dataset_format(store):
    store.enqueue(_items(2), "v1")
    store.resolve(store.pending()[0].id, verdict=True, tags=["IsRacist"])
    rows = store.training_rows()
    assert len(rows) == 1
    assert set(rows[0]) == {"CommentId", "VideoId", "Text", "IsHatespeech", "tags"}
    assert rows[0]["IsHatespeech"] is True


def test_purge_removes_only_reviewed_items_older_than_the_retention_window(store):
    store.enqueue(_items(2), "v1")
    store.resolve(store.pending()[0].id, verdict=False)
    assert store.purge_older_than(days=1) == 0          # recien revisado: se conserva
    assert store.purge_older_than(days=-1) == 1         # ventana ya vencida
    assert store.stats()["total"] == 1                  # el pendiente nunca se borra


class FakePipeline:
    def predict_proba(self, texts):
        p = np.array([0.95 if "hate" in t else 0.05 for t in texts])
        return np.column_stack([1 - p, p])


META = {"model_version": "fake", "thresholds": {"t_low": 0.5, "t_high": 0.9}}


def _client(store=None):
    return TestClient(create_app(FakePipeline(), META, review_store=store, monitor_autostart=False))


def test_endpoints_return_503_when_the_queue_is_disabled():
    with _client() as c:
        assert c.get("/health").json()["review_enabled"] is False
        assert c.post("/review/queue", json={"items": [{"text": "hola"}]}).status_code == 503
        assert c.get("/review/pending").status_code == 503


def test_queue_endpoint_only_enqueues_what_needs_a_human_by_default(store):
    with _client(store) as c:
        body = {"items": [{"text": "so much hate"}, {"text": "lovely video"}]}
        assert c.post("/review/queue", json=body).json() == {"encolados": 1, "candidatos": 1}
        assert c.get("/health").json()["review_enabled"] is True
    assert store.pending()[0].text == "so much hate"


def test_queue_endpoint_can_take_everything_when_asked(store):
    with _client(store) as c:
        body = {"items": [{"text": "so much hate"}, {"text": "lovely video"}], "only_flagged": False}
        assert c.post("/review/queue", json=body).json()["encolados"] == 2


def test_full_cycle_through_the_api(store):
    with _client(store) as c:
        c.post("/review/queue", json={"items": [{"text": "so much hate", "video_id": "dQw4w9WgXcQ"}]})
        pending = c.get("/review/pending").json()
        assert pending["items"][0]["video_id"] == "dQw4w9WgXcQ"
        assert "IsRacist" in pending["etiquetas_disponibles"]
        item_id = pending["items"][0]["id"]
        r = c.post(f"/review/{item_id}", json={"is_hatespeech": True, "tags": ["IsRacist"], "reviewer": "jose"})
        assert r.status_code == 200
        assert c.get("/review/pending").json()["items"] == []
        assert c.get("/review/stats").json()["es_odio"] == 1


def test_api_rejects_unknown_tags_and_missing_items(store):
    with _client(store) as c:
        c.post("/review/queue", json={"items": [{"text": "so much hate"}]})
        item_id = c.get("/review/pending").json()["items"][0]["id"]
        assert c.post(f"/review/{item_id}", json={"is_hatespeech": True, "tags": ["Inventada"]}).status_code == 422
        assert c.post("/review/99999", json={"is_hatespeech": True}).status_code == 404
        assert c.post("/review/queue", json={"items": []}).status_code == 422
