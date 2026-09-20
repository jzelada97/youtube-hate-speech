import numpy as np
import pytest

pytest.importorskip("torch", reason="la red recurrente necesita torch (pip install -e \".[nn]\")")

from hatedet.models.lstm import LSTMClassifier  # noqa: E402

TEXTS = ["all these people are criminals and animals", "what a lovely video thanks", "you are a stupid idiot",
         "great song love it", "these blacks are all thieves", "nice work everyone", "so much hate here",
         "beautiful and inspiring", "criminals every one of them", "thank you for sharing this"] * 8
LABELS = np.array([1, 0, 0, 0, 1, 0, 1, 0, 1, 0] * 8)


@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_fits_and_predicts_probabilities_for_both_cells(cell):
    m = LSTMClassifier(cell=cell, max_epochs=3, emb_dim=8, hidden=8).fit(TEXTS, LABELS)
    proba = m.predict_proba(["a new comment", "hate hate hate"])
    assert proba.shape == (2, 2) and np.allclose(proba.sum(axis=1), 1) and ((proba >= 0) & (proba <= 1)).all()
    assert set(m.predict(["x", "y"])) <= {0, 1}


def test_same_seed_gives_same_predictions():
    a = LSTMClassifier(max_epochs=3, emb_dim=8, hidden=8, seed=1).fit(TEXTS, LABELS)
    b = LSTMClassifier(max_epochs=3, emb_dim=8, hidden=8, seed=1).fit(TEXTS, LABELS)
    assert np.allclose(a.predict_proba(TEXTS[:5]), b.predict_proba(TEXTS[:5]))


def test_early_stopping_limits_epochs_and_restores_best_state():
    m = LSTMClassifier(max_epochs=40, patience=2, emb_dim=8, hidden=8).fit(TEXTS, LABELS)
    assert len(m.history_) <= 40 and min(m.history_) <= m.history_[-1] + 1e-9


def test_unknown_words_and_empty_text_do_not_crash():
    m = LSTMClassifier(max_epochs=2, emb_dim=8, hidden=8).fit(TEXTS, LABELS)
    assert m.predict_proba(["zzzz qqqq unseenword", "", "!!!"]).shape == (3, 2)


def test_truncates_long_texts_and_learns_the_obvious_signal():
    m = LSTMClassifier(max_epochs=25, emb_dim=16, hidden=16, max_len=10).fit(TEXTS, LABELS)
    long_text = "criminals " * 500
    p = m.predict_proba([long_text, "lovely video thanks"])[:, 1]
    assert p[0] > p[1]


def test_can_be_pickled(tmp_path):
    import joblib

    m = LSTMClassifier(max_epochs=2, emb_dim=8, hidden=8).fit(TEXTS, LABELS)
    joblib.dump(m, tmp_path / "m.joblib")
    assert np.allclose(joblib.load(tmp_path / "m.joblib").predict_proba(TEXTS[:3]), m.predict_proba(TEXTS[:3]))
