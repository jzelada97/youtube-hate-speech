import numpy as np
import pytest

from hatedet.data.loader import load_raw_comments
from hatedet.models.ensemble import DEFAULT_ENSEMBLE_PARAMS, MEMBERS, build_ensemble, build_stacking, build_voting

TEXTS = ["all these people are criminals and animals", "what a lovely video thanks", "you are a stupid idiot",
         "great song love it", "these blacks are all thieves", "nice work everyone", "so much hate here",
         "beautiful and inspiring", "criminals every one of them", "thank you for sharing this"] * 6
LABELS = np.array([1, 0, 0, 0, 1, 0, 1, 0, 1, 0] * 6)


@pytest.mark.parametrize("name", list(MEMBERS))
def test_every_member_is_a_complete_pipeline_from_raw_text(name):
    model = MEMBERS[name]().fit(TEXTS, LABELS)
    proba = model.predict_proba(["a brand new comment"])
    assert proba.shape == (1, 2) and abs(proba.sum() - 1) < 1e-6


def test_default_ensemble_is_a_soft_weighted_vote_of_three_members():
    ens = build_ensemble()
    assert [n for n, _ in ens.estimators] == ["lr_word", "lr_char", "nb_word"]
    assert ens.voting == "soft" and list(ens.weights) == [2, 1, 1]


def test_ensemble_params_override_defaults_and_do_not_mutate_them():
    before = dict(DEFAULT_ENSEMBLE_PARAMS)
    ens = build_ensemble({"w_lr_word": 4, "lr_char_C": 0.05, "lr_char_max_features": 1000})
    assert list(ens.weights) == [4, 1, 1]
    assert ens.estimators[1][1].named_steps["clf"].C == 0.05
    assert ens.estimators[1][1].named_steps["tfidf"].max_features == 1000
    assert DEFAULT_ENSEMBLE_PARAMS == before


def test_ensemble_fits_and_serialises_as_a_single_object(tmp_path):
    import joblib

    ens = build_ensemble().fit(TEXTS, LABELS)
    path = tmp_path / "e.joblib"
    joblib.dump(ens, path)
    loaded = joblib.load(path)
    sample = ["so much hate here", "lovely"]
    assert np.allclose(ens.predict_proba(sample), loaded.predict_proba(sample))


def test_voting_and_stacking_builders():
    assert len(build_voting(("lr_word", "lr_char")).estimators) == 2
    assert build_stacking(("lr_word", "nb_word")).stack_method == "predict_proba"


@pytest.mark.requires_data
def test_ensemble_on_real_data_learns_something():
    df = load_raw_comments()
    y = df.IsHatespeech.astype(int).values
    ens = build_ensemble().fit(df.Text[:600], y[:600])
    p = ens.predict_proba(df.Text[600:])[:, 1]
    assert p[y[600:] == 1].mean() > p[y[600:] == 0].mean()
