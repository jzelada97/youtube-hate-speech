import numpy as np

from hatedet.models.bands import band_for, choose_thresholds


def test_band_for_boundaries():
    assert band_for(0.10, 0.5, 0.9) == "permitir"
    assert band_for(0.50, 0.5, 0.9) == "revisar"
    assert band_for(0.89, 0.5, 0.9) == "revisar"
    assert band_for(0.90, 0.5, 0.9) == "ocultar"


def test_choose_thresholds_meets_target_recall_on_separable_data():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(100), np.zeros(400)].astype(int)
    scores = np.r_[rng.uniform(0.6, 1.0, 100), rng.uniform(0.0, 0.5, 400)]
    t = choose_thresholds(y, scores, target_recall=0.9, target_precision=0.9)
    assert t["recall_at_t_low"] >= 0.9 and t["precision_at_t_high"] >= 0.9
    assert t["t_low"] <= t["t_high"]


def test_choose_thresholds_unreachable_precision_falls_back_safely():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    scores = rng.uniform(0, 1, 300)  # sin senal: precision alta inalcanzable
    t = choose_thresholds(y, scores, target_recall=0.7, target_precision=0.99)
    assert t["t_high"] >= t["t_low"]
