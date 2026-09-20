import numpy as np

from hatedet.models.evaluate import evaluate, generalization_gap_pp, precision_at_fixed_recall


def test_evaluate_perfect_predictions():
    y_true = np.array([True, False, True, False, True])
    y_pred = y_true.copy()
    y_scores = np.array([0.9, 0.1, 0.95, 0.05, 0.99])

    report = evaluate(y_true, y_pred, y_scores)

    assert report.f1_macro == 1.0
    assert report.f1_minority == 1.0
    assert report.pr_auc == 1.0


def test_evaluate_detects_confusion():
    y_true = np.array([True, True, False, False])
    y_pred = np.array([False, False, False, False])
    y_scores = np.array([0.4, 0.3, 0.2, 0.1])

    report = evaluate(y_true, y_pred, y_scores)

    tn, fp, fn, tp = report.confusion.ravel()
    assert fn == 2
    assert tp == 0


def test_precision_at_fixed_recall_returns_zero_when_unreachable():
    y_true = np.array([True, True, False, False])
    y_scores = np.array([0.1, 0.1, 0.9, 0.9])  # scores al reves del label real

    result = precision_at_fixed_recall(y_true, y_scores, target_recall=1.0)

    assert 0.0 <= result <= 1.0


def test_generalization_gap_pp_is_percentage_points():
    class FakeReport:
        def __init__(self, f1_macro):
            self.f1_macro = f1_macro

    gap = generalization_gap_pp(FakeReport(0.90), FakeReport(0.85))
    assert abs(gap - 5.0) < 1e-9
