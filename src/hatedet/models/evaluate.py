"""Metricas de evaluacion para clasificacion binaria desbalanceada.

Con ~14% de positivos, accuracy es enganosa (un modelo que siempre predice "no odio" ya
acertaria ~86%). Se usan metricas que penalizan explicitamente fallar en la clase minoritaria.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)


@dataclass
class ClassificationReport:
    f1_macro: float
    f1_minority: float
    pr_auc: float
    precision_at_recall_90: float
    confusion: np.ndarray

    def __str__(self) -> str:
        tn, fp, fn, tp = self.confusion.ravel()
        return (
            f"F1-macro:               {self.f1_macro:.3f}\n"
            f"F1 clase minoritaria:    {self.f1_minority:.3f}\n"
            f"PR-AUC:                  {self.pr_auc:.3f}\n"
            f"Precision @ recall=0.90: {self.precision_at_recall_90:.3f}\n"
            f"Matriz de confusion -> TN={tn} FP={fp} FN={fn} TP={tp}"
        )


def precision_at_fixed_recall(y_true, y_scores, target_recall: float = 0.90) -> float:
    """Precision alcanzable manteniendo al menos `target_recall` de recall.

    Relevante para el sistema de bandas de decision (permitir/revisar/eliminar): el cliente
    puede fijar cuanto recall quiere garantizar y esto responde que precision paga por ello.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    valid = recall >= target_recall
    if not valid.any():
        return 0.0
    return float(precision[valid].max())


def best_macro_f1(y_true, y_scores) -> float:
    """Mejor F1-macro sobre todos los umbrales posibles: mide el ranking, sin depender de un corte concreto.

    Es la misma definicion que usan las comparaciones de la CV pareada (scripts/eval_lstm.py), para que las cifras
    sean comparables con la referencia congelada (reports/pre_bert_reference.json).
    """
    y = np.asarray(y_true).astype(bool)
    scores = np.asarray(y_scores)
    best = 0.0
    for threshold in np.unique(scores):
        pred = scores >= threshold
        f1s = []
        for cls in (True, False):
            tp = np.sum((pred == cls) & (y == cls))
            fp = np.sum((pred == cls) & (y != cls))
            fn = np.sum((pred != cls) & (y == cls))
            f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
        best = max(best, float(np.mean(f1s)))
    return best


def evaluate(y_true, y_pred, y_scores) -> ClassificationReport:
    return ClassificationReport(
        f1_macro=f1_score(y_true, y_pred, average="macro"),
        f1_minority=f1_score(y_true, y_pred, pos_label=True),
        pr_auc=average_precision_score(y_true, y_scores),
        precision_at_recall_90=precision_at_fixed_recall(y_true, y_scores),
        confusion=confusion_matrix(y_true, y_pred),
    )


def generalization_gap_pp(train_report: ClassificationReport, test_report: ClassificationReport) -> float:
    """Diferencia train-test en F1-macro, en puntos porcentuales (NFR-1 del cliente: < 5)."""
    return (train_report.f1_macro - test_report.f1_macro) * 100
