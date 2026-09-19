"""Bandas de decision (permitir / revisar / ocultar) a partir de la probabilidad de odio.

El sistema RECOMIENDA, no ejecuta acciones. Los umbrales se eligen con probabilidades fuera de muestra:
  t_low  = umbral mas alto que aun alcanza `target_recall` de recall (a partir de aqui, revisar).
  t_high = umbral mas bajo con precision >= `target_precision` (a partir de aqui, ocultar).
Con este modelo la precision maxima alcanzable es baja (~0.4-0.5): "ocultar" NO es una decision
fiable por si sola, y `revisar` es la banda principal (ver docs/DECISIONES.md, seccion 12).
"""

import numpy as np
from sklearn.metrics import precision_recall_curve

BAND_ALLOW, BAND_REVIEW, BAND_HIDE = "permitir", "revisar", "ocultar"


def choose_thresholds(
    y_true, scores, target_recall: float = 0.70, target_precision: float = 0.50
) -> dict:
    y_true, scores = np.asarray(y_true).astype(int), np.asarray(scores)
    precision, recall, thr = precision_recall_curve(y_true, scores)
    ok_r = np.where(recall[:-1] >= target_recall)[0]
    t_low = float(thr[ok_r[-1]]) if len(ok_r) else float(thr[0])
    ok_p = np.where(precision[:-1] >= target_precision)[0]
    t_high = float(thr[ok_p[0]]) if len(ok_p) else 1.0
    t_high = max(t_high, t_low)
    at = lambda t: (scores >= t)
    return {
        "t_low": round(t_low, 4),
        "t_high": round(t_high, 4),
        "recall_at_t_low": round(float(at(t_low)[y_true == 1].mean()), 3),
        "precision_at_t_low": round(float(y_true[at(t_low)].mean()) if at(t_low).any() else 0.0, 3),
        "flagged_fraction_at_t_low": round(float(at(t_low).mean()), 3),
        "precision_at_t_high": round(float(y_true[at(t_high)].mean()) if at(t_high).any() else 0.0, 3),
        "recall_at_t_high": round(float(at(t_high)[y_true == 1].mean()), 3),
        "flagged_fraction_at_t_high": round(float(at(t_high).mean()), 3),
    }


def band_for(score: float, t_low: float, t_high: float) -> str:
    if score >= t_high:
        return BAND_HIDE
    if score >= t_low:
        return BAND_REVIEW
    return BAND_ALLOW
