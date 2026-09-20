"""Marca comentarios con etiqueta IsHatespeech sospechosa (confident learning).

Ejecutar: python scripts/find_label_issues.py
Genera data/interim/label_issues.csv (ordenado por sospecha) para revision humana.
No modifica data/raw: solo propone candidatos, la decision de cambiar una etiqueta es humana.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues
from sklearn.model_selection import RepeatedStratifiedKFold

from hatedet.data.loader import load_raw_comments
from hatedet.data.schema import COMMENT_ID_COLUMN, TARGET_COLUMN, TEXT_COLUMN
from hatedet.models.baseline import build_baseline_pipeline

OUT = Path("data/interim/label_issues.csv")


def out_of_fold_probs(df: pd.DataFrame, n_repeats: int = 3) -> np.ndarray:
    """P(odio) fuera de muestra, promediada sobre varias particiones CV para estabilidad."""
    y = df[TARGET_COLUMN].values.astype(int)
    probs, counts = np.zeros(len(df)), np.zeros(len(df))
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=42)
    for tr, te in cv.split(df[TEXT_COLUMN], y):
        model = build_baseline_pipeline()
        model.fit(df[TEXT_COLUMN].iloc[tr], y[tr])
        probs[te] += model.predict_proba(df[TEXT_COLUMN].iloc[te])[:, 1]
        counts[te] += 1
    return probs / counts


def main():
    df = load_raw_comments()
    y = df[TARGET_COLUMN].values.astype(int)
    p1 = out_of_fold_probs(df)
    pred_probs = np.column_stack([1 - p1, p1])

    idx = find_label_issues(
        labels=y, pred_probs=pred_probs, return_indices_ranked_by="self_confidence"
    )
    issues = df.loc[idx, [COMMENT_ID_COLUMN, TEXT_COLUMN, TARGET_COLUMN, "IsToxic", "IsRacist"]].copy()
    issues["p_odio_modelo"] = p1[idx].round(3)
    issues["sugerencia"] = np.where(issues[TARGET_COLUMN], "quizas NO es odio", "quizas SI es odio")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    issues.to_csv(OUT, index=False)

    print(f"Etiquetas sospechosas: {len(issues)} de {len(df)} ({len(issues) / len(df):.1%})")
    print(issues["sugerencia"].value_counts().to_string())
    print(f"Guardado en {OUT}")


if __name__ == "__main__":
    main()
