"""Evalua si la data augmentation ayuda, con CV repetida PAREADA. Ejecutar: python scripts/eval_augmentation.py

Regla clave anti-fuga: la augmentation se aplica SOLO al fold de entrenamiento; el fold de validacion
es siempre real. Todas las variantes usan exactamente las mismas particiones (comparacion pareada).
"""

import sys

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from hatedet.data.augment import augment_train
from hatedet.data.loader import load_raw_comments
from hatedet.data.schema import TARGET_COLUMN, TEXT_COLUMN
from hatedet.models.baseline import build_baseline_pipeline

sys.stdout.reconfigure(encoding="utf-8")

VARIANTS = {
    "sin augmentation (baseline)": dict(),
    "negativos dificiles (40)": dict(n_hard_negatives=40),
    "EDA ambas clases x1": dict(n_eda_copies=1),
    "EDA ambas clases x2": dict(n_eda_copies=2),
    "EDA solo odio x3": dict(n_eda_copies=3, eda_classes="hate"),
    "EDA ambas x1 + negativos (40)": dict(n_eda_copies=1, n_hard_negatives=40),
}


def main():
    df = load_raw_comments()
    y = df[TARGET_COLUMN].values.astype(int)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=11)
    scores = {v: {"f1_macro": [], "f1_odio": [], "pr_auc": []} for v in VARIANTS}

    for fold, (tr, te) in enumerate(cv.split(df, y)):
        train, val = df.iloc[tr], df.iloc[te]
        for name, kw in VARIANTS.items():
            aug = augment_train(train, seed=fold, **kw)
            model = build_baseline_pipeline().fit(aug[TEXT_COLUMN], aug[TARGET_COLUMN].astype(int))
            pred = model.predict(val[TEXT_COLUMN])
            prob = model.predict_proba(val[TEXT_COLUMN])[:, 1]
            s = scores[name]
            s["f1_macro"].append(f1_score(val[TARGET_COLUMN], pred, average="macro"))
            s["f1_odio"].append(f1_score(val[TARGET_COLUMN], pred))
            s["pr_auc"].append(average_precision_score(val[TARGET_COLUMN], prob))

    base = "sin augmentation (baseline)"
    print(f"{'variante':32s} {'F1-macro':>9s} {'F1-odio':>9s} {'PR-AUC':>9s}   (medias sobre 30 folds)")
    for name, s in scores.items():
        print(f"{name:32s} {np.mean(s['f1_macro']):9.3f} {np.mean(s['f1_odio']):9.3f} {np.mean(s['pr_auc']):9.3f}")
    print("\nDiferencia pareada frente al baseline:")
    for name, s in scores.items():
        if name == base:
            continue
        for m in ("f1_macro", "f1_odio", "pr_auc"):
            d = np.array(s[m]) - np.array(scores[base][m])
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            print(f"  {name:32s} {m:9s} dif={d.mean():+.3f}  mejora {int((d > 0).sum()):2d}/30 folds  p={p:.3f}")


if __name__ == "__main__":
    main()
