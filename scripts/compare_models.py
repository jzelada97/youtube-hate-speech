"""Compara modelos individuales y ensembles con CV repetida PAREADA. Ejecutar: python scripts/compare_models.py

Mismas particiones para todos los modelos; la aumentacion (EDA ambas clases x1, la del modelo servido) se
aplica solo al fold de entrenamiento. Guarda las puntuaciones por fold en reports/model_comparison.json.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.model_selection import RepeatedStratifiedKFold

from hatedet.data.augment import augment_train
from hatedet.data.loader import load_raw_comments
from hatedet.data.schema import TARGET_COLUMN, TEXT_COLUMN
from hatedet.models.ensemble import MEMBERS, build_stacking, build_voting

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

def best_macro_f1(y, prob):
    """F1-macro con el mejor umbral posible (optimista, pero igual de generoso con todos los modelos)."""
    y = np.asarray(y).astype(bool)
    best = 0.0
    for t in np.unique(prob):
        pred = prob >= t
        f = []
        for cls in (True, False):
            tp = np.sum((pred == cls) & (y == cls)); fp = np.sum((pred == cls) & (y != cls)); fn = np.sum((pred != cls) & (y == cls))
            f.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
        best = max(best, np.mean(f))
    return float(best)


def precision_at_recall(y, prob, target=0.70):
    p, r, _ = precision_recall_curve(y, prob)
    ok = r[:-1] >= target
    return float(p[:-1][ok].max()) if ok.any() else 0.0


FACTORIES = {
    "baseline (lr_word)": MEMBERS["lr_word"],
    "voting 2 (lr_word+lr_char)": lambda: build_voting(("lr_word", "lr_char")),
    "voting 3 (lr_word+lr_char+nb_word)": lambda: build_voting(("lr_word", "lr_char", "nb_word")),
    "voting 3 ponderado (2,1,1)": lambda: build_voting(("lr_word", "lr_char", "nb_word"), weights=(2, 1, 1)),
    "voting 4 (+svm_char)": lambda: build_voting(("lr_word", "lr_char", "nb_word", "svm_char")),
    "voting 5 (+et_word)": lambda: build_voting(("lr_word", "lr_char", "nb_word", "svm_char", "et_word")),
}


def one_fold(fold, tr, te, df):
    train, val = df.iloc[tr], df.iloc[te]
    aug = augment_train(train, n_eda_copies=1, seed=fold)
    out = {}
    for name, make in FACTORIES.items():
        m = make().fit(aug[TEXT_COLUMN], aug[TARGET_COLUMN].astype(int))
        prob = m.predict_proba(val[TEXT_COLUMN])[:, 1]
        pred = prob >= 0.5
        yv = val[TARGET_COLUMN]
        out[name] = (f1_score(yv, pred, average="macro"), best_macro_f1(yv, prob),
                     average_precision_score(yv, prob), precision_at_recall(yv, prob))
    return out


def main():
    df = load_raw_comments()
    y = df[TARGET_COLUMN].values.astype(int)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=11)
    folds = list(cv.split(df, y))
    res = Parallel(n_jobs=-1)(delayed(one_fold)(i, tr, te, df) for i, (tr, te) in enumerate(folds))
    S = {n: {"f1_macro_at_0.5": [r[n][0] for r in res], "f1_macro_best_thr": [r[n][1] for r in res],
             "pr_auc": [r[n][2] for r in res], "precision_at_recall70": [r[n][3] for r in res]} for n in FACTORIES}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/model_comparison.json").write_text(json.dumps(S, indent=1))

    base = "baseline (lr_word)"
    keys = ("f1_macro_at_0.5", "f1_macro_best_thr", "pr_auc", "precision_at_recall70")
    print(f"{'modelo':36s}" + "".join(f"{k:>22s}" for k in keys))
    for n, sc in S.items():
        print(f"{n:36s}" + "".join(f"{np.mean(sc[k]):22.3f}" for k in keys))
    print("Diferencia pareada frente al baseline (media, folds que mejoran /20, p de Wilcoxon):")
    for n, sc in S.items():
        if n == base:
            continue
        line = f"{n:36s}"
        for k in keys[1:]:
            d = np.array(sc[k]) - np.array(S[base][k])
            p = wilcoxon(d).pvalue if np.any(d) else 1.0
            line += f" {k}:{d.mean():+.3f}({int((d > 0).sum())}/20,p={p:.3f})"
        print(line)


if __name__ == "__main__":
    main()
