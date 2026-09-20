"""Evalua las redes recurrentes frente a TF-IDF con la misma CV pareada. Ejecutar: python scripts/eval_lstm.py

Mismas particiones para todos los modelos. Las redes recurrentes se entrenan SIN Easy Data Augmentation (barajar o
borrar palabras destruye el orden, que es justo lo que una red recurrente aprovecha); el baseline y el ensemble
usan su propia configuracion (EDA). Salida: reports/lstm_comparison.json.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import RepeatedStratifiedKFold

from hatedet.data.augment import augment_train
from hatedet.data.loader import load_raw_comments
from hatedet.data.schema import TARGET_COLUMN, TEXT_COLUMN
from hatedet.models.ensemble import MEMBERS, build_ensemble
from hatedet.models.glove import glove_initializer
from hatedet.models.lstm import LSTMClassifier

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")


def best_macro_f1(y, prob):
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


# (fabrica, usa_EDA)
MODELS = {
    "baseline (lr_word + EDA)": (MEMBERS["lr_word"], True),
    "ensemble (votacion + EDA)": (build_ensemble, True),
    "BiLSTM desde cero": (lambda: LSTMClassifier(cell="lstm"), False),
    "BiGRU desde cero": (lambda: LSTMClassifier(cell="gru"), False),
    "BiLSTM pequena (mas regularizada)": (lambda: LSTMClassifier(cell="lstm", emb_dim=32, hidden=24, dropout=0.6, weight_decay=5e-2), False),
    "BiLSTM + GloVe congelado": (lambda: LSTMClassifier(cell="lstm", emb_dim=300, hidden=48, pretrained=glove_initializer, freeze_embeddings=True), False),
    "BiLSTM + GloVe afinado": (lambda: LSTMClassifier(cell="lstm", emb_dim=300, hidden=48, pretrained=glove_initializer, freeze_embeddings=False, lr=1e-3), False),
}
if len(sys.argv) > 1 and sys.argv[1] == "glove":
    MODELS = {k: v for k, v in MODELS.items() if k.startswith(("baseline", "ensemble", "BiLSTM desde cero")) or "GloVe" in k}


def main():
    df = load_raw_comments()
    y = df[TARGET_COLUMN].values.astype(int)
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=11).split(df, y))
    S = {n: {"f1_macro_best_thr": [], "pr_auc": [], "precision_at_recall70": []} for n in MODELS}
    for i, (tr, te) in enumerate(folds):
        train, val = df.iloc[tr], df.iloc[te]
        aug = augment_train(train, n_eda_copies=1, seed=i)
        for name, (make, eda) in MODELS.items():
            data = aug if eda else train
            m = make().fit(data[TEXT_COLUMN], data[TARGET_COLUMN].astype(int))
            prob = m.predict_proba(val[TEXT_COLUMN])[:, 1]
            yv = val[TARGET_COLUMN]
            S[name]["f1_macro_best_thr"].append(best_macro_f1(yv, prob))
            S[name]["pr_auc"].append(average_precision_score(yv, prob))
            S[name]["precision_at_recall70"].append(precision_at_recall(yv, prob))
        print(f"fold {i + 1}/{len(folds)} hecho", flush=True)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/lstm_glove_comparison.json" if "GloVe" in " ".join(S) else "reports/lstm_comparison.json").write_text(json.dumps(S, indent=1))

    keys = ("f1_macro_best_thr", "pr_auc", "precision_at_recall70")
    print(f"\n{'modelo':38s}" + "".join(f"{k:>22s}" for k in keys))
    for n, sc in S.items():
        print(f"{n:38s}" + "".join(f"{np.mean(sc[k]):22.3f}" for k in keys))
    base = "baseline (lr_word + EDA)"
    print("\nDiferencia pareada frente al baseline (media, folds que mejoran /10, p de Wilcoxon):")
    for n, sc in S.items():
        if n == base:
            continue
        line = f"{n:38s}"
        for k in keys:
            d = np.array(sc[k]) - np.array(S[base][k])
            p = wilcoxon(d).pvalue if np.any(d) else 1.0
            line += f" {k}:{d.mean():+.3f}({int((d > 0).sum())}/10,p={p:.3f})"
        print(line)


if __name__ == "__main__":
    main()
