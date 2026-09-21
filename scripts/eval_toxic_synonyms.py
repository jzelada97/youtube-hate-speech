"""Evalua la aumentacion con sinonimos toxicos (docs/DECISIONES.md, seccion 6.9).

    python scripts/eval_toxic_synonyms.py

Mismo protocolo que la referencia congelada (reports/pre_bert_reference.json): CV estratificada repetida pareada de
10 particiones (5 folds x 2 repeticiones, random_state=11) sobre las 997 filas, mismas particiones para todo, y la
aumentacion se genera SOLO con el trozo de entrenamiento. Modelos: baseline (lr_word) y ensemble servido
(conf/ensemble_params.json), que es la vara contra la que se decide.

Salidas: reports/toxic_synonyms_comparison.json y, para lectura humana,
data/interim/toxic_syn_review_sample.csv (con comentarios reales: fuera de git).
"""

import json
import random
import sys
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold

from hatedet.data.augment import augment_train
from hatedet.data.loader import load_raw_comments
from hatedet.data.schema import TARGET_COLUMN, TEXT_COLUMN
from hatedet.data.toxic_synonyms import toxic_variants
from hatedet.models.ensemble import MEMBERS, build_ensemble
from hatedet.models.evaluate import best_macro_f1, precision_at_fixed_recall
from hatedet.models.training import ensemble_params

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

SAMPLE_PATH = Path("data/interim/toxic_syn_review_sample.csv")
REPORT_PATH = Path("reports/toxic_synonyms_comparison.json")
REFERENCE_PATH = Path("reports/pre_bert_reference.json")

MODELS = {
    "baseline (lr_word)": MEMBERS["lr_word"],
    "ensemble servido (params optimizados)": lambda: build_ensemble(ensemble_params()),
}
CONFIGS = {
    "sin aumentacion": {},
    "EDA x1 ambas (referencia adoptada)": dict(n_eda_copies=1),
    "SIN x1 ambas": dict(n_toxsyn_copies=1),
    "SIN x3 ambas": dict(n_toxsyn_copies=3),
    "SIN x3 solo odio": dict(n_toxsyn_copies=3, toxsyn_classes="hate"),
    "EDA x1 + SIN x1 ambas": dict(n_eda_copies=1, n_toxsyn_copies=1),
    "EDA x1 + SIN x3 ambas": dict(n_eda_copies=1, n_toxsyn_copies=3),
}
REFERENCE = "EDA x1 ambas (referencia adoptada)"


def write_review_sample(df: pd.DataFrame, n_per_class: int = 30) -> None:
    """Muestra para lectura humana (criterio 4 de la seccion 6.9), estratificada por clase, con la columna vacia."""
    rng = random.Random(7)
    rows = []
    for label in (True, False):
        pool = []
        for t, y in zip(df[TEXT_COLUMN], df[TARGET_COLUMN]):
            if bool(y) == label:
                # crc32 y no hash(): el hash de un str cambia entre ejecuciones y la muestra no seria reproducible.
                v = toxic_variants(t, 1, random.Random(zlib.crc32(t.encode("utf-8"))))
                if v:
                    pool.append((t, v[0]))
        for original, variant in rng.sample(pool, min(n_per_class, len(pool))):
            rows.append({"etiqueta_original": "odio" if label else "no_odio", "original": original,
                         "variante": variant, "conserva_sentido_y_etiqueta (si/no)": ""})
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(SAMPLE_PATH, index=False, encoding="utf-8-sig")
    print(f"muestra para lectura humana: {SAMPLE_PATH} ({len(rows)} filas)")


def main() -> None:
    df = load_raw_comments()
    y = df[TARGET_COLUMN].values.astype(int)
    write_review_sample(df)
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=11).split(df, y))
    S = {m: {c: {"f1_macro_best_thr": [], "pr_auc": [], "precision_at_recall70": []} for c in CONFIGS} for m in MODELS}
    for i, (tr, te) in enumerate(folds):
        train, val = df.iloc[tr], df.iloc[te]
        yv = val[TARGET_COLUMN]
        for config, kwargs in CONFIGS.items():
            data = augment_train(train, seed=i, **kwargs)
            for model_name, make in MODELS.items():
                prob = make().fit(data[TEXT_COLUMN], data[TARGET_COLUMN].astype(int)).predict_proba(val[TEXT_COLUMN])[:, 1]
                s = S[model_name][config]
                s["f1_macro_best_thr"].append(best_macro_f1(yv, prob))
                s["pr_auc"].append(float(average_precision_score(yv, prob)))
                s["precision_at_recall70"].append(precision_at_fixed_recall(yv, prob, 0.70))
        n_syn = int((augment_train(train, seed=i, n_toxsyn_copies=3)["kind"] == "sinonimo_toxico").sum())
        print(f"particion {i + 1}/{len(folds)} hecha ({n_syn} ejemplos sinteticos con x3)", flush=True)

    # Comprobacion de que las particiones y el pipeline son los de la referencia congelada.
    if REFERENCE_PATH.exists():
        ref = json.loads(REFERENCE_PATH.read_text())["cv_pareada_10_particiones"]
        stored = np.array(ref["baseline (lr_word + EDA)"]["pr_auc"]["por_particion"])
        now = np.array(S["baseline (lr_word)"][REFERENCE]["pr_auc"])
        print(f"comprobacion baseline: diferencia maxima con la referencia congelada = {np.abs(stored - now).max():.5f}")

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(S, indent=1), encoding="utf-8")

    keys = ("f1_macro_best_thr", "pr_auc", "precision_at_recall70")
    for model_name, cfgs in S.items():
        print(f"\n=== {model_name} ===")
        print(f"{'variante':38s}" + "".join(f"{k:>22s}" for k in keys))
        for c, sc in cfgs.items():
            print(f"{c:38s}" + "".join(f"{np.mean(sc[k]):22.4f}" for k in keys))
        print(f"\nDiferencia pareada frente a '{REFERENCE}' (media, particiones que mejoran /10, p de Wilcoxon):")
        for c, sc in cfgs.items():
            if c == REFERENCE:
                continue
            line = f"{c:38s}"
            for k in keys:
                d = np.array(sc[k]) - np.array(cfgs[REFERENCE][k])
                p = wilcoxon(d).pvalue if np.any(d) else 1.0
                line += f" {k}:{d.mean():+.4f}({int((d > 0).sum())}/10,p={p:.3f})"
            print(line)


if __name__ == "__main__":
    main()
