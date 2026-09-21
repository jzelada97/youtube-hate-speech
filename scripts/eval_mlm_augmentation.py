"""Evalua la aumentacion con un modelo de lenguaje enmascarado (docs/DECISIONES.md, seccion 6.7).

    python scripts/eval_mlm_augmentation.py [--regen] [--variants 5]

Mismo protocolo que la referencia congelada (reports/pre_bert_reference.json): CV estratificada repetida pareada de
10 particiones (5 folds x 2 repeticiones, random_state=11) sobre las 997 filas, mismas particiones para todo, y la
aumentacion se genera SOLO con el trozo de entrenamiento (el de validacion es siempre real).

Modelos:
  - baseline: lr_word (el de las secciones anteriores).
  - ensemble por defecto: build_ensemble() sin parametros. Es el que salia en la tabla de la CV pareada previa.
  - ensemble servido: build_ensemble(conf/ensemble_params.json), el optimizado con Optuna que cumple el gap < 5 pp.
    ESTE es el que se sirve, y la vara contra la que se decide.

Salidas: reports/mlm_augmentation_comparison.json y, para lectura humana, data/interim/mlm_review_sample.csv y
data/interim/mlm_variants.json (con texto de comentarios reales: fuera de git).
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold

from hatedet.data.augment import augment_train
from hatedet.data.loader import load_raw_comments
from hatedet.data.schema import TARGET_COLUMN, TEXT_COLUMN
from hatedet.models.ensemble import MEMBERS, build_ensemble
from hatedet.models.evaluate import best_macro_f1, precision_at_fixed_recall
from hatedet.models.training import ensemble_params

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

VARIANTS_PATH = Path("data/interim/mlm_variants.json")
SAMPLE_PATH = Path("data/interim/mlm_review_sample.csv")
REPORT_PATH = Path("reports/mlm_augmentation_comparison.json")
REFERENCE_PATH = Path("reports/pre_bert_reference.json")

MODELS = {
    "baseline (lr_word)": MEMBERS["lr_word"],
    "ensemble por defecto": build_ensemble,
    "ensemble servido (params optimizados)": lambda: build_ensemble(ensemble_params()),
}

# nombre -> argumentos de augment_train. EDA de referencia: 1 copia por comentario, ambas clases (lo que se sirve hoy).
CONFIGS = {
    "sin aumentacion": {},
    "EDA x1 ambas (referencia adoptada)": dict(n_eda_copies=1),
    "MLM x1 ambas": dict(n_mlm_copies=1),
    "MLM x3 ambas": dict(n_mlm_copies=3),
    "MLM x3 solo odio": dict(n_mlm_copies=3, mlm_classes="hate"),
    "EDA x1 + MLM x1 ambas": dict(n_eda_copies=1, n_mlm_copies=1),
}
REFERENCE = "EDA x1 ambas (referencia adoptada)"
# La aumentacion se compara sobre el modelo servido y el baseline; el ensemble por defecto solo con EDA (para
# reconciliar con la tabla previa).
RUN = {
    "baseline (lr_word)": list(CONFIGS),
    "ensemble por defecto": [REFERENCE],
    "ensemble servido (params optimizados)": list(CONFIGS),
}


def load_or_generate_variants(texts: list[str], n_variants: int, regen: bool) -> dict[str, list[str]]:
    if VARIANTS_PATH.exists() and not regen:
        cached = json.loads(VARIANTS_PATH.read_text(encoding="utf-8"))
        if all(t in cached for t in texts):
            print(f"variantes cargadas de {VARIANTS_PATH} ({sum(bool(v) for v in cached.values())} comentarios con variantes)")
            return cached
    from hatedet.data.mlm_augment import MLMAugmenter

    print("generando variantes con DistilBERT (una sola vez; se reutilizan en todas las particiones)...", flush=True)
    variants = MLMAugmenter().generate(texts, n_variants=n_variants)
    out = dict(zip(texts, variants))
    VARIANTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    VARIANTS_PATH.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def write_review_sample(df: pd.DataFrame, variants: dict[str, list[str]], n_per_class: int = 30) -> None:
    """Muestra para lectura humana (criterio 4 de la seccion 6.7), estratificada por clase, con la columna vacia."""
    rows = []
    rng = np.random.default_rng(7)
    for label in (True, False):
        pool = [(t, v) for t, y in zip(df[TEXT_COLUMN], df[TARGET_COLUMN]) if bool(y) == label for v in variants[t][:1]]
        for i in rng.choice(len(pool), size=min(n_per_class, len(pool)), replace=False):
            rows.append({"etiqueta_original": "odio" if label else "no_odio", "original": pool[i][0],
                         "variante": pool[i][1], "conserva_sentido_y_etiqueta (si/no)": ""})
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(SAMPLE_PATH, index=False, encoding="utf-8-sig")
    print(f"muestra para lectura humana: {SAMPLE_PATH} ({len(rows)} filas)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regen", action="store_true", help="Regenera las variantes aunque haya cache")
    parser.add_argument("--variants", type=int, default=5, help="Variantes por comentario a generar")
    args = parser.parse_args()

    df = load_raw_comments()
    y = df[TARGET_COLUMN].values.astype(int)
    variants = load_or_generate_variants(df[TEXT_COLUMN].tolist(), args.variants, args.regen)
    write_review_sample(df, variants)
    n_with = sum(bool(v) for v in variants.values())
    print(f"comentarios con al menos una variante: {n_with}/{len(df)}; variantes totales: {sum(map(len, variants.values()))}\n")

    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=11).split(df, y))
    S: dict[str, dict[str, dict[str, list[float]]]] = {m: {c: {"f1_macro_best_thr": [], "pr_auc": [], "precision_at_recall70": []}
                                                            for c in cfgs} for m, cfgs in RUN.items()}
    for i, (tr, te) in enumerate(folds):
        train, val = df.iloc[tr], df.iloc[te]
        yv = val[TARGET_COLUMN]
        for config in CONFIGS:
            if not any(config in cfgs for cfgs in RUN.values()):
                continue
            aug = augment_train(train, seed=i, mlm_variants=variants, **CONFIGS[config])
            data = aug
            for model_name, make in MODELS.items():
                if config not in RUN[model_name]:
                    continue
                prob = make().fit(data[TEXT_COLUMN], data[TARGET_COLUMN].astype(int)).predict_proba(val[TEXT_COLUMN])[:, 1]
                s = S[model_name][config]
                s["f1_macro_best_thr"].append(best_macro_f1(yv, prob))
                s["pr_auc"].append(float(average_precision_score(yv, prob)))
                s["precision_at_recall70"].append(precision_at_fixed_recall(yv, prob, 0.70))
        print(f"particion {i + 1}/{len(folds)} hecha", flush=True)

    # Comprobacion de que las particiones y el pipeline son los de la referencia congelada.
    if REFERENCE_PATH.exists():
        ref = json.loads(REFERENCE_PATH.read_text())["cv_pareada_10_particiones"]
        for model_name, ref_name in [("baseline (lr_word)", "baseline (lr_word + EDA)"),
                                     ("ensemble por defecto", "ensemble (votacion + EDA)")]:
            stored = np.array(ref[ref_name]["pr_auc"]["por_particion"])
            now = np.array(S[model_name][REFERENCE]["pr_auc"])
            print(f"comprobacion {model_name}: diferencia maxima con la referencia congelada = {np.abs(stored - now).max():.5f}")

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(S, indent=1), encoding="utf-8")

    keys = ("f1_macro_best_thr", "pr_auc", "precision_at_recall70")
    for model_name, cfgs in S.items():
        print(f"\n=== {model_name} ===")
        print(f"{'variante':38s}" + "".join(f"{k:>22s}" for k in keys))
        for c, sc in cfgs.items():
            print(f"{c:38s}" + "".join(f"{np.mean(sc[k]):22.4f}" for k in keys))
        if REFERENCE in cfgs and len(cfgs) > 1:
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
