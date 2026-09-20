"""Entrena el modelo servible de punta a punta. Ejecutar: python scripts/train.py [--model baseline|ensemble]

Reproducible con un solo comando (NFR-3):
  1. Split estratificado train/val/test (semilla fija). El TEST no se toca hasta el final.
  2. CV de 5 folds sobre train+val, con augmentation SOLO en el fold de entrenamiento, para obtener
     probabilidades fuera de muestra -> umbrales de las bandas de decision y estimacion honesta.
  3. Modelo final entrenado con train+val (con augmentation) y evaluado UNA vez en test.
  4. Serializa el Pipeline completo + metadata (umbrales, metricas) en models/.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from hatedet.data.augment import augment_train
from hatedet.data.loader import load_raw_comments
from hatedet.data.schema import TARGET_COLUMN, TEXT_COLUMN
from hatedet.data.splitter import stratified_split
from hatedet.models.baseline import build_baseline_pipeline
from hatedet.models.ensemble import build_ensemble
from hatedet.models.bands import choose_thresholds
from hatedet.models.evaluate import evaluate, generalization_gap_pp

MODELS_DIR = Path("models")
PARAMS_PATH = Path("conf/ensemble_params.json")
AUGMENTATION = dict(n_eda_copies=1, eda_classes="both")


def ensemble_params():
    return json.loads(PARAMS_PATH.read_text()) if PARAMS_PATH.exists() else None


MODELS = {
    "baseline": ("baseline_v3", build_baseline_pipeline),
    "ensemble": ("ensemble_v1", lambda: build_ensemble(ensemble_params())),
}


def fit_with_augmentation(train: pd.DataFrame, seed: int, factory):
    aug = augment_train(train, seed=seed, **AUGMENTATION)
    return factory().fit(aug[TEXT_COLUMN], aug[TARGET_COLUMN].astype(int))


def out_of_fold_scores(df: pd.DataFrame, factory) -> np.ndarray:
    y = df[TARGET_COLUMN].values.astype(int)
    oof = np.zeros(len(df))
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(df, y)):
        model = fit_with_augmentation(df.iloc[tr], seed=fold, factory=factory)
        oof[te] = model.predict_proba(df.iloc[te][TEXT_COLUMN])[:, 1]
    return oof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="ensemble")
    parser_kind = parser.parse_args().model
    version, factory = MODELS[parser_kind]
    model_path = MODELS_DIR / f"{version}.joblib"
    metadata_path = MODELS_DIR / f"{version}.metadata.json"
    print(f"Entrenando {version}")
    df = load_raw_comments()
    train, val, test = stratified_split(df)
    dev = pd.concat([train, val], ignore_index=True)
    print(f"Desarrollo (train+val): {len(dev)} | Test (intacto): {len(test)}")

    oof = out_of_fold_scores(dev, factory)
    y_dev = dev[TARGET_COLUMN].values.astype(int)
    cv_report = evaluate(y_dev, (oof >= 0.5).astype(int), oof)
    thresholds = choose_thresholds(y_dev, oof)
    print("\n--- CV fuera de muestra (estimacion honesta, umbral 0.5) ---")
    print(cv_report)
    print(f"\nUmbrales de bandas: {thresholds}")

    model = fit_with_augmentation(dev, seed=42, factory=factory)
    reports = {}
    for name, split in [("dev (train+val, en muestra)", dev), ("test", test)]:
        scores = model.predict_proba(split[TEXT_COLUMN])[:, 1]
        reports[name] = evaluate(split[TARGET_COLUMN], (scores >= 0.5), scores)
        print(f"\n--- {name} ---\n{reports[name]}")

    gap = generalization_gap_pp(reports["dev (train+val, en muestra)"], reports["test"])
    print(f"\nGap F1-macro en muestra - test: {gap:.2f} pp (limite cliente: 5 pp)")
    gap_cv = generalization_gap_pp(reports["dev (train+val, en muestra)"], cv_report)
    print(f"Gap F1-macro en muestra - CV:   {gap_cv:.2f} pp")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    test_scores = model.predict_proba(test[TEXT_COLUMN])[:, 1]
    metadata = {
        "model_version": version,
        "model_kind": parser_kind,
        "target_column": TARGET_COLUMN,
        "augmentation": AUGMENTATION,
        "n_dev": len(dev),
        "n_test": len(test),
        "thresholds": thresholds,
        "f1_macro_cv": cv_report.f1_macro,
        "f1_hate_cv": cv_report.f1_minority,
        "pr_auc_cv": cv_report.pr_auc,
        "f1_macro_test": reports["test"].f1_macro,
        "f1_hate_test": reports["test"].f1_minority,
        "pr_auc_test": reports["test"].pr_auc,
        "generalization_gap_pp_in_sample_vs_test": gap,
        "generalization_gap_pp_in_sample_vs_cv": gap_cv,
        "meets_gap_requirement": bool(gap < 5 and gap_cv < 5),
        "test_flagged_fraction_review_or_hide": float((test_scores >= thresholds["t_low"]).mean()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"\nModelo guardado en {model_path}, metadata en {metadata_path}")


if __name__ == "__main__":
    main()
