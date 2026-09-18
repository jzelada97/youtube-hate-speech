"""Entrena el modelo baseline de punta a punta. Ejecutar: python scripts/train_baseline.py

Reproducible con un solo comando (NFR-3): carga datos, split estratificado, entrena el Pipeline
baseline, evalua en train/val/test y serializa el artefacto final en models/.
"""

import json
from pathlib import Path

import joblib

from hatedet.data.loader import load_raw_comments
from hatedet.data.schema import TARGET_COLUMN, TEXT_COLUMN
from hatedet.data.splitter import stratified_split
from hatedet.models.baseline import build_baseline_pipeline
from hatedet.models.evaluate import evaluate, generalization_gap_pp

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "baseline_v2.joblib"
METADATA_PATH = MODELS_DIR / "baseline_v2.metadata.json"


def main():
    df = load_raw_comments()
    train, val, test = stratified_split(df)

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(
        f"Tasa de positivos -> train={train[TARGET_COLUMN].mean():.3f} "
        f"val={val[TARGET_COLUMN].mean():.3f} test={test[TARGET_COLUMN].mean():.3f}"
    )

    pipeline = build_baseline_pipeline()
    pipeline.fit(train[TEXT_COLUMN], train[TARGET_COLUMN])

    reports = {}
    for name, split in [("train", train), ("val", val), ("test", test)]:
        y_true = split[TARGET_COLUMN]
        y_pred = pipeline.predict(split[TEXT_COLUMN])
        y_scores = pipeline.predict_proba(split[TEXT_COLUMN])[:, 1]
        reports[name] = evaluate(y_true, y_pred, y_scores)
        print(f"\n--- {name} ---")
        print(reports[name])

    gap = generalization_gap_pp(reports["train"], reports["test"])
    print(f"\nGap de generalizacion (F1-macro train-test): {gap:.2f} pp (limite cliente: 5 pp)")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    metadata = {
        "model_version": "baseline_v2",
        "target_column": TARGET_COLUMN,
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "f1_macro_train": reports["train"].f1_macro,
        "f1_macro_val": reports["val"].f1_macro,
        "f1_macro_test": reports["test"].f1_macro,
        "generalization_gap_pp": gap,
        "meets_gap_requirement": gap < 5,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"\nModelo guardado en {MODEL_PATH}, metadata en {METADATA_PATH}")


if __name__ == "__main__":
    main()
