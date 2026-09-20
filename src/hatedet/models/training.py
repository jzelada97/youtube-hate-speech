"""Piezas de entrenamiento compartidas por scripts/train.py y scripts/retrain.py.

Viven en el paquete (y no en scripts/) porque las necesitan dos entradas distintas y `scripts/` no es
un paquete importable: un script no debe importar de otro script.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from hatedet.data.augment import augment_train
from hatedet.data.schema import TARGET_COLUMN, TEXT_COLUMN
from hatedet.models.baseline import build_baseline_pipeline
from hatedet.models.ensemble import build_ensemble

PARAMS_PATH = Path("conf/ensemble_params.json")
AUGMENTATION = dict(n_eda_copies=1, eda_classes="both")


def ensemble_params():
    return json.loads(PARAMS_PATH.read_text()) if PARAMS_PATH.exists() else None


def build_lstm():
    """Importacion perezosa: torch es opcional (extra `nn`) y el modelo servido no lo necesita."""
    from hatedet.models.glove import glove_initializer
    from hatedet.models.lstm import LSTMClassifier

    return LSTMClassifier(cell="lstm", emb_dim=300, hidden=48, pretrained=glove_initializer, freeze_embeddings=True)


# (version, fabrica, aumentacion). La red recurrente NO usa EDA: barajar o borrar palabras destruye el orden.
MODELS = {
    "baseline": ("baseline_v3", build_baseline_pipeline, AUGMENTATION),
    "ensemble": ("ensemble_v1", lambda: build_ensemble(ensemble_params()), AUGMENTATION),
    "lstm": ("lstm_glove_v1", build_lstm, {}),
}


def fit_with_augmentation(train: pd.DataFrame, seed: int, factory, augmentation=AUGMENTATION):
    """Aumenta SOLO el trozo de entrenamiento y ajusta el modelo."""
    aug = augment_train(train, seed=seed, **augmentation)
    return factory().fit(aug[TEXT_COLUMN], aug[TARGET_COLUMN].astype(int))


def out_of_fold_scores(df: pd.DataFrame, factory, augmentation=AUGMENTATION) -> np.ndarray:
    """Probabilidades fuera de muestra: base de los umbrales de banda y de la estimacion honesta."""
    y = df[TARGET_COLUMN].values.astype(int)
    oof = np.zeros(len(df))
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(df, y)):
        model = fit_with_augmentation(df.iloc[tr], seed=fold, factory=factory, augmentation=augmentation)
        oof[te] = model.predict_proba(df.iloc[te][TEXT_COLUMN])[:, 1]
    return oof
