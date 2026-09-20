"""Splits de train/val/test.

Decisión clave: el split principal es estratificado por comentario (no por vídeo), porque con
solo 13 vídeos un split agrupado por VideoId no permite estratificar bien la clase minoritaria
(IsHatespeech, ~14%) y reduciría aún más un dataset ya pequeño. Como contrapartida se acepta un
riesgo de fuga leve (comentarios del mismo vídeo/hilo pueden compartir vocabulario en train y test).

Para cuantificar ese riesgo se ofrece `video_holdout_split`, un split alternativo que deja vídeos
completos fuera de entrenamiento, usado solo como chequeo de robustez en el informe de evaluación,
no como split de referencia para seleccionar hiperparámetros.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from hatedet.data.schema import TARGET_COLUMN, VIDEO_ID_COLUMN

SEED = 42


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split estratificado por TARGET_COLUMN en train/val/test."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET_COLUMN], random_state=seed
    )
    relative_val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        stratify=train_val[TARGET_COLUMN],
        random_state=seed,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def video_holdout_split(
    df: pd.DataFrame, n_test_videos: int = 3, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split de robustez: deja `n_test_videos` vídeos completos fuera de entrenamiento.

    Solo para diagnóstico de fuga de información, no para tuning de hiperparámetros.
    """
    rng = np.random.default_rng(seed)
    video_ids = df[VIDEO_ID_COLUMN].unique()
    test_videos = rng.choice(video_ids, size=n_test_videos, replace=False)
    test_mask = df[VIDEO_ID_COLUMN].isin(test_videos)
    return df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)
