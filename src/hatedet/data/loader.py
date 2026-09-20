"""Ingesta del dataset crudo de comentarios de YouTube."""

from pathlib import Path

import pandas as pd

from hatedet.data.schema import REQUIRED_COLUMNS, TEXT_COLUMN, validate_schema

DEFAULT_RAW_PATH = Path("data/raw/youtoxic_english_1000.csv")


def load_raw_comments(path: Path = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """Carga el CSV crudo, valida el esquema y elimina duplicados exactos de texto.

    Se deduplica aquí (no en el split) porque un duplicado exacto en train y test
    inflaría artificialmente las métricas de test.
    """
    df = pd.read_csv(path)
    validate_schema(df)
    df = df.drop_duplicates(subset=[TEXT_COLUMN]).reset_index(drop=True)
    return df[REQUIRED_COLUMNS]
