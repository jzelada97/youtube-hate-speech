"""Contrato de columnas del dataset crudo de comentarios de YouTube."""

TEXT_COLUMN = "Text"
VIDEO_ID_COLUMN = "VideoId"
COMMENT_ID_COLUMN = "CommentId"

TARGET_COLUMN = "IsHatespeech"

LABEL_COLUMNS = [
    "IsToxic",
    "IsAbusive",
    "IsThreat",
    "IsProvocative",
    "IsObscene",
    "IsHatespeech",
    "IsRacist",
    "IsNationalist",
    "IsSexist",
    "IsHomophobic",
    "IsReligiousHate",
    "IsRadicalism",
]

REQUIRED_COLUMNS = [COMMENT_ID_COLUMN, VIDEO_ID_COLUMN, TEXT_COLUMN, *LABEL_COLUMNS]


def validate_schema(df) -> None:
    """Lanza ValueError si al DataFrame le faltan columnas requeridas."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el dataset: {missing}")
