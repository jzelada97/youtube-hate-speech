"""Pipeline baseline: limpieza + normalizacion + TF-IDF + regresion logistica.

Un unico sklearn.Pipeline serializable de punta a punta (texto crudo -> prediccion), para que el
mismo objeto entrenado sea el que se sirva en inferencia (ver docs/DECISIONES.md, seccion 6.2).
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from hatedet.nlp.cleaner import TextCleaner
from hatedet.nlp.group_masker import GroupMasker
from hatedet.nlp.normalizer import Normalizer


def build_baseline_pipeline(
    normalizer_method: str | None = "stem",
    max_features: int = 5_000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 5,
    C: float = 0.3,
    l1_ratio: float = 1.0,
    mask_groups: bool = True,
) -> Pipeline:
    """Construye el pipeline baseline sin entrenarlo.

    Hiperparametros elegidos por busqueda en validacion cruzada (ver docs/DECISIONES.md, seccion
    10): penalizacion L1 (selecciona features, no solo las encoge) es la que consigue bajar el
    gap train/test de ~20pp a ~2.6pp sin perder F1-macro de validacion. Quedan expuestos como
    parametros para el tuning con Optuna del nivel Medio, no como constantes fijas en el codigo.
    """
    return Pipeline(
        steps=[
            ("cleaner", TextCleaner()),
            ("group_masker", GroupMasker() if mask_groups else "passthrough"),
            ("normalizer", Normalizer(method=normalizer_method)),
            (
                "tfidf",
                TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df),
            ),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    l1_ratio=l1_ratio,
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
