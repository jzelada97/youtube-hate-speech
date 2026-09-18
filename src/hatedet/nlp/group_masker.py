"""Enmascarado de terminos de identidad (raza, religion, nacionalidad) por un token neutro.

Reemplazar "black", "muslims", "white"... por `grp` hace que el modelo aprenda patrones de
contexto compartidos entre grupos ("all grp are ...", "grp always ...") en vez de asociar una
palabra concreta a odio. Reduce el sesgo de identidad y comparte estadistica entre grupos, algo
valioso con solo ~140 positivos (ver docs/DECISIONES.md, seccion 10.7).

El lexico se fijo a priori por categorias y NO incluye palabras como "terrorist" o "isis", que
no son identidades. Es una lista cerrada e intencionadamente conservadora; ampliarla requiere
volver a validar con validacion cruzada.
"""

import re

from sklearn.base import BaseEstimator, TransformerMixin

MASK_TOKEN = "grp"

IDENTITY_GROUPS: dict[str, str] = {
    "black": r"blacks?|africans?|negro(?:es)?",
    "white": r"whites?|caucasians?|crackers?",
    "arab_muslim": r"arabs?|muslims?|islam\w*",
    "jewish": r"jews?|jewish|zionists?",
    "latino_asian": r"mexicans?|latinos?|hispanics?|asians?|chinese|indians?",
    "christian": r"christians?|catholics?",
}

IDENTITY_PATTERN = re.compile(
    r"\b(?:" + "|".join(f"(?:{p})" for p in IDENTITY_GROUPS.values()) + r")\b"
)


def mask_identity_terms(text: str) -> str:
    """Sustituye cada termino de identidad por el token `grp` (espera texto ya en minusculas)."""
    return IDENTITY_PATTERN.sub(f" {MASK_TOKEN} ", text)


class GroupMasker(BaseEstimator, TransformerMixin):
    """Transformer sin estado que aplica `mask_identity_terms`. Va tras `TextCleaner`."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [mask_identity_terms(t) for t in X]
