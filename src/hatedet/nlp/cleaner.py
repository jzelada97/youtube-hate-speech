"""Limpieza de texto como transformer de scikit-learn.

Vive dentro del mismo Pipeline que el vectorizador y el modelo (ver decisión de arquitectura:
"sin skew train/serve" en docs/architecture.md) para que el mismo objeto serializado se use en
entrenamiento e inferencia.
"""

from sklearn.base import BaseEstimator, TransformerMixin

from hatedet.nlp.regex_rules import (
    collapse_repeated_chars,
    collapse_whitespace,
    normalize_leetspeak,
    remove_html_tags,
    remove_mentions,
    remove_urls,
    strip_non_alphanumeric,
    unescape_html_entities,
)


def clean_text(text: str) -> str:
    text = remove_html_tags(text)
    text = unescape_html_entities(text)
    text = text.lower()
    text = remove_urls(text)
    text = remove_mentions(text)
    text = normalize_leetspeak(text)
    text = collapse_repeated_chars(text)
    text = strip_non_alphanumeric(text)
    text = collapse_whitespace(text)
    return text


class TextCleaner(BaseEstimator, TransformerMixin):
    """Aplica `clean_text` a una serie/lista de textos. Sin estado -> fit es no-op."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [clean_text(t) for t in X]
