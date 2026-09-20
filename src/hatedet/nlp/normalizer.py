"""Normalización de texto: stopwords + stemming o lematización.

Se implementan ambas técnicas y se dejan como parámetro (`method`) en lugar de aplicar una
combinación fija, porque el objetivo es compararlas por ablación (notebooks/02) y quedarse solo
con la que mejor generalice, no con ambas en producción (ver docs/DECISIONES.md).
"""

from functools import lru_cache

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.base import BaseEstimator, TransformerMixin

_STOPWORDS = set(stopwords.words("english"))
_STEMMER = SnowballStemmer("english")


@lru_cache(maxsize=1)
def _spacy_lemmatizer():
    import spacy

    # Se desactivan parser/NER: solo hace falta tokenización + lematización, y es
    # varias veces más rápido sobre 1000 comentarios cortos.
    return spacy.load("en_core_web_sm", disable=["parser", "ner"])


def _stem_tokens(tokens: list[str]) -> list[str]:
    return [_STEMMER.stem(t) for t in tokens]


def _lemmatize_texts(texts: list[str]) -> list[list[str]]:
    nlp = _spacy_lemmatizer()
    return [[tok.lemma_ for tok in doc] for doc in nlp.pipe(texts)]


class Normalizer(BaseEstimator, TransformerMixin):
    """Quita stopwords y aplica stemming o lematización sobre texto ya limpiado.

    Parameters
    ----------
    method: "stem" | "lemma" | None
        None desactiva stemming/lematización (deja solo el filtrado de stopwords).
    remove_stopwords: bool
    """

    def __init__(self, method: str | None = "stem", remove_stopwords: bool = True):
        self.method = method
        self.remove_stopwords = remove_stopwords

    def fit(self, X, y=None):
        return self

    def transform(self, X: list[str]) -> list[str]:
        texts = list(X)

        if self.method == "lemma":
            token_lists = _lemmatize_texts(texts)
        else:
            token_lists = [t.split() for t in texts]
            if self.method == "stem":
                token_lists = [_stem_tokens(toks) for toks in token_lists]

        if self.remove_stopwords:
            token_lists = [[t for t in toks if t not in _STOPWORDS] for toks in token_lists]

        return [" ".join(toks) for toks in token_lists]
