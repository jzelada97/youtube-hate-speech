"""Modelos base y ensembles (Nivel Medio).

Cada miembro es un `Pipeline` completo (texto crudo -> probabilidad) con su propia representacion, de modo
que el ensemble (`VotingClassifier` / `StackingClassifier`) sigue siendo UN solo objeto serializable que recibe
texto crudo: mismo principio de "sin skew train/serve" que el baseline. La diversidad entre miembros
(palabras vs caracteres, lineal vs bayesiano vs arboles) es lo que hace que sus errores se compensen.
"""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from hatedet.nlp.cleaner import TextCleaner
from hatedet.nlp.group_masker import GroupMasker
from hatedet.nlp.normalizer import Normalizer


def _prep():
    return [("cleaner", TextCleaner()), ("group_masker", GroupMasker())]


def _word_tfidf(min_df=5, ngram_range=(1, 2)):
    return [
        ("normalizer", Normalizer(method="stem")),
        ("tfidf", TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_features=5000)),
    ]


def _char_tfidf(min_df=3, max_features=20000):
    return [
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=min_df,
                                  sublinear_tf=True, max_features=max_features)),
    ]


def lr_word(C: float = 0.3, min_df: int = 5, ngram_max: int = 2) -> Pipeline:
    """El baseline: TF-IDF de palabras + regresion logistica L1."""
    clf = LogisticRegression(C=C, l1_ratio=1.0, solver="liblinear", class_weight="balanced",
                             max_iter=2000, random_state=42)
    return Pipeline(_prep() + _word_tfidf(min_df, (1, ngram_max)) + [("clf", clf)])


def lr_char(C: float = 1.0, min_df: int = 3, max_features: int = 20000) -> Pipeline:
    """TF-IDF de n-gramas de caracteres (robusto a leetspeak/erratas) + regresion logistica L2."""
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=42)
    return Pipeline(_prep() + _char_tfidf(min_df, max_features) + [("clf", clf)])


def nb_word(alpha: float = 0.5, min_df: int = 2) -> Pipeline:
    """Naive Bayes complementario (pensado para texto desbalanceado)."""
    return Pipeline(_prep() + _word_tfidf(min_df=min_df) + [("clf", ComplementNB(alpha=alpha))])


def svm_char(C: float = 0.1) -> Pipeline:
    """SVM lineal sobre caracteres, calibrado para que devuelva probabilidades."""
    svm = LinearSVC(C=C, class_weight="balanced", random_state=42)
    return Pipeline(_prep() + _char_tfidf() + [("clf", CalibratedClassifierCV(svm, cv=3))])


def et_word(n_estimators: int = 300) -> Pipeline:
    """Arboles aleatorios extremos: un tipo de modelo muy distinto a los lineales."""
    clf = ExtraTreesClassifier(n_estimators=n_estimators, min_samples_leaf=2,
                               class_weight="balanced_subsample", random_state=42, n_jobs=1)
    return Pipeline(_prep() + _word_tfidf() + [("clf", clf)])


MEMBERS = {"lr_word": lr_word, "lr_char": lr_char, "nb_word": nb_word, "svm_char": svm_char, "et_word": et_word}


def build_voting(members=("lr_word", "lr_char", "nb_word"), weights=None) -> VotingClassifier:
    """Votacion 'soft': promedia las probabilidades de los miembros."""
    return VotingClassifier([(m, MEMBERS[m]()) for m in members], voting="soft", weights=weights)


def build_stacking(members=("lr_word", "lr_char", "nb_word", "svm_char")) -> StackingClassifier:
    """Stacking: un meta-modelo (regresion logistica) aprende a combinar las probabilidades de los miembros."""
    return StackingClassifier(
        [(m, MEMBERS[m]()) for m in members],
        final_estimator=LogisticRegression(class_weight="balanced", max_iter=1000),
        stack_method="predict_proba", cv=5,
    )


DEFAULT_ENSEMBLE_PARAMS = {
    "lr_word_C": 0.3, "lr_word_min_df": 5, "lr_word_ngram_max": 2,
    "lr_char_C": 1.0, "lr_char_min_df": 3, "lr_char_max_features": 20000,
    "nb_alpha": 0.5, "nb_min_df": 2,
    "w_lr_word": 2, "w_lr_char": 1, "w_nb": 1,
}


def build_ensemble(params: dict | None = None) -> VotingClassifier:
    """Ensemble del Nivel Medio: votacion 'soft' ponderada de regresion logistica (palabras), regresion logistica
    (caracteres) y Naive Bayes complementario. `params` sobreescribe DEFAULT_ENSEMBLE_PARAMS (lo usa Optuna)."""
    p = {**DEFAULT_ENSEMBLE_PARAMS, **(params or {})}
    return VotingClassifier(
        [
            ("lr_word", lr_word(p["lr_word_C"], p["lr_word_min_df"], p["lr_word_ngram_max"])),
            ("lr_char", lr_char(p["lr_char_C"], p["lr_char_min_df"], p["lr_char_max_features"])),
            ("nb_word", nb_word(p["nb_alpha"], p["nb_min_df"])),
        ],
        voting="soft", weights=[p["w_lr_word"], p["w_lr_char"], p["w_nb"]],
    )
