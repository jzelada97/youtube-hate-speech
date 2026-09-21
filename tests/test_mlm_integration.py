"""augment_train con variantes de un modelo de lenguaje enmascarado (no necesita transformers: usa un dict)."""

import pandas as pd

from hatedet.data.augment import augment_train
from hatedet.data.schema import TARGET_COLUMN, TEXT_COLUMN

VARIANTS = {
    "the quiet street was lovely": ["the quiet village was lovely", "the quiet street was pleasant"],
    "these people ruin everything here": ["these people ruin everything there"],
    "texto que esta solo en validacion": ["esta variante nunca debe entrar"],
}


def _train():
    return pd.DataFrame({
        TEXT_COLUMN: ["the quiet street was lovely", "these people ruin everything here"],
        TARGET_COLUMN: [False, True],
    })


def test_variants_inherit_the_label_of_the_comment_they_come_from():
    out = augment_train(_train(), mlm_variants=VARIANTS, n_mlm_copies=2)
    mlm = out[out["kind"] == "mlm"]
    assert len(mlm) == 3 and mlm["is_synthetic"].all()
    assert set(mlm[mlm[TEXT_COLUMN].str.contains("ruin")][TARGET_COLUMN]) == {True}
    assert set(mlm[~mlm[TEXT_COLUMN].str.contains("ruin")][TARGET_COLUMN]) == {False}


def test_variants_of_comments_outside_train_are_never_used():
    """Es la garantia contra la filtracion: solo se consultan los textos que estan en el trozo de entrenamiento."""
    out = augment_train(_train(), mlm_variants=VARIANTS, n_mlm_copies=5)
    assert not out[TEXT_COLUMN].str.contains("nunca debe entrar").any()


def test_n_mlm_copies_limits_how_many_variants_per_comment():
    out = augment_train(_train(), mlm_variants=VARIANTS, n_mlm_copies=1)
    assert (out["kind"] == "mlm").sum() == 2


def test_hate_only_adds_variants_of_the_minority_class():
    out = augment_train(_train(), mlm_variants=VARIANTS, n_mlm_copies=2, mlm_classes="hate")
    mlm = out[out["kind"] == "mlm"]
    assert len(mlm) == 1 and bool(mlm[TARGET_COLUMN].iloc[0]) is True


def test_default_behaviour_is_unchanged_without_variants():
    plain = augment_train(_train(), n_eda_copies=1, seed=3)
    with_none = augment_train(_train(), n_eda_copies=1, seed=3, mlm_variants=None, n_mlm_copies=2)
    assert plain.equals(with_none)
    assert "mlm" not in set(plain["kind"])


def test_mlm_and_eda_can_be_combined_and_originals_stay_first_and_untouched():
    out = augment_train(_train(), n_eda_copies=1, mlm_variants=VARIANTS, n_mlm_copies=1, seed=1)
    assert {"real", "eda", "mlm"} <= set(out["kind"])
    assert list(out[out["kind"] == "real"][TEXT_COLUMN]) == list(_train()[TEXT_COLUMN])
