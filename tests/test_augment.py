import random

import pandas as pd

from hatedet.data.augment import augment_train, hard_negatives


def _train():
    return pd.DataFrame({
        "Text": ["all blacks are idiots", "nice video", "another fine comment", "all good here"],
        "IsHatespeech": [True, False, False, False],
    })


def test_no_augmentation_returns_only_real_rows():
    out = augment_train(_train())
    assert len(out) == 4 and not out.is_synthetic.any()


def test_original_rows_are_never_modified():
    train = _train()
    out = augment_train(train, n_hard_negatives=5, n_eda_copies=2)
    real = out[~out.is_synthetic]
    assert real.Text.tolist() == train.Text.tolist()
    assert real.IsHatespeech.tolist() == train.IsHatespeech.tolist()


def test_hard_negatives_are_not_hate_and_have_requested_count():
    assert len(hard_negatives(10, random.Random(0))) == 10
    out = augment_train(_train(), n_hard_negatives=10)
    neg = out[out.kind == "negativo_dificil"]
    assert len(neg) == 10 and not neg.IsHatespeech.any()


def test_eda_default_augments_both_classes_keeping_labels():
    out = augment_train(_train(), n_eda_copies=1)
    eda = out[out.kind == "eda"]
    assert len(eda) == 4 and eda.IsHatespeech.sum() == 1


def test_eda_hate_only_option():
    out = augment_train(_train(), n_eda_copies=2, eda_classes="hate")
    eda = out[out.kind == "eda"]
    assert len(eda) == 2 and eda.IsHatespeech.all()


def test_deterministic_for_same_seed():
    a = augment_train(_train(), n_hard_negatives=5, n_eda_copies=2, seed=3)
    b = augment_train(_train(), n_hard_negatives=5, n_eda_copies=2, seed=3)
    assert a.Text.tolist() == b.Text.tolist()
