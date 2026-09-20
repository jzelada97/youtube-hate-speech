import random

import pandas as pd
import pytest

from hatedet.data.augment import augment_train
from hatedet.data.easy_augment import (
    easy_augment, is_protected, random_deletion, random_insertion, random_swap,
)


def test_group_and_lexicon_terms_are_protected():
    assert is_protected("blacks") and is_protected("Muslims,") and is_protected("idiots!")
    assert not is_protected("table")


def test_deletion_never_removes_protected_tokens():
    tokens = "all blacks are idiots and criminals today".split()
    for seed in range(20):
        out = random_deletion(tokens, p=0.99, rng=random.Random(seed))
        assert "blacks" in out and "idiots" in out


def test_swap_keeps_protected_tokens_in_place():
    tokens = "the blacks are idiots but nice people here".split()
    for seed in range(20):
        out = random_swap(tokens, n=5, rng=random.Random(seed))
        assert out[1] == "blacks" and out[3] == "idiots"
        assert sorted(out) == sorted(tokens)


def test_insertion_adds_words_without_inserting_protected_ones():
    tokens = "blacks are idiots and wonderful people".split()
    out = random_insertion(tokens, n=3, rng=random.Random(0))
    assert len(out) == len(tokens) + 3
    assert out.count("blacks") == 1 and out.count("idiots") == 1


def test_easy_augment_unknown_operation_raises():
    with pytest.raises(ValueError):
        easy_augment("hello world", "teleport", 0.1, random.Random(0))


def test_augment_train_eda_hate_only_adds_hate_rows_with_same_label():
    train = pd.DataFrame({
        "Text": ["all blacks are idiots", "nice video today", "another fine comment"],
        "IsHatespeech": [True, False, False],
    })
    out = augment_train(train, n_eda_copies=2, eda_classes="hate")
    eda = out[out.kind == "eda"]
    assert len(eda) == 2 and eda.IsHatespeech.all()


def test_augment_train_eda_both_classes_preserves_each_label():
    train = pd.DataFrame({
        "Text": ["all blacks are idiots", "nice video today"],
        "IsHatespeech": [True, False],
    })
    out = augment_train(train, n_eda_copies=1, eda_classes="both")
    eda = out[out.kind == "eda"]
    assert len(eda) == 2 and sorted(eda.IsHatespeech.tolist()) == [False, True]
