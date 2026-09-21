import random
import re

import pandas as pd
import pytest

from hatedet.data.augment import augment_train
from hatedet.data.schema import TARGET_COLUMN, TEXT_COLUMN
from hatedet.data.toxic_synonyms import (
    GROUPS, corpus_support, find_spans, swap_span, toxic_variants,
)


def variants(text, n=8, seed=1):
    return toxic_variants(text, n, random.Random(seed))


def test_swaps_a_toxic_term_for_an_equivalent_one_and_keeps_the_rest():
    text = "you are a stupid idiot and everyone knows it"
    out = variants(text)
    assert out and all(v != text for v in out)
    for v in out:
        assert v.split()[:3] == ["you", "are", "a"] or v.split()[:3] == ["you", "are", "an"]
        assert v.endswith("and everyone knows it")


def test_keeps_capitalisation_and_surrounding_punctuation():
    out = variants("What an IDIOT! Truly, a Moron.", n=10)
    assert out
    for v in out:
        assert v.endswith("."), v
        assert v.split()[0:2] == ["What", "an"] or v.split()[0:2] == ["What", "a"], v
    # Cuando se cambia el termino en mayusculas, el nuevo tambien va en mayusculas y conserva el "!".
    changed_first = [v for v in out if "IDIOT!" not in v]
    assert changed_first and all(re.search(r"\b[A-Z]{3,}!", v) for v in changed_first)
    # Y el segundo, que iba capitalizado, sigue capitalizado.
    changed_second = [v for v in out if "Moron." not in v]
    assert changed_second and all(re.search(r"\b[A-Z][a-z]+\.$", v) for v in changed_second)


def test_a_an_agrees_with_the_new_word():
    for v in variants("he is an asshole and a moron and an idiot", n=12):
        for article, word in re.findall(r"\b(a|an) (\w+)", v):
            assert article == ("an" if word[0] in "aeiou" else "a"), v


def test_phrases_are_swapped_as_a_whole():
    out = variants("he is such a piece of shit, honestly", n=6)
    assert out
    for v in out:
        assert v.startswith("he is such a piece of ") or v.startswith("he is such a sack of ")
        assert v.endswith(", honestly")
    assert any("garbage" in v or "crap" in v or "sack of" in v for v in out)


def test_singular_and_plural_groups_never_mix():
    plural_only = set(GROUPS["matones_plural"])
    for v in variants("those thugs should be jailed", n=10):
        assert v.split()[1] in plural_only


def test_words_outside_the_lexicon_and_identity_terms_are_never_touched():
    text = "these blacks are lazy and idiots and the mayor said so"
    for v in variants(text, n=10):
        words = v.split()
        assert words[:3] == ["these", "blacks", "are"] and "lazy" in words and words[-4:] == ["the", "mayor", "said", "so"]


def test_text_without_lexicon_terms_returns_nothing():
    assert variants("what a lovely video, thanks for sharing") == []


def test_variants_are_distinct_different_from_the_original_and_at_most_n():
    out = variants("idiots and morons and jerks", n=5)
    assert len(out) == len(set(out)) <= 5 and "idiots and morons and jerks" not in out


def test_same_seed_gives_same_variants():
    assert variants("what an idiot", seed=3) == variants("what an idiot", seed=3)
    assert variants("what an idiot", seed=3) != variants("what an idiot", seed=4)


def test_swap_span_changes_length_correctly_for_phrases():
    tokens = "a piece of shit here".split()
    assert swap_span(tokens, 1, 4, "idiot") == "an idiot here".split()
    assert find_spans(tokens) == [(1, 4, "frase_mierda")]


def test_corpus_support_counts_comments_not_occurrences():
    counts = corpus_support(["idiot idiot idiot", "an idiot", "nothing here"])
    assert counts["idiot"] == 2 and counts["moron"] == 0


# --- integracion con augment_train --------------------------------------------------------------------------------
def _train():
    return pd.DataFrame({
        TEXT_COLUMN: ["these thugs are idiots", "a lovely day at the park", "what a stupid comment"],
        TARGET_COLUMN: [True, False, False],
    })


def test_augment_train_adds_labelled_toxic_synonym_rows_only_where_there_are_terms():
    out = augment_train(_train(), n_toxsyn_copies=2, seed=5)
    syn = out[out["kind"] == "sinonimo_toxico"]
    assert len(syn) > 0 and syn["is_synthetic"].all()
    assert not syn[TEXT_COLUMN].str.contains("lovely").any()
    hate = syn[syn[TEXT_COLUMN].str.contains("thugs|punks|hoodlums|goons|gangsters")]
    assert set(hate[TARGET_COLUMN]) == {True}
    assert set(syn[~syn[TEXT_COLUMN].str.contains("thugs|punks|hoodlums|goons|gangsters|idiots|morons|jerks|fools|"
                                                  "losers|bastards|pricks|assholes|imbeciles|dumbasses|jackasses")][TARGET_COLUMN]) <= {False}


def test_hate_only_restricts_to_the_minority_class():
    out = augment_train(_train(), n_toxsyn_copies=3, toxsyn_classes="hate", seed=5)
    assert set(out[out["kind"] == "sinonimo_toxico"][TARGET_COLUMN]) == {True}


def test_disabled_by_default_and_independent_of_eda_randomness():
    plain = augment_train(_train(), n_eda_copies=1, seed=9)
    assert "sinonimo_toxico" not in set(plain["kind"])
    both = augment_train(_train(), n_eda_copies=1, n_toxsyn_copies=2, seed=9)
    eda_plain = plain[plain["kind"] == "eda"][TEXT_COLUMN].tolist()
    eda_both = both[both["kind"] == "eda"][TEXT_COLUMN].tolist()
    assert eda_plain == eda_both, "activar los sinonimos no debe cambiar lo que genera EDA"
