"""Generador de variantes con un modelo enmascarado. Usa un BERT diminuto de pesos aleatorios creado en un directorio
temporal: comprueba las REGLAS (que se tapa, que se propone, determinismo), no la calidad linguistica, y no necesita red.
"""

import pytest

pytest.importorskip("torch", reason="necesita torch (pip install -e \".[nn]\")")
pytest.importorskip("transformers", reason="necesita transformers (pip install -e \".[nn]\")")

from transformers import BertConfig, BertForMaskedLM, BertTokenizer  # noqa: E402

from hatedet.data.mlm_augment import POLARITY, MLMAugmenter  # noqa: E402

CONTENT = ("wonderful neighbours friendly helpful quiet street garden village market river bridge morning evening "
           "season concert festival library museum children teachers doctors farmers yesterday together").split()
FUNCTION = "the a of and to in was were is are these those all every not never they them".split()
SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_bert")
    vocab = SPECIAL + sorted(set(CONTENT + FUNCTION + ["blacks", "good", "bad", "idiots"]))
    # transformers 5: el vocabulario se pasa como dict {token: id} (ya no se lee de un vocab_file).
    BertTokenizer(vocab={w: i for i, w in enumerate(vocab)}, do_lower_case=True).save_pretrained(path)
    BertForMaskedLM(BertConfig(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=64,
    )).save_pretrained(path)
    return str(path)


def make(model_dir, **kw):
    # min_prob=0: con pesos aleatorios las probabilidades son casi uniformes y el umbral real las descartaria todas.
    return MLMAugmenter(model_dir, min_prob=0.0, top_k=12, mask_fraction=kw.pop("mask_fraction", 0.3), **kw)


SENTENCE = "the wonderful neighbours cleaned the quiet street yesterday together"


def test_generates_variants_that_differ_from_the_original_and_keep_the_length(model_dir):
    variants = make(model_dir).generate([SENTENCE], n_variants=6)[0]
    assert variants, "deberia generar alguna variante"
    assert all(v != SENTENCE and len(v.split()) == len(SENTENCE.split()) for v in variants)


def test_function_words_negations_and_quantifiers_are_never_touched(model_dir):
    """Son justo la estructura que porta el odio ("all these X are ..."): no se tocan nunca."""
    text = "all these wonderful neighbours are never quiet and they always clean the street"
    # posiciones de: all, these, are, never, and, they, always, the
    positions, keep = [0, 1, 4, 5, 7, 8, 9, 11], ["all", "these", "are", "never", "and", "they", "always", "the"]
    variants = make(model_dir, mask_fraction=0.5).generate([text], n_variants=12)[0]
    assert variants
    for v in variants:
        words = v.split()
        assert [words[i] for i in positions] == keep


def test_protected_words_are_never_masked_or_proposed(model_dir):
    """Terminos de grupo y lexico toxico: los mismos que protege EDA."""
    text = "these blacks and idiots are wonderful neighbours and friendly helpful teachers"
    for v in make(model_dir, mask_fraction=0.5).generate([text], n_variants=12)[0]:
        assert "blacks" in v.split() and "idiots" in v.split()


def test_polarity_words_are_never_masked(model_dir):
    assert {"good", "bad", "kill"} <= POLARITY
    text = "the good neighbours were friendly and the bad weather was quiet yesterday"
    for v in make(model_dir, mask_fraction=0.6).generate([text], n_variants=12)[0]:
        assert "good" in v.split() and "bad" in v.split()


def test_same_seed_gives_same_variants_and_a_different_seed_gives_others(model_dir):
    a = make(model_dir, seed=1).generate([SENTENCE], n_variants=4)
    assert a == make(model_dir, seed=1).generate([SENTENCE], n_variants=4)
    assert a != make(model_dir, seed=2).generate([SENTENCE], n_variants=4)


def test_texts_that_cannot_be_augmented_return_no_variants(model_dir):
    aug = make(model_dir, max_tokens=12)
    too_short = "nice street"
    only_function_words = "these are all they were not the"
    too_long = " ".join(["wonderful neighbours"] * 30)
    assert aug.generate([too_short, only_function_words, too_long], n_variants=3) == [[], [], []]


def test_generation_for_one_text_does_not_depend_on_the_others_in_the_batch(model_dir):
    """Garantia contra la filtracion: precalcular una vez y reutilizar en cualquier particion es valido."""
    aug = make(model_dir)
    # El indice del texto forma parte de la semilla, asi que se compara SIEMPRE en el mismo indice (0).
    alone = aug.generate([SENTENCE], n_variants=3)[0]
    with_a = aug.generate([SENTENCE, "the friendly teachers helped the quiet children at the library"], 3)[0]
    with_b = aug.generate([SENTENCE, "farmers brought wonderful garden vegetables to the village market"], 3)[0]
    assert alone == with_a == with_b
