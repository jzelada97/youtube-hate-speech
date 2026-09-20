from hatedet.nlp.normalizer import Normalizer


def test_stemming_reduces_related_words_to_same_root():
    norm = Normalizer(method="stem", remove_stopwords=False)
    out = norm.transform(["running runs run"])
    tokens = out[0].split()
    assert len(set(tokens)) == 1


def test_lemmatization_normalizes_plurals_and_verb_forms():
    norm = Normalizer(method="lemma", remove_stopwords=False)
    out = norm.transform(["the dogs were running"])
    assert "dog" in out[0]
    assert "run" in out[0]


def test_stopword_removal():
    norm = Normalizer(method=None, remove_stopwords=True)
    out = norm.transform(["this is the worst video ever"])
    assert "the" not in out[0].split()
    assert "is" not in out[0].split()
    assert "worst" in out[0]


def test_no_stopword_removal_keeps_all_tokens():
    norm = Normalizer(method=None, remove_stopwords=False)
    text = "this is the worst video ever"
    out = norm.transform([text])
    assert out[0] == text
