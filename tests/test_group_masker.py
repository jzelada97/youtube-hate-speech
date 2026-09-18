from hatedet.nlp.group_masker import GroupMasker, mask_identity_terms


def test_masks_identity_terms_across_categories():
    out = mask_identity_terms("blacks and whites and muslims and jews")
    assert out.split() == ["grp", "and", "grp", "and", "grp", "and", "grp"]


def test_respects_word_boundaries():
    assert "grp" not in mask_identity_terms("i ate a blackberry with whitewash").split()


def test_does_not_mask_non_identity_words_like_terrorist():
    assert "terrorist" in mask_identity_terms("that terrorist attacked").split()


def test_group_masker_transformer_is_stateless_and_batches():
    out = GroupMasker().fit_transform(["all blacks are", "nothing here"])
    assert "grp" in out[0].split()
    assert out[1] == "nothing here"
