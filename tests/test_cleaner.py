from hatedet.nlp.cleaner import TextCleaner, clean_text


def test_clean_text_lowercases_and_strips_urls_and_mentions():
    result = clean_text("Check THIS out @john http://example.com/x now!!!")
    assert "http" not in result
    assert "@john" not in result
    assert result == result.lower()
    assert "check" in result
    assert "now" in result


def test_clean_text_handles_html_entities_and_tags():
    result = clean_text("<b>Tom &amp; Jerry</b>")
    assert "<" not in result and ">" not in result
    assert "tom" in result and "jerry" in result


def test_clean_text_normalizes_leetspeak_and_repeated_chars():
    result = clean_text("H4TE this soooooo much")
    assert "hate" in result
    assert "soo" in result
    assert "soooooo" not in result


def test_text_cleaner_transformer_is_stateless_and_batches():
    cleaner = TextCleaner()
    out = cleaner.fit_transform(["Hello WORLD", "@user check http://x.com"])
    assert out == ["hello world", "check"]
