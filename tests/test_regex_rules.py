from hatedet.nlp import regex_rules as rr


def test_remove_urls():
    assert rr.remove_urls("check http://x.com/a?b=1 now") == "check   now"
    assert rr.remove_urls("go to www.example.com please") == "go to   please"


def test_remove_mentions():
    assert rr.remove_mentions("hey @john_doe stop") == "hey   stop"


def test_remove_html_tags():
    assert rr.remove_html_tags("<b>bold</b> text") == " bold  text"


def test_unescape_html_entities():
    assert rr.unescape_html_entities("Tom &amp; Jerry &lt;3") == "Tom & Jerry <3"


def test_normalize_leetspeak():
    assert rr.normalize_leetspeak("h4t3 sp33ch") == "hate speech"


def test_collapse_repeated_chars():
    assert rr.collapse_repeated_chars("looooove") == "loove"
    assert rr.collapse_repeated_chars("aa") == "aa"


def test_strip_non_alphanumeric():
    assert rr.strip_non_alphanumeric("it's 100% true!!!") == "it's 100  true   "


def test_collapse_whitespace():
    assert rr.collapse_whitespace("  a   b\t\nc  ") == "a b c"
