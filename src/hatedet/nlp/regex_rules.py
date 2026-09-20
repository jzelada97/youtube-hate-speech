"""Expresiones regulares de limpieza de comentarios.

Cada regex es una función pura (str -> str) para poder testear y componer por separado.
"""

import html
import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_REPEATED_CHARS_RE = re.compile(r"(.)\1{2,}")
_NON_ALPHANUM_RE = re.compile(r"[^a-z0-9\s']")
_MULTI_SPACE_RE = re.compile(r"\s+")

# Sustituciones leetspeak habituales en comentarios de odio para evadir filtros de palabras.
_LEETSPEAK_MAP = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "@": "a",
        "$": "s",
    }
)


def remove_html_tags(text: str) -> str:
    return _HTML_TAG_RE.sub(" ", text)


def remove_urls(text: str) -> str:
    return _URL_RE.sub(" ", text)


def remove_mentions(text: str) -> str:
    return _MENTION_RE.sub(" ", text)


def unescape_html_entities(text: str) -> str:
    return html.unescape(text)


def normalize_leetspeak(text: str) -> str:
    return text.translate(_LEETSPEAK_MAP)


def collapse_repeated_chars(text: str) -> str:
    """'looooove' -> 'loove' (conserva la duplicación como señal de énfasis, sin ruido extremo)."""
    return _REPEATED_CHARS_RE.sub(r"\1\1", text)


def strip_non_alphanumeric(text: str) -> str:
    return _NON_ALPHANUM_RE.sub(" ", text)


def collapse_whitespace(text: str) -> str:
    return _MULTI_SPACE_RE.sub(" ", text).strip()
