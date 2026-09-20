"""Easy Data Augmentation (Wei & Zou, 2019): borrado, intercambio e insercion aleatorios de palabras.

Operan sobre tokens separados por espacios y NUNCA tocan palabras protegidas (terminos de grupo y el
lexico toxico), para no eliminar justo lo que hace que un comentario sea de odio. Desviacion respecto al
paper: la insercion duplica una palabra de la propia frase en lugar de insertar un sinonimo, porque los
sinonimos de WordNet no mejoraron y anaden una dependencia (ver docs/DECISIONES.md, seccion 5.9).
"""

import random
import re

from hatedet.data.lexicon import LEXICON
from hatedet.nlp.group_masker import IDENTITY_GROUPS

_PROTECTED = re.compile(
    r"^(?:" + "|".join(f"(?:{p})" for p in [*IDENTITY_GROUPS.values(), *(e.pattern for e in LEXICON)]) + r")$",
    re.I,
)
_PUNCT = re.compile(r"^\W+|\W+$")
OPERATIONS = ("deletion", "swap", "insertion")


def is_protected(token: str) -> bool:
    return bool(_PROTECTED.match(_PUNCT.sub("", token)))


def random_deletion(tokens: list[str], p: float, rng: random.Random) -> list[str]:
    if len(tokens) <= 1:
        return tokens
    kept = [t for t in tokens if is_protected(t) or rng.random() > p]
    return kept or [rng.choice(tokens)]


def random_swap(tokens: list[str], n: int, rng: random.Random) -> list[str]:
    out = tokens[:]
    free = [i for i, t in enumerate(out) if not is_protected(t)]
    for _ in range(n):
        if len(free) < 2:
            break
        i, j = rng.sample(free, 2)
        out[i], out[j] = out[j], out[i]
    return out


def random_insertion(tokens: list[str], n: int, rng: random.Random) -> list[str]:
    out = tokens[:]
    free = [t for t in out if not is_protected(t) and len(_PUNCT.sub("", t)) > 3]
    for _ in range(n):
        if not free:
            break
        out.insert(rng.randrange(len(out) + 1), rng.choice(free))
    return out


def easy_augment(text: str, operation: str, alpha: float, rng: random.Random) -> str:
    tokens = text.split()
    n = max(1, round(alpha * len(tokens)))
    if operation == "deletion":
        tokens = random_deletion(tokens, alpha, rng)
    elif operation == "swap":
        tokens = random_swap(tokens, n, rng)
    elif operation == "insertion":
        tokens = random_insertion(tokens, n, rng)
    else:
        raise ValueError(f"operacion desconocida: {operation}")
    return " ".join(tokens)
