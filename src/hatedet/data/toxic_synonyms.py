"""Aumentacion por sinonimos TOXICOS: cambia un termino toxico por otro equivalente (toxico -> toxico).

Conserva la etiqueta por construccion: no se voltea nada, solo se intercambian palabras del mismo tipo, numero y
registro ("idiot" <-> "moron", "thugs" <-> "punks"). Es lo que no cubrieron los experimentos anteriores, que
PROTEGIAN estas palabras (EDA, BERT enmascarado) o dependian de WordNet, que apenas marca 35 palabras ofensivas y no
conoce los insultos que dominan este corpus (ver docs/DECISIONES.md, secciones 6.4 y 6.9).

Objetivo: con ~140 positivos el modelo asocia el odio a palabras concretas ("thugs") y falla ante equivalentes
("hoodlums"). Intercambiar terminos equivalentes reparte ese peso entre ellos.

Solo palabras y frases fijas cortas: la parafrasis de frases enteras exige un modelo de lenguaje (seccion 6.8).
Los grupos son un BORRADOR redactado a partir del vocabulario del corpus (y del lexico de lexicon.py); se
revisa con una persona antes de darlos por buenos.
"""

import random
import re

# grupo -> terminos intercambiables entre si. Un grupo = misma categoria gramatical, numero y gravedad parecida.
GROUPS: dict[str, list[str]] = {
    "insulto_singular": ["idiot", "moron", "imbecile", "fool", "jackass", "dumbass", "asshole", "bastard", "prick",
                         "jerk", "loser"],
    "insulto_plural": ["idiots", "morons", "imbeciles", "fools", "jackasses", "dumbasses", "assholes", "bastards",
                       "pricks", "jerks", "losers"],
    "adjetivo_estupido": ["stupid", "dumb", "idiotic", "moronic", "brainless", "ignorant"],
    "obscenidad": ["shit", "crap", "bullshit"],
    "intensificador": ["fucking", "damn", "goddamn", "freaking", "bloody"],
    # Se excluyen a proposito los primates (monkeys, apes...): en un comentario no racista serian una alusion racial.
    "deshumanizante_plural": ["animals", "savages", "parasites", "beasts", "brutes"],
    "matones_singular": ["thug", "punk", "hoodlum", "goon", "gangster"],
    "matones_plural": ["thugs", "punks", "hoodlums", "goons", "gangsters"],
    "delincuente_singular": ["criminal", "thief", "looter", "rioter", "crook", "robber"],
    "delincuente_plural": ["criminals", "thieves", "looters", "rioters", "crooks", "robbers"],
    "atributo_inutil": ["useless", "worthless", "pathetic"],
    "atributo_asqueroso": ["disgusting", "vile", "nasty", "filthy"],
    "frase_mierda": ["piece of shit", "piece of crap", "piece of garbage", "sack of shit"],
}

_SPLIT = re.compile(r"^(\W*)(.*?)(\W*)$", re.S)
_MEMBER_TO_GROUP: dict[str, str] = {w: g for g, ws in GROUPS.items() for w in ws}
_MAX_WORDS = max(len(w.split()) for ws in GROUPS.values() for w in ws)


def _parts(token: str) -> tuple[str, str, str]:
    return _SPLIT.match(token).groups()


def find_spans(tokens: list[str]) -> list[tuple[int, int, str]]:
    """Tramos (inicio, fin_exclusivo, grupo) con un termino del lexico. Las frases largas tienen prioridad."""
    cores = [_parts(t)[1].lower() for t in tokens]
    spans, i = [], 0
    while i < len(tokens):
        for size in range(min(_MAX_WORDS, len(tokens) - i), 0, -1):
            phrase = " ".join(cores[i:i + size])
            group = _MEMBER_TO_GROUP.get(phrase)
            if group:
                spans.append((i, i + size, group))
                i += size
                break
        else:
            i += 1
    return spans


def _match_case(original: str, new: str) -> str:
    if len(original) > 1 and original.isupper():
        return new.upper()
    if original[:1].isupper():
        return new[:1].upper() + new[1:]
    return new


def _fix_article(tokens: list[str], start: int) -> None:
    """"a idiot" -> "an idiot" cuando el termino nuevo empieza por otra clase de sonido."""
    if start == 0:
        return
    pre, core, suf = _parts(tokens[start - 1])
    if core.lower() in ("a", "an"):
        following = _parts(tokens[start])[1].lower()
        article = "an" if following[:1] in "aeiou" else "a"
        tokens[start - 1] = pre + _match_case(core, article) + suf


def swap_span(tokens: list[str], start: int, end: int, new: str) -> list[str]:
    """Sustituye tokens[start:end] por `new` conservando puntuacion y mayusculas del original."""
    pre = _parts(tokens[start])[0]
    suf = _parts(tokens[end - 1])[2]
    original_core = " ".join(_parts(t)[1] for t in tokens[start:end])
    words = _match_case(original_core, new).split()
    replaced = tokens[:start] + words + tokens[end:]
    replaced[start] = pre + replaced[start]
    replaced[start + len(words) - 1] += suf
    _fix_article(replaced, start)
    return replaced


def toxic_variants(text: str, n: int, rng: random.Random, p_swap: float = 0.7, attempts: int = 12) -> list[str]:
    """Hasta `n` variantes distintas del texto con terminos toxicos cambiados por equivalentes (puede ser 0)."""
    tokens = text.split()
    spans = find_spans(tokens)
    if not spans:
        return []
    out: list[str] = []
    for _ in range(attempts * n):
        if len(out) == n:
            break
        chosen = [s for s in spans if rng.random() < p_swap] or [rng.choice(spans)]
        result = tokens
        # De derecha a izquierda: cambiar el largo de un tramo no descoloca los indices de los anteriores.
        for start, end, group in sorted(chosen, reverse=True):
            current = " ".join(_parts(t)[1] for t in tokens[start:end]).lower()
            options = [w for w in GROUPS[group] if w != current]
            result = swap_span(result, start, end, rng.choice(options))
        variant = " ".join(result)
        if variant != text and variant not in out:
            out.append(variant)
    return out


def corpus_support(texts: list[str]) -> dict[str, int]:
    """Cuantos comentarios contienen cada termino del lexico (para saber cuales tienen apoyo en el vocabulario)."""
    counts = {w: 0 for w in _MEMBER_TO_GROUP}
    for text in texts:
        tokens = text.split()
        seen = {" ".join(_parts(t)[1].lower() for t in tokens[a:b]) for a, b, _ in find_spans(tokens)}
        for w in seen:
            counts[w] += 1
    return counts
