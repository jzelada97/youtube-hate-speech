"""Patrones de contexto alrededor de menciones a grupos de identidad. Ejecutar: python scripts/identity_patterns.py

Responde: cuando un comentario nombra un grupo (raza, religion...), que palabras y estructuras lo
rodean en los comentarios de odio frente a los que no lo son? Solo descriptivo; n pequeno, no concluyente.
"""

import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

from hatedet.data.loader import load_raw_comments
from hatedet.nlp.group_masker import IDENTITY_GROUPS, IDENTITY_PATTERN

sys.stdout.reconfigure(encoding="utf-8")

WINDOW = 3
MIN_COUNT = 5
ALL = "|".join(f"(?:{p})" for p in IDENTITY_GROUPS.values())
TOKEN = re.compile(r"[a-z']+")

TEMPLATES = {
    "det/cuantificador + GRUPO (these|those|all|typical|you|them|us)": rf"\b(?:these|those|all|typical|you|them|us)\s+(?:{ALL})\b",
    "GRUPO + copulativo (are|is|were|was)": rf"\b(?:{ALL})\s+(?:are|is|were|was)\b",
    "GRUPO + always/never/only/just/all": rf"\b(?:{ALL})\s+(?:always|never|only|just|all)\b",
    "GRUPO + need/should/must": rf"\b(?:{ALL})\s+(?:need|should|must|have to|ought)\b",
    "GRUPO + 'lives matter'": rf"\b(?:{ALL})\s+lives\s+matter",
    "some/many/most/not all + GRUPO": rf"\b(?:not all|some|many|most)\s+(?:{ALL})\b",
    "GRUPO + verbo violento": rf"\b(?:{ALL})\s+(?:\w+\s+)?(?:kill|kills|murder|rape|loot|riot|steal|rob)",
}


def log_odds(hate: Counter, other: Counter) -> pd.DataFrame:
    P, N = sum(hate.values()) + 1, sum(other.values()) + 1
    rows = [
        (w, hate[w], other[w], np.log(((hate[w] + 1) / P) / ((other[w] + 1) / N)))
        for w in set(hate) | set(other)
        if hate[w] + other[w] >= MIN_COUNT
    ]
    return pd.DataFrame(rows, columns=["palabra", "odio", "no_odio", "log_odds"]).sort_values("log_odds")


def main():
    df = load_raw_comments()
    low = df.Text.str.lower()
    mention = low.str.contains(IDENTITY_PATTERN, regex=True)
    hate = df.IsHatespeech

    print(f"Mencionan un grupo: {mention.sum()} ({mention.mean():.0%})")
    print(f"  tasa de odio con mencion: {hate[mention].mean():.1%} | sin mencion: {hate[~mention].mean():.1%}")
    print(f"  odio SIN mencion: {int((hate & ~mention).sum())} de {int(hate.sum())}")
    print(f"  no-odio CON mencion: {int((~hate & mention).sum())} de {int((~hate).sum())}\n")

    print("Tasa de odio por categoria:")
    for name, rx in IDENTITY_GROUPS.items():
        m = low.str.contains(rf"\b(?:{rx})\b", regex=True)
        print(f"  {name:13s} n={m.sum():3d}  odio={hate[m].mean() if m.sum() else float('nan'):.0%}")

    before = {True: Counter(), False: Counter()}
    after = {True: Counter(), False: Counter()}
    for text, h in zip(low, hate):
        toks = TOKEN.findall(text)
        for i, w in enumerate(toks):
            if IDENTITY_PATTERN.fullmatch(w):
                before[bool(h)].update(toks[max(0, i - WINDOW) : i])
                after[bool(h)].update(toks[i + 1 : i + 1 + WINDOW])

    for label, cnt in [("ANTES", before), ("DESPUES", after)]:
        d = log_odds(cnt[True], cnt[False])
        print(f"\n{label} del grupo (ventana {WINDOW}, min {MIN_COUNT} apariciones) - asociadas a ODIO:")
        print(d.tail(10).iloc[::-1].to_string(index=False))
        print("asociadas a NO odio:")
        print(d.head(6).to_string(index=False))

    print("\nPlantillas (n, %odio):")
    for name, rx in TEMPLATES.items():
        m = low.str.contains(rx, regex=True)
        print(f"  n={m.sum():3d} odio={hate[m].mean() if m.sum() else float('nan'):4.0%}  {name}")


if __name__ == "__main__":
    main()
