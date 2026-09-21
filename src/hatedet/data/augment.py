"""Data augmentation de texto para el conjunto de ENTRENAMIENTO (nunca para val/test).

Dos tecnicas (ver docs/DECISIONES.md, seccion 5.9), las unicas con mejora medida en CV pareada:
  1. Negativos dificiles por plantilla: nombran un grupo de forma explicita pero neutra/positiva.
  2. Easy Data Augmentation (borrado / intercambio / insercion con palabras protegidas).

Se descartaron el reemplazo por sinonimos (WordNet), los pares contrastivos y el volteo lexico
(efecto ~0 o etiquetas dudosas); el codigo completo esta en la rama `experiment/augmentation-flip`.
Se aplican DESPUES de partir los datos y solo al train, para no contaminar la evaluacion.
"""

import random

import pandas as pd

from hatedet.data.easy_augment import OPERATIONS, easy_augment
from hatedet.data.schema import TARGET_COLUMN, TEXT_COLUMN
from hatedet.data.toxic_synonyms import toxic_variants

_GROUP_FORMS = [
    ("Black people", "Black"), ("white people", "white"), ("Muslims", "Muslim"),
    ("Jewish people", "Jewish"), ("Asian people", "Asian"), ("Latino people", "Latino"),
]

HARD_NEGATIVE_FRAMES = [
    "Some {g} are doctors and teachers in my town.",
    "{g} lives matter, and so does everyone else's.",
    "I read a study about {g} communities and their history.",
    "Many {g} joined the peaceful march yesterday.",
    "I have {g} friends and they are great people.",
    "The documentary explains the culture of {g} very well.",
    "My neighbours are {g} and they helped us move house.",
    "A local charity run by {g} is collecting food for families.",
    "The mayor met with leaders of the {g} community today.",
    "Great video, thanks for showing how {g} celebrate this festival.",
    "There are {g} on both sides of this debate, so let's listen to everyone.",
    "As one of the {g} in this city I feel welcome here.",
    "The school invited {g} parents to talk about their traditions.",
    "It is unfair to judge all {g} by the actions of one person.",
    "Statistics show that {g} are affected by this policy as well.",
    "The book tells the story of a {a} family during the war.",
    "Why do people say bad things about {g}? They are just people like us.",
    "This song was written by a {a} artist and I love it.",
    "The article discusses how {g} are represented in the media.",
    "Respect to {g} who work hard every day for their families.",
    "I disagree with the video, but that has nothing to do with {g}.",
    "Both {g} and others took part in the clean-up after the storm.",
    "A {a} nurse saved my father's life last year.",
    "The museum has a new exhibition on {g} history.",
    "Ask any of the {g} who live here and they will tell you it is a good street.",
    "The coach picked players from all backgrounds, including {g}.",
    "Let's not start a fight about {g}; the video is about something else.",
    "Some {g} agree with this decision and some do not.",
    "The council is building a community centre for {g} and everyone else.",
    "Honestly the comments about {g} here make me sad, we should be better than this.",
    "The film shows {g} and other neighbours working together.",
    "Thank you to the {g} volunteers who helped at the hospital.",
    "This channel has many {g} viewers and they are always polite.",
    "I asked some {g} what they think and opinions were mixed.",
    "Proud of my {g} classmates for winning the science prize.",
    "The judge treated {g} and everyone else exactly the same.",
    "Not all {g} think alike, just like not all of us do.",
    "Interesting talk on how {g} came to live in this region.",
]


def hard_negatives(n: int, rng: random.Random) -> list[str]:
    frames = HARD_NEGATIVE_FRAMES[:]
    rng.shuffle(frames)
    out = []
    for i in range(n):
        plural, adj = rng.choice(_GROUP_FORMS)
        out.append(frames[i % len(frames)].format(g=plural, a=adj))
    return out


def augment_train(
    train: pd.DataFrame,
    n_hard_negatives: int = 0,
    n_eda_copies: int = 0,
    eda_alpha: float = 0.1,
    eda_classes: str = "both",
    seed: int = 42,
    mlm_variants: dict[str, list[str]] | None = None,
    n_mlm_copies: int = 0,
    mlm_classes: str = "both",
    n_toxsyn_copies: int = 0,
    toxsyn_classes: str = "both",
) -> pd.DataFrame:
    """Devuelve `train` + filas sinteticas (columnas `is_synthetic`, `kind`). Los originales no se tocan.

    n_eda_copies: copias EDA por comentario (una operacion por copia, en rotacion) de `eda_classes`
      ("both" o "hate"); la etiqueta de cada copia es la del original.
    n_hard_negatives: frases que mencionan un grupo sin ser odio (etiqueta False).
    mlm_variants / n_mlm_copies / mlm_classes: variantes generadas con un modelo de lenguaje enmascarado
      (data/mlm_augment.py, docs seccion 6.7), precalculadas como {texto original: [variantes]}. Se usan las
      primeras `n_mlm_copies` de cada comentario de `mlm_classes` que este en `train`, con su misma etiqueta. Solo
      pueden entrar variantes de comentarios de `train`: las de validacion no se consultan nunca.
    n_toxsyn_copies / toxsyn_classes: hasta `n_toxsyn_copies` variantes por comentario cambiando terminos toxicos por
      equivalentes (data/toxic_synonyms.py, docs seccion 6.9); toxico -> toxico, con la etiqueta del original. Solo
      genera para los comentarios que contienen algun termino del lexico. Usa su propia semilla: activarlo o no no
      altera el aleatorio de EDA.
    """
    rng = random.Random(seed)
    base = train[[TEXT_COLUMN, TARGET_COLUMN]].assign(is_synthetic=False, kind="real")
    rows = []
    if n_toxsyn_copies:
        tox_rng = random.Random(f"{seed}-toxsyn")
        src = train if toxsyn_classes == "both" else train[train[TARGET_COLUMN].astype(bool)]
        rows += [
            {TEXT_COLUMN: variant, TARGET_COLUMN: bool(y), "is_synthetic": True, "kind": "sinonimo_toxico"}
            for t, y in zip(src[TEXT_COLUMN], src[TARGET_COLUMN])
            for variant in toxic_variants(t, n_toxsyn_copies, tox_rng)
        ]
    if n_mlm_copies and mlm_variants:
        src = train if mlm_classes == "both" else train[train[TARGET_COLUMN].astype(bool)]
        rows += [
            {TEXT_COLUMN: variant, TARGET_COLUMN: bool(y), "is_synthetic": True, "kind": "mlm"}
            for t, y in zip(src[TEXT_COLUMN], src[TARGET_COLUMN]) for variant in mlm_variants.get(t, [])[:n_mlm_copies]
        ]
    if n_eda_copies:
        src = train if eda_classes == "both" else train[train[TARGET_COLUMN].astype(bool)]
        rows += [
            {TEXT_COLUMN: easy_augment(t, OPERATIONS[i % len(OPERATIONS)], eda_alpha, rng),
             TARGET_COLUMN: bool(y), "is_synthetic": True, "kind": "eda"}
            for t, y in zip(src[TEXT_COLUMN], src[TARGET_COLUMN]) for i in range(n_eda_copies)
        ]
    rows += [{TEXT_COLUMN: t, TARGET_COLUMN: False, "is_synthetic": True, "kind": "negativo_dificil"}
             for t in hard_negatives(n_hard_negatives, rng)]
    return pd.concat([base, pd.DataFrame(rows)], ignore_index=True) if rows else base
