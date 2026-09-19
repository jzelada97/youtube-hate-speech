"""Lexico toxico. En esta rama se usa SOLO para proteger palabras clave en Easy Data Augmentation.

Las columnas `neutral`/bloqueadores pertenecen al experimento de volteo de etiqueta, descartado
(ver rama `experiment/augmentation-flip` y docs/DECISIONES.md, seccion 5.9).

Original: lexico curado toxico -> neutro para volteo de etiqueta (aumentacion contrafactual).

Borrador redactado a partir de las palabras mas frecuentes/asociadas a toxicidad en el corpus y validado
por una persona (ver docs/DECISIONES.md, seccion 5.9). `neutral=""` significa borrar la palabra;
`neutral=None` significa "no se puede neutralizar de forma fiable": su presencia BLOQUEA el volteo.
Las entradas de confianza "baja" estan apagadas por defecto y se activan a mano tras decidirlo.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LexiconEntry:
    pattern: str
    neutral: str | None
    kind: str
    confidence: str  # "alta" | "media" | "baja"


LEXICON: list[LexiconEntry] = [
    LexiconEntry(r"fucking|fuckin|fukin", "", "obscenidad-intensificador", "alta"),
    LexiconEntry(r"bullshit", "nonsense", "obscenidad", "alta"),
    LexiconEntry(r"shit|crap", "stuff", "obscenidad", "alta"),
    LexiconEntry(r"damn|goddamn", "", "obscenidad-intensificador", "media"),
    LexiconEntry(r"asshole|assholes|jackass|dumbass|dumb ass", "person", "insulto", "alta"),
    LexiconEntry(r"bitch|bitches", "person", "insulto", "alta"),
    LexiconEntry(r"bastard|bastards", "person", "insulto", "alta"),
    LexiconEntry(r"motherfucker|motherfuckers", "person", "insulto", "alta"),
    LexiconEntry(r"cunt|cunts", "person", "insulto", "alta"),
    LexiconEntry(r"prick|pricks|douche|douchebag", "person", "insulto", "alta"),
    LexiconEntry(r"idiot|idiots|moron|morons|imbecile|fool|fools", "person", "insulto", "alta"),
    LexiconEntry(r"stupid|dumb|ignorant|moronic", "mistaken", "adjetivo-insulto", "media"),
    LexiconEntry(r"retard|retarded|retards", "person", "insulto-capacitista", "alta"),
    LexiconEntry(r"scum|lowlife|lowlifes|vermin|filth", "people", "deshumanizante", "media"),
    LexiconEntry(
        r"animal|animals|monkey|monkeys|ape|apes|chimp|chimps|savage|savages|parasite|parasites",
        "people", "deshumanizante", "media",
    ),
    LexiconEntry(r"nigga|niggas|niggaz|nigger|niggers|coon|coons", "man", "slur-racial", "media"),
    LexiconEntry(r"lazy", "hardworking", "atributo-negativo", "alta"),
    LexiconEntry(r"useless|worthless|pathetic", "useful", "atributo-negativo", "media"),
    LexiconEntry(r"disgusting|vile|nasty|filthy|dirty", "pleasant", "atributo-negativo", "media"),
    LexiconEntry(r"commie|commies|libtard|libtards", "person", "insulto-politico", "media"),
    # --- confianza baja: apagadas por defecto, decision pendiente del usuario ---
    LexiconEntry(r"thug|thugs|hoodlum|goon|punk|gangster|ghetto", "man", "insulto-tema-racial", "baja"),
    LexiconEntry(
        r"criminal|criminals|thief|thieves|crook|looter|looters|rioter|rioters",
        "citizen", "atributo-descriptivo", "baja",
    ),
    # --- bloqueadores: sin reemplazo neutro fiable; su presencia impide voltear a limpio ---
    LexiconEntry(r"fuck|fucks|fucked|kill|kills|killed|murder|die|dies|death|dead|shoot|shot", None, "bloqueador", "alta"),
]

# Para voltear a toxico: sustantivos genericos plurales -> insultos plurales.
GENERIC_PLURAL_NOUNS = r"people|guys|folks|dudes"
INSULT_PLURALS = ["idiots", "morons", "assholes"]
