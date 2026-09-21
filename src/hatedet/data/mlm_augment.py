"""Aumentacion con un modelo de lenguaje enmascarado (BERT): generar variaciones de comentarios ya etiquetados.

El modelo preentrenado solo TRANSFORMA texto (rellena palabras tapadas segun el contexto); NUNCA decide etiquetas:
cada variante hereda la etiqueta del comentario del que sale. El detector que se sirve sigue siendo el clasico.
Ver docs/DECISIONES.md, secciones 6.1 y 6.7.

Como se cuida que la variante conserve el sentido y la etiqueta:
- Solo se tapan palabras de CONTENIDO: nunca palabras vacias (que incluyen negaciones y cuantificadores como
  "all", "never", "these", justo la estructura que porta el odio, seccion 3.4), nunca palabras protegidas (terminos
  de grupo y lexico toxico: las mismas que protege EDA) y nunca un conjunto de palabras de polaridad fuerte.
- Las candidatas se filtran igual: ni palabras vacias, ni protegidas, ni de polaridad. Asi el modelo no puede
  meter un insulto en un comentario limpio ni cambiar "good" por "bad".
- La generacion de cada comentario depende solo de ese comentario (el modelo esta fijo), asi que se puede
  precalcular una vez y reutilizar en cualquier particion sin filtrar informacion de validacion.
"""

import random
import re
from collections.abc import Sequence

from hatedet.data.easy_augment import is_protected

_CORE = re.compile(r"^(\W*)(.*?)(\W*)$", re.S)
_ALPHA = re.compile(r"^[a-z]{3,}$")

# Polaridad y sentido fuertes: sustituir estas palabras suele cambiar el significado (antonimos: "good" -> "bad").
# Lista deliberadamente parcial: la ultima palabra la tiene la CV pareada y la lectura humana de una muestra.
POLARITY = frozenset("""
good bad great terrible awful horrible best worst better worse nice evil fine happy sad angry love hate like dislike
right wrong true false agree disagree support oppose safe dangerous smart dumb lazy clean dirty poor rich
kill killed killing die died dying death dead murder shot shoot violent violence peaceful peace war crime criminal
criminals racist racism
""".split())


# Negaciones y cuantificadores que la lista de palabras vacias de NLTK NO incluye ("never", "always", "every"...)
# y que son justo la estructura que porta el odio ("all these X are always ..."): se conservan siempre.
STRUCTURE = frozenset("""
never always every everyone everybody everything none nobody nothing neither cannot without nowhere ever
whole entire any anyone anybody many most few typical
""".split())


def _stopwords() -> frozenset[str]:
    from nltk.corpus import stopwords

    return frozenset(stopwords.words("english")) | STRUCTURE


class MLMAugmenter:
    """Genera variantes de un texto sustituyendo palabras de contenido por las que propone un modelo enmascarado.

    `model_name` puede ser un id de Hugging Face (p. ej. "distilbert-base-uncased", se descarga una vez) o una ruta
    local. Corre en CPU y no envia texto a ningun servicio.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        mask_fraction: float = 0.15,
        top_k: int = 15,
        min_prob: float = 0.05,
        max_tokens: int = 128,
        batch_size: int = 16,
        seed: int = 42,
    ):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).eval()
        self.mask_fraction, self.top_k, self.min_prob = mask_fraction, top_k, min_prob
        self.max_tokens, self.batch_size, self.seed = max_tokens, batch_size, seed
        self._stop = _stopwords()

    # -- reglas de que se puede tapar / proponer -------------------------------------------------------------
    def _can_mask(self, word: str) -> bool:
        core = _CORE.match(word).group(2).lower()
        return bool(_ALPHA.match(core)) and core not in self._stop and core not in POLARITY and not is_protected(word)

    def _acceptable(self, candidate: str | None, original_core: str) -> bool:
        if candidate is None:  # id sin palabra asociada en el tokenizador: nunca es una candidata valida
            return False
        return bool(
            _ALPHA.match(candidate) and candidate != original_core and candidate not in self._stop
            and candidate not in POLARITY and not is_protected(candidate)
        )

    def _fits(self, text: str) -> bool:
        return len(self.tokenizer(text, add_special_tokens=True)["input_ids"]) <= self.max_tokens

    # -- generacion -------------------------------------------------------------------------------------------
    def _plan(self, text: str, variant: int, index: int) -> tuple[list[str], list[int], list[str]] | None:
        """Palabras con [MASK] puesto, posiciones tapadas y palabra original de cada hueco (en minusculas).

        None si el texto no se puede aumentar (muy corto, sin palabras de contenido o mas largo que `max_tokens`).
        """
        words = text.split()
        candidates = [i for i, w in enumerate(words) if self._can_mask(w)]
        if len(words) < 4 or not candidates or not self._fits(text):
            return None
        rng = random.Random(f"{self.seed}-{index}-{variant}")
        k = min(len(candidates), max(1, round(self.mask_fraction * len(words))))
        chosen = sorted(rng.sample(candidates, k))
        masked, originals = words[:], []
        for i in chosen:
            pre, core, suf = _CORE.match(words[i]).groups()
            masked[i] = f"{pre}{self.tokenizer.mask_token}{suf}"
            originals.append(core.lower())
        return masked, chosen, originals

    def generate(self, texts: Sequence[str], n_variants: int = 3) -> list[list[str]]:
        """Para cada texto, hasta `n_variants` variantes distintas del original (puede ser menos, o ninguna)."""
        jobs = []  # (indice_texto, variante, palabras_tapadas, posiciones, originales)
        for ti, text in enumerate(texts):
            for v in range(n_variants):
                plan = self._plan(text, v, ti)
                if plan:
                    jobs.append((ti, v, *plan))
        out: list[list[str]] = [[] for _ in texts]
        for start in range(0, len(jobs), self.batch_size):
            batch = jobs[start:start + self.batch_size]
            for (ti, v, masked, chosen, _), new_words in zip(batch, self._fill(batch)):
                words = texts[ti].split()
                variant = words[:]
                for pos, replacement in zip(chosen, new_words):
                    if replacement:
                        pre, _, suf = _CORE.match(words[pos]).groups()
                        variant[pos] = f"{pre}{replacement}{suf}"
                joined = " ".join(variant)
                if joined != " ".join(words) and joined not in out[ti]:
                    out[ti].append(joined)
        return out

    def _fill(self, batch) -> list[list[str | None]]:
        """Para cada trabajo del lote, la palabra elegida en cada hueco (None = mantener la original)."""
        torch = self._torch
        enc = self.tokenizer([" ".join(m) for _, _, m, _, _ in batch], return_tensors="pt", padding=True,
                             truncation=True, max_length=self.max_tokens)
        with torch.no_grad():
            logits = self.model(**enc).logits
        results = []
        for row, (ti, v, masked, chosen, originals) in enumerate(batch):
            holes = (enc["input_ids"][row] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            if len(holes) != len(chosen):  # la truncacion se comio algun hueco: se descarta la variante
                results.append([None] * len(chosen))
                continue
            rng = random.Random(f"{self.seed}-{ti}-{v}-pick")
            row_out = []
            for hole, original in zip(holes, originals):
                probs = torch.softmax(logits[row, hole], dim=-1)
                top = torch.topk(probs, self.top_k)
                options = []
                for p, tok_id in zip(top.values.tolist(), top.indices.tolist()):
                    word = self.tokenizer.convert_ids_to_tokens(tok_id)
                    # Se excluye la propia palabra original: es casi siempre la candidata mas probable y no cambia nada.
                    if p >= self.min_prob and self._acceptable(word, original):
                        options.append((word, p))
                row_out.append(rng.choices([w for w, _ in options], [p for _, p in options])[0] if options else None)
            results.append(row_out)
        return results
