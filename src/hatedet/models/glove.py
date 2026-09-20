"""Embeddings GloVe preentrenados (6B tokens, 300 dimensiones) para inicializar la red recurrente.

Se descargan una vez de Hugging Face (formato de sentence-transformers, sin dependencias extra) a
data/external/glove/ (no se versiona). Es conocimiento del lenguaje aprendido FUERA de nuestro dataset (Wikipedia y
Gigaword), NO etiquetas de odio: transforma texto, no decide la clase.
"""

import json
from pathlib import Path

import httpx
import numpy as np
import torch

GLOVE_DIR = Path("data/external/glove")
BASE_URL = ("https://huggingface.co/sentence-transformers/average_word_embeddings_glove.6B.300d/"
            "resolve/main/0_WordEmbeddings/")
FILES = {"vocab": "whitespacetokenizer_config.json", "weights": "pytorch_model.bin"}
EMB_DIM = 300

_CACHE: dict = {}


def ensure_downloaded(directory: Path = GLOVE_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        target = directory / name
        if target.exists() and target.stat().st_size > 0:
            continue
        tmp = target.with_suffix(target.suffix + ".part")
        with httpx.stream("GET", BASE_URL + name, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
        tmp.replace(target)
    return directory


def load_glove(directory: Path = GLOVE_DIR) -> tuple[dict[str, int], np.ndarray]:
    if "glove" not in _CACHE:
        d = ensure_downloaded(directory)
        words = json.loads((d / FILES["vocab"]).read_text(encoding="utf-8"))["vocab"]
        state = torch.load(d / FILES["weights"], map_location="cpu", weights_only=True)
        weights = next(v for k, v in state.items() if k.endswith("weight")).numpy()
        _CACHE["glove"] = ({w: i for i, w in enumerate(words)}, weights)
    return _CACHE["glove"]


def glove_initializer(vocab: dict[str, int], emb_dim: int) -> np.ndarray:
    """Matriz (len(vocab)+2, emb_dim) alineada con `LSTMClassifier`: fila 0 = PAD, 1 = UNK, 2.. = palabras."""
    if emb_dim != EMB_DIM:
        raise ValueError(f"GloVe tiene {EMB_DIM} dimensiones; se pidio emb_dim={emb_dim}")
    index, weights = load_glove()
    rng = np.random.default_rng(0)
    matrix = rng.normal(0, 0.1, (len(vocab) + 2, emb_dim)).astype(np.float32)
    matrix[0] = 0.0
    matrix[1] = weights.mean(axis=0)
    for word, i in vocab.items():
        j = index.get(word)
        if j is not None:
            matrix[i] = weights[j]
    return matrix


def coverage(vocab: dict[str, int]) -> float:
    index, _ = load_glove()
    return sum(w in index for w in vocab) / max(1, len(vocab))
