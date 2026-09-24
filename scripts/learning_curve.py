"""Curva de aprendizaje del baseline: cuanto ganariamos con mas datos del mismo tipo.

    python scripts/learning_curve.py

Es la medicion que sostiene el argumento central del proyecto (docs/DECISIONES.md, seccion 3.6): el techo es de
representacion y etiquetas, no de volumen. Hasta ahora la tabla estaba en el informe SIN codigo que la generara,
asi que no era reproducible ni verificable. Este script la regenera y la persiste.

Protocolo: CV estratificada repetida (5 folds x 3 repeticiones, semilla 42) sobre el conjunto de DESARROLLO
(train+val, 797 filas; el test no se toca). Para cada tamano se submuestrea ESTRATIFICADAMENTE el trozo de
entrenamiento de cada fold, de modo que la proporcion de odio no cambie con el tamano; el fold de validacion es
siempre completo y real. Se reportan F1-macro de validacion y de entrenamiento: la distancia entre ambas curvas
es el sobreajuste, y su altura conjunta es el techo.

Salidas: reports/learning_curve.json y reports/figures/learning_curve.png
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from hatedet.data.loader import load_raw_comments
from hatedet.data.schema import TARGET_COLUMN, TEXT_COLUMN
from hatedet.data.splitter import stratified_split
from hatedet.models.baseline import build_baseline_pipeline

sys.stdout.reconfigure(encoding="utf-8")

FRACTIONS = (0.15, 0.30, 0.50, 0.75, 1.0)
N_SPLITS, N_REPEATS, SEED = 5, 3, 42
JSON_PATH = Path("reports/learning_curve.json")
FIG_PATH = Path("reports/figures/learning_curve.png")


def subsample(idx: np.ndarray, y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Submuestreo estratificado: la tasa de odio no debe cambiar con el tamano, o no mediriamos el tamano."""
    if fraction >= 1.0:
        return idx
    keep, _ = train_test_split(idx, train_size=fraction, stratify=y[idx], random_state=seed)
    return keep


def main() -> None:
    df = load_raw_comments()
    train, val, _ = stratified_split(df)
    import pandas as pd

    dev = pd.concat([train, val], ignore_index=True)
    y = dev[TARGET_COLUMN].values.astype(int)
    texts = dev[TEXT_COLUMN].values
    print(f"Desarrollo: {len(dev)} filas, {y.sum()} de odio (el test queda intacto)\n")

    folds = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(dev, y))
    rows = []
    for fraction in FRACTIONS:
        val_scores, train_scores, sizes, positives = [], [], [], []
        for i, (tr, te) in enumerate(folds):
            sub = subsample(tr, y, fraction, seed=SEED + i)
            model = build_baseline_pipeline().fit(texts[sub], y[sub])
            val_scores.append(f1_score(y[te], model.predict(texts[te]), average="macro"))
            train_scores.append(f1_score(y[sub], model.predict(texts[sub]), average="macro"))
            sizes.append(len(sub))
            positives.append(int(y[sub].sum()))
        rows.append({
            "fraccion": fraction,
            "n_entrenamiento": int(round(np.mean(sizes))),
            "n_positivos": int(round(np.mean(positives))),
            "f1_macro_validacion": round(float(np.mean(val_scores)), 4),
            "f1_macro_entrenamiento": round(float(np.mean(train_scores)), 4),
            "desviacion_validacion": round(float(np.std(val_scores)), 4),
        })
        r = rows[-1]
        print(f"  {r['n_entrenamiento']:4d} ejemplos ({r['n_positivos']:3d} positivos): "
              f"validacion {r['f1_macro_validacion']:.4f} +- {r['desviacion_validacion']:.4f} | "
              f"entrenamiento {r['f1_macro_entrenamiento']:.4f}", flush=True)

    first, last = rows[0], rows[-1]
    ganancia = (last["f1_macro_validacion"] - first["f1_macro_validacion"]) * 100
    factor = last["n_entrenamiento"] / first["n_entrenamiento"]
    print(f"\nMultiplicar los datos por {factor:.1f} sube el F1-macro {ganancia:+.1f} pp.")

    JSON_PATH.parent.mkdir(exist_ok=True)
    JSON_PATH.write_text(json.dumps({
        "protocolo": f"CV estratificada repetida {N_SPLITS}x{N_REPEATS}, semilla {SEED}, submuestreo estratificado",
        "modelo": "build_baseline_pipeline() sin aumentacion",
        "curva": rows,
        "lectura": (f"Multiplicar los datos por {factor:.1f} sube el F1-macro {ganancia:+.1f} pp. Las curvas de "
                    "entrenamiento y validacion estan pegadas: regimen de alto sesgo, techo por representacion y "
                    "etiquetas, no por falta de datos."),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = [r["n_entrenamiento"] for r in rows]
    va = [r["f1_macro_validacion"] for r in rows]
    tr_ = [r["f1_macro_entrenamiento"] for r in rows]
    sd = [r["desviacion_validacion"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(n, tr_, "o--", color="#9aa0a6", label="Entrenamiento")
    ax.plot(n, va, "o-", color="#1a73e8", linewidth=2, label="Validación (fuera de muestra)")
    ax.fill_between(n, np.array(va) - np.array(sd), np.array(va) + np.array(sd), color="#1a73e8", alpha=0.15)
    ax.set_xlabel("Comentarios de entrenamiento")
    ax.set_ylabel("F1-macro")
    ax.set_title("Curva de aprendizaje: el techo no es el volumen de datos")
    ax.set_ylim(0.60, 0.80)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    for x, yv, p in zip(n, va, [r["n_positivos"] for r in rows]):
        ax.annotate(f"{p} pos.", (x, yv), textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=8, color="#5f6368")
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    print(f"Escritos {JSON_PATH} y {FIG_PATH}")


if __name__ == "__main__":
    main()
