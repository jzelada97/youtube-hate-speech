"""Los coeficientes vivos del modelo servido: que palabras usa de verdad para decidir.

    python scripts/show_coefficients.py

`lr_word` es el miembro con mas peso del voto (4 de 8) y es una regresion logistica sobre TF-IDF: cada palabra del
vocabulario tiene un coeficiente que dice cuanto empuja hacia "odio". La penalizacion L1, la que se uso para cerrar
el sobreajuste (docs/DECISIONES.md, seccion 7.3), lleva la mayoria a CERO exactamente. Este script muestra los que
sobreviven, que resulta que son doce de 777 (seccion 13.4).

Salidas: reports/coefficients.json y reports/figures/coefficients.png (material para la presentacion).
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

MODEL = Path("models/ensemble_v1.joblib")
JSON_PATH = Path("reports/coefficients.json")
FIG_PATH = Path("reports/figures/coefficients.png")


def main() -> None:
    ensemble = joblib.load(MODEL)
    lr = ensemble.named_estimators_["lr_word"]
    coef = lr.named_steps["clf"].coef_[0]
    names = np.array(lr.named_steps["tfidf"].get_feature_names_out())
    intercept = float(lr.named_steps["clf"].intercept_[0])

    alive = np.flatnonzero(coef)
    order = alive[np.argsort(-coef[alive])]
    print(f"Modelo servido: {ensemble.weights} como pesos del voto, lr_word con el mayor\n")
    print(f"Vocabulario: {len(coef)} variables | coeficientes NO nulos: {len(alive)} "
          f"({len(alive) / len(coef):.1%}) | el resto los anulo L1\n")
    for i in order:
        print(f"   {names[i]:12s} {coef[i]:+.4f}")
    print(f"   {'(intercepto)':12s} {intercept:+.4f}")

    JSON_PATH.parent.mkdir(exist_ok=True)
    JSON_PATH.write_text(json.dumps({
        "modelo": "ensemble_v1 / miembro lr_word",
        "pesos_del_voto": list(ensemble.weights),
        "n_variables": int(len(coef)),
        "n_no_nulos": int(len(alive)),
        "intercepto": round(intercept, 4),
        "coeficientes": {str(names[i]): round(float(coef[i]), 4) for i in order},
        "lectura": ("L1 dejo vivas 12 de 777 variables. El coeficiente mayor es `grp`, el token con el que se "
                    "sustituyen los terminos de identidad: el modelo es, en buena parte, un detector de menciones "
                    "a grupos (ver seccion 13.4)."),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    words = [str(names[i]) for i in order][::-1]
    values = [float(coef[i]) for i in order][::-1]
    colors = ["#c5221f" if v > 0 else "#1a73e8" for v in values]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bars = ax.barh(words, values, color=colors)
    # El token de identidad es el protagonista: se destaca.
    for bar, w in zip(bars, words):
        if w == "grp":
            bar.set_edgecolor("black")
            bar.set_linewidth(1.8)
    ax.axvline(0, color="#3c4043", linewidth=0.9)
    ax.set_xlabel("Coeficiente (empuja hacia «odio» →)")
    ax.set_title(f"Lo único que usa el modelo: {len(alive)} palabras de {len(coef)}\n"
                 "(la penalización L1 anuló el resto)", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    for v, w in zip(values, words):
        ax.annotate(f"{v:+.2f}", (v, w), xytext=(6 if v > 0 else -6, 0), textcoords="offset points",
                    va="center", ha="left" if v > 0 else "right", fontsize=8.5)
    ax.set_xlim(min(values) - 1.2, max(values) + 1.4)
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    print(f"\nEscritos {JSON_PATH} y {FIG_PATH}")


if __name__ == "__main__":
    main()
