"""Compara estrategias de busqueda de hiperparametros sobre el problema real.

Ejecutar: python scripts/hpo_sampler_comparison.py
Requiere: pip install -e ".[hpo]"

Espacio (todo numerico, para poder comparar TODOS los metodos en igualdad):
  log10(C) en [-3, 1.5] · min_df entero en [1, 20] · ngram_max en {1, 2}
Objetivo: F1-macro medio de una CV estratificada de 5 folds con FOLDS FIJOS (objetivo determinista:
aisla la calidad del muestreador del ruido de la CV). El texto se normaliza una sola vez (es independiente
de estos hiperparametros) para que cada evaluacion cueste ~0.3 s.
Metodos: Random, TPE, CMA-ES y GPSampler (Optuna; el GPSampler requiere torch) y un proceso gaussiano +
Expected Improvement propio (scikit-learn), todos con el mismo presupuesto. Se repite con varias semillas y se compara la mejor-hasta-ahora por intento.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
from scipy.stats import norm, qmc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from hatedet.data.loader import load_raw_comments
from hatedet.nlp.cleaner import TextCleaner
from hatedet.nlp.group_masker import GroupMasker
from hatedet.nlp.normalizer import Normalizer

sys.stdout.reconfigure(encoding="utf-8")
optuna.logging.set_verbosity(optuna.logging.ERROR)

FIG_DIR = Path("reports/figures")
BUDGET, SEEDS, N_INIT = 40, 8, 10
LOGC_LO, LOGC_HI, DF_LO, DF_HI = -3.0, 1.5, 1, 20

df = load_raw_comments()
TEXTS = Normalizer(method="stem").transform(GroupMasker().transform(TextCleaner().transform(df.Text)))
Y = df.IsHatespeech.values.astype(int)
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=0).split(TEXTS, Y))
_CACHE: dict = {}


def objective_value(logc: float, min_df: int, ngram_max: int) -> float:
    key = (round(logc, 4), int(min_df), int(ngram_max))
    if key in _CACHE:
        return _CACHE[key]
    scores = []
    for tr, te in FOLDS:
        vec = TfidfVectorizer(ngram_range=(1, ngram_max), min_df=int(min_df), max_features=5000)
        xtr = vec.fit_transform([TEXTS[i] for i in tr])
        xte = vec.transform([TEXTS[i] for i in te])
        clf = LogisticRegression(C=10 ** logc, l1_ratio=1.0, solver="liblinear",
                                 class_weight="balanced", max_iter=2000, random_state=0).fit(xtr, Y[tr])
        scores.append(f1_score(Y[te], clf.predict(xte), average="macro"))
    _CACHE[key] = float(np.mean(scores))
    return _CACHE[key]


def running_best(values):
    return np.maximum.accumulate(values)


def run_optuna(sampler, budget):
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def obj(trial):
        return objective_value(
            trial.suggest_float("logc", LOGC_LO, LOGC_HI),
            trial.suggest_int("min_df", DF_LO, DF_HI),
            trial.suggest_int("ngram_max", 1, 2),
        )

    study.optimize(obj, n_trials=budget)
    return running_best([t.value for t in study.trials])


def from_unit(u):
    return LOGC_LO + u[0] * (LOGC_HI - LOGC_LO), int(round(DF_LO + u[1] * (DF_HI - DF_LO))), int(round(1 + u[2]))


def run_gp(seed, budget, n_init=N_INIT):
    rng = np.random.default_rng(seed)
    x = qmc.Sobol(3, seed=seed).random(n_init)
    y = [objective_value(*from_unit(u)) for u in x]
    while len(y) < budget:
        kernel = ConstantKernel(1.0) * Matern(length_scale=[0.3] * 3, nu=2.5) + WhiteKernel(1e-4, (1e-8, 1e-1))
        gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=2,
                                      random_state=int(rng.integers(1 << 30))).fit(x, y)
        cand = qmc.Sobol(3, seed=int(rng.integers(1 << 30))).random(2048)
        mu, sd = gp.predict(cand, return_std=True)
        z = (mu - max(y)) / np.maximum(sd, 1e-9)
        ei = (mu - max(y)) * norm.cdf(z) + sd * norm.pdf(z)
        u = cand[int(np.argmax(ei))]
        x = np.vstack([x, u])
        y.append(objective_value(*from_unit(u)))
    return running_best(y)


def landscape():
    logcs = np.round(np.arange(LOGC_LO, LOGC_HI + 1e-9, 0.25), 2)
    dfs = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
    grid = np.zeros((2, len(dfs), len(logcs)))
    for g in (1, 2):
        for i, d in enumerate(dfs):
            for j, c in enumerate(logcs):
                grid[g - 1, i, j] = objective_value(c, d, g)
    return logcs, dfs, grid


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("1) Paisaje completo (rejilla) ...", flush=True)
    logcs, dfs, grid = landscape()
    print(f"   {len(_CACHE)} evaluaciones en {time.time() - t0:.0f} s | mejor en rejilla = {grid.max():.4f}")
    total_var = grid.var()
    importance = {
        "log10(C)": grid.mean(axis=(0, 1)).var() / total_var,
        "min_df": grid.mean(axis=(0, 2)).var() / total_var,
        "ngram_max": grid.mean(axis=(1, 2)).var() / total_var,
    }
    print("   Importancia de efecto principal (varianza explicada por cada hiperparametro solo):")
    for k, v in sorted(importance.items(), key=lambda kv: -kv[1]):
        print(f"     {k:10s} {v:6.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for g, ax in enumerate(axes):
        im = ax.imshow(grid[g], origin="lower", aspect="auto", cmap="viridis", vmin=grid.min(), vmax=grid.max())
        ax.set_xticks(range(0, len(logcs), 4), [f"{c:g}" for c in logcs[::4]])
        ax.set_yticks(range(len(dfs)), dfs)
        ax.set_xlabel("log10(C)")
        ax.set_title(f"F1-macro (CV) · ngram_max={g + 1}")
    axes[0].set_ylabel("min_df")
    fig.colorbar(im, ax=axes, label="F1-macro")
    fig.savefig(FIG_DIR / "hpo_landscape.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print("2) Comparando muestreadores ...", flush=True)
    curves = {"Aleatorio": [], "TPE": [], "CMA-ES": [], "GP propio (sklearn) + EI": [], "GP de Optuna (GPSampler)": []}
    for seed in range(SEEDS):
        curves["Aleatorio"].append(run_optuna(optuna.samplers.RandomSampler(seed=seed), BUDGET))
        curves["TPE"].append(run_optuna(optuna.samplers.TPESampler(seed=seed, n_startup_trials=N_INIT), BUDGET))
        curves["CMA-ES"].append(run_optuna(optuna.samplers.CmaEsSampler(seed=seed), BUDGET))
        curves["GP propio (sklearn) + EI"].append(run_gp(seed, BUDGET))
        curves["GP de Optuna (GPSampler)"].append(
            run_optuna(optuna.samplers.GPSampler(seed=seed, n_startup_trials=N_INIT), BUDGET)
        )
        print(f"   semilla {seed + 1}/{SEEDS} hecha ({time.time() - t0:.0f} s)", flush=True)

    best_known = max(max(_CACHE.values()), grid.max())
    (FIG_DIR.parent / "hpo_results.json").write_text(json.dumps({
        "best_known": best_known, "budget": BUDGET, "seeds": SEEDS,
        "importance": importance, "curves": {k: [list(map(float, c)) for c in v] for k, v in curves.items()},
    }, indent=1))
    print(f"\nMejor F1-macro visto por cualquier metodo: {best_known:.4f}")
    print(f"Regret medio (mejor conocido - mejor encontrado) tras N intentos, {SEEDS} semillas:")
    checkpoints = [10, 20, 30, 40]
    print(f"{'metodo':26s}" + "".join(f"{f'n={n}':>12s}" for n in checkpoints))
    for name, cs in curves.items():
        arr = np.array(cs)
        print(f"{name:26s}" + "".join(f"{(best_known - arr[:, n - 1]).mean():12.4f}" for n in checkpoints))
    print("\nIntentos medios hasta llegar a 0.005 del mejor conocido (max 40 = no llego):")
    for name, cs in curves.items():
        hits = [int(np.argmax(np.array(c) >= best_known - 0.005)) + 1 if (np.array(c) >= best_known - 0.005).any() else BUDGET
                for c in cs]
        print(f"  {name:26s} {np.mean(hits):5.1f}  (seeds que llegan: {sum(np.array(c).max() >= best_known - 0.005 for c in cs)}/{SEEDS})")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"Aleatorio": "#888780", "TPE": "#1baf7a", "CMA-ES": "#eb6834",
              "GP propio (sklearn) + EI": "#2a78d6", "GP de Optuna (GPSampler)": "#6250d6"}
    x = np.arange(1, BUDGET + 1)
    for name, cs in curves.items():
        arr = np.array(cs)
        ax.plot(x, np.median(arr, axis=0), color=colors[name], lw=2, label=name)
        ax.fill_between(x, np.percentile(arr, 25, axis=0), np.percentile(arr, 75, axis=0), color=colors[name], alpha=0.12)
    ax.axhline(best_known, color="k", ls=":", lw=1)
    ax.set_xlabel("Intentos (evaluaciones de la CV)")
    ax.set_ylabel("Mejor F1-macro encontrado hasta ahora")
    ax.set_title(f"Convergencia (mediana e IQR, {SEEDS} semillas)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.savefig(FIG_DIR / "hpo_convergence.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFiguras en {FIG_DIR}/ (hpo_landscape.png, hpo_convergence.png). Total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
