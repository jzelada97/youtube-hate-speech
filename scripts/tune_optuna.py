"""Optimiza con Optuna (TPE) los hiperparametros del ensemble CON RESTRICCION DE SOBREAJUSTE.
Ejecutar: python scripts/tune_optuna.py

Objetivo: maximizar el PR-AUC medio de una CV estratificada 5x2 (calidad de ranking, independiente del umbral).
Restriccion: gap medio (F1-macro en el propio entrenamiento - F1-macro en validacion, umbral 0.5) <= 5 puntos
porcentuales, el requisito del cliente. Sin la restriccion, la primera version de esta busqueda ignoraba el
sobreajuste y elegia un ensemble con 19 pp de gap. La aumentacion (EDA) se aplica solo al fold de entrenamiento.

Validacion final con particiones NUEVAS y pareada frente a (a) el baseline y (b) el ensemble por defecto, para
vigilar el "winner's curse". Regla de adopcion: gap <= 5 pp Y PR-AUC no peor que el baseline.
Salidas: conf/ensemble_params.json (parametros elegidos, versionado) y reports/optuna_ensemble.json.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from hatedet.data.augment import augment_train
from hatedet.data.loader import load_raw_comments
from hatedet.data.schema import TARGET_COLUMN, TEXT_COLUMN
from hatedet.models.baseline import build_baseline_pipeline
from hatedet.models.ensemble import DEFAULT_ENSEMBLE_PARAMS, build_ensemble

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS, GAP_LIMIT = 40, 0.05
DF = load_raw_comments()
Y = DF[TARGET_COLUMN].values.astype(int)


def fold_metrics(make_model, fold, tr, te):
    train = DF.iloc[tr]
    aug = augment_train(train, n_eda_copies=1, seed=fold)
    model = make_model().fit(aug[TEXT_COLUMN], aug[TARGET_COLUMN].astype(int))
    prob_val = model.predict_proba(DF.iloc[te][TEXT_COLUMN])[:, 1]
    prob_tr = model.predict_proba(train[TEXT_COLUMN])[:, 1]
    f1_tr = f1_score(Y[tr], prob_tr >= 0.5, average="macro")
    f1_val = f1_score(Y[te], prob_val >= 0.5, average="macro")
    return average_precision_score(Y[te], prob_val), f1_tr - f1_val, f1_val


def cv(make_model, seed, repeats):
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed).split(DF, Y))
    res = Parallel(n_jobs=-1)(delayed(fold_metrics)(make_model, i, tr, te) for i, (tr, te) in enumerate(folds))
    return np.array(res)  # columnas: pr_auc, gap, f1_val


def suggest(trial):
    return {
        "lr_word_C": trial.suggest_float("lr_word_C", 0.02, 3.0, log=True),
        "lr_word_min_df": trial.suggest_int("lr_word_min_df", 2, 10),
        "lr_word_ngram_max": trial.suggest_int("lr_word_ngram_max", 1, 2),
        "lr_char_C": trial.suggest_float("lr_char_C", 0.003, 10.0, log=True),
        "lr_char_min_df": trial.suggest_int("lr_char_min_df", 2, 10),
        "lr_char_max_features": trial.suggest_int("lr_char_max_features", 1000, 20000, log=True),
        "nb_alpha": trial.suggest_float("nb_alpha", 0.1, 10.0, log=True),
        "nb_min_df": trial.suggest_int("nb_min_df", 2, 10),
        "w_lr_word": trial.suggest_int("w_lr_word", 1, 4),
        "w_lr_char": trial.suggest_int("w_lr_char", 1, 3),
        "w_nb": trial.suggest_int("w_nb", 1, 2),
    }


def objective(trial):
    m = cv(lambda: build_ensemble(suggest(trial)), seed=0, repeats=2)
    trial.set_user_attr("gap", float(m[:, 1].mean()))
    trial.set_user_attr("constraint", float(m[:, 1].mean() - GAP_LIMIT))
    return float(m[:, 0].mean())


def main():
    Path("reports").mkdir(exist_ok=True)
    study = optuna.create_study(
        direction="maximize", study_name="ensemble_constrained", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=12,
                                           constraints_func=lambda t: [t.user_attrs["constraint"]]),
        storage="sqlite:///reports/optuna_study.db",
    )
    study.optimize(objective, n_trials=max(0, N_TRIALS - len(study.trials)))

    feasible = [t for t in study.trials if t.user_attrs.get("gap", 1) <= GAP_LIMIT and t.value is not None]
    print(f"Intentos: {len(study.trials)} | factibles (gap <= 5 pp): {len(feasible)}")
    if not feasible:
        print("Ningun intento cumple el gap: se mantiene el baseline como modelo servido.")
        adopted, best = None, None
    else:
        top = max(feasible, key=lambda t: t.value)
        best = {**DEFAULT_ENSEMBLE_PARAMS, **top.params}
        print(f"Mejor factible: PR-AUC {top.value:.4f}, gap {top.user_attrs['gap'] * 100:.1f} pp")

    default = cv(lambda: build_ensemble(DEFAULT_ENSEMBLE_PARAMS), seed=2024, repeats=4)
    baseline = cv(build_baseline_pipeline, seed=2024, repeats=4)
    print(f"[particiones nuevas 5x4] baseline: PR-AUC {baseline[:,0].mean():.4f}, gap {baseline[:,1].mean()*100:.1f} pp, F1 {baseline[:,2].mean():.3f}")
    print(f"[particiones nuevas 5x4] ensemble por defecto: PR-AUC {default[:,0].mean():.4f}, gap {default[:,1].mean()*100:.1f} pp")
    out = {"n_trials": len(study.trials), "feasible": len(feasible), "gap_limit_pp": GAP_LIMIT * 100,
           "baseline": {"pr_auc": float(baseline[:, 0].mean()), "gap_pp": float(baseline[:, 1].mean() * 100)},
           "default_ensemble": {"pr_auc": float(default[:, 0].mean()), "gap_pp": float(default[:, 1].mean() * 100)}}
    if best:
        tuned = cv(lambda: build_ensemble(best), seed=2024, repeats=4)
        d = tuned[:, 0] - baseline[:, 0]
        p = wilcoxon(d).pvalue if np.any(d) else 1.0
        ok = tuned[:, 1].mean() <= GAP_LIMIT and d.mean() >= -0.005
        print(f"[particiones nuevas 5x4] ensemble optimizado: PR-AUC {tuned[:,0].mean():.4f}, gap {tuned[:,1].mean()*100:.1f} pp, "
              f"F1 {tuned[:,2].mean():.3f} | vs baseline: {d.mean():+.4f} ({int((d > 0).sum())}/20, p={p:.3f})")
        out["tuned"] = {"params": best, "pr_auc": float(tuned[:, 0].mean()), "gap_pp": float(tuned[:, 1].mean() * 100),
                        "f1": float(tuned[:, 2].mean()), "diff_vs_baseline": float(d.mean()),
                        "folds_better": int((d > 0).sum()), "wilcoxon_p": float(p)}
        out["adopted"] = "ensemble optimizado" if ok else "baseline (el ensemble no cumple gap<=5pp o empeora el ranking)"
        Path("conf").mkdir(exist_ok=True)
        if ok:
            Path("conf/ensemble_params.json").write_text(json.dumps(best, indent=2))
    else:
        out["adopted"] = "baseline (ningun ensemble cumple el gap)"
    Path("reports/optuna_ensemble.json").write_text(json.dumps(out, indent=1))
    print("Decision:", out["adopted"])
    try:
        imp = optuna.importance.get_param_importances(study)
        print("Importancia (fANOVA):", {k: f"{v:.0%}" for k, v in list(imp.items())[:6]})
    except Exception as e:
        print("Importancia no disponible:", e)


if __name__ == "__main__":
    main()
