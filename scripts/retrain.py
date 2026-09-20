"""Reentrena el modelo con los veredictos de la cola de revision y lo promociona SOLO si supera el filtro.

  python scripts/retrain.py [--model ensemble|baseline] [--db data/review.db] [--dry-run]

Por que un script aparte y no `train.py --con-revisiones`: reentrenar es una decision, no un paso mas.
Lo que se juega es sustituir el modelo que atiende peticiones, asi que hay un filtro explicito y el
resultado queda por escrito en reports/.

Protocolo de comparacion (el mismo rigor que el resto del informe, ver docs/DECISIONES.md seccion 4):
  - CV repetida estratificada sobre el DATASET ORIGINAL del cliente. Los folds de validacion son
    siempre datos del cliente: nunca nos examinamos con nuestras propias etiquetas de revision.
  - En cada fold se entrenan dos modelos con la MISMA receta: uno solo con el trozo original de
    entrenamiento (el actual) y otro anadiendole todas las filas revisadas (el candidato).
  - Comparacion pareada fold a fold + Wilcoxon. Asi la pregunta que se responde es exactamente
    "anadir las revisiones, mejora?", y no "he tenido suerte con la particion?".

Filtro de promocion (los tres a la vez):
  1. El modelo final cumple el requisito del cliente: gap < 5 pp (en muestra vs CV y vs test).
  2. El candidato no empeora de forma significativa en PR-AUC frente al actual.
  3. Hay al menos MIN_REVIEWS veredictos nuevos (con menos, el ruido supera a la senal).
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from hatedet.data.loader import load_raw_comments
from hatedet.data.schema import TARGET_COLUMN, TEXT_COLUMN
from hatedet.data.splitter import stratified_split
from hatedet.db.store import ReviewStore
from hatedet.models.bands import choose_thresholds
from hatedet.models.evaluate import evaluate, generalization_gap_pp
from hatedet.models.training import MODELS, fit_with_augmentation, out_of_fold_scores

MIN_REVIEWS = 20
N_SPLITS, N_REPEATS, SEED = 5, 2, 7
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")


def reviewed_frame(store: ReviewStore) -> pd.DataFrame:
    """Veredictos humanos como filas del dataset. Vacio si aun no hay ninguno."""
    rows = store.training_rows()
    if not rows:
        return pd.DataFrame(columns=[TEXT_COLUMN, TARGET_COLUMN])
    df = pd.DataFrame(rows)
    return df[[TEXT_COLUMN, TARGET_COLUMN]].drop_duplicates(subset=[TEXT_COLUMN])


def paired_comparison(original: pd.DataFrame, extra: pd.DataFrame, factory, augmentation) -> dict:
    """PR-AUC y F1-macro fold a fold, con y sin las filas revisadas. Validacion siempre con datos del cliente."""
    y = original[TARGET_COLUMN].values.astype(int)
    folds = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    actual, candidate = {"pr_auc": [], "f1": []}, {"pr_auc": [], "f1": []}

    for i, (tr, te) in enumerate(folds.split(original, y)):
        train_part, holdout = original.iloc[tr], original.iloc[te]
        y_true = holdout[TARGET_COLUMN].values.astype(int)
        for data, bucket in [(train_part, actual), (pd.concat([train_part, extra], ignore_index=True), candidate)]:
            model = fit_with_augmentation(data, seed=i, factory=factory, augmentation=augmentation)
            scores = model.predict_proba(holdout[TEXT_COLUMN])[:, 1]
            bucket["pr_auc"].append(average_precision_score(y_true, scores))
            bucket["f1"].append(f1_score(y_true, scores >= 0.5, average="macro"))

    out = {}
    for metric in ("pr_auc", "f1"):
        a, c = np.array(actual[metric]), np.array(candidate[metric])
        diff = c - a
        out[metric] = {
            "actual": round(float(a.mean()), 4), "candidato": round(float(c.mean()), 4),
            "delta": round(float(diff.mean()), 4),
            "folds_que_mejoran": f"{int((diff > 0).sum())}/{len(diff)}",
            "p_wilcoxon": round(float(wilcoxon(c, a).pvalue), 4) if np.any(diff != 0) else None,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="ensemble")
    parser.add_argument("--db", default="data/review.db", help="Cola de revision (ver db/store.py)")
    parser.add_argument("--dry-run", action="store_true", help="Evalua y escribe el informe, sin promocionar")
    args = parser.parse_args()

    version, factory, augmentation = MODELS[args.model]
    store = ReviewStore(args.db)
    extra = reviewed_frame(store)
    stats = store.stats()
    print(f"Cola de revision: {stats}")
    print(f"Veredictos utilizables: {len(extra)} (minimo para reentrenar: {MIN_REVIEWS})")

    if len(extra) < MIN_REVIEWS:
        print("\nNo hay suficientes veredictos nuevos. No se reentrena: con tan pocas filas el ruido")
        print("supera a la senal y solo se conseguiria mover el modelo sin saber si a mejor o a peor.")
        return 1

    df = load_raw_comments()
    train, val, test = stratified_split(df)
    dev = pd.concat([train, val], ignore_index=True)
    print(f"\nDesarrollo (original): {len(dev)} | Test (intacto): {len(test)}")

    print(f"\nComparacion pareada ({N_SPLITS}x{N_REPEATS} folds sobre datos del cliente)...")
    comparison = paired_comparison(dev, extra, factory, augmentation)
    for metric, r in comparison.items():
        print(f"  {metric:8s} actual {r['actual']}  candidato {r['candidato']}  "
              f"delta {r['delta']:+.4f}  {r['folds_que_mejoran']}  p={r['p_wilcoxon']}")

    dev_plus = pd.concat([dev, extra], ignore_index=True)
    oof = out_of_fold_scores(dev_plus, factory, augmentation)
    y_dev = dev_plus[TARGET_COLUMN].values.astype(int)
    cv_report = evaluate(y_dev, (oof >= 0.5).astype(int), oof)
    thresholds = choose_thresholds(y_dev, oof)

    model = fit_with_augmentation(dev_plus, seed=42, factory=factory, augmentation=augmentation)
    in_sample = evaluate(dev_plus[TARGET_COLUMN], model.predict(dev_plus[TEXT_COLUMN]),
                         model.predict_proba(dev_plus[TEXT_COLUMN])[:, 1])
    test_scores = model.predict_proba(test[TEXT_COLUMN])[:, 1]
    test_report = evaluate(test[TARGET_COLUMN], (test_scores >= 0.5), test_scores)
    gap, gap_cv = generalization_gap_pp(in_sample, test_report), generalization_gap_pp(in_sample, cv_report)
    print(f"\nCandidato: F1-macro CV {cv_report.f1_macro:.4f} | PR-AUC CV {cv_report.pr_auc:.4f}")
    print(f"Gap en muestra-test {gap:.2f} pp | en muestra-CV {gap_cv:.2f} pp (limite 5 pp)")

    pr = comparison["pr_auc"]
    empeora = pr["delta"] < 0 and pr["p_wilcoxon"] is not None and pr["p_wilcoxon"] < 0.05
    cumple_gap = gap < 5 and gap_cv < 5
    promote = cumple_gap and not empeora and not args.dry_run

    razones = []
    if not cumple_gap:
        razones.append(f"no cumple el requisito de gap ({gap:.2f} / {gap_cv:.2f} pp)")
    if empeora:
        razones.append(f"empeora el PR-AUC de forma significativa ({pr['delta']:+.4f}, p={pr['p_wilcoxon']})")
    if args.dry_run:
        razones.append("--dry-run: solo evaluacion")

    REPORTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "fecha": stamp, "modelo": version, "n_revisiones": len(extra), "cola": stats,
        "comparacion_pareada": comparison,
        "candidato": {"f1_macro_cv": cv_report.f1_macro, "pr_auc_cv": cv_report.pr_auc,
                      "f1_macro_test": test_report.f1_macro, "gap_pp_vs_test": gap, "gap_pp_vs_cv": gap_cv},
        "promocionado": promote, "razones": razones,
    }
    report_path = REPORTS_DIR / f"retrain_{stamp}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    if promote:
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(model, MODELS_DIR / f"{version}.joblib")
        meta_path = MODELS_DIR / f"{version}.metadata.json"
        metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        metadata.update({
            "model_version": version, "retrained_at": stamp, "n_dev": len(dev_plus),
            "n_human_reviews": len(extra), "thresholds": thresholds,
            "f1_macro_cv": cv_report.f1_macro, "pr_auc_cv": cv_report.pr_auc,
            "f1_macro_test": test_report.f1_macro,
            "generalization_gap_pp_in_sample_vs_test": gap,
            "generalization_gap_pp_in_sample_vs_cv": gap_cv,
            "meets_gap_requirement": True,
        })
        meta_path.write_text(json.dumps(metadata, indent=2))
        print(f"\nPROMOCIONADO. Modelo y metadata actualizados. Informe: {report_path}")
        print("Reinicia la API para que cargue el modelo nuevo.")
    else:
        print(f"\nNO promocionado: {'; '.join(razones)}. Informe: {report_path}")
        print("El modelo en produccion no se ha tocado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
